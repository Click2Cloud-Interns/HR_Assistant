import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import uuid
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from database import (
    get_beneficiary_by_aadhaar,
    get_beneficiary_details,
    get_beneficiary_transactions
)

load_dotenv()

AZURE_CLIENT = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
AZURE_DEPLOYMENT    = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_SA_NAME       = os.getenv("AZURE_SA_NAME")
AZURE_SA_ACCESSKEY  = os.getenv("AZURE_SA_ACCESSKEY")

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3,  "april": 4,
    "may": 5,     "june": 6,     "july": 7,   "august": 8,
    "september": 9,"october": 10, "november": 11,"december": 12
}

# ─────────────────────────────────────────────────────────────
# Session stores  (module-level → shared across all calls)
# ─────────────────────────────────────────────────────────────
SESSION_HISTORY = {}   # sid → list of {user, bot}
SESSION_CONSENT = {}   # sid → state dict (see _init_state)

def _init_state(sid: str) -> dict:
    if sid not in SESSION_CONSENT:
        SESSION_CONSENT[sid] = {
            "dpip_done":      False,
            "dpip_accepted":  False,
            # "pending" → show Aadhaar prompt
            # "awaiting" → waiting for number/image
            # "awaiting_confirm" → waiting for yes/correction
            # "done" → fully verified
            "aadhaar_step":   "pending",
            "aadhaar_number": None,
            "aadhaar_data":   None,
        }
    if sid not in SESSION_HISTORY:
        SESSION_HISTORY[sid] = []
    return SESSION_CONSENT[sid]


# ─────────────────────────────────────────────────────────────
# Exported helper – called from main.py to SKIP the DPIP +
# Aadhaar flow when main.py has already handled those steps.
# ─────────────────────────────────────────────────────────────
def mark_dpip_done(session_id: str):
    """
    Pre-mark DPIP + Aadhaar as done.
    Call from main.py BEFORE post_chat() to bypass the internal
    DPIP consent and Aadhaar collection steps.
    """
    s = _init_state(session_id)
    s["dpip_accepted"] = True
    s["dpip_done"]     = True
    s["aadhaar_step"]  = "done"
    print(f"✅ DPIP + Aadhaar pre-marked as done for session: {session_id}")


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Request model  (used when main.py calls post_chat directly)
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:        str
    aadhaar_number: Optional[str] = None
    session_id:     str
    language:       str = "english"


# ─────────────────────────────────────────────────────────────
# Language helpers
# ─────────────────────────────────────────────────────────────
YES_WORDS        = {"yes","y","ho","होय","हां","haan","हो","hoy","ha",
                    "ok","okay","हाँ","hou","hau","haa","han","aahe","ahe",
                    "thik","theek","ji","bilkul","zaroor"}
NO_WORDS         = {"no","n","nahi","नाही","नहीं","nope"}
CORRECTION_WORDS = ["durusti","दुरुस्ती","correction","wrong","incorrect",
                    "चुकीचे","गलत","edit","change","बदल","sudhaar","सुधार"]

def _is_yes(msg: str) -> bool:
    return msg.strip().lower() in YES_WORDS

def _is_no(msg: str) -> bool:
    return msg.strip().lower() in NO_WORDS

def _is_correction(msg: str) -> bool:
    m = msg.strip().lower()
    return any(w in m for w in CORRECTION_WORDS)

def _norm_lang(language: str, user_message: str = "",
               session_history: list = []) -> str:
    lang = (language or "english").strip().lower()
    if lang in ("mr","mar","marathi"):  return "marathi"
    if lang in ("hi","hindi"):          return "hindi"
    if re.search(r"[\u0900-\u097F]", user_message or ""):
        return "marathi"
    if session_history:
        last = session_history[-1]
        combo = f"{last.get('user','')} {last.get('bot','')}"
        if re.search(r"[\u0900-\u097F]", combo):
            return "marathi"
    return "english"


# ─────────────────────────────────────────────────────────────
# OCR helpers
# ─────────────────────────────────────────────────────────────
def _ocr(content: bytes, ext: str) -> str:
    try:
        from utils import extract_text_from_bytes
        return extract_text_from_bytes(content, ext)
    except Exception as e:
        print(f"❌ OCR error: {e}")
        return ""


def _has_both_sides(text: str) -> bool:
    """Return True only if OCR text contains indicators from BOTH front AND back."""
    t = text.lower()
    front = [
        bool(re.search(r'\d{4}\s?\d{4}\s?\d{4}', text)),
        bool(re.search(r'\b(dob|date of birth|जन्म|born|year of birth|yob)\b', t)),
        bool(re.search(r'\b(male|female|पुरुष|महिला|transgender)\b', t)),
    ]
    back = [
        bool(re.search(r'\b(address|पत्ता|address)\b', t)),
        bool(re.search(r'\b(s/o|d/o|w/o|c/o|son of|daughter of|wife of)\b', t)),
        bool(re.search(r'\b(pin|pincode|dist|district|state|taluka|village|po )\b', t)),
        bool(re.search(r'\b(uidai|unique identification|माझी ओळख|qr|barcode)\b', t)),
    ]
    has_front = sum(front) >= 2
    has_back  = sum(back)  >= 1
    print(f"  front={sum(front)}/3  back={sum(back)}/4")
    return has_front and has_back


def _parse_aadhaar(text: str) -> dict:
    data = {}

    try:
        from utils import (
            extract_aadhaar_front_details,
            extract_aadhaar_back_details
        )

        front_data = extract_aadhaar_front_details(text)
        back_data  = extract_aadhaar_back_details(text)

        # Aadhaar number
        if front_data.get("AadhaarNo"):
            data["aadhaar_number"] = front_data.get("AadhaarNo")

        # Name
        if front_data.get("FullName"):
            data["full_name"] = front_data.get("FullName")

        # DOB
        dob_obj = front_data.get("DateOfBirth")
        if dob_obj:
            if hasattr(dob_obj, "strftime"):
                data["dob"] = dob_obj.strftime("%d/%m/%Y")
            else:
                data["dob"] = str(dob_obj)

        # Address
        if back_data.get("Address"):
            data["address"] = back_data.get("Address")

        print("Structured Aadhaar Extracted:")
        print(data)

    except Exception as e:
        print("❌ Structured Aadhaar extraction failed:", e)

    return data
# ─────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────
def _validate_query(user_message: str) -> dict:
    prompt = f"""
You are a query validator for POST-REGISTRATION topics only.
ALLOWED: application status, transaction/payment history, document verification.
NOT ALLOWED: general scheme info, eligibility, how to apply, geography, general knowledge.
User question: "{user_message}"
Return ONLY valid JSON: {{"is_valid": true, "reason": "brief"}}
"""
    try:
        r = AZURE_CLIENT.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system",  "content": "Return only valid JSON, no markdown."},
                {"role": "user",    "content": prompt}
            ],
            temperature=0, max_tokens=100
        )
        content = r.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n",1)[1] if "\n" in content else content[3:]
            content = content.rsplit("```",1)[0].strip()
        result = json.loads(content)
    except Exception:
        return {"is_valid": True, "rejection_messages": {}}

    if "is_valid" not in result:
        return {"is_valid": True, "rejection_messages": {}}

    rejection_messages = {
        "hindi":   "मैं केवल आवेदन की स्थिति, लेनदेन इतिहास और दस्तावेज़ सत्यापन में मदद कर सकता हूं।",
        "marathi": "मी फक्त अर्जाची स्थिती, व्यवहार इतिहास आणि कागदपत्र सत्यापनासाठी मदत करू शकतो.",
        "english": "I can only help with application status, transaction history, and document verification."
    }
    return {"is_valid": result.get("is_valid", True),
            "rejection_messages": rejection_messages}


def _call_llm(prompt: str) -> str:
    system = (
        "You are a POST-REGISTRATION database assistant for Ladli Behna Yojana.\n"
        "Answer ONLY from the database data provided. Rules:\n"
        "- NEVER answer general knowledge questions\n"
        "- Only discuss: application status, transactions, document verification\n"
        "- Be concise. Do NOT add extra details unless requested\n"
        "- If user asks about a linked document and hasn't provided its value, ask for it\n"
        "- If user provides the value, compare with DB and state match/mismatch clearly\n"
        "- Never ask for last 4 digits or partial identifiers\n"
    )
    r = AZURE_CLIENT.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"system","content":system},
                  {"role":"user",  "content":prompt}],
        temperature=0.3, max_tokens=800
    )
    return r.choices[0].message.content.strip()


def _txn_intent(msg: str) -> dict:
    prompt = f"""
Extract transaction intent. Return strict JSON only.
Rules: transaction_flag=1 only if user asks about payments/transactions.
User: "{msg}"
JSON: {{"transaction_flag":0,"month_list":null,"start_month":null,"end_month":null,"last_n_months":null}}
"""
    r = AZURE_CLIENT.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"system","content":"Return valid JSON only."},
                  {"role":"user",  "content":prompt}],
        temperature=0, max_tokens=200
    )
    return json.loads(r.choices[0].message.content)


def _upload_chart(df: pd.DataFrame) -> str:
    plt.figure(figsize=(10,5))
    plt.bar(df["PaymentMonth"], df["Amount"])
    plt.xticks(rotation=45); plt.xlabel("Month"); plt.ylabel("Amount")
    plt.title("Transaction History"); plt.tight_layout()
    fname = f"transactions_{uuid.uuid4()}.png"
    plt.savefig(fname); plt.close()
    svc = BlobServiceClient(
        account_url=f"https://{AZURE_SA_NAME}.blob.core.windows.net",
        credential=AZURE_SA_ACCESSKEY
    )
    cc = svc.get_container_client("charts")
    try: cc.create_container()
    except: pass
    with open(fname,"rb") as d:
        cc.get_blob_client(fname).upload_blob(d, overwrite=True)
    os.remove(fname)
    sas = generate_blob_sas(
        account_name=AZURE_SA_NAME, container_name="charts", blob_name=fname,
        account_key=AZURE_SA_ACCESSKEY, permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow()+timedelta(hours=1000)
    )
    return f"https://{AZURE_SA_NAME}.blob.core.windows.net/charts/{fname}?{sas}"


# ─────────────────────────────────────────────────────────────
# Core DB query + LLM answer
# ─────────────────────────────────────────────────────────────
def _db_answer(sid: str, user_msg: str, aadhaar: str,
               lang: str, is_aadhaar_only: bool) -> dict:
    hist = SESSION_HISTORY.setdefault(sid, [])
    chart_url = None

    bid = get_beneficiary_by_aadhaar(aadhaar)
    if not bid:
        nf = {
            "marathi": "या आधार क्रमांकावर कोणताही अर्ज आढळला नाही. कृपया आधार क्रमांक पुन्हा तपासा.",
            "hindi":   "इस आधार नंबर से कोई अर्ज नहीं मिला। नंबर सही है या नहीं चेक करें।",
            "english": "No application found with this Aadhaar number. Please verify it."
        }
        return {"response":{"response":nf.get(lang,nf["english"]),
                            "transaction_chart_url":None,"history":hist},
                "mode":"post_application"}

    ben  = get_beneficiary_details(bid)
    txns = get_beneficiary_transactions(bid)

    if is_aadhaar_only:
        status = (ben or {}).get("ApplicationStatus","UNKNOWN")
        sm = {"marathi": f'तुमच्या अर्जाची स्थिती "{str(status).upper()}" आहे.',
              "hindi":   f'आपके आवेदन की स्थिति "{str(status).upper()}" है।',
              "english": f'Your application status is "{str(status).upper()}".' }
        return {"response":{"response":sm.get(lang,sm["english"]),
                            "transaction_chart_url":None,
                            "history":hist[-5:]},"mode":"post_application"}

    df = pd.DataFrame(txns)
    if df.empty or "TransactionDate" not in df.columns:
        df = pd.DataFrame(columns=["TransactionDate","Amount","PaymentMonth"])
    else:
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

    intent = _txn_intent(user_msg)
    if intent["transaction_flag"] == 1:
        if intent["last_n_months"]:
            ed = datetime.today()
            sd = ed - relativedelta(months=intent["last_n_months"])
            df = df[(df["TransactionDate"]>=sd)&(df["TransactionDate"]<=ed)]
        elif intent["start_month"] and intent["end_month"]:
            sm2 = MONTH_MAP[intent["start_month"]]
            em  = MONTH_MAP[intent["end_month"]]
            df["TxnMonth"] = df["TransactionDate"].dt.month
            df = df[(df["TxnMonth"]>=sm2)&(df["TxnMonth"]<=em)]
        elif intent["month_list"]:
            months = [MONTH_MAP[m] for m in intent["month_list"]]
            df["TxnMonth"] = df["TransactionDate"].dt.month
            df = df[df["TxnMonth"].isin(months)]
        if not df.empty:
            chart_url = _upload_chart(df)

    db_ctx = f"Beneficiary:\n{ben}\n\nTransactions:\n{df.to_dict(orient='records')}"
    prompt  = f"Conversation history:\n{hist[-5:]}\n\nDatabase:\n{db_ctx}\n\nQuestion:\n{user_msg}"
    reply   = _call_llm(prompt)
    hist.append({"user":user_msg,"bot":reply})

    return {"response":{"response":reply,"transaction_chart_url":chart_url,
                        "history":hist[-5:]},"mode":"post_application"}


# ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
#  CORE LOGIC FUNCTION  ← called by both post_chat() and
#                          the HTTP endpoint
# ══════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────
def _handle_post_chat(
    sid:     str,
    msg:     str,
    lang:    str,
    passed_aadhaar: Optional[str] = None,
    file_bytes:     Optional[bytes] = None,
    file_ext:       Optional[str]   = None,
) -> dict:
    """
    Central logic for post-registration chat.
    Called by:
      - post_chat(ChatRequest)  — from main.py (no file)
      - post_application_chat_form() — HTTP endpoint with optional file
    """
    state = _init_state(sid)

    def _r(text: str, mode: str = "post_application") -> dict:
        return {
            "response": {
                "response": text,
                "transaction_chart_url": None,
                "history": SESSION_HISTORY.get(sid, [])[-5:]
            },
            "mode": mode
        }

    # ══════════════════════════════════════════════════════════
    # FAST PATH — called from main.py after mark_dpip_done()
    # Both dpip_done AND aadhaar_step=="done" are pre-set.
    # ══════════════════════════════════════════════════════════
    if state["dpip_done"] and state["aadhaar_step"] == "done":
        aadhaar = passed_aadhaar or state.get("aadhaar_number")
        if not aadhaar:
            return _r({
                "marathi": "कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi":   "कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Please enter your 12-digit Aadhaar number."
            }.get(lang,"Please enter your 12-digit Aadhaar number."),
            mode="post_application_awaiting_aadhaar")

        is12 = bool(re.match(r'^\d{12}$', msg.strip()))
        if not is12:
            v = _validate_query(msg)
            if not v["is_valid"]:
                return _r(v["rejection_messages"].get(lang, v["rejection_messages"]["english"]))
        return _db_answer(sid, msg, aadhaar, lang, is12)

    # ══════════════════════════════════════════════════════════
    # STEP 0 — DPIP CONSENT
    # ══════════════════════════════════════════════════════════
    if not state["dpip_done"]:
        ui = msg.strip().lower()

        if not state["dpip_accepted"] and not _is_yes(ui) and not _is_no(ui):
            # Show consent message
            dpip = {
                "marathi": (
                    "आपली ओळख पडताळण्यासाठी आम्हाला आपली वैयक्तिक माहिती "
                    "(आधार क्रमांक इ.) गोळा करणे आवश्यक आहे.\n\n"
                    "ही माहिती Digital Personal Data Protection Act (DPDP) "
                    "नियमांनुसार सुरक्षित ठेवली जाईल.\n\n"
                    "आपण माहिती सामायिक करण्यास संमती देता का?\n\n"
                    "कृपया उत्तर द्या: होय / नाही"
                ),
                "hindi": (
                    "आपकी पहचान सत्यापित करने के लिए हमें आपकी व्यक्तिगत जानकारी "
                    "(आधार नंबर आदि) एकत्र करनी होगी।\n\n"
                    "यह जानकारी DPDP नियमों के अनुसार सुरक्षित रखी जाएगी।\n\n"
                    "क्या आप जानकारी साझा करने की सहमति देते हैं?\n\n"
                    "कृपया उत्तर दें: हां / नहीं"
                ),
                "english": (
                    "To verify your identity we need to collect your personal information "
                    "(Aadhaar number, etc.).\n\n"
                    "This will be kept secure per DPDP rules.\n\n"
                    "Do you consent to share your information?\n\n"
                    "Please answer: Yes / No"
                )
            }
            return _r(dpip.get(lang, dpip["english"]), mode="post_application_dpip")

        if _is_yes(ui):
            state["dpip_accepted"] = True
            state["dpip_done"]     = True
            state["aadhaar_step"]  = "awaiting"
            # fall through to STEP 1

        elif _is_no(ui):
            state["dpip_done"] = True
            bye = {
                "marathi": "धन्यवाद. आपण कधीही परत येऊ शकता.\n\nलाडकी बहिण योजना – आपल्या सशक्तीकरणासाठी.",
                "hindi":   "धन्यवाद। आप कभी भी वापस आ सकते हैं।\n\nलाडकी बहन योजना – आपके सशक्तिकरण के लिए।",
                "english": "Thank you. You can come back anytime.\n\nLadki Bahin Yojana – For your empowerment."
            }
            return _r(bye.get(lang, bye["english"]))

    # ══════════════════════════════════════════════════════════
    # STEP 1 — AADHAAR COLLECTION
    # ══════════════════════════════════════════════════════════
    if state["dpip_done"] and state["aadhaar_step"] == "awaiting":

        aadhaar_prompt = {
            "marathi": (
                "अर्जाची स्थिती तपासण्यासाठी कृपया खालीलपैकी एक करा:\n\n"
                "• आपला १२ अंकी आधार क्रमांक टाइप करा\n"
                "• किंवा आधार कार्डचा फोटो अपलोड करा\n\n"
                "📸 फोटो अपलोड करताना:\n"
                "आधार कार्डची पुढची (Front) आणि मागची (Back) बाजू "
                "एकाच फोटोमध्ये एकत्र करून अपलोड करा."
            ),
            "hindi": (
                "अर्ज की स्थिति जांचने के लिए कृपया निम्नलिखित में से एक करें:\n\n"
                "• अपना १२ अंकी आधार नंबर टाइप करें\n"
                "• या आधार कार्ड का फोटो अपलोड करें\n\n"
                "📸 फोटो अपलोड करते समय:\n"
                "आधार कार्ड का आगे (Front) और पीछे (Back) दोनों तरफ "
                "एक ही फोटो में अपलोड करें।"
            ),
            "english": (
                "To check your application status, please do one of the following:\n\n"
                "• Type your 12-digit Aadhaar number\n"
                "• Or upload a photo of your Aadhaar card\n\n"
                "📸 When uploading:\n"
                "Include BOTH Front and Back sides in a single image."
            )
        }

        # ── File uploaded ──
        if file_bytes and file_ext:
            if file_ext not in (".jpg",".jpeg",".png",".pdf"):
                return _r({
                    "marathi": "❌ अवैध फाइल प्रकार. फक्त JPG, PNG किंवा PDF अपलोड करा.",
                    "hindi":   "❌ अमान्य फ़ाइल प्रकार। केवल JPG, PNG या PDF।",
                    "english": "❌ Invalid file type. Upload JPG, PNG or PDF only."
                }.get(lang,"Invalid file type."), mode="post_application_awaiting_aadhaar")

            raw = _ocr(file_bytes, file_ext)
            print(f"OCR (first 300): {raw[:300]}")

            if not raw.strip():
                return _r({
                    "marathi": "📸 फोटो अस्पष्ट आहे. कृपया स्पष्ट फोटो अपलोड करा.",
                    "hindi":   "📸 फोटो अस्पष्ट है। स्पष्ट फोटो अपलोड करें।",
                    "english": "📸 The photo is unclear. Please upload a clearer image."
                }.get(lang), mode="post_application_awaiting_aadhaar")

            if not _has_both_sides(raw):
                return _r({
                    "marathi": (
                        "❌ फक्त आधार कार्डची एक बाजू आढळली.\n\n"
                        "📸 कृपया आधार कार्डची पुढची (Front) आणि मागची (Back) "
                        "दोन्ही बाजू एकाच फोटोमध्ये अपलोड करा."
                    ),
                    "hindi": (
                        "❌ आधार कार्ड का केवल एक तरफ मिला।\n\n"
                        "📸 कृपया Front और Back दोनों तरफ एक ही फोटो में अपलोड करें।"
                    ),
                    "english": (
                        "❌ Only one side of the Aadhaar card was detected.\n\n"
                        "📸 Please upload a single image with BOTH Front and Back sides."
                    )
                }.get(lang), mode="post_application_awaiting_aadhaar")

            data = _parse_aadhaar(raw)
            if not data.get("aadhaar_number"):
                return _r({
                    "marathi": "❌ फोटोमध्ये आधार क्रमांक सापडला नाही. स्पष्ट फोटो अपलोड करा किंवा क्रमांक टाइप करा.",
                    "hindi":   "❌ फ़ोटो में आधार नंबर नहीं मिला। स्पष्ट फ़ोटो अपलोड करें या नंबर टाइप करें।",
                    "english": "❌ Aadhaar number not found in photo. Upload a clearer image or type your number."
                }.get(lang), mode="post_application_awaiting_aadhaar")

            state["aadhaar_number"] = data["aadhaar_number"]
            state["aadhaar_data"]   = data
            state["aadhaar_step"]   = "awaiting_confirm"

            num  = data["aadhaar_number"]
            name = data.get("full_name","—")
            dob  = data.get("dob","—")
            address = data.get("address","—")

            confirm = {
                "marathi": (
                    f"आधार तपशील प्राप्त झाले:\n\n"
                    f" आधार क्रमांक: XXXX-XXXX-{num[-4:]}\n"
                    f" नाव: {name}\n"
                    f" जन्मतारीख: {dob}\n"
                    f"पत्ता: {address}\n\n"
                    f"ही माहिती योग्य आहे का?\n"
                    f"योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:"
                ),
                "hindi": (
                    f"आधार विवरण प्राप्त हुआ:\n\n"
                    f" आधार नंबर: XXXX-XXXX-{num[-4:]}\n"
                    f" नाम: {name}\n"
                    f" जन्म तिथि: {dob}\n"
                    f" पता: {address}\n\n"
                    f"क्या यह सही है?\n"
                    f"सही हो तो 'हां', बदलाव के लिए 'सुधार' टाइप करें:"
                ),
                "english": (
                    f"Aadhaar details received:\n\n"
                    f" Aadhaar: XXXX-XXXX-{num[-4:]}\n"
                    f" Name: {name}\n"
                    f" DOB: {dob}\n"
                    f" Address: {address}\n\n"
                    f"Is this correct?\n"
                    f"Type 'Yes' to confirm or 'Correction' to re-enter:"
                )
            }
            return _r(confirm.get(lang, confirm["english"]),
                      mode="post_application_aadhaar_confirm")

        # ── Typed 12-digit number ──
        m12 = re.fullmatch(r"\d{12}", msg.strip())
        if m12:
            num = m12.group()
            state["aadhaar_number"] = num
            state["aadhaar_data"]   = {"aadhaar_number": num}
            state["aadhaar_step"]   = "awaiting_confirm"

            confirm = {
                "marathi": (
                    f"आधार क्रमांक नोंदवला: XXXX-XXXX-{num[-4:]}\n\n"
                    f"हा क्रमांक योग्य आहे का?\n"
                    f"'होय' किंवा 'दुरुस्ती' टाइप करा:"
                ),
                "hindi": (
                    f"आधार नंबर दर्ज किया: XXXX-XXXX-{num[-4:]}\n\n"
                    f"क्या यह सही है?\n"
                    f"'हां' या 'सुधार' टाइप करें:"
                ),
                "english": (
                    f"Aadhaar recorded: XXXX-XXXX-{num[-4:]}\n\n"
                    f"Is this correct?\n"
                    f"Type 'Yes' to confirm or 'Correction' to re-enter:"
                )
            }
            return _r(confirm.get(lang, confirm["english"]),
                      mode="post_application_aadhaar_confirm")

        # ── Neither ── show the prompt
        return _r(aadhaar_prompt.get(lang, aadhaar_prompt["english"]),
                  mode="post_application_awaiting_aadhaar")

    # ══════════════════════════════════════════════════════════
    # STEP 2 — AADHAAR CONFIRMATION
    # ══════════════════════════════════════════════════════════
    if state["dpip_done"] and state["aadhaar_step"] == "awaiting_confirm":
        ui = msg.strip().lower()

        if _is_yes(ui):
            state["aadhaar_step"] = "done"
            aadhaar = state["aadhaar_number"]
            return _db_answer(sid, "check application status", aadhaar, lang, True)

        if _is_correction(ui):
            # Move to field-level correction — keep existing data, just ask which field
            state["aadhaar_step"] = "awaiting_field_correction"
            ask_field = {
                "marathi": "कृपया कोणती माहिती दुरुस्त करायची आहे ते सांगा (नाव / जन्मतारीख / पत्ता):",
                "hindi":   "कृपया बताएं कौन सी जानकारी सुधारनी है (नाम / जन्मतिथि / पता):",
                "english": "Please tell us which detail to correct (name / dob / address):"
            }
            return _r(ask_field.get(lang, ask_field["english"]),
                      mode="post_application_field_correction")   
        ask = {
            "marathi": "कृपया 'होय' किंवा 'दुरुस्ती' असे उत्तर द्या.",
            "hindi":   "कृपया 'हां' या 'सुधार' में उत्तर दें।",
            "english": "Please reply 'Yes' to confirm or 'Correction' to re-enter."
        }
        return _r(ask.get(lang, ask["english"]),
                  mode="post_application_aadhaar_confirm")

    # ══════════════════════════════════════════════════════════
    # STEP 2.5 — FIELD-LEVEL CORRECTION
    # ══════════════════════════════════════════════════════════
    if state["dpip_done"] and state["aadhaar_step"] == "awaiting_field_correction":
        data = state.get("aadhaar_data", {}) or {}
        ui   = msg.strip().lower()

        # Sub-step A: which field to correct?
        if not state.get("_correction_field"):
            if any(w in ui for w in ["name", "नाव", "naam"]):
                state["_correction_field"] = "full_name"
                ask = {
                    "marathi": "कृपया योग्य नाव टाइप करा:",
                    "hindi":   "कृपया सही नाम टाइप करें:",
                    "english": "Please type the correct name:"
                }
                return _r(ask.get(lang, ask["english"]), mode="post_application_field_correction")  # ← CHANGED

            elif any(w in ui for w in ["dob", "date", "जन्म", "birth", "जन्मतारीख"]):
                state["_correction_field"] = "dob"
                ask = {
                    "marathi": "कृपया योग्य जन्मतारीख टाइप करा (DD/MM/YYYY):",
                    "hindi":   "कृपया सही जन्मतिथि टाइप करें (DD/MM/YYYY):",
                    "english": "Please type the correct date of birth (DD/MM/YYYY):"
                }
                return _r(ask.get(lang, ask["english"]), mode="post_application_field_correction")  # ← CHANGED

            elif any(w in ui for w in ["address", "पत्ता", "पता", "addr"]):
                state["_correction_field"] = "address"
                ask = {
                    "marathi": "कृपया योग्य पत्ता टाइप करा:",
                    "hindi":   "कृपया सही पता टाइप करें:",
                    "english": "Please type the correct address:"
                }
                return _r(ask.get(lang, ask["english"]), mode="post_application_field_correction")  # ← CHANGED

            else:
                ask_again = {
                    "marathi": "कृपया स्पष्ट करा: नाव / जन्मतारीख / पत्ता यापैकी काय दुरुस्त करायचे?",
                    "hindi":   "कृपया स्पष्ट करें: नाम / जन्मतिथि / पता में से क्या सुधारना है?",
                    "english": "Please clarify: which field — name / dob / address?"
                }
                return _r(ask_again.get(lang, ask_again["english"]), mode="post_application_field_correction")  # ← CHANGED

        # Sub-step B: user typed the corrected value
        else:
            field = state["_correction_field"]
            data[field] = msg.strip()
            state["aadhaar_data"] = data
            state["_correction_field"] = None
            state["aadhaar_step"] = "awaiting_confirm"

            num  = data.get("aadhaar_number", "")
            name = data.get("full_name", "—")
            dob  = data.get("dob", "—")
            addr = data.get("address", "—")

            confirm = {
                "marathi": (
                    f"आधार तपशील:\n\n"
                    f"नाव: {name}\n"
                    f"जन्मतारीख: {dob}\n"
                    f"पत्ता: {addr}\n\n"
                    f"ही माहिती योग्य आहे का?\n"
                    f"योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:"
                ),
                "hindi": (
                    f"आधार विवरण:\n\n"
                    f"नाम: {name}\n"
                    f"जन्मतिथि: {dob}\n"
                    f"पता: {addr}\n\n"
                    f"क्या यह सही है?\n"
                    f"'हां' या 'सुधार' टाइप करें:"
                ),
                "english": (
                    f"Aadhaar details:\n\n"
                    f"Name: {name}\n"
                    f"DOB: {dob}\n"
                    f"Address: {addr}\n\n"
                    f"Is this correct?\n"
                    f"Type 'Yes' to confirm or 'Correction' to re-enter:"
                )
            }
            return _r(confirm.get(lang, confirm["english"]), mode="post_application_aadhaar_confirm")  # ← back to confirm mode after correction
        
    # ══════════════════════════════════════════════════════════
    # STEP 3 — ONGOING CONVERSATION (aadhaar_step == "done")
    # ══════════════════════════════════════════════════════════
    if state["dpip_done"] and state["aadhaar_step"] == "done":
        aadhaar = passed_aadhaar or state.get("aadhaar_number")
        if not aadhaar:
            return _r("Please provide your Aadhaar number.",
                      mode="post_application_awaiting_aadhaar")
        is12 = bool(re.match(r'^\d{12}$', msg.strip()))
        if not is12:
            v = _validate_query(msg)
            if not v["is_valid"]:
                return _r(v["rejection_messages"].get(lang, v["rejection_messages"]["english"]))
        return _db_answer(sid, msg, aadhaar, lang, is12)

    # Should never reach here
    return _r("Something went wrong. Please try again.")


# ─────────────────────────────────────────────────────────────
# PUBLIC FUNCTION — called from main.py with ChatRequest
# This is a SYNCHRONOUS wrapper so main.py can call it normally.
# ─────────────────────────────────────────────────────────────
def post_chat(req: ChatRequest) -> dict:
    """
    Called synchronously from main.py:
        from api.post_registration import post_chat, ChatRequest
        res = post_chat(ChatRequest(...))
    """
    sid  = req.session_id
    msg  = req.message
    lang = _norm_lang(req.language, msg, SESSION_HISTORY.get(sid, []))
    return _handle_post_chat(
        sid=sid, msg=msg, lang=lang,
        passed_aadhaar=req.aadhaar_number,
        file_bytes=None, file_ext=None
    )


# ─────────────────────────────────────────────────────────────
# HTTP ENDPOINT — direct API calls (with optional file upload)
# ─────────────────────────────────────────────────────────────
@app.post("/post-application-chat")
async def post_application_chat_form(
    message:        str            = Form(...),
    session_id:     str            = Form(...),
    language:       str            = Form("english"),
    aadhaar_number: Optional[str]  = Form(None),
    file:           Optional[UploadFile] = File(None),
):
    sid  = session_id
    msg  = message
    lang = _norm_lang(language, msg, SESSION_HISTORY.get(sid, []))

    file_bytes = None
    file_ext   = None
    if file and file.filename:
        file_bytes = await file.read()
        file_ext   = os.path.splitext(file.filename)[1].lower()

    return await _handle_post_chat(
        sid=sid, msg=msg, lang=lang,
        passed_aadhaar=aadhaar_number,
        file_bytes=file_bytes, file_ext=file_ext
    )
