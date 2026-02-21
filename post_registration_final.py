import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from azure.storage.blob import BlobServiceClient
import uuid
from datetime import datetime
from dateutil.relativedelta import relativedelta
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import timedelta

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
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

AZURE_SA_NAME = os.getenv("AZURE_SA_NAME")
AZURE_SA_ACCESSKEY = os.getenv("AZURE_SA_ACCESSKEY")

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

def validate_post_registration_query(user_message: str) -> dict:
    validation_prompt = f"""
You are a query validator. Determine if this question is about POST-REGISTRATION topics ONLY.

ALLOWED topics:
- Application status (approved/rejected/pending)
- Transaction/payment history and amounts
- Document verification status (bank/mobile/Aadhaar linking)
- Beneficiary-specific database information

NOT ALLOWED topics (reject immediately):
- General scheme information or overview
- Eligibility criteria or requirements
- How to apply for the scheme
- Required documents for application
- Geographical information (districts, cities, states, addresses)
- General knowledge questions
- Other government schemes
- Any question that doesn't need the beneficiary's database records

User question: "{user_message}"

Return ONLY valid JSON with no additional text:
{{
  "is_valid": true,
  "reason": "brief reason"
}}
"""
    try:
        response = AZURE_CLIENT.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a query validator. Return only valid JSON with no markdown, no backticks, no additional text."},
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            content = content.rsplit("```", 1)[0].strip()
        result = json.loads(content)
        if "is_valid" not in result:
            return {"is_valid": True, "rejection_messages": {}}
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in validation: {e}")
        return {"is_valid": True, "rejection_messages": {}}
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return {"is_valid": True, "rejection_messages": {}}

    rejection_messages = {
        "hindi": "मैं केवल आवेदन की स्थिति, लेनदेन इतिहास और दस्तावेज़ सत्यापन में मदद कर सकता हूं। कृपया इन विषयों के बारे में पूछें।",
        "marathi": "मी फक्त अर्जाची स्थिती, व्यवहार इतिहास आणि कागदपत्र सत्यापनासाठी मदत करू शकतो. कृपया या विषयांबद्दल विचारा.",
        "english": "I can only help with application status, transaction history, and document verification. Please ask about these topics."
    }
    return {
        "is_valid": result.get("is_valid", True),
        "rejection_messages": rejection_messages
    }


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_HISTORY = {}
SESSION_CONSENT = {}


class ChatRequest(BaseModel):
    message: str
    aadhaar_number: Optional[str] = None
    session_id: str
    language: str = "english"


def call_llm(prompt: str) -> str:
    system_prompt = (
        "You are a POST-REGISTRATION database assistant for Ladli Behna Yojana.\n"
        "You can ONLY answer questions using the provided database context.\n"
        "Rules:\n"
        "- Answer ONLY from the database data provided\n"
        "- If database context is empty, say you cannot help\n"
        "- NEVER answer general knowledge questions\n"
        "- NEVER provide scheme information, eligibility criteria, or geographical data\n"
        "- Only discuss: application status, transactions, document verification\n"
        "- Answer ONLY what is explicitly asked\n"
        "- Be concise and precise\n"
        "- Do NOT add extra details unless requested\n"
        "- If the user asks whether a specific document (bank account, mobile number, or Aadhaar) "
        "is linked, seeded, updated, or verified correctly AND the user has NOT yet provided "
        "that document value, you MUST ask ONLY for that same document\n"
        "- HOWEVER, if the user HAS provided the document value in the conversation "
        "or in the current message, AND the database context contains the corresponding value, "
        "you MUST compare the user-provided value with the database value\n"
        "- If both values match, clearly confirm that it is linked correctly\n"
        "- If they do not match, clearly state that it is not linked correctly\n"
        "- Never ask again for the document once it has been provided\n"
        "- Never ask for Aadhaar when the question is about bank account or mobile number\n"
        "- Do NOT ask for last 4 digits or partial identifiers\n"
    )
    response = AZURE_CLIENT.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()


def extract_transaction_intent_llm(user_prompt: str):
    intent_prompt = f"""
Extract transaction intent from the user message.

Rules:
- transaction_flag = 1 only if the user asks about payments or transactions
- If user says "last N months", set last_n_months = N
- If user mentions individual months, use month_list
- If user mentions a range like "June to September", use start_month and end_month
- ONLY ONE of (month_list) OR (start/end) OR (last_n_months) can be non-null
- Month names must be lowercase full names
- Return STRICT JSON only

User message:
"{user_prompt}"

JSON:
{{
  "transaction_flag": 0 or 1,
  "month_list": ["june","july"] or null,
  "start_month": "june" or null,
  "end_month": "september" or null,
  "last_n_months": number or null
}}
"""
    response = AZURE_CLIENT.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": intent_prompt}
        ],
        temperature=0,
        max_tokens=200
    )
    return json.loads(response.choices[0].message.content)


def upload_chart(df: pd.DataFrame):
    print("Generating chart...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.bar(df["PaymentMonth"], df["Amount"])
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.title("Transaction History")
    plt.tight_layout()

    file_name = f"transactions_{uuid.uuid4()}.png"
    plt.savefig(file_name)
    plt.close()

    blob_service = BlobServiceClient(
        account_url=f"https://{AZURE_SA_NAME}.blob.core.windows.net",
        credential=AZURE_SA_ACCESSKEY
    )
    container_name = "charts"
    container_client = blob_service.get_container_client(container_name)
    try:
        container_client.create_container()
    except Exception:
        pass

    blob_client = container_client.get_blob_client(file_name)
    with open(file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    os.remove(file_name)

    sas_token = generate_blob_sas(
        account_name=AZURE_SA_NAME,
        container_name=container_name,
        blob_name=file_name,
        account_key=AZURE_SA_ACCESSKEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1000)
    )
    return f"https://{AZURE_SA_NAME}.blob.core.windows.net/{container_name}/{file_name}?{sas_token}"


@app.post("/post-application-chat")
def post_chat(req: ChatRequest):
    session_id = req.session_id
    user_message = req.message
    aadhaar_number = req.aadhaar_number
    language = req.language

    import re

    language_norm = (language or "english").strip().lower()
    if language_norm in ["mr", "mar", "marathi"]:
        language_norm = "marathi"
    elif language_norm in ["hi", "hindi"]:
        language_norm = "hindi"
    else:
        language_norm = "english"

    if language_norm == "english" and re.search(r"[\u0900-\u097F]", user_message or ""):
        language_norm = "marathi"
    elif language_norm == "english":
        last_history = SESSION_HISTORY.get(session_id, [])
        if last_history:
            last_turn = last_history[-1]
            last_text = f"{last_turn.get('user','')} {last_turn.get('bot','')}"
            if re.search(r"[\u0900-\u097F]", last_text):
                language_norm = "marathi"

    is_aadhaar_only = bool(re.match(r'^\d{12}$', user_message.strip()))

    if is_aadhaar_only:
        message_aadhaar = user_message.strip()
        if not aadhaar_number or aadhaar_number != message_aadhaar:
            aadhaar_number = message_aadhaar

    # ── STEP 0: DPIP CONSENT ──
    if session_id not in SESSION_CONSENT:
        SESSION_CONSENT[session_id] = {"dpip_accepted": False, "dpip_done": False}

    if not SESSION_CONSENT[session_id].get("dpip_done"):
        user_input = user_message.strip().lower()
        yes_words = ["yes", "y", "ho", "होय", "हां", "haan", "हो"]
        no_words = ["no", "n", "nahi", "नाही", "नहीं"]

        if not SESSION_CONSENT[session_id].get("dpip_accepted") and user_input not in yes_words and user_input not in no_words:
            dpip_messages = {
                "marathi": "आपली ओळख पडताळण्यासाठी आम्हाला आपली वैयक्तिक माहिती (आधार क्रमांक इ.) गोळा करणे आवश्यक आहे.\n\nही माहिती Digital Personal Data Protection Act (DPDP) नियमांनुसार सुरक्षित ठेवली जाईल.\n\nआपण माहिती सामायिक करण्यास संमती देता का?\n\nकृपया उत्तर द्या: होय / नाही",
                "hindi": "आपकी पहचान सत्यापित करने के लिए हमें आपकी व्यक्तिगत जानकारी (आधार नंबर आदि) एकत्र करनी होगी।\n\nयह जानकारी Digital Personal Data Protection Act (DPDP) नियमों के अनुसार सुरक्षित रखी जाएगी।\n\nक्या आप जानकारी साझा करने की सहमति देते हैं?\n\nकृपया उत्तर दें: हां / नहीं",
                "english": "To verify your identity, we need to collect your personal information (Aadhaar number, etc.).\n\nThis information will be kept secure according to Digital Personal Data Protection Act (DPDP) rules.\n\nDo you consent to share your information for verification?\n\nPlease answer: Yes / No"
            }
            if session_id not in SESSION_HISTORY:
                SESSION_HISTORY[session_id] = []
            return {
                "response": {
                    "response": dpip_messages.get(language_norm, dpip_messages["english"]),
                    "transaction_chart_url": None,
                    "history": []
                },
                "mode": "post_application"
            }

        elif user_input in yes_words:
            SESSION_CONSENT[session_id]["dpip_accepted"] = True
            SESSION_CONSENT[session_id]["dpip_done"] = True

        elif user_input in no_words:
            SESSION_CONSENT[session_id]["dpip_done"] = True
            end_messages = {
                "marathi": "धन्यवाद. आपण कधीही परत येऊ शकता.\n\nलाडकी बहिण योजना – आपल्या सशक्तीकरणासाठी.",
                "hindi": "धन्यवाद। आप कभी भी वापस आ सकते हैं।\n\nलाडकी बहन योजना – आपके सशक्तिकरण के लिए।",
                "english": "Thank you. You can come back anytime.\n\nLadki Bahin Yojana – For your empowerment."
            }
            if session_id not in SESSION_HISTORY:
                SESSION_HISTORY[session_id] = []
            return {
                "response": {
                    "response": end_messages.get(language_norm, end_messages["english"]),
                    "transaction_chart_url": None,
                    "history": []
                },
                "mode": "post_application"
            }

    # ── STEP 1: AADHAAR CHECK ──

    if not aadhaar_number:
        aadhaar_request_messages = {
            "marathi": "कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
            "hindi": "कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
            "english": "Please enter your 12-digit Aadhaar number."
        }
        if session_id not in SESSION_HISTORY:
            SESSION_HISTORY[session_id] = []
        return {
            "response": {
                "response": aadhaar_request_messages.get(language_norm, aadhaar_request_messages["english"]),
                "transaction_chart_url": None,
                "history": SESSION_HISTORY.get(session_id, [])[-5:]
            },
            "mode": "post_application_awaiting_aadhaar"
        }

    if not is_aadhaar_only:
        validation = validate_post_registration_query(user_message)
        if not validation["is_valid"]:
            rejection_msg = validation["rejection_messages"].get(
                language_norm, validation["rejection_messages"]["english"]
            )
            if session_id not in SESSION_HISTORY:
                SESSION_HISTORY[session_id] = []
            return {
                "response": {
                    "response": rejection_msg,
                    "transaction_chart_url": None,
                    "history": SESSION_HISTORY.get(session_id, [])[-5:]
                },
                "mode": "post_application"
            }

    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []

    db_context = ""
    chart_url = None

    beneficiary_id = get_beneficiary_by_aadhaar(aadhaar_number)
    if not beneficiary_id:
        not_found = {
            "marathi": "या आधार क्रमांकावर कोणताही अर्ज आढळला नाही. कृपया आधार क्रमांक पुन्हा तपासा. अर्ज केलेला नसेल तर कृपया प्रथम अर्ज करा.",
            "hindi": "इस आधार नंबर से कोई अर्ज नहीं मिला। एक बार चेक कर लीजिए कि नंबर सही है या नहीं। अगर अभी अर्ज नहीं किया है तो पहले वो करना होगा।",
            "english": "I couldn't find any application with this Aadhaar number. Please check if the number is correct. If you haven't applied yet, you'll need to do that first."
        }
        return {
            "response": {
                "response": not_found.get(language_norm, not_found["english"]),
                "transaction_chart_url": None,
                "history": SESSION_HISTORY[session_id]
            },
            "mode": "post_application"
        }

    print(f"Found BeneficiaryId: {beneficiary_id}")

    beneficiary = get_beneficiary_details(beneficiary_id)
    print("Beneficiary Details:", beneficiary)
    transactions = get_beneficiary_transactions(beneficiary_id)

    if is_aadhaar_only:
        status = (beneficiary or {}).get("ApplicationStatus") or "UNKNOWN"
        status_upper = str(status).upper()
        status_messages = {
            "marathi": f'तुमच्या अर्जाची स्थिती "{status_upper}" आहे.',
            "hindi": f'आपके आवेदन की स्थिति "{status_upper}" है।',
            "english": f'Your application status is "{status_upper}".'
        }
        return {
            "response": {
                "response": status_messages.get(language_norm, status_messages["english"]),
                "transaction_chart_url": None,
                "history": SESSION_HISTORY.get(session_id, [])[-5:]
            },
            "mode": "post_application"
        }

    transc_df = pd.DataFrame(transactions)

    if transc_df.empty or "TransactionDate" not in transc_df.columns:
        print(f"⚠️ No transactions found for beneficiary {beneficiary_id}")
        transc_df = pd.DataFrame(columns=["TransactionDate", "Amount", "PaymentMonth"])
    else:
        transc_df["TransactionDate"] = pd.to_datetime(transc_df["TransactionDate"])

    intent = extract_transaction_intent_llm(user_message)

    if intent["transaction_flag"] == 1:
        if intent["last_n_months"]:
            end_date = datetime.today()
            start_date = end_date - relativedelta(months=intent["last_n_months"])
            transc_df = transc_df[
                (transc_df["TransactionDate"] >= start_date) &
                (transc_df["TransactionDate"] <= end_date)
            ]
        elif intent["start_month"] and intent["end_month"]:
            sm = MONTH_MAP[intent["start_month"]]
            em = MONTH_MAP[intent["end_month"]]
            transc_df["TxnMonth"] = transc_df["TransactionDate"].dt.month
            transc_df = transc_df[
                (transc_df["TxnMonth"] >= sm) &
                (transc_df["TxnMonth"] <= em)
            ]
        elif intent["month_list"]:
            months = [MONTH_MAP[m] for m in intent["month_list"]]
            transc_df["TxnMonth"] = transc_df["TransactionDate"].dt.month
            transc_df = transc_df[transc_df["TxnMonth"].isin(months)]

        chart_url = upload_chart(transc_df)

    db_context = f"""
Beneficiary:
{beneficiary}

Transactions:
{transc_df.to_dict(orient="records")}
"""

    last_5_history = SESSION_HISTORY[session_id][-5:]

    prompt = f"""
Conversation history:
{last_5_history}

Database data:
{db_context}

User question:
{user_message}
"""

    bot_reply = call_llm(prompt)

    SESSION_HISTORY[session_id].append({
        "user": user_message,
        "bot": bot_reply
    })

    return {
        "response": {
            "response": bot_reply,
            "transaction_chart_url": chart_url,
            "history": SESSION_HISTORY[session_id][-5:]
        },
        "mode": "post_application"
    }
