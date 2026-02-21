"""
Ladki Bahin Yojana - Eligibility Agent FastAPI Backend
POC Demo for Maharashtra Government Scheme
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any, List, Set
from openai import AzureOpenAI
import requests
from dotenv import load_dotenv
from eligibility_rules import ELIGIBILITY_RULES, ELIGIBILITY_QUESTIONS
import asyncio
import audioop
import base64
from datetime import datetime, date
import json
import os
import traceback
from typing import Optional
from urllib.parse import parse_qs
from starlette.websockets import WebSocketState, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, Request, HTTPException, UploadFile, File, Form
from pathlib import Path
from starlette.responses import HTMLResponse
from plivo import plivoxml
from config import create_azure_speech_recognizer, azure_text_to_speech
from database import get_user_by_phone
from models import (
    ChatRequest,
    ChatResponse,
)
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

app = FastAPI(
    title="Ladki Bahin Yojana - Eligibility Agent API",
    description="AI-powered eligibility checker for Maharashtra's Mukhyamantri Majhi Ladki Bahin Yojana",
    version="1.0.0"
)

# CORS - Allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Store conversation sessions (in-memory for POC)
sessions: Dict[str, Dict] = {}
HOST_URL = os.getenv('HOST_URL', 'wss://your-domain.com')

voice_sessions = {}

call_center_clients: Set[WebSocket] = set()

class ConnectCallModel(BaseModel):
    ActionMode: int
    UserID: int
    TokenID: Optional[int] = None

# System prompt for the Eligibility Agent
SYSTEM_PROMPT = """You are the "Ladki Bahin Eligibility Agent" (लाडकी बहीण पात्रता सहाय्यक) - an AI assistant for Maharashtra's Mukhyamantri Majhi Ladki Bahin Yojana scheme.

Your capabilities:
1. Check eligibility for the scheme based on user's information
2. Provide step-by-step guidance for application
3. Answer questions about the scheme in English, Hindi, or Marathi
4. Explain eligibility and ineligibility criteria clearly

SCHEME RULES:
- Monthly Benefit: ₹1,500 (proposed increase to ₹2,100)
- Age: 21-65 years
- Gender: Female only
- Residency: Maharashtra permanent resident
- Income: Annual family income ≤ ₹2.5 lakh
- Marital Status: Married, Widowed, Divorced, Abandoned, or Unmarried (1 per family)
- Max 2 women per household can receive benefits

INELIGIBILITY (if ANY apply, person is NOT eligible):
- Any family member pays income tax
- Any family member is permanent government employee
- Any family member receives government pension
- Any family member is MP/MLA/Board Chairman/Director
- Family owns four-wheeler (tractor exempted)
- Already receiving ₹1,500+ from another government scheme

REQUIRED DOCUMENTS:
- Aadhaar Card (linked to bank & mobile)
- Bank Passbook (own account, not joint, DBT enabled)
- Passport Photo
- Residency Proof (Domicile/15-yr old Ration Card/Voter ID)
- Income Certificate (only for white ration card holders)

BEHAVIOR:
1. ENTIRE response must be in user's language (English/Hindi/Marathi) - including verdict
2. Be EXTREMELY concise - max 2-3 short sentences or bullet points
3. NO greetings, pleasantries, or filler words
4. NO explanations unless user explicitly asks "why" or "explain"
5. Ask ONE eligibility question at a time
6. Use conversational, friendly, LOCAL Marathi - NOT formal/robotic Marathi

MARATHI LANGUAGE STYLE:
- Use colloquial, everyday Marathi that people actually speak
- Prefer: "तुमच्या घरी" over "तुमच्या कुटुंबात"
- Prefer: "आहे का?" over "आहे की नाही?"
- Prefer: "मिळतंय का?" over "प्राप्त होत आहे का?"
- Prefer: "भरतो का?" over "भरतात का?"
- Use natural contractions and casual phrasing
- Sound like a helpful neighbor, not a government form

CRITICAL TRANSLATION RULES FOR MARATHI:
- For "unmarried/single" marital status, ALWAYS use "अविवाहित" (avivahit)
- NEVER use "अपूर्ण" (apurna) or "परिणीत" (parinit) for unmarried status
- Use only these exact terms for marital status:
  * Unmarried/Single = अविवाहित (avivahit)
  * Married = विवाहित (vivahit)
  * Widow = विधवा (vidhwa)
  * Divorced = घटस्फोटित (ghatasphoti)

When asking about marital status in Marathi, phrase it as:
"तुमची वैवाहिक स्थिती काय आहे? (अविवाहित/विवाहित/विधवा/घटस्फोटित)"

QUESTION EXAMPLES (Friendly Marathi):
- "तुमच्या घरची वार्षिक कमाई 2.5 लाखांपेक्षा कमी आहे का?"
- "तुमचं स्वतःचं बँक खातं आहे का?"
- "तुमच्या घरातलं कोणी इन्कम टॅक्स भरतो का?"
- "तुमच्या घरातलं कोणी सरकारी नोकरीत आहे का?"
- "तुमच्या घरातल्या कोणाला पेन्शन मिळतं का?"
- "तुमच्या घरातलं कोणी आमदार, खासदार अशा पदावर आहे का?"
- "तुमच्या घरी चारचाकी गाडी आहे का? (ट्रॅक्टर नाही)"
- "तुम्हाला आधीच दुसऱ्या कोणत्या योजनेतून महिन्याला 1500 रुपये किंवा त्यापेक्षा जास्त मिळतंय का?"
- "तुमची वैवाहिक स्थिती काय आहे?"

QUESTION SEQUENCE:
Ask these questions ONE BY ONE in friendly, conversational Marathi:
1. Family annual income less than ₹2.5 lakh?
2. Do you have own bank account?
3. Does any family member pay income tax?
4. Any family member government employee?
5. Any family member receives pension?
6. Any family member MP/MLA/political position?
7. Family owns four-wheeler (tractor exempt)?
8. Receiving ₹1500+ from other government scheme?
9. Marital status?

After ALL questions answered, you will be provided with the final verdict.
"""

def validate_pre_registration_query(user_message: str) -> dict:
    """Check if query is relevant to pre-registration/eligibility"""
    
    validation_prompt = f"""
You are a query validator. Determine if this question is about PRE-REGISTRATION/ELIGIBILITY topics ONLY.

ALLOWED topics:
- Scheme information and overview
- Eligibility criteria and requirements
- Required documents for application
- Application process and how to apply
- Benefits and features of the scheme
- Age, income, residency requirements
- Ineligibility criteria

NOT ALLOWED topics (reject immediately):
- Application status checks (not applied yet)
- Payment/transaction queries (not registered yet)
- Document verification status (not applied yet)
- Geographical information (districts, cities, talukas, villages, addresses)
- General knowledge questions unrelated to the scheme
- Other government schemes
- Historical facts or general world knowledge

User question: "{user_message}"

Return ONLY valid JSON with no additional text:
{{
  "is_valid": true,
  "reason": "brief reason"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
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
            print(f"⚠️ Invalid validation response structure: {result}")
            return {
                "is_valid": True,
                "rejection_messages": {}
            }
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in validation: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return {
            "is_valid": True,
            "rejection_messages": {}
        }
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return {
            "is_valid": True,
            "rejection_messages": {}
        }
    
    rejection_messages = {
        "hindi": "मैं केवल लाडली बहना योजना की पात्रता, आवश्यक दस्तावेज़ और आवेदन प्रक्रिया में मदद कर सकता हूं। कृपया इन विषयों के बारे में पूछें।",
        "marathi": "मी फक्त लाडली बहिण योजनेची पात्रता, आवश्यक कागदपत्रे आणि अर्ज प्रक्रियेत मदत करू शकतो. कृपया या विषयांबद्दल विचारा.",
        "english": "I can only help with Ladli Behna Yojana eligibility, required documents, and application process. Please ask about these topics."
    }
    
    return {
        "is_valid": result.get("is_valid", True),
        "rejection_messages": rejection_messages
    }

class EligibilityCheckRequest(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    income: Optional[float] = None
    maharashtra_resident: Optional[bool] = None
    income_tax_payer: Optional[bool] = False
    govt_employee: Optional[bool] = False
    govt_pension: Optional[bool] = False
    political_position: Optional[bool] = False
    four_wheeler: Optional[bool] = False
    existing_benefit: Optional[bool] = False


class ResetRequest(BaseModel):
    session_id: str = "default"


def get_ai_response(session_id: str, user_message: str, aadhaar_data: Optional[Dict[str, Any]] = None, user_lang: Optional[str] = None, file_uploaded: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get response from Azure OpenAI for the eligibility agent"""

    import re as _re

    # ─────────────────────────────────────────────
    # Initialize session if new
    # ─────────────────────────────────────────────
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "eligibility_status": None,
            "checked_criteria": {},
            "language": user_lang,
            "dpip_accepted": False,
            "dpip_done": False,
            "aadhaar_done": False,
            "temp_aadhaar_data": None,
            "aadhaar_confirmed": False,
            "correction_field": None,
            "awaiting_correction_field": False
        }

    # ─────────────────────────────────────────────
    # STEP 0: DPIP CONSENT (Marathi, before anything)
    # ─────────────────────────────────────────────
    if not sessions[session_id].get("dpip_done"):

        if not sessions[session_id].get("dpip_accepted"):

            # First load — show consent message
            if not user_message:
                return {
                    "response": (
                        "आपली ओळख पडताळण्यासाठी आम्हाला आपली वैयक्तिक माहिती "
                        "(आधार क्रमांक, उत्पन्न तपशील इ.) गोळा करणे आवश्यक आहे.\n\n"
                        "ही माहिती Digital Personal Data Protection Act (DPDP) "
                        "नियमांनुसार सुरक्षित ठेवली जाईल.\n\n"
                        "आपण माहिती सामायिक करण्यास संमती देता का?\n\n"
                        "कृपया उत्तर द्या: होय / नाही"
                    ),
                    "is_complete": False
                }

            user_input = user_message.strip().lower()
            yes_words = ["yes", "y", "ho", "होय", "हां", "haan", "हो"]
            no_words  = ["no", "n", "nahi", "नाही", "नहीं"]

            if user_input in yes_words:
                sessions[session_id]["dpip_accepted"] = True
                sessions[session_id]["dpip_done"] = True
                return {
                    "response": (
                        "पात्रता तपासणी सुरू करण्यासाठी कृपया खालीलपैकी एक करा:\n\n"
                        "• आपला १२ अंकी आधार क्रमांक टाइप करा\n"
                        "• किंवा आधार कार्डचा फोटो अपलोड करा\n\n"
                        "📸 फोटो अपलोड करताना: आधार कार्डची पुढची (Front) आणि "
                        "मागची (Back) बाजू एकाच फोटोमध्ये एकत्र करून अपलोड करा."
                    ),
                    "is_complete": False
                }

            elif user_input in no_words:
                sessions[session_id]["dpip_done"] = True
                return {
                    "response": (
                        "धन्यवाद. आपण कधीही परत येऊ शकता.\n\n"
                        "लाडकी बहिण योजना – आपल्या सशक्तीकरणासाठी.\n\n"
                        "════════════════════════════════════════════\n"
                        "पात्रता तपासणी प्रक्रिया संपली.\n"
                        "════════════════════════════════════════════"
                    ),
                    "is_complete": True
                }

            else:
                return {
                    "response": "कृपया होय किंवा नाही असे उत्तर द्या:",
                    "is_complete": False
                }

    # ─────────────────────────────────────────────
    # STEP 0.5: AADHAAR COLLECTION
    # (runs after DPIP accepted, before eligibility AI)
    # ─────────────────────────────────────────────
    if not sessions[session_id].get("aadhaar_done"):

        temp = sessions[session_id].get("temp_aadhaar_data")

        # ── We already have temp data → handle confirmation / correction ──
        if temp:
            fields     = temp.get("fields", {})
            user_input = user_message.strip().lower()

            yes_words        = ["yes", "ho", "y", "होय", "हो", "हां", "haan"]
            correction_words = ["correction", "durusti", "दुरुस्ती", "सुधार"]

            # Sub-step A: waiting to know WHICH field to correct
            if sessions[session_id].get("awaiting_correction_field"):
                msg = _re.sub(r'[^\w\u0900-\u097F]', '', user_message.strip().lower())

                if any(w in msg for w in ["name", "नाव", "naam"]):
                    sessions[session_id]["correction_field"] = "name"
                elif any(w in msg for w in ["dob", "dateofbirth", "birth", "जन्मतारीख", "जन्म"]):
                    sessions[session_id]["correction_field"] = "dob"
                elif any(w in msg for w in ["address", "पत्ता", "पता"]):
                    sessions[session_id]["correction_field"] = "address"
                else:
                    return {
                        "response": "कृपया कोणती माहिती दुरुस्त करायची आहे ते सांगा (नाव / जन्मतारीख / पत्ता):",
                        "is_complete": False
                    }

                field_labels = {"name": "नाव", "dob": "जन्मतारीख", "address": "पत्ता"}
                label = field_labels[sessions[session_id]["correction_field"]]
                sessions[session_id]["awaiting_correction_field"] = False
                return {
                    "response": f"कृपया योग्य {label} टाइप करा:",
                    "is_complete": False
                }

            # Sub-step B: waiting for the corrected VALUE
            if sessions[session_id].get("correction_field"):
                field = sessions[session_id]["correction_field"]
                fields[field] = user_message.strip()
                sessions[session_id]["temp_aadhaar_data"]["fields"] = fields
                sessions[session_id]["correction_field"] = None

                return {
                    "response": (
                        f"आधार तपशील:\n\n"
                        f"नाव: {fields.get('name', 'N/A')}\n"
                        f"जन्मतारीख: {fields.get('dob', 'N/A')}\n"
                        f"पत्ता: {fields.get('address', 'N/A')}\n\n"
                        "ही माहिती योग्य आहे का?\n\n"
                        "योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:"
                    ),
                    "is_complete": False
                }

            # Sub-step C: user says YES → confirm and fall through to eligibility AI
            if user_input in yes_words:
                from database import db_manager
                # Calculate age from dob if possible
                dob_str = fields.get("dob", "")
                age_val = ""
                if dob_str:
                    try:
                        if "-" in dob_str:
                            dob_dt = datetime.strptime(dob_str, "%d-%m-%Y")
                        else:
                            dob_dt = datetime.strptime(dob_str, "%d/%m/%Y")
                        today = datetime.now()
                        age_val = str(
                            today.year - dob_dt.year -
                            ((today.month, today.day) < (dob_dt.month, dob_dt.day))
                        )
                    except Exception:
                        age_val = ""

                confirmed_aadhaar_data = {
                    "full_name":      fields.get("name", ""),
                    "date_of_birth":  fields.get("dob", ""),
                    "age":            fields.get("age", age_val),
                    "gender":         fields.get("gender", "F"),
                    "address":        fields.get("address", ""),
                    "district":       fields.get("district", ""),
                    "aadhaar_number": fields.get("aadhaar_number", "")
                }

                sessions[session_id]["aadhaar_done"]      = True
                sessions[session_id]["aadhaar_confirmed"] = True
                sessions[session_id]["aadhaar_data"]      = confirmed_aadhaar_data
                sessions[session_id]["temp_aadhaar_data"] = None

                # Use confirmed data as aadhaar_data for the AI call below
                aadhaar_data = confirmed_aadhaar_data
                user_message = ""   # trigger AI to ask first eligibility question

            # Sub-step D: user says CORRECTION
            elif user_input in correction_words:
                sessions[session_id]["awaiting_correction_field"] = True
                return {
                    "response": "कृपया कोणती माहिती दुरुस्त करायची आहे ते सांगा (नाव / जन्मतारीख / पत्ता):",
                    "is_complete": False
                }

            else:
                return {
                    "response": "कृपया 'होय' किंवा 'दुरुस्ती' असे उत्तर द्या:",
                    "is_complete": False
                }

        else:
            # ── No temp data yet — handle typed number OR uploaded image ──

            # Case 0: file uploaded directly (image of Aadhaar card)
            if file_uploaded:
                # Import the same OCR functions that utils.py uses
                from utils import (
                    extract_text_from_bytes,
                    extract_aadhaar_front_details,
                    extract_aadhaar_back_details
                )

                # ── Run OCR on the uploaded image ──
                raw_text = extract_text_from_bytes(
                    file_uploaded["content"],
                    file_uploaded["extension"]
                )

                if not raw_text.strip():
                    return {
                        "response": (
                            "📸 फोटो अस्पष्ट आहे. कृपया स्पष्ट फोटो अपलोड करा.\n\n"
                            "आधार कार्डची पुढची (Front) आणि मागची (Back) बाजू "
                            "एकाच फोटोमध्ये एकत्र करून अपलोड करा."
                        ),
                        "is_complete": False
                    }

                # ── FRONT SIDE CHECK: 12-digit Aadhaar number ──
                aadhaar_num_in_text = _re.search(r'\d{4}\s?\d{4}\s?\d{4}', raw_text)
                has_front = bool(aadhaar_num_in_text)

                # ── BACK SIDE CHECK: address label or address keywords ──
                back_keywords = [
                    "address", "पत्ता", "पता",
                    "s/o", "d/o", "w/o", "c/o",
                    "house", "village", "vill", "dist", "district",
                    "pin", "state", "at post", "ward", "nagar",
                    "road", "street", "taluka", "tehsil", "sector",
                    "near", "behind", "opposite", "post office"
                ]
                has_back = bool(
                    any(kw in raw_text.lower() for kw in back_keywords)
                )

                # ── REJECT: not a valid Aadhaar image at all ──
                if not has_front and not has_back:
                    return {
                        "response": (
                            "हा आधार कार्डचा फोटो नाही किंवा अस्पष्ट आहे!\n\n"
                            "📸 कृपया आधार कार्डची पुढची (Front) आणि मागची (Back) बाजू "
                            "एकाच फोटोमध्ये एकत्र करून अपलोड करा."
                        ),
                        "is_complete": False
                    }

                # ── REJECT: only back side uploaded ──
                if not has_front:
                    return {
                        "response": (
                            "❌ फक्त मागची बाजू (Back) मिळाली!\n\n"
                            "📸 कृपया आधार कार्डची पुढची (Front) आणि मागची (Back) बाजू "
                            "एकाच फोटोमध्ये एकत्र करून पुन्हा अपलोड करा.\n\n"
                            "पुढची बाजू: नाव, जन्मतारीख आणि आधार क्रमांक असतो."
                        ),
                        "is_complete": False
                    }

                # ── REJECT: only front side uploaded ──
                if not has_back:
                    return {
                        "response": (
                            "❌ फक्त पुढची बाजू (Front) मिळाली!\n\n"
                            "📸 कृपया आधार कार्डची पुढची (Front) आणि मागची (Back) बाजू "
                            "एकाच फोटोमध्ये एकत्र करून पुन्हा अपलोड करा.\n\n"
                            "मागची बाजू: पत्ता असतो."
                        ),
                        "is_complete": False
                    }

                # ── Both sides present — extract fields ──
                front_data = extract_aadhaar_front_details(raw_text)
                back_data  = extract_aadhaar_back_details(raw_text)

                # Get Aadhaar number
                raw_aadhaar = front_data.get("AadhaarNo", "")
                if not raw_aadhaar and aadhaar_num_in_text:
                    raw_aadhaar = _re.sub(r'\s', '', aadhaar_num_in_text.group())

                # Get DOB and calculate age
                dob_raw = ""
                dob_obj = front_data.get("DateOfBirth")
                if dob_obj:
                    if hasattr(dob_obj, 'strftime'):
                        dob_raw = dob_obj.strftime("%d/%m/%Y")
                    else:
                        dob_raw = str(dob_obj)

                age_val = ""
                if dob_raw:
                    try:
                        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                            try:
                                dob_dt = datetime.strptime(dob_raw, fmt)
                                today = datetime.now()
                                age_val = str(
                                    today.year - dob_dt.year -
                                    ((today.month, today.day) < (dob_dt.month, dob_dt.day))
                                )
                                break
                            except Exception:
                                continue
                    except Exception:
                        age_val = ""

                sessions[session_id]["temp_aadhaar_data"] = {
                    "source": "ocr",
                    "fields": {
                        "name":           front_data.get("FullName", ""),
                        "dob":            dob_raw,
                        "address":        back_data.get("Address", ""),
                        "aadhaar_number": raw_aadhaar,
                        "gender":         front_data.get("Gender", "F"),
                        "district":       back_data.get("District", ""),
                        "age":            age_val
                    }
                }

                f = sessions[session_id]["temp_aadhaar_data"]["fields"]
                return {
                    "response": (
                        f"आधार तपशील प्राप्त झाले:\n\n"
                        f"नाव: {f['name'] or 'N/A'}\n"
                        f"जन्मतारीख: {f['dob'] or 'N/A'}\n"
                        f"पत्ता: {f['address'] or 'N/A'}\n\n"
                        "ही माहिती योग्य आहे का?\n\n"
                        "योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:"
                    ),
                    "is_complete": False
                }

            # Case 1: caller passed aadhaar_data (image processed externally via OCR)
            if aadhaar_data:
                fields = {
                    "name":           aadhaar_data.get("full_name", ""),
                    "dob":            aadhaar_data.get("date_of_birth", ""),
                    "age":            aadhaar_data.get("age", ""),
                    "address":        aadhaar_data.get("address", ""),
                    "aadhaar_number": aadhaar_data.get("aadhaar_number", ""),
                    "gender":         aadhaar_data.get("gender", "F"),
                    "district":       aadhaar_data.get("district", "")
                }
                sessions[session_id]["temp_aadhaar_data"] = {"source": "ocr", "fields": fields}
                return {
                    "response": (
                        f"आधार तपशील प्राप्त झाले:\n\n"
                        f"नाव: {fields['name'] or 'N/A'}\n"
                        f"जन्मतारीख: {fields['dob'] or 'N/A'}\n"
                        f"पत्ता: {fields['address'] or 'N/A'}\n\n"
                        "ही माहिती योग्य आहे का?\n\n"
                        "योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:"
                    ),
                    "is_complete": False
                }

            # Case 2: user typed a 12-digit Aadhaar number
            if user_message:
                cleaned = _re.sub(r'\s+', '', user_message.strip())

                if _re.fullmatch(r'\d{12}', cleaned):
                    try:
                        from database import db_manager
                        aadhaar_db = db_manager.get_aadhaar_details(cleaned) if db_manager else None
                    except Exception:
                        aadhaar_db = None

                    if not aadhaar_db:
                        return {
                            "response": (
                                "हा आधार क्रमांक सापडला नाही. "
                                "कृपया आधार कार्डचा फोटो अपलोड करा.\n\n"
                                "📸 फोटो अपलोड करताना: आधार कार्डची पुढची (Front) आणि "
                                "मागची (Back) बाजू एकाच फोटोमध्ये एकत्र करून अपलोड करा."
                            ),
                            "is_complete": False
                        }

                    name    = aadhaar_db.get("FullName", "")
                    dob     = aadhaar_db.get("DateOfBirth", "")
                    address = aadhaar_db.get("Address", "")
                    district = aadhaar_db.get("City", "")
                    if hasattr(dob, 'strftime'):
                        dob = dob.strftime("%d/%m/%Y")
                    else:
                        dob = str(dob) if dob else ""

                    sessions[session_id]["temp_aadhaar_data"] = {
                        "source": "manual_number",
                        "fields": {
                            "name": name, "dob": dob,
                            "address": address, "district": district,
                            "aadhaar_number": cleaned, "gender": "F", "age": ""
                        }
                    }
                    return {
                        "response": (
                            f"आधार तपशील प्राप्त झाले:\n\n"
                            f"नाव: {name or 'N/A'}\n"
                            f"जन्मतारीख: {dob or 'N/A'}\n"
                            f"पत्ता: {address or 'N/A'}\n\n"
                            "ही माहिती योग्य आहे का?\n\n"
                            "योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:"
                        ),
                        "is_complete": False
                    }

                else:
                    # Something typed but not a valid 12-digit number — re-prompt
                    return {
                        "response": (
                            "पात्रता तपासणी सुरू करण्यासाठी कृपया खालीलपैकी एक करा:\n\n"
                            "• आपला १२ अंकी आधार क्रमांक टाइप करा\n"
                            "• किंवा आधार कार्डचा फोटो अपलोड करा\n\n"
                            "📸 फोटो अपलोड करताना: आधार कार्डची पुढची (Front) आणि "
                            "मागची (Back) बाजू एकाच फोटोमध्ये एकत्र करून अपलोड करा."
                        ),
                        "is_complete": False
                    }

            # Case 3: no input at all — show Aadhaar prompt
            return {
                "response": (
                    "पात्रता तपासणी सुरू करण्यासाठी कृपया खालीलपैकी एक करा:\n\n"
                    "• आपला १२ अंकी आधार क्रमांक टाइप करा\n"
                    "• किंवा आधार कार्डचा फोटो अपलोड करा\n\n"
                    "📸 फोटो अपलोड करताना: आधार कार्डची पुढची (Front) आणि "
                    "मागची (Back) बाजू एकाच फोटोमध्ये एकत्र करून अपलोड करा."
                ),
                "is_complete": False
            }

    # ─────────────────────────────────────────────
    # FROM HERE: original eligibility AI logic — UNCHANGED
    # ─────────────────────────────────────────────

    # Use stored aadhaar_data if not passed in this call
    if not aadhaar_data and sessions[session_id].get("aadhaar_data"):
        aadhaar_data = sessions[session_id]["aadhaar_data"]

    # ✅ Smart Validation: Check if this looks like a simple response vs a new query
    is_simple_response = any([
        # Single word responses
        len(user_message.strip().split()) <= 3,
        # Yes/No responses
        user_message.strip().lower() in ["yes", "no", "ho", "hoy", "नाही", "हाँ", "नहीं", "होय"],
        # Numbers (age, income, etc.)
        user_message.strip().replace(".", "").replace(",", "").isdigit(),
        # Marital status responses
        any(word in user_message.lower() for word in ["married", "unmarried", "widow", "divorced", "विवाहित", "अविवाहित", "विधवा"]),
        # Already in conversation (has message history)
        len(sessions[session_id]["messages"]) > 0
    ])
    
    # ✅ ONLY validate if it's NOT a simple response
    if not is_simple_response:
        validation = validate_pre_registration_query(user_message)
        if not validation["is_valid"]:
            # Detect language if not already set
            if not user_lang:
                from utils import detect_language
                user_lang = detect_language(user_message)
            
            rejection_msg = validation["rejection_messages"].get(user_lang, validation["rejection_messages"]["english"])
            return {
                "response": rejection_msg,
                "is_complete": False
            }
    
    # Prioritize passed language
    if user_lang:
        sessions[session_id]["language"] = user_lang
    
    # Detect language from user's message if still not set
    if not sessions[session_id].get("language") and user_message.strip():
        from utils import detect_language
        detected_lang = detect_language(user_message)
        sessions[session_id]["language"] = detected_lang
        print(f"🔍 Detected language for session {session_id}: {detected_lang}")
    
    user_language = sessions[session_id].get("language", "marathi")
    
    # Build system prompt - start with original
    system_prompt = SYSTEM_PROMPT
    
    # Add language instruction
    language_instruction = f"""

CRITICAL LANGUAGE INSTRUCTION:
The user is communicating in {user_language.upper()}.
You MUST respond ENTIRELY in {user_language.upper()} language.
ALL instructions, questions, and verdicts below MUST be translated to {user_language.upper()}.
"""
    
    system_prompt = system_prompt + language_instruction
    
    # If aadhaar_data is provided, enhance the system prompt
    if aadhaar_data:
        age = aadhaar_data.get('age', 'N/A')
        
        aadhaar_context = f"""

AADHAAR VERIFIED INFORMATION:
The user has already provided and verified their Aadhaar card. You have the following information:
- Full Name: {aadhaar_data.get('full_name', 'N/A')}
- Age: {age} years
- Gender: {aadhaar_data.get('gender', 'N/A')}
- District: {aadhaar_data.get('district', 'N/A')}
- Address: {aadhaar_data.get('address', 'Maharashtra')}
- Residency: Maharashtra permanent resident (verified via Aadhaar)

CRITICAL INSTRUCTIONS FOR AADHAAR-VERIFIED USERS (translate all questions to {user_language.upper()}):
1. DO NOT ask about Name, Age, Gender, Maharashtra residency, or District - these are ALREADY VERIFIED
2. DO NOT ask "Is the above Aadhaar information correct?" - it has already been confirmed
3. IMMEDIATELY start asking the remaining eligibility questions in this EXACT order:
   a. Is your family's annual income less than ₹2.50 lakh? (if NO → NOT ELIGIBLE)
   b. Do you have your own bank account? (if NO → may need assistance, but not disqualifying)
   c. Does any family member pay income tax? (if YES → IMMEDIATELY declare NOT ELIGIBLE)
   d. Are you currently receiving ₹1500 or more per month from any other government scheme? (if YES → IMMEDIATELY declare NOT ELIGIBLE)
   e. Does your family own a four-wheeler vehicle? (excluding tractor) (if YES → IMMEDIATELY declare NOT ELIGIBLE)
   f. What is your marital status? (Options: Unmarried/Married/Widow/Divorced)

4. CRITICAL STOPPING RULE: 
   - If user answers YES to questions (c), (d), or (e) → STOP immediately
   - Do NOT ask remaining questions
   - Immediately provide verdict: "You are NOT ELIGIBLE. Reason: [specific reason]"
   
5. Ask ONE question at a time in the sequence above
6. Keep responses SHORT and conversational (2-3 sentences maximum)
7. For eligibility determination, use the verified Aadhaar data:
   - Gender: {aadhaar_data.get('gender', 'N/A')} (already verified)
   - Age: {age} years (already verified - check if between 21-65)
   - Maharashtra residency: Confirmed (already verified)

8. if all questions are answered correctly, provide final verdict in {user_language.upper()}:
   "Based on the information provided, you are ELIGIBLE for Ladki Bahin Yojana."
   else if any criteria fails immediately provide final verdict in {user_language.upper()}:
   "You are NOT ELIGIBLE for Ladki Bahin Yojana. Reason: [specific reason]"

9. Follow all other eligibility rules and behavior guidelines as specified in the main system prompt above
"""
        system_prompt = system_prompt + aadhaar_context
    
    # Add user message to history
    sessions[session_id]["messages"].append({
        "role": "user",
        "content": user_message
    })
    
    try:
        # Build messages with system prompt
        messages_with_system = [{"role": "system", "content": system_prompt}] + sessions[session_id]["messages"]
        
        # Call Azure OpenAI API
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            max_tokens=1024,
            messages=messages_with_system
        )
        
        assistant_message = response.choices[0].message.content

        # Add assistant response to history
        sessions[session_id]["messages"].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # ✅ CHECK IF ELIGIBILITY FLOW IS COMPLETE
        verdict_patterns = [
            "eligible", "not eligible", "ineligible",
            "पात्र", "अपात्र",
        ]

        reason_keywords = [
            "कारण", "reason", "because",
            "वर्षांच्या पात्रतेमध्ये नाही",
            "उत्पन्न मर्यादा ओलांडली",
            "पात्रता नाही"
        ]

        response_lower = assistant_message.lower()

        has_verdict_pattern = any(pattern in response_lower for pattern in verdict_patterns)
        has_reason = any(keyword in assistant_message for keyword in reason_keywords)

        has_verdict_format = ("पात्र:" in assistant_message or "अपात्र:" in assistant_message or 
                            "eligible:" in response_lower or "ineligible:" in response_lower or
                            "not eligible:" in response_lower)

        is_question = assistant_message.strip().endswith("?")

        is_flow_complete = (
            (has_verdict_pattern and has_reason) or 
            has_verdict_format or
            (has_verdict_pattern and len(assistant_message) < 100 and not is_question)
        )

        return {
            "response": assistant_message,
            "is_complete": is_flow_complete
        }
        
    except Exception as e:
        return {
            "response": f"Error: {str(e)}. Please check your API key.",
            "is_complete": False
        }


def check_eligibility_rule(criteria: str, value: Any) -> tuple:
    """Check a specific eligibility rule"""
    rules = ELIGIBILITY_RULES
    
    if criteria == "age":
        age = int(value)
        if rules["age"]["min"] <= age <= rules["age"]["max"]:
            return True, "Age criteria met ✅"
        return False, f"Age must be between {rules['age']['min']} and {rules['age']['max']} years ❌"
    
    elif criteria == "income":
        income = float(value)
        if income <= rules["income"]["max_annual"]:
            return True, "Income criteria met ✅"
        return False, f"Annual family income exceeds ₹{rules['income']['max_annual']:,} ❌"
    
    elif criteria == "gender":
        if value.lower() in ["female", "f", "महिला", "स्त्री"]:
            return True, "Gender criteria met ✅"
        return False, "Only female applicants are eligible ❌"
    
    elif criteria == "residency":
        if value.lower() in ["yes", "हो", "हां", "maharashtra"]:
            return True, "Residency criteria met ✅"
        return False, "Must be Maharashtra permanent resident ❌"
    
    return None, "Unknown criteria"


@app.get("/")
async def root():
    """Serve the main application"""
    return FileResponse("frontend/index.html")


@app.post("/api/chat")
async def chat(
    session_id: str = Form(...),
    message: str = Form(""),
    file: Optional[UploadFile] = File(None),
    doc_type: str = Form("aadhaar")
):
    """Handle chat messages with optional file upload"""
    file_uploaded = None
    if file and file.filename:
        file_content = await file.read()
        file_extension = Path(file.filename).suffix
        file_uploaded = {
            "content": file_content,
            "name": file.filename,
            "extension": file_extension,
            "doc_type": doc_type
        }

    response = get_ai_response(
        session_id=session_id,
        user_message=message,
        file_uploaded=file_uploaded
    )

    return {"response": response, "session_id": session_id}


@app.post("/api/check-eligibility")
async def check_eligibility(request: EligibilityCheckRequest):
    """Direct eligibility check API"""
    results = {
        "eligible": True,
        "checks": [],
        "failed_criteria": []
    }
    
    if request.age is not None:
        passed, msg = check_eligibility_rule("age", request.age)
        results["checks"].append({"criterion": "age", "passed": passed, "message": msg})
        if not passed:
            results["eligible"] = False
            results["failed_criteria"].append("age")
    
    if request.gender is not None:
        passed, msg = check_eligibility_rule("gender", request.gender)
        results["checks"].append({"criterion": "gender", "passed": passed, "message": msg})
        if not passed:
            results["eligible"] = False
            results["failed_criteria"].append("gender")
    
    if request.income is not None:
        passed, msg = check_eligibility_rule("income", request.income)
        results["checks"].append({"criterion": "income", "passed": passed, "message": msg})
        if not passed:
            results["eligible"] = False
            results["failed_criteria"].append("income")
    
    if request.maharashtra_resident is not None:
        passed, msg = check_eligibility_rule("residency", "yes" if request.maharashtra_resident else "no")
        results["checks"].append({"criterion": "residency", "passed": passed, "message": msg})
        if not passed:
            results["eligible"] = False
            results["failed_criteria"].append("residency")
    
    ineligibility_checks = [
        ("income_tax_payer", request.income_tax_payer, "Family member pays income tax"),
        ("govt_employee", request.govt_employee, "Family member is government employee"),
        ("govt_pension", request.govt_pension, "Family member receives government pension"),
        ("political_position", request.political_position, "Family member holds political position"),
        ("four_wheeler", request.four_wheeler, "Family owns four-wheeler vehicle"),
        ("existing_benefit", request.existing_benefit, "Already receiving ₹1500+ from another scheme")
    ]
    
    for key, value, description in ineligibility_checks:
        if value:
            results["eligible"] = False
            results["failed_criteria"].append(key)
            results["checks"].append({
                "criterion": key,
                "passed": False,
                "message": f"{description} - NOT ELIGIBLE ❌"
            })
    
    if results["eligible"]:
        results["verdict"] = "🎉 Congratulations! You appear to be ELIGIBLE for Ladki Bahin Yojana!"
        results["next_steps"] = [
            "Visit ladakibahin.maharashtra.gov.in to apply",
            "Keep your Aadhaar card, bank passbook, and residency proof ready",
            "Complete e-KYC verification",
            "For help, call 181 or 1800-120-8040"
        ]
    else:
        results["verdict"] = "❌ Sorry, based on the information provided, you are NOT ELIGIBLE for Ladki Bahin Yojana."
        results["reason"] = f"Failed criteria: {', '.join(results['failed_criteria'])}"
    
    return results


@app.get("/api/questions")
async def get_questions():
    """Get eligibility check questions"""
    return ELIGIBILITY_QUESTIONS


@app.get("/api/rules")
async def get_rules():
    """Get eligibility rules"""
    return ELIGIBILITY_RULES


@app.post("/api/reset")
async def reset_session(request: ResetRequest):
    """Reset a chat session"""
    if request.session_id in sessions:
        del sessions[request.session_id]
    
    return {"status": "Session reset successfully"}


@app.get("/api/speech-token")
async def get_speech_token():
    """Get Azure Speech token with enhanced error handling and multi-language support"""
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = os.getenv("AZURE_SPEECH_REGION", "centralindia")
    
    if not speech_key:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Azure Speech API Key not configured",
                "details": "Please set AZURE_SPEECH_KEY in your .env file",
                "success": False
            }
        )
    
    if not service_region:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Azure Speech Region not configured",
                "details": "Please set AZURE_SPEECH_REGION in your .env file (e.g., centralindia)",
                "success": False
            }
        )

    fetch_token_url = f"https://{service_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    headers = {
        'Ocp-Apim-Subscription-Key': speech_key
    }
    
    try:
        response = requests.post(fetch_token_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return {
                "token": response.text,
                "region": service_region,
                "success": True
            }
        elif response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Invalid Azure Speech API Key",
                    "details": "The API key is incorrect. Please verify AZURE_SPEECH_KEY",
                    "success": False
                }
            )
        elif response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Access Forbidden",
                    "details": "Check your Azure Speech Service permissions and subscription",
                    "success": False
                }
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail={
                    "error": "Failed to retrieve token",
                    "status_code": response.status_code,
                    "details": response.text,
                    "success": False
                }
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "Request timeout",
                "details": "Azure Speech Service took too long to respond. Please try again",
                "success": False
            }
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Connection error",
                "details": "Unable to connect to Azure Speech Service. Check your internet connection",
                "success": False
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Unexpected error",
                "details": str(e),
                "success": False
            }
        )

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming Plivo call and setup voice interface"""
    try:
        form_data = await request.form()
        caller_phone = form_data.get("From", "unknown")
        call_uuid = form_data.get("CallUUID", f"call_{datetime.now().timestamp()}")

        user_details = get_user_by_phone(caller_phone)
        beneficiary_id = user_details.get("BeneficiaryId", f"unknown_{datetime.now().timestamp()}")

        session_data = {
            "beneficiary_id": beneficiary_id,
            "caller_phone": caller_phone,
            "call_uuid": call_uuid,
            "session_id": f"{beneficiary_id}_{call_uuid}",
            "call_start": datetime.now(),
            "conversation_history": []
        }

        voice_sessions[str(beneficiary_id)] = session_data

        print(f"✅ Session created for beneficiary_id: {beneficiary_id}")

        safe_user_info = {}
        if user_details:
            for key, value in user_details.items():
                if isinstance(value, (datetime, date)):
                    safe_user_info[key] = value.isoformat()
                else:
                    safe_user_info[key] = value

        await broadcast_to_call_center({
            "type": "call_started",
            "call": {
                "beneficiary_id": beneficiary_id,
                "caller_phone": caller_phone,
                "call_uuid": call_uuid,
                "session_id": f"{beneficiary_id}_{call_uuid}",
                "call_start": session_data["call_start"].isoformat(),
                "conversation_history": [],
                "user_info": safe_user_info.get("FullName", "Unknown User")
            }
        })

        print(f"📢 Broadcasted call_started for {beneficiary_id}")

        await asyncio.sleep(0.1)

        user_name = ""
        if user_details and user_details.get("FullName"):
            user_name = user_details["FullName"].split()[0]

        greeting = "नमस्कार"
        if user_name:
            greeting += f" {user_name}"
        greeting += (
            "! लाडकी बहिणी योजनेच्या व्हॉईस सहाय्यकात आपले स्वागत आहे. "
            "मी आज आपली कशी मदत करू?"
        )

        response = plivoxml.ResponseElement()
        response.add(plivoxml.SpeakElement(
            greeting,
            voice="Polly.Aditi",
            language="mr-IN"
        ))

        ws_url = f"{HOST_URL}/media-stream?beneficiary_id={beneficiary_id}"
        print(f"🔗 WebSocket URL: {ws_url}")

        response.add(plivoxml.StreamElement(
            ws_url,
            bidirectional=True,
            streamTimeout=86400,
            keepCallAlive=True,
            contentType="audio/x-mulaw;rate=8000",
            audioTrack="inbound"
        ))

        xml_response = '<?xml version="1.0" encoding="UTF-8"?>\n' + response.to_string()
        return HTMLResponse(xml_response, media_type="application/xml")

    except Exception as e:
        print(f"❌ Error in incoming call: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle Plivo media stream for voice interaction"""

    await websocket.accept()
    print("✅ WebSocket accepted")

    query_params = parse_qs(websocket.url.query)
    beneficiary_id_param = query_params.get("beneficiary_id", [None])[0]

    if not beneficiary_id_param:
        await websocket.close(code=1008, reason="Missing beneficiary_id")
        return

    beneficiary_id_str = str(beneficiary_id_param)
    print(f"🔍 Looking for session: {beneficiary_id_str}")

    session = None
    for _ in range(20):
        if beneficiary_id_str in voice_sessions:
            session = voice_sessions[beneficiary_id_str]
            break
        await asyncio.sleep(0.5)

    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return

    print(f"🎙️ Voice session started for beneficiary_id {beneficiary_id_str}")

    recognizer, stream = create_azure_speech_recognizer()

    processing_response = False
    loop = asyncio.get_running_loop()

    def recognizing_handler(evt):
        partial = evt.result.text.strip()
        if partial:
            print(f"[Partial] {partial}")

    def recognized_handler(evt):
        nonlocal processing_response

        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return

        final_text = evt.result.text.strip()

        if not final_text:
            print("🔇 Empty speech detected, skipping AI call")
            return

        if processing_response:
            print("⚠️ Already processing, skipping...")
            return

        processing_response = True

        detected_lang = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        )

        print(f"🗣️ [{detected_lang}] User said: {final_text}")

        user_message = {
            "role": "user",
            "message": final_text,
            "timestamp": datetime.now().isoformat()
        }
        session["conversation_history"].append(user_message)

        asyncio.run_coroutine_threadsafe(
            broadcast_to_call_center({
                "type": "transcript_update",
                "beneficiary_id": beneficiary_id_str,
                "conversation_history": session["conversation_history"]
            }),
            loop
        )

        async def process_chat():
            nonlocal processing_response
            try:
                reply = get_ai_response(
                    session_id=session["session_id"],
                    user_message=final_text,
                )

                print("Assistant:", reply)

                assistant_message = {
                    "role": "bot",
                    "message": reply,
                    "timestamp": datetime.now().isoformat()
                }
                session["conversation_history"].append(assistant_message)

                await broadcast_to_call_center({
                    "type": "transcript_update",
                    "beneficiary_id": beneficiary_id_str,
                    "conversation_history": session["conversation_history"]
                })

                audio = azure_text_to_speech(reply)
                audio_b64 = base64.b64encode(audio).decode("utf-8")

                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "event": "playAudio",
                        "media": {
                            "contentType": "audio/x-mulaw",
                            "sampleRate": 8000,
                            "payload": audio_b64
                        }
                    })

            except Exception as e:
                print(f"❌ Chat error: {e}")
                traceback.print_exc()
            finally:
                processing_response = False

        asyncio.run_coroutine_threadsafe(process_chat(), loop)

    recognizer.recognizing.connect(recognizing_handler)
    recognizer.recognized.connect(recognized_handler)
    recognizer.start_continuous_recognition()

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            if data.get("event") == "media":
                audio = base64.b64decode(data["media"]["payload"])
                audio = audioop.ulaw2lin(audio, 2)
                audio = audioop.ratecv(audio, 2, 1, 8000, 16000, None)[0]
                stream.write(audio)

            elif data.get("event") == "stop":
                break

    finally:
        recognizer.stop_continuous_recognition()
        stream.close()
        if beneficiary_id_str in voice_sessions:
            await broadcast_to_call_center({
                "type": "call_ended",
                "beneficiary_id": beneficiary_id_str
            })
            del voice_sessions[beneficiary_id_str]
        print(f"📞 Session closed for beneficiary_id {beneficiary_id_str}")


def serialize_for_json(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


@app.websocket("/call-center-ws")
async def call_center_websocket(websocket: WebSocket):
    """WebSocket endpoint for call center dashboard"""
    await websocket.accept()
    call_center_clients.add(websocket)
    print(f"✅ Call center client connected. Total: {len(call_center_clients)}")

    try:
        active_calls_data = []
        for beneficiary_id, session in voice_sessions.items():
            user_info = get_user_by_phone(session["caller_phone"])

            user_name = "Unknown User"
            if user_info and user_info.get("FullName"):
                user_name = user_info["FullName"]

            active_calls_data.append({
                "beneficiary_id": session["beneficiary_id"],
                "caller_phone": session["caller_phone"],
                "call_uuid": session["call_uuid"],
                "session_id": session["session_id"],
                "call_start": session["call_start"].isoformat(),
                "conversation_history": session["conversation_history"],
                "user_info": user_name
            })

        await websocket.send_json({
            "type": "initial_state",
            "active_calls": active_calls_data
        })

        print(f"📤 Sent initial_state with {len(active_calls_data)} active calls")

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if message:
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break

    except WebSocketDisconnect:
        print("📊 Call center client disconnected normally")
    except Exception as e:
        print(f"❌ Call center WebSocket error: {e}")
        traceback.print_exc()
    finally:
        call_center_clients.discard(websocket)
        print(f"📊 Call center client removed. Total: {len(call_center_clients)}")


async def broadcast_to_call_center(message: dict):
    """Broadcast message to all connected call center clients"""
    if not call_center_clients:
        print(f"⚠️ No call center clients connected to receive: {message.get('type')}")
        return

    print(f"📢 Broadcasting {message.get('type')} to {len(call_center_clients)} clients")

    disconnected_clients = set()
    for client in call_center_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            print(f"❌ Failed to send to client: {e}")
            disconnected_clients.add(client)

    for client in disconnected_clients:
        call_center_clients.discard(client)

    if disconnected_clients:
        print(f"🧹 Removed {len(disconnected_clients)} disconnected clients")


@app.get("/call-center", response_class=HTMLResponse)
async def call_center_dashboard():
    """Serve the call center dashboard"""
    with open("static/call_center.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🌸 Ladki Bahin Yojana - Eligibility Agent API (FastAPI)")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nMake sure to set Azure OpenAI credentials in .env file")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
