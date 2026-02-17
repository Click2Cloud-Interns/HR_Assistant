from fastapi import FastAPI, Form, File, UploadFile, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os
from typing import Optional
import json
import re
from openai import AzureOpenAI
from pathlib import Path
from api.pre_registration import get_ai_response
from api.post_registration import post_chat
from api.post_registration import ChatRequest
from api.registration import get_bot_response
from api.registration import initialize_blob_storage
from utils import detect_language, process_aadhaar_details, get_multilingual_message, format_aadhaar_confirmation
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load ENV
# --------------------------------------------------
load_dotenv()

# eligibilty_instance=EligibilityCheckRequest()
# --------------------------------------------------
# Azure OpenAI Client
# --------------------------------------------------
AZURE_CLIENT = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(title="Ladki Bahin Yojana - Smart Chat Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


# Root route to serve frontend
@app.get("/")
async def root():
    return {"status": "ok", "message": "Ladki Bahin Yojana API is running"}


SESSION_MODE = {}
SESSION_DATA = {}  # New: Store complete session state

def initialize_session(session_id: str):
    """Initialize session state for new users"""
    if session_id not in SESSION_DATA:
        SESSION_DATA[session_id] = {
            "menu_selected": None,
            "current_mode": None,
            "aadhaar_verified": False,
            "language": None,
            "aadhaar_data": None,
            "pre_filled_fields": [],
            "original_message": None
        }
    return SESSION_DATA[session_id]

def classify_menu_selection(user_message: str) -> str:
    """
    Classify user's menu selection
    Returns: "eligibility" | "form_filling" | "post_application"
    """
    user_msg_lower = user_message.lower().strip()
    
    # ✅ CHECK POST APPLICATION FIRST (before form_filling)
    # Post application / Status keywords
    if any(keyword in user_msg_lower for keyword in [
        "स्थिती", "status", "payment", "installment", "पेमेंट",
        "किस्त", "स्टेटस", "स्थिति", "पाहायची", "देखना", "check"
    ]):
        return "post_application"
    
    # Complaint keywords
    if any(keyword in user_msg_lower for keyword in [
        "तक्रार", "complaint", "issue", "problem", "शिकायत"
    ]):
        return "post_application"
    
    # Eligibility keywords
    if any(keyword in user_msg_lower for keyword in [
        "पात्रता", "पात्र", "eligible", "eligibility", "qualify",
        "योग्य", "योग्यता", "तपासायची", "check eligibility"
    ]):
        return "eligibility"
    
    # Form filling keywords (CHECK THIS LAST)
    if any(keyword in user_msg_lower for keyword in [
        "अर्ज", "नवीन", "apply", "application", "form", "fill",
        "आवेदन", "फॉर्म", "भरायचा", "करायचा", "नया"
    ]):
        # ✅ IMPORTANT: Check if it's about status, not new application
        if any(word in user_msg_lower for word in ["स्थिती", "status", "पाहायची", "देखना"]):
            return "post_application"
        return "form_filling"
    
 
def get_aadhaar_confirmation_message(aadhaar_data: dict, language: str) -> str:
    """Generate Aadhaar confirmation question in user's language"""
    # ✅ Just pass the dict and language - no need to extract fields here
    return format_aadhaar_confirmation(aadhaar_data, language)

def is_affirmative_response(message: str) -> bool:
    """Check if user response is YES"""
    msg_lower = message.lower().strip()
    yes_keywords = [
    # English
    "yes", "yeah", "yep", "yup", "yah", "y", "k",
    "correct", "right", "ok", "okay",
    "sure", "fine", "good", "done", "agreed",
    
    # Marathi Devanagari
    "होय", "हो", "हाँ", "हां", "ठीक", "ठिक", 
    "बरोबर", "सही", "आहे", "होऊ", "हौ", "अवश्य",
    "जी हाँ", "ठीक है", "सही है", "बिल्कुल",
    
    # Marathi Romanized
    "hoy", "ho", "hau", "hou", "haa", "ha", "haan", "han",
    "thik", "theek", "barobar", "sahi", "aahe", "ahe",
    
    # Hindi Devanagari  
    "जी", "जरूर", "ओके",
    
    # Hindi Romanized
    "ji", "bilkul", "zaroor", "jarur",
    
    # Common phrases
    "thik hai", "theek hai", "sahi hai", 
    "thik ahe", "barobar ahe", "sahi ahe",
    "yes ji", "ok ji", "hoy hoy"
]
    return any(keyword in msg_lower for keyword in yes_keywords)


def is_negative_response(message: str) -> bool:
    """Check if user response is NO"""
    msg_lower = message.lower().strip()
    no_keywords = [
        "no", "nope", "wrong", "incorrect", "नाही", "नहीं", "चुकीचे", "गलत"
    ]
    return any(keyword in msg_lower for keyword in no_keywords)

async def handle_aadhaar_flow(
    session: dict,
    message: str,
    prev_res_mode: str,
    target_mode: str,  # "form_filling" or "post_application" or "intent_detection"
    session_id: str,
    prev_res: str,
    doc_type: str,
    file: UploadFile
):
    """
    Unified handler for Aadhaar verification and confirmation across all routes.
    Returns: response dict
    """
    # Detect language
    if not session.get("language"):
        session["language"] = detect_language(message)
    
    # ===== VERIFICATION MODE =====
    if prev_res_mode == f"{target_mode}_aadhaar_verify":
        print(f"{target_mode} - Processing Aadhaar")
        
        aadhaar_response = await process_aadhaar_details(
            message=message, session_id=session_id, prev_res=prev_res, doc_type=doc_type, 
            file=file, prev_res_mode=prev_res_mode
        )
        
        if aadhaar_response.get("both_sides_complete"):
            session["aadhaar_verified"] = True
            session["aadhaar_data"] = aadhaar_response.get("data")
            
            confirmation_msg = get_aadhaar_confirmation_message(
                session["aadhaar_data"], session["language"]
            )
            
            return {
                "response": {"response": confirmation_msg},
                "mode": f"{target_mode}_aadhaar_confirm"
            }
        else:
            return {
                "response": {"response": aadhaar_response.get("message")},
                "mode": f"{target_mode}_aadhaar_verify",
                "data": aadhaar_response.get("data")
            }
    
    # ===== CONFIRMATION MODE =====
    elif prev_res_mode == f"{target_mode}_aadhaar_confirm":
        print(f"{target_mode} - Handling confirmation")
        
        if is_affirmative_response(message):
            # User confirmed - proceed to target mode
            
            # ✅ NEW: Handle intent_detection mode
            if target_mode == "intent_detection":
                print("Aadhaar confirmed - now detecting intent from original_message")
                
                # Try to classify the original message
                original_msg = session.get("original_message", "")
                menu_choice = classify_menu_selection(original_msg)
                
                # If valid intent detected
                if menu_choice == "eligibility":
                    session["menu_selected"] = "eligibility"
                    session["current_mode"] = "eligibility"
                    
                    ai_result = get_ai_response(
                        session_id=session_id,
                        user_message=message,
                        aadhaar_data=session.get("aadhaar_data")
                    )

                    ai_response = ai_result["response"]
                    is_flow_complete = ai_result.get("is_complete", False)

                    response_mode = "eligibility_flow_complete" if is_flow_complete else "eligibility"

                    return {
                        "response": {"response": ai_response},
                        "mode": response_mode,
                        "lang": session.get("language", "marathi")  # ✅ Add user's language
                    }
                
                elif menu_choice == "form_filling":
                    session["menu_selected"] = "form_filling"
                    SESSION_MODE[session_id] = "form_filling"
                    
                    bot_response = get_bot_response(
                        session_id=session_id,
                        user_message="",
                        file_uploaded=None
                    )
                    
                    return {
                        "response": bot_response,
                        "mode": "form_filling"
                    }
                
                elif menu_choice == "post_application":
                    session["menu_selected"] = "post_application"
                    
                    # ✅ Get full Aadhaar number
                    aadhaar_number = session.get("aadhaar_data", {}).get("aadhaar_number", "")
                    
                    res = post_chat(ChatRequest(
                        session_id=session_id,
                        message=original_msg,
                        aadhaar_number=aadhaar_number ,
                        language=session.get("language","english")
                    ))
                    return res
                
                # ✅ NO CLEAR INTENT - Ask for clarification
                else:
                    print("No clear intent detected - asking user")
                    clarification_messages = {
                        "marathi": "कृपया मला तुमचा प्रश्न विचारा.",
                        "hindi": "कृपया मुझे अपना प्रश्न पूछें।",
                        "english": "Please ask me your question."
                    }
                    
                    clarification_msg = clarification_messages.get(
                        session.get("language", "english"), 
                        clarification_messages["english"]
                    )
                    
                    return {
                        "response": {"response": clarification_msg},
                        "mode": "awaiting_intent"
                    }
            
            elif target_mode == "form_filling":
                session["menu_selected"] = "form_filling"
                SESSION_MODE[session_id] = "form_filling"
                
                # ✅ Aadhaar already verified - proceed to form filling
                bot_response = get_bot_response(
                    session_id=session_id,
                    user_message="",
                    file_uploaded=None
                )
                
                return {
                    "response": {
                        "response": bot_response.get("response"),
                        "type": bot_response.get("type", "info")
                    },
                    "mode": "form_filling"
                }
            
            # EXISTING: Handle post_application mode
            else:  # post_application
                original_msg = session.get("original_message", "check application status")
                
                # Get full Aadhaar number
                aadhaar_number = session.get("aadhaar_data", {}).get("aadhaar_number", "")
                
                res = post_chat(ChatRequest(
                    session_id=session_id,
                    message=original_msg,
                    aadhaar_number=aadhaar_number,
                    language=session.get("language", "english")   
                ))
                return res
        
        elif is_negative_response(message):
            # Re-verify
            session["aadhaar_verified"] = False
            session["aadhaar_data"] = None
            return {
                "response": {"response": get_multilingual_message("aadhaar_request", session["language"])},
                "mode": f"{target_mode}_aadhaar_verify"
            }
        else:
            # Ask again
            return {
                "response": {"response": get_multilingual_message("clarify_yes_no", session["language"])},
                "mode": f"{target_mode}_aadhaar_confirm"
            }

# --------------------------------------------------
# SMART ROUTER SYSTEM PROMPT (UPDATED)
# --------------------------------------------------
ROUTER_SYSTEM_PROMPT = """
You are a smart intent router for the Maharashtra Government scheme
"Ladki Bahin Yojana".

Your job is to classify the user’s intent into EXACTLY ONE of the
following flags:

- "eligibility"
- "form_filling"
- "post_application"

You will be given:
1) The previous assistant response (prev_res) – may be empty or null
2) The current user message

You MUST consider BOTH together to determine intent.

--------------------------------------------------
CRITICAL OVERRIDE RULE – FORM FILLING (VERY IMPORTANT)
--------------------------------------------------

Route to "form_filling" ONLY if the user EXPLICITLY expresses intent
to START or DO a NEW APPLICATION.

Explicit intent phrases include (English / Hindi / Marathi examples):

- "I want to apply"
- "Apply for Ladki Bahin"
- "Start application"
- "New application"
- "Fill the form"
- "Submit application"
- "अर्ज करायचा आहे"
- "नवीन अर्ज"
- "लाडकी बहीण अर्ज भरायचा आहे"

❗ NEVER infer "form_filling" from:
- User providing data
- User answering questions
- User uploading documents
- Assistant requesting information

--------------------------------------------------
POST-APPLICATION CONTEXT OVERRIDE (CRITICAL)
--------------------------------------------------

If the previous assistant response is asking to:
- check / verify / validate / confirm
- link or confirm linkage
- fetch application status
- check payment, installment, or transaction
- troubleshoot issues after submission

AND the user responds by providing ANY of the following:
- mobile number
- Aadhaar number
- bank account number
- IFSC code
- yes / no / numeric confirmation

THEN you MUST route to "post_application",
EVEN IF the user never explicitly said they applied.

This rule OVERRIDES the "eligibility" flag.

--------------------------------------------------
FLAG DEFINITIONS
--------------------------------------------------

flag_type = "eligibility"
Use when:
- User is asking whether they qualify for the scheme
- User is checking eligibility rules or conditions
- Assistant is asking questions like:
  - age
  - income
  - marital status
  - residence
  - family details
- User provides personal info ONLY for eligibility determination
- Conversation is clearly BEFORE application

-------------------------------------

flag_type = "post_application"
Use when:
- User asks about application status
- Questions about payments or installments
- Aadhaar / mobile / bank / IFSC linkage AFTER submission
- Verification, validation, or tracking of an application
- User responds to verification-style questions
- Assistant uses words like:
  "check", "verify", "linked", "status", "payment",
  "installment", "transaction", "pending", "approved", "rejected"

-------------------------------------

flag_type = "form_filling"
Use ONLY when:
- User clearly and explicitly wants to APPLY
- User instructs to start, fill, or submit a NEW application

--------------------------------------------------
STRICT RULES (NON-NEGOTIABLE)
--------------------------------------------------

- Choose ONLY ONE flag
- NEVER guess or infer "form_filling"
- Data entry alone does NOT imply application intent
- Use prev_res to detect whether the user is responding
- Follow OVERRIDE rules before FLAG DEFINITIONS
- Return ONLY valid JSON
- No explanation, no extra text, no markdown

--------------------------------------------------
Output format:
{
  "flag_type": "eligibility" | "form_filling" | "post_application"
}

"""
CALL_CENTER__CHATBOT_ROUTER_SYSTEM_PROMPT = """
You are a smart intent router for the Maharashtra Government scheme
"Ladki Bahin Yojana".

Your job is to classify the user’s intent into EXACTLY ONE of the
following flags:

- "eligibility"
- "post_application"

You will be given:
1) The previous assistant response (prev_res) – may be empty or null
2) The current user message

You MUST consider BOTH together to determine intent.

--------------------------------------------------
POST-APPLICATION CONTEXT OVERRIDE (CRITICAL)
--------------------------------------------------

If the previous assistant response is asking to:
- check / verify / validate / confirm
- link or confirm linkage
- fetch application status
- check payment, installment, or transaction
- troubleshoot issues after submission

AND the user responds by providing ANY of the following:
- mobile number
- Aadhaar number
- bank account number
- IFSC code
- yes / no / numeric confirmation

THEN you MUST route to "post_application",
EVEN IF the user never explicitly said they applied.

This rule OVERRIDES all other rules.

--------------------------------------------------
FLAG DEFINITIONS
--------------------------------------------------

flag_type = "eligibility"
Use when:
- User is asking whether they qualify for the scheme
- User is checking eligibility rules or conditions
- Assistant is asking questions like:
  - age
  - income
  - marital status
  - residence
  - family details
- User provides personal info ONLY for eligibility determination
- Conversation is clearly BEFORE application
- User expresses interest but does NOT talk about application status,
  payments, verification, or submission

-------------------------------------

flag_type = "post_application"
Use when:
- User asks about application status
- Questions about payments or installments
- Aadhaar / mobile / bank / IFSC linkage AFTER submission
- Verification, validation, or tracking of an application
- User responds to verification-style questions
- Assistant uses words like:
  "check", "verify", "linked", "status", "payment",
  "installment", "transaction", "pending", "approved", "rejected"

--------------------------------------------------
STRICT RULES (NON-NEGOTIABLE)
--------------------------------------------------

- Choose ONLY ONE flag
- Use prev_res to detect whether the user is responding
- Data sharing alone does NOT imply post-application
  UNLESS override conditions are met
- Follow OVERRIDE rules before FLAG DEFINITIONS
- Return ONLY valid JSON
- No explanation, no extra text, no markdown

--------------------------------------------------
Output format:
{
  "flag_type": "eligibility" | "post_application"
}

"""


# --------------------------------------------------
# ROUTER FUNCTION (UPDATED)
# --------------------------------------------------
def route_message(message: str, prev_res: Optional[str]):
    user_payload = f"""
Previous assistant response:
{prev_res or "None"}

Current user message:
{message}
"""

    response = AZURE_CLIENT.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload}
        ],
        temperature=0,
        max_tokens=50
    )

    return json.loads(response.choices[0].message.content)


def route_message_call_center(message: str, prev_res: Optional[str]):
    user_payload = f"""
Previous assistant response:
{prev_res or "None"}

Current user message:
{message}
"""

    response = AZURE_CLIENT.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": CALL_CENTER__CHATBOT_ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload}
        ],
        temperature=0,
        max_tokens=50
    )

    return json.loads(response.choices[0].message.content)


@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Starting application initialization...")

        if not initialize_blob_storage():
            logger.warning("⚠️ Blob storage initialization failed. Document uploads will not work.")

        logger.info("✅ Application startup complete!")

    except Exception as e:
        logger.error(f"❌ FATAL ERROR during startup: {e}")
        raise


# --------------------------------------------------
# ROUTER API (UPDATED INPUT)
# --------------------------------------------------
@app.post("/smart-chat-router-ladki-bahin")
async def smart_chat_router(
        message: str = Form(...),
        session_id: str = Form(...),
        prev_res: Optional[str] = Form(None),
        doc_type: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        prev_res_mode: Optional[str] = Form(None)
):
    """
    Smart router for Ladki Bahin Yojana chatbot
    Uses previous response for better routing
    """
    print(f"Received message: {message}")
    # Initialize session
    session = initialize_session(session_id)
    
    # ==========================================
    # STEP 1.5: HANDLE EXIT MODE - USER WANTS TO RESTART
    # ==========================================
    if prev_res_mode == "exit":
        print("User exited previously - checking if they want to apply again")
        
        menu_choice = classify_menu_selection(message)
        
        # If user wants to apply, reset the form session completely
        if menu_choice == "form_filling":
            print("User wants to apply again after exit - resetting form session")
            session["menu_selected"] = "form_filling"
            SESSION_MODE[session_id] = "form_filling"
            
            # Reset the form session by removing old session data from registration.py
            from api.registration import sessions as reg_sessions
            if session_id in reg_sessions:
                del reg_sessions[session_id]
            
            # Start from Aadhaar verification (VERIFICATION MODE)
            return {
                "response": {"response": get_multilingual_message("aadhaar_request", session.get("language", "english"))},
                "mode": "form_filling_aadhaar_verify"
            }
        
        elif menu_choice == "eligibility":
            print("User wants to check eligibility after exit")
            session["menu_selected"] = "eligibility"
            
            ai_result = get_ai_response(
                session_id=session_id,
                user_message=message,
                aadhaar_data=session.get("aadhaar_data")
            )

            ai_response = ai_result["response"]
            is_flow_complete = ai_result.get("is_complete", False)
            response_mode = "eligibility_flow_complete" if is_flow_complete else "eligibility"

            return {
                "response": {"response": ai_response},
                "mode": response_mode,
                "lang": session.get("language", "marathi")
            }
        
        elif menu_choice == "post_application":
            print("User wants to check post-application status after exit")
            session["menu_selected"] = "post_application"
            
            aadhaar_number = session.get("aadhaar_data", {}).get("aadhaar_number", "")
            
            res = post_chat(ChatRequest(
                session_id=session_id,
                message=message,
                aadhaar_number=aadhaar_number,
                language=session.get("language", "english")
            ))
            return res
    
    # ==========================================
    # STEP 2: HANDLE FORM FILLING MODE
    # ==========================================
    if prev_res_mode == "form_filling":
        print("Previous mode was form filling")

        user_msg = message.strip().lower()

        # ✅ CHECK FOR EXIT FIRST
        exit_keywords = ["exit", "quit", "cancel", "abort", "stop", "end", "बाहेर", "बाहेर निघा", "रद्द", "थांबा", "समाप्त", "बाहर", "बाहर निकलें", "रद्द करें", "रोकें", "समाप्त करें"]
        
        should_exit = any(keyword in user_msg for keyword in exit_keywords)
        
        if should_exit:
            print("User chose to exit form filling")
            user_language = session.get("language", "english")
            
            exit_messages = {
                "marathi": "नोंदणी एजंटशी संवाद साधल्याबद्दल धन्यवाद.",
                "hindi": "पंजीकरण एजेंट के साथ बातचीत करने के लिए धन्यवाद।",
                "english": "Thank you for interacting with registration agent."
            }
            
            final_msg_registration = exit_messages.get(user_language, exit_messages["english"])
            
            return {
                "response": {"response": final_msg_registration},
                "mode": "exit"
            }

        file_uploaded = None
        if file and file.filename:
            # ✅ VALIDATE FILE TYPE BEFORE PROCESSING
            file_extension = Path(file.filename).suffix.lower()
            allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
            
            if file_extension not in allowed_extensions:
                user_language = session.get("language", "english")
                
                error_messages = {
                    "marathi": "❌ अवैध फाइल प्रकार! फक्त PDF, JPG, JPEG आणि PNG फाइल्स परवानगी आहे.",
                    "hindi": "❌ अमान्य फ़ाइल प्रकार! केवल PDF, JPG, JPEG और PNG फ़ाइलों की अनुमति है।",
                    "english": "❌ Invalid file type! Only PDF, JPG, JPEG, and PNG files are allowed."
                }
                
                return {
                    "response": {
                        "response": error_messages.get(user_language, error_messages["english"]),
                        "type": "error"
                    },
                    "mode": "form_filling"
                }
            
            # ✅ READ FILE CONTENT
            normalized_doc_type = None

            if doc_type:
                dt = doc_type.strip().lower()

                if "aadhaar" in dt:
                    normalized_doc_type = "aadhaar"
                elif "pan" in dt:
                    normalized_doc_type = "pan_card"
                elif "bank" in dt:
                    normalized_doc_type = "bank_passbook"
                else:
                    normalized_doc_type = dt
            else:
                normalized_doc_type = None

            file_uploaded = {
                "content": file.file.read(),
                "name": file.filename,
                "extension": file_extension,
                "doc_type": normalized_doc_type
            }


        # -----------------------------
        # 1. SUBMIT
        # -----------------------------
        if user_msg == "submit":
            print("User chose to submit form")
            bot_response = get_bot_response(
                session_id,
                message,
                file_uploaded
            )
            # Keep original format
            return {
                "response": bot_response,
                "mode": "Submit"
            }

         # ✅ CONTINUE FORM FILLING (for any other message)
        else:
            print("Continue form filling or call get_bot_response")
            try:
                bot_response = get_bot_response(
                    session_id,
                    message,
                    file_uploaded
                )
                print(f"✅ bot_response received: {bot_response}")
                # Keep original format
                return {
                    "response": bot_response,
                    "mode": "form_filling"
                }
            except Exception as e:
                import traceback
                print(f"❌ ERROR in get_bot_response:")
                print(f"❌ Error message: {str(e)}")
                print(f"❌ Full traceback:")
                traceback.print_exc()
                
                # Return error in user's language
                error_msg = f"Error: {str(e)}"
                return {
                    "response": {"response": error_msg},
                    "mode": "form_filling"
                }
    
    # ==========================================
    # STEP 3: HANDLE ELIGIBILITY AADHAAR MODE
    # ==========================================
    if prev_res_mode == "eligibility_aadhaar":
        print("Processing Aadhaar for eligibility")
        
        # Process Aadhaar (number or upload)
        aadhaar_response = await process_aadhaar_details(
            message=message,
            session_id=session_id,
            prev_res=prev_res,
            doc_type=doc_type,
            file=file,
            prev_res_mode=prev_res_mode
        )
        
        # Check if both sides complete
        if aadhaar_response.get("both_sides_complete"):
            print("Aadhaar verification complete - moving to eligibility")
            
            # Store Aadhaar data in session
            session["aadhaar_verified"] = True
            session["aadhaar_data"] = aadhaar_response.get("data")
            session["current_mode"] = "eligibility"
            
            # Ensure language is set if not already
            if not session.get("language"):
                session["language"] = detect_language(message)
            
            # Format confirmation message
            confirmation_msg = format_aadhaar_confirmation(
                session["aadhaar_data"], 
                session["language"]
            )

            # Match form_filling format (direct string)
            return {
                "response": {"response": confirmation_msg},
                "mode": "eligibility",
                "aadhaar_data": session["aadhaar_data"]
            }
        else:
            # Still processing Aadhaar (waiting for other side)
            # Match form_filling format (direct string)
            return {
                "response": {"response": aadhaar_response.get("message")},
                "mode": "eligibility_aadhaar",
                "data": aadhaar_response.get("data")
            }
        
    # ==========================================
    # STEP 2.5: HANDLE DPIP CONSENT MODE
    # ==========================================
    if prev_res_mode == "dpip_consent":
        print("Handling DPIP consent response")

        # DO NOT modify session["step"] here
        # Let registration.py control its own state

        bot_response = get_bot_response(
            session_id=session_id,
            user_message=message,
            file_uploaded=None
        )

        return {
            "response": {
                "response": bot_response.get("response"),
                "type": bot_response.get("type", "info"),
                "waiting_for": bot_response.get("waiting_for")
            },
            "mode": "form_filling"
        }
    
    # ==========================================
    # STEP 2.6: HANDLE AADHAAR INITIAL UPLOAD (with OCR)
    # ==========================================
    if prev_res_mode in ["upload_aadhaar_initial", "verifying_aadhaar", "confirm_aadhaar_details", "aadhaar_correction"]:
        print(f"Handling Aadhaar verification step: {prev_res_mode}")
        
        file_uploaded = None
        if file and file.filename:
            file_extension = Path(file.filename).suffix.lower()
            file_uploaded = {
                "content": file.file.read(),
                "name": file.filename,
                "extension": file_extension,
                "doc_type": "aadhaar"
            }
        
        bot_response = get_bot_response(
            session_id=session_id,
            user_message=message,
            file_uploaded=file_uploaded
        )
        
        return {
            "response": {
                "response": bot_response.get("response"),
                "type": bot_response.get("type", "info")
            },
            "mode": "form_filling"
        }
    
    # ==========================================
    # STEP 2.7: HANDLE PAN CARD UPLOAD
    # ==========================================
    if prev_res_mode in ["upload_pan_card", "pan_not_linked"]:
        print(f"Handling PAN card step: {prev_res_mode}")
        
        file_uploaded = None
        if file and file.filename:
            file_extension = Path(file.filename).suffix.lower()
            file_uploaded = {
                "content": file.file.read(),
                "name": file.filename,
                "extension": file_extension,
                "doc_type": "pan_card"
            }
        
        bot_response = get_bot_response(
            session_id=session_id,
            user_message=message,
            file_uploaded=file_uploaded
        )
        
        return {
            "response": {
                "response": bot_response.get("response"),
                "type": bot_response.get("type", "info")
            },
            "mode": "form_filling"
        }
    
    # ==========================================
    # STEP 4: HANDLE ELIGIBILITY MODE
    # ==========================================
    if prev_res_mode == "eligibility":
        print("In eligibility mode - checking if user wants to switch")
        
        #  CHECK IF USER WANTS TO SWITCH MODES
        menu_choice = classify_menu_selection(message)
        
        # If user explicitly wants to switch to form filling
        if menu_choice == "form_filling":
            print("User wants to switch to form filling from eligibility")
            session["menu_selected"] = "form_filling"
            session["original_message"] = message
        
            return {
                "response": {"response": get_multilingual_message("aadhaar_request", session["language"])},
                "mode": "form_filling_aadhaar_verify"
            }
        
        # If user explicitly wants to switch to post application
        elif menu_choice == "post_application":
            print("User wants to switch to post application from eligibility")
            session["menu_selected"] = "post_application"
            session["original_message"] = message
            
            # ALWAYS ASK FOR NEW AADHAAR - Don't reuse from other modes
            return {
                "response": {"response": get_multilingual_message("aadhaar_request", session["language"])},
                "mode": "post_application_aadhaar_verify"
            }
        
        # Otherwise, continue eligibility conversation
        else:
            print("Continuing eligibility conversation")
            
            ai_result = get_ai_response(
                session_id=session_id,
                user_message=message,
                aadhaar_data=session.get("aadhaar_data")
            )

            # Extract response and completion status
            ai_response = ai_result["response"]
            is_flow_complete = ai_result.get("is_complete", False)

            # Set mode based on completion status
            response_mode = "eligibility_flow_complete" if is_flow_complete else "eligibility"

            # Keep original nested format for eligibility
            return {
                "response": {
                    "response": ai_response
                },
                "mode": response_mode,
                "lang": session.get("language", "marathi")
            }
    
    # ==========================================
    # STEP 4.1: HANDLE AADHAAR FLOWS
    # ==========================================
    if prev_res_mode in [
        "form_filling_aadhaar_verify", "form_filling_aadhaar_confirm",
        "post_application_aadhaar_verify", "post_application_aadhaar_confirm",
        "intent_detection_aadhaar_verify", "intent_detection_aadhaar_confirm"  # ✅ ADD THIS
    ]:
        # Determine target mode
        if "form_filling" in prev_res_mode:
            target = "form_filling"
        elif "post_application" in prev_res_mode:
            target = "post_application"
        elif "intent_detection" in prev_res_mode:
            target = "intent_detection"  # ✅ ADD THIS
        
        return await handle_aadhaar_flow(
            session, message, prev_res_mode, target,
            session_id, prev_res, doc_type, file
        )
    
    # ==========================================
    # STEP 4.2: HANDLE AWAITING INTENT MODE
    # ==========================================
    if prev_res_mode == "awaiting_intent":
        print("User responded after clarification request")
        
        # Classify the new message
        menu_choice = classify_menu_selection(message)
        session["original_message"] = message
        
        if menu_choice == "eligibility":
            print("Intent: Eligibility")
            session["menu_selected"] = "eligibility"
            
            ai_result = get_ai_response(
                session_id=session_id,
                user_message=message,
                aadhaar_data=session.get("aadhaar_data")
            )

            ai_response = ai_result["response"]
            is_flow_complete = ai_result.get("is_complete", False)

            response_mode = "eligibility_flow_complete" if is_flow_complete else "eligibility"

            return {
                "response": {"response": ai_response},
                "mode": response_mode,
                "lang": session.get("language", "marathi")  # ✅ Add user's language
            }
        
        elif menu_choice == "form_filling":
            print("Intent: Form Filling")
            session["menu_selected"] = "form_filling"
            SESSION_MODE[session_id] = "form_filling"
            
            bot_response = get_bot_response(
                session_id=session_id,
                user_message="",
                file_uploaded=None
            )
            
            return {
                "response": {
                    "response": bot_response.get("response"),
                    "type": bot_response.get("type", "info")
                },
                "mode": "form_filling"
            }
        
        elif menu_choice == "post_application":
            print("Intent: Post Application")
            session["menu_selected"] = "post_application"
            
            # Get full Aadhaar number
            aadhaar_number = session.get("aadhaar_data", {}).get("aadhaar_number", "")
            
            res = post_chat(ChatRequest(
                session_id=session_id,
                message=message,
                aadhaar_number=aadhaar_number  ,
                language=session.get("language", "english") 
            ))
            return res
        
        else:
            # Still unclear - ask again
            return {
                "response": "I didn't understand. Please clearly tell me if you want to check eligibility, apply, or check status.",
                "mode": "awaiting_intent"
            }


    # ===================================================
    # STEP 4.3: HANDLE POST APPLICATION AWAITING AADHAAR
    # ===================================================
    if prev_res_mode == "post_application_awaiting_aadhaar":
        print("Post application - checking Aadhaar number")
        
        # Validate Aadhaar number format
        aadhaar_match = re.fullmatch(r"\d{12}", message.strip())
        
        if not aadhaar_match:
            # Invalid format - ask again
            error_messages = {
                "marathi": "अवैध आधार क्रमांक. कृपया १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "अमान्य आधार नंबर। कृपया १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Invalid Aadhaar number. Please enter a 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": error_messages.get(
                        session.get("language", "english"),
                        error_messages["english"]
                    )
                },
                "mode": "post_application_awaiting_aadhaar"
            }
        
        # Valid Aadhaar number - use full number
        aadhaar_number = aadhaar_match.group()
        
        # Store in session for future use
        session["post_app_aadhaar_number"] = aadhaar_number
        
        # Call post_chat with the full Aadhaar number
        original_msg = session.get("original_message", "check application status")
        
        res = post_chat(ChatRequest(
            session_id=session_id,
            message=original_msg,
            aadhaar_number=aadhaar_number  
        ))
        
        return res
    # ==========================================
    # STEP 5: MENU SELECTION (FIRST USER MESSAGE)
    # ==========================================
    if session["menu_selected"] is None:
        print("First user message - detecting menu selection")
        
        menu_choice = classify_menu_selection(message)
        session["original_message"] = message
        
        # -------- ELIGIBILITY SELECTED --------
        if menu_choice == "eligibility":
            print("Menu: Eligibility selected - requesting Aadhaar")
            session["menu_selected"] = "eligibility"
            session["current_mode"] = "eligibility_aadhaar"
            session["language"] = detect_language(message)
            
            # Return Aadhaar request message
            aadhaar_request = get_multilingual_message("aadhaar_request", session["language"])
            
            return {
                "response": {
                    "response": aadhaar_request
                },
                "mode": "eligibility_aadhaar"
            }
        
         # -------- FORM FILLING SELECTED --------
        elif menu_choice == "form_filling":
            print("Menu: Form filling selected - starting with DPIP consent")
            session["menu_selected"] = "form_filling"
            session["language"] = detect_language(message)
            SESSION_MODE[session_id] = "form_filling"
            
            # ✅ START WITH DPIP CONSENT (new flow)
            try:
                bot_response = get_bot_response(
                    session_id=session_id,
                    user_message="",  # Empty to trigger initial DPIP message
                    file_uploaded=None
                )
                print(f"✅ [FORM FILLING] bot_response type: {type(bot_response)}")
                print(f"✅ [FORM FILLING] bot_response keys: {bot_response.keys() if isinstance(bot_response, dict) else 'not a dict'}")
                print(f"✅ [FORM FILLING] bot_response['response']: {str(bot_response.get('response', 'NO RESPONSE KEY'))[:200]}")
                
                # ✅ FIX: Return in the format frontend expects
                # bot_response is already a dict with "response", "type", "waiting_for"
                # We need to wrap ONLY the "response" field
                return {
                    "response": {
                        "response": bot_response.get("response"),
                        "type": bot_response.get("type", "info"),
                        "waiting_for": bot_response.get("waiting_for")
                    },
                    "mode": "dpip_consent"  # ✅ Use proper mode name
                }
                
            except Exception as e:
                import traceback
                print(f"❌ ERROR calling get_bot_response:")
                print(f"❌ Error: {str(e)}")
                traceback.print_exc()
                
                return {
                    "response": {
                        "response": f"Debug Error: {str(e)}",
                        "type": "error"
                    },
                    "mode": "form_filling"
                }
        
        # -------- POST APPLICATION SELECTED --------
        elif menu_choice == "post_application":
            print("Menu: Post application selected")
            session["menu_selected"] = "post_application"
            session["language"] = detect_language(message)   
                
            # Directly ask for Aadhaar number
            aadhaar_request_messages = {
                "marathi": "कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Please enter your 12-digit Aadhaar number."
            }
            
            return {
            "response": {
                "response": aadhaar_request_messages.get(
                    session["language"], 
                    aadhaar_request_messages["english"]
                )
            },
            "mode": "post_application_awaiting_aadhaar"
            }
        
        # -------- UNKNOWN / NO CLEAR INTENT --------
        # User didn't select a clear menu option - request Aadhaar first
        else:
            print("Menu: No clear intent - requesting Aadhaar first")
            session["menu_selected"] = None  # Don't set yet, will decide after Aadhaar
            session["current_mode"] = "intent_detection_aadhaar"
            session["language"] = detect_language(message)
            
            # Request Aadhaar to start
            aadhaar_request = get_multilingual_message("aadhaar_request", session["language"])
            
            return {
                "response": {
                    "response": aadhaar_request
                },
                "mode": "intent_detection_aadhaar_verify"
            }
    
    # ==========================================
    # STEP 6: FALLBACK - USE ROUTER
    # ==========================================
    print("Fallback: Using router")
    routing_result = route_message(message, prev_res)
    print(f"Routing result: {routing_result}")

    route = routing_result["flag_type"]

    if route == 'eligibility':
        print("Routed to Eligibility Agent")

        ai_result = get_ai_response(
            session_id=session_id,
            user_message=message,
            aadhaar_data=session.get("aadhaar_data"),
            user_lang=session.get("language")
        )

        ai_response = ai_result["response"]
        is_flow_complete = ai_result.get("is_complete", False)

        response_mode = "eligibility_flow_complete" if is_flow_complete else "eligibility"

        return {
            "response": {"response": ai_response},
            "mode": response_mode,
            "lang": session.get("language", "marathi")
        }

    elif route == 'form_filling':
        print("Routed to Form Filling Agent")
        SESSION_MODE[session_id] = "form_filling"

        # ✅ DIRECTLY CALL registration.py
        bot_response = get_bot_response(
            session_id=session_id,
            user_message="",  # Empty message to trigger initial greeting
            file_uploaded=None
        )

        return {
            "response": {
                "response": bot_response.get("response"),
                "type": bot_response.get("type", "info")
            },
            "mode": "form_filling"
        }

    elif route == 'post_application':
        print("Routed to Post Application Agent")
        
        # ✅ Get full Aadhaar from session if available
        aadhaar_number = session.get("post_app_aadhaar_number")
        
        res_post_application = post_chat(ChatRequest(
            session_id=session_id,
            message=message,
            aadhaar_number=aadhaar_number,  
            language=session.get("language", "english") 
        ))
        print(f"Post Application Agent Response: {res_post_application}")

        return res_post_application

    # Keep original fallback format
    return {
        "session_id": session_id,
        "flag_type": routing_result["flag_type"]
    }

@app.post("/call-center-smart-chat-router-ladki-bahin")
async def call_center_smart_chat_router(
        message: str = Form(...),
        session_id: str = Form(...),
        prev_res: Optional[str] = Form(None),
        prev_res_mode: Optional[str] = Form(None)
):
    """
    Call center router - IDENTICAL to main chatbot
    Only difference: Aadhaar collection via NUMBER instead of OCR upload
    """
    print(f"[CALL CENTER] Received message: {message}")
    
    # Initialize session
    session = initialize_session(session_id)
    
    # ==========================================
    # STEP 1: HANDLE ELIGIBILITY AADHAAR MODE
    # ==========================================
    if prev_res_mode == "eligibility_aadhaar":
        print("[CALL CENTER] Collecting Aadhaar for eligibility")
        
        # Detect language if not set
        if not session.get("language"):
            session["language"] = detect_language(message)
        
        # Validate Aadhaar number format
        aadhaar_match = re.fullmatch(r"\d{12}", message.strip())
        
        if not aadhaar_match:
            error_messages = {
                "marathi": "अवैध आधार क्रमांक. कृपया १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "अमान्य आधार नंबर। कृपया १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Invalid Aadhaar number. Please enter a 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": error_messages.get(
                        session["language"],
                        error_messages["english"]
                    )
                },
                "mode": "eligibility_aadhaar"
            }
        
        # Valid Aadhaar - store and move to eligibility
        aadhaar_number = aadhaar_match.group()
        
        # Create aadhaar_data structure (same as chatbot after OCR)
        session["aadhaar_data"] = {
            "aadhaar_number": aadhaar_number,
            "full_name": "User",  # Placeholder, will be asked in conversation
            "age": None,
            "district": None
        }
        session["aadhaar_verified"] = True
        session["current_mode"] = "eligibility"
        
        # Generate confirmation message (same as chatbot)
        confirmation_msg = get_multilingual_message(
            "aadhaar_confirmation_number",  # Different key for number-based confirmation
            session["language"],
            aadhaar_number=f"XXXX-XXXX-{aadhaar_number[-4:]}"  # Mask for privacy
        )
        
        # If that key doesn't exist, use simple confirmation
        if "aadhaar_confirmation_number" not in confirmation_msg or confirmation_msg == "aadhaar_confirmation_number":
            confirmation_messages = {
                "marathi": f"आधार क्रमांक {aadhaar_number[-4:]} नोंदवला आहे. आता पात्रता तपासू.",
                "hindi": f"आधार नंबर {aadhaar_number[-4:]} दर्ज किया गया। अब पात्रता जांच शुरू करते हैं।",
                "english": f"Aadhaar number ending in {aadhaar_number[-4:]} recorded. Let's check eligibility."
            }
            confirmation_msg = confirmation_messages.get(session["language"], confirmation_messages["english"])
        
        return {
            "response": {"response": confirmation_msg},
            "mode": "eligibility",
            "aadhaar_data": session["aadhaar_data"]
        }
    
    # ==========================================
    # STEP 2: HANDLE ELIGIBILITY MODE
    # ==========================================
    if prev_res_mode == "eligibility":
        menu_choice = classify_menu_selection(message)
        
        if menu_choice == "post_application":
            print("Switching to post-application - requesting Aadhaar")
            session["menu_selected"] = "post_application"
            session["original_message"] = message
            
            # ✅ ALWAYS ASK FOR AADHAAR - Don't reuse from eligibility
            if not session.get("language"):
                session["language"] = detect_language(message)
            
            aadhaar_request_messages = {
                "marathi": "कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Please enter your 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": aadhaar_request_messages.get(
                        session["language"],
                        aadhaar_request_messages["english"]
                    )
                },
                "mode": "post_application_awaiting_aadhaar"
            }
            
        # Continue eligibility conversation (IDENTICAL to chatbot)
        else:
            ai_result = get_ai_response(
                session_id=session_id,
                user_message=message,
                aadhaar_data=session.get("aadhaar_data"),
                user_lang=session.get("language")
            )

            ai_response = ai_result["response"]
            is_flow_complete = ai_result.get("is_complete", False)
            response_mode = "eligibility_flow_complete" if is_flow_complete else "eligibility"

            return {
                "response": {"response": ai_response},
                "mode": response_mode,
                "lang": session.get("language", "marathi")
            }
    
    # ==========================================
    # STEP 3: HANDLE ELIGIBILITY FLOW COMPLETE
    # ==========================================
    if prev_res_mode == "eligibility_flow_complete":
        menu_choice = classify_menu_selection(message)
        
        if menu_choice == "post_application":
            print("Eligibility complete - switching to post-application")
            session["menu_selected"] = "post_application"
            session["original_message"] = message
            
            # ✅ ALWAYS ASK FOR AADHAAR
            if not session.get("language"):
                session["language"] = detect_language(message)
            
            aadhaar_request_messages = {
                "marathi": "कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Please enter your 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": aadhaar_request_messages.get(
                        session["language"],
                        aadhaar_request_messages["english"]
                    )
                },
                "mode": "post_application_awaiting_aadhaar"
            }
        
        # Continue conversation
        ai_result = get_ai_response(
            session_id=session_id,
            user_message=message,
            aadhaar_data=session.get("aadhaar_data"),
            user_lang=session.get("language")
        )

        ai_response = ai_result["response"]
        is_flow_complete = ai_result.get("is_complete", False)
        response_mode = "eligibility_flow_complete" if is_flow_complete else "eligibility"

        return {
            "response": {"response": ai_response},
            "mode": response_mode,
            "lang": session.get("language", "marathi")
        }
    
    # ==========================================
    # STEP 4: HANDLE POST APPLICATION AWAITING AADHAAR
    # ==========================================
    if prev_res_mode == "post_application_awaiting_aadhaar":
        print("[CALL CENTER] Validating Aadhaar for post-application")
        
        aadhaar_match = re.fullmatch(r"\d{12}", message.strip())
        
        if not aadhaar_match:
            if not session.get("language"):
                session["language"] = detect_language(message)
            
            error_messages = {
                "marathi": "अवैध आधार क्रमांक. कृपया १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "अमान्य आधार नंबर। कृपया १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Invalid Aadhaar number. Please enter a 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": error_messages.get(
                        session.get("language", "english"),
                        error_messages["english"]
                    )
                },
                "mode": "post_application_awaiting_aadhaar"
            }
        
        aadhaar_number = aadhaar_match.group()
        session["post_app_aadhaar_number"] = aadhaar_number
        
        original_msg = session.get("original_message", "check application status")
        
        res = post_chat(ChatRequest(
            session_id=session_id,
            message=original_msg,
            aadhaar_number=aadhaar_number,
            language=session.get("language", "english")
        ))
        
        return res
    
    # ==========================================
    # STEP 5: MENU SELECTION (FIRST MESSAGE)
    # ==========================================
    if session["menu_selected"] is None:
        print("[CALL CENTER] First message - detecting intent")
        
        menu_choice = classify_menu_selection(message)
        session["original_message"] = message
        
        # -------- ELIGIBILITY SELECTED --------
        if menu_choice == "eligibility":
            print("[CALL CENTER] Eligibility selected - requesting Aadhaar")
            session["menu_selected"] = "eligibility"
            session["current_mode"] = "eligibility_aadhaar"
            session["language"] = detect_language(message)
            
            # Ask for Aadhaar number (SAME FLOW as chatbot, different input method)
            aadhaar_request_messages = {
                "marathi": "पात्रता तपासण्यासाठी, कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "पात्रता जांचने के लिए, कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "To check eligibility, please enter your 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": aadhaar_request_messages.get(
                        session["language"],
                        aadhaar_request_messages["english"]
                    )
                },
                "mode": "eligibility_aadhaar"
            }
        
        # -------- POST APPLICATION SELECTED --------
        elif menu_choice == "post_application":
            print("[CALL CENTER] Post-application selected")
            session["menu_selected"] = "post_application"
            session["language"] = detect_language(message)
            
            aadhaar_request_messages = {
                "marathi": "कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Please enter your 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": aadhaar_request_messages.get(
                        session["language"],
                        aadhaar_request_messages["english"]
                    )
                },
                "mode": "post_application_awaiting_aadhaar"
            }
        
        # -------- UNKNOWN INTENT --------
        else:
            print("[CALL CENTER] No clear intent - asking for clarification")
            session["language"] = detect_language(message)
            
            clarification_messages = {
                "marathi": "कृपया मला सांगा - तुम्हाला पात्रता तपासायची आहे की अर्जाची स्थिती पाहायची आहे?",
                "hindi": "कृपया मुझे बताएं - आप पात्रता जांचना चाहते हैं या आवेदन की स्थिति देखना चाहते हैं?",
                "english": "Please tell me - do you want to check eligibility or check application status?"
            }
            
            return {
                "response": {
                    "response": clarification_messages.get(
                        session["language"],
                        clarification_messages["english"]
                    )
                },
                "mode": "awaiting_intent"
            }
    
    # ==========================================
    # STEP 6: HANDLE AWAITING INTENT
    # ==========================================
    if prev_res_mode == "awaiting_intent":
        print("[CALL CENTER] User clarified intent")
        
        menu_choice = classify_menu_selection(message)
        session["original_message"] = message
        
        if menu_choice == "eligibility":
            session["menu_selected"] = "eligibility"
            session["current_mode"] = "eligibility_aadhaar"
            
            aadhaar_request_messages = {
                "marathi": "पात्रता तपासण्यासाठी, कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "पात्रता जांचने के लिए, कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "To check eligibility, please enter your 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": aadhaar_request_messages.get(
                        session["language"],
                        aadhaar_request_messages["english"]
                    )
                },
                "mode": "eligibility_aadhaar"
            }
        
        elif menu_choice == "post_application":
            session["menu_selected"] = "post_application"
            
            aadhaar_request_messages = {
                "marathi": "कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Please enter your 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": aadhaar_request_messages.get(
                        session["language"],
                        aadhaar_request_messages["english"]
                    )
                },
                "mode": "post_application_awaiting_aadhaar"
            }
        
        else:
            clarification_messages = {
                "marathi": "कृपया स्पष्टपणे सांगा - पात्रता की स्थिती?",
                "hindi": "कृपया स्पष्ट रूप से बताएं - पात्रता या स्थिति?",
                "english": "Please clearly state - eligibility or status?"
            }
            
            return {
                "response": {
                    "response": clarification_messages.get(
                        session["language"],
                        clarification_messages["english"]
                    )
                },
                "mode": "awaiting_intent"
            }
    
    # ==========================================
    # STEP 7: FALLBACK - USE ROUTER
    # ==========================================
    print("[CALL CENTER] Fallback - using router")
    
    routing_result = route_message_call_center(message, prev_res)
    route = routing_result["flag_type"]

    if route == 'eligibility':
        print("[CALL CENTER] Routed to eligibility")
        
        if not session.get("aadhaar_verified"):
            session["menu_selected"] = "eligibility"
            session["current_mode"] = "eligibility_aadhaar"
            
            if not session.get("language"):
                session["language"] = detect_language(message)
            
            aadhaar_request_messages = {
                "marathi": "पात्रता तपासण्यासाठी, कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "पात्रता जांचने के लिए, कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "To check eligibility, please enter your 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": aadhaar_request_messages.get(
                        session["language"],
                        aadhaar_request_messages["english"]
                    )
                },
                "mode": "eligibility_aadhaar"
            }
        
        # Already have Aadhaar, continue
        ai_result = get_ai_response(
            session_id=session_id,
            user_message=message,
            aadhaar_data=session.get("aadhaar_data"),
            user_lang=session.get("language")
        )

        ai_response = ai_result["response"]
        is_flow_complete = ai_result.get("is_complete", False)
        response_mode = "eligibility_flow_complete" if is_flow_complete else "eligibility"

        return {
            "response": {"response": ai_response},
            "mode": response_mode,
            "lang": session.get("language", "marathi")
        }

    elif route == 'post_application':
        print("[CALL CENTER] Routed to post-application")

        # Allow user to override Aadhaar by entering a new 12-digit number at any time
        aadhaar_match = re.fullmatch(r"\d{12}", message.strip())
        if aadhaar_match:
            session["post_app_aadhaar_number"] = aadhaar_match.group()
            original_msg = session.get("original_message") or "check application status"

            res_post_application = post_chat(ChatRequest(
                session_id=session_id,
                message=original_msg,
                aadhaar_number=session["post_app_aadhaar_number"],
                language=session.get("language", "english")
            ))
            return res_post_application
        
        if not session.get("post_app_aadhaar_number"):
            session["original_message"] = message
            
            if not session.get("language"):
                session["language"] = detect_language(message)
            
            aadhaar_request_messages = {
                "marathi": "कृपया तुमचा १२ अंकी आधार क्रमांक प्रविष्ट करा.",
                "hindi": "कृपया अपना १२ अंकी आधार नंबर दर्ज करें।",
                "english": "Please enter your 12-digit Aadhaar number."
            }
            
            return {
                "response": {
                    "response": aadhaar_request_messages.get(
                        session["language"],
                        aadhaar_request_messages["english"]
                    )
                },
                "mode": "post_application_awaiting_aadhaar"
            }
        
        res_post_application = post_chat(ChatRequest(
            session_id=session_id,
            message=message,
            aadhaar_number=session["post_app_aadhaar_number"],
            language=session.get("language", "english")
        ))
        
        return res_post_application

    return {
        "session_id": session_id,
        "flag_type": routing_result["flag_type"]
    }


# --------------------------------------------------
# RUN
# --------------------------------------------------
# --------------------------------------------------
# SPEECH TOKEN API
# --------------------------------------------------
@app.get("/api/speech-token")
async def get_speech_token():
    """Get Azure Speech token with enhanced error handling"""
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = os.getenv("AZURE_SPEECH_REGION", "centralindia")

    if not speech_key:
        raise HTTPException(status_code=500, detail="Azure Speech API Key not configured")

    if not service_region:
        raise HTTPException(status_code=500, detail="Azure Speech Region not configured")

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
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to retrieve token: {response.text}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to speech service: {str(e)}"
        )

from fastapi.responses import Response
from api.text_to_speech import text_to_speech_gemini
# ... existing code ...

@app.post("/api/tts")
async def generate_tts(request: dict):
    """
    Generate speech from text using Gemini 2.5 TTS
    """
    try:
        text = request.get("text")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
            
        # Detect language logic can be enhanced, defaulting to 'en-IN' or 'hi-IN' based on text if needed
        # For now, let's pass a default or let the function handle it. 
        # The existing function signature is text_to_speech(text, language_code="en-IN", ...)
        
        audio_content = text_to_speech_gemini(filename="output.wav", api_key=api_key, text=text)
        
        if not audio_content:
             raise HTTPException(status_code=500, detail="Failed to generate audio")

        return Response(content=audio_content, media_type="audio/mp3")

    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=9015, reload=False)
