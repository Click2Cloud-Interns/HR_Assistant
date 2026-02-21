# utils.py
import io
import re
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import os
from openai import AzureOpenAI


from database import db_manager

# ============== Azure OpenAI Setup ==============
try:
    openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    print("✅ Azure OpenAI client initialized for Aadhaar parsing")
except Exception as e:
    openai_client = None
    AZURE_DEPLOYMENT = None
    print(f"⚠️ Azure OpenAI not available: {e}")

# ============== Router Setup ==============
router = APIRouter(prefix="/api", tags=["Aadhaar"])

# In-memory session storage (use Redis in production)
sessions = {}

def detect_language(text: str) -> str:
    """Detect language from user text using Azure OpenAI"""
    
    # Skip detection for numeric strings (Aadhaar, Mobile, etc.)
    if text.strip().isdigit():
        return "marathi" # Default to Marathi for numeric inputs in this app context
    
    if not openai_client:
        # Simple fallback without OpenAI
        text_lower = text.lower().strip()
        
        # Marathi/Hindi transliteration mapping (common short inputs)
        marathi_latin = ['ho', 'hao', 'hoay', 'hau', 'nahi', 'nahay', 'aahe']
        hindi_latin = ['ha', 'haan', 'nhi', 'nahi', 'hai']
        
        # Marathi-specific words (unique to Marathi, NOT in Hindi)
        marathi_unique = ['आहे', 'तुम्ही', 'आपण', 'होय', 'तपासायची', 'करायचा', 'पाहिजे', 'झाले', 'काय', 'कसे']
        marathi_count = sum(1 for word in marathi_unique if word in text)
        marathi_count += sum(1 for word in marathi_latin if word in text_lower.split())
        
        # Hindi-specific words (unique to Hindi, NOT in Marathi)
        hindi_unique = ['है', 'हैं', 'आप', 'मैं', 'हाँ', 'चाहिए', 'करना', 'क्या', 'कैसे', 'था']
        hindi_count = sum(1 for word in hindi_unique if word in text)
        hindi_count += sum(1 for word in hindi_latin if word in text_lower.split())
        
        # English words
        english_words = ['yes', 'no', 'is', 'are', 'the', 'i', 'you', 'what', 'how']
        english_count = sum(1 for word in text_lower.split() if word in english_words)
        
        # Determine language based on counts
        if marathi_count > hindi_count and marathi_count > english_count:
            return "marathi"
        elif hindi_count > english_count:
            return "hindi"
        elif english_count > 0:
            return "english"
        
        # Default to marathi if no clear match
        return "marathi"
    
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert language detector for Indian languages (Marathi, Hindi, English).

IDENTIFICATION RULES:
1. Marathi:
   - Script: आहे, होय, नाही, तुम्ही, पाहिजे
   - Latin: "ho", "hau", "hao", "hoay", "nahi", "aahe", "mala", "ka"
2. Hindi:
   - Script: है, हूँ, हाँ, नहीं, आप, चाहिए
   - Latin: "ha", "haan", "nhi", "hai", "mujhe", "kya"
3. English:
   - "yes", "no", "is", "are", "you", "who", "what"

CRITICAL: 
- If text is "ho", "hao", or "hau", it is MARATHI.
- If text is "ha" or "haan", it is HINDI.
- Only return "english" if it's clearly English (like "yes", "correct", "ok").

Return ONLY one word in lowercase: 'marathi' OR 'hindi' OR 'english'."""
                },
                {
                    "role": "user",
                    "content": f"Detect language for: {text}"
                }
            ],
            temperature=0,
            max_tokens=5
        )
        
        detected = response.choices[0].message.content.strip().lower()
        print(f"🔍 Language detected: '{detected}' for input: '{text[:50]}...'")
        
        # Validate response
        if detected in ["marathi", "hindi", "english"]:
            return detected
        
        # If invalid response, fallback
        print(f"⚠️ Invalid language detection response: {detected}")
        return "english"
        
    except Exception as e:
        print(f"⚠️ Language detection failed: {e}")
        return "english"

def get_multilingual_message(message_key: str, language: str, **kwargs) -> str:
    """
    Get localized message using Azure OpenAI.
    Supports template variables for dynamic content.
    """
    if not openai_client:
        # Fallback messages in English
        fallback = {
            "aadhaar_request": "please upload your Aadhaar card or enter your Aadhaar number.",
            "aadhaar_not_found": "Your Aadhaar number was not found in the database. Please upload your Aadhaar card: First upload the front side, then upload the back side.",
            "front_processed": "Aadhaar front side successfully processed. Now please upload the back side.",
            "back_processed": "Back side processed. Please upload the front side.",
            "both_complete": "Both sides of Aadhaar card successfully processed. Data has been saved.",
            "image_error": "Could not read text from image. Please upload a clear image.",
            "side_detection_error": "Could not identify Aadhaar card. Please upload a clear image.",
            "aadhaar_verified": "Aadhaar verified. Your information has been retrieved.",
            "aadhaar_not_in_db": "Aadhaar number not found in database. Please upload your Aadhaar card.",
            "front_side_uploaded": "Aadhaar front side processed. Please upload the back side.",
            "back_side_uploaded": "Back side processed. Please upload the front side.",
            "aadhaar_already_verified": "Aadhaar already verified.",
            "aadhaar_input_request": "Please enter your 12-digit Aadhaar number or upload your Aadhaar card.",
            "aadhaar_confirmation": "Aadhaar verified! According to Aadhaar, your name is {name}, your age is {age} years, and your address is in {district} district of Maharashtra. Now I will ask you some questions to check your eligibility.",
            "invalid_file_type": "Invalid file type. Only {extensions} are allowed.",  # ✅ NEW
            "ocr_extraction_failed": "Could not extract text from image. Please upload a clear image.",  # ✅ NEW
            "side_detection_failed": "Could not identify which side of Aadhaar card. Please upload a clear image."  # ✅ NEW
        }
        msg = fallback.get(message_key, "Operation completed.")
        return msg.format(**kwargs) if kwargs else msg
    
   # English message templates
    templates = {
        "aadhaar_request": "please upload your Aadhaar card or enter your 12 Digit Aadhaar number.",
        "aadhaar_not_found": "Your Aadhaar number was not found in the database. Please upload your Aadhaar card: First upload the front side, then upload the back side.",
        "front_processed": "Aadhaar front side successfully processed. Now please upload the back side.",
        "back_processed": "Back side processed. Please upload the front side.",
        "both_complete": "Both sides of Aadhaar card successfully processed. Data has been saved.",
        "image_error": "Could not read text from image. Please upload a clear image.",
        "side_detection_error": "Could not identify Aadhaar card. Please upload a clear image.",
        "aadhaar_verified": "Aadhaar verified. Your information has been retrieved.",
        "aadhaar_not_in_db": "Aadhaar number not found in database. Please upload your Aadhaar card.",
        "front_side_uploaded": "Aadhaar front side processed. Please upload the back side.",
        "back_side_uploaded": "Back side processed. Please upload the front side.",
        "aadhaar_already_verified": "Aadhaar already verified.",
        "aadhaar_input_request": "Please enter your 12-digit Aadhaar number or upload your Aadhaar card.",
        "invalid_file_type": "Invalid file type. Only {extensions} are allowed.",
        "ocr_extraction_failed": "Could not extract text from image. Please upload a clear image.",
        "side_detection_failed": "Could not identify which side of Aadhaar card. Please upload a clear image.",
        "invalid_aadhaar": "Invalid Aadhaar number format. Please enter a valid 12-digit Aadhaar number.",
        "clarify_yes_no": "Please answer with 'yes' or 'no'.",
        "aadhaar_confirmation": "Is the information from your Aadhaar correct? Please confirm. According to Aadhaar, your name is {name}, your age is {age} years, and your address is in {district} district of Maharashtra, and your pincode is {pincode}.",
    }
    
    english_message = templates.get(message_key, "Operation completed.")
    
    # Format with kwargs if provided
    if kwargs:
        english_message = english_message.format(**kwargs)
    
    # If language is English, return as-is
    if language == "english":
        return english_message
    
    try:
        target_lang = "Marathi" if language == "marathi" else "Hindi"
        
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": f"Translate to {target_lang}. Maintain all formatting and numbers. Return ONLY the translation."
                },
                {
                    "role": "user",
                    "content": english_message
                }
            ],
            temperature=0,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return english_message

# ============== OCR & Extraction Functions ==============

def extract_text_from_bytes(file_bytes: bytes, extension: str) -> str:
    try:
        if extension.lower() == ".pdf":
            images = convert_from_bytes(file_bytes, dpi=300)
            return "\n".join(
                pytesseract.image_to_string(img, lang="eng+hin")
                for img in images
            )
        else:
            img = Image.open(io.BytesIO(file_bytes))
            return pytesseract.image_to_string(img, lang="eng+hin")
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


def detect_aadhaar_side(text: str) -> str: 
    text_lower = text.lower()
    
    # Back side has address information - CHECK THIS FIRST
    if re.search(r'Address|पत्ता|पता', text, re.IGNORECASE):
        return "back"
    
    # Front side has name, DOB, and gender
    has_dob = bool(re.search(r'\b\d{2}/\d{2}/\d{4}\b', text))
    has_gender = bool(re.search(r'\b(?:Male|Female|पुरुष|महिला|स्त्री)\b', text, re.IGNORECASE))
    has_name_label = bool(re.search(r'\b(?:Name|नाम|नांव)\b', text, re.IGNORECASE))
    
    # If has DOB or gender or name label, it's front
    if has_dob or has_gender or has_name_label:
        return "front"
    
    return "unknown"


def parse_with_ai(text: str, document_side: str) -> Dict[str, Any]:
    if not openai_client:
        print("⚠️ Azure OpenAI not configured. Using basic extraction.")
        return None
    
    if document_side == "front":
        prompt = f"""Extract information from this Aadhaar card FRONT side OCR text.

Return ONLY a JSON object with these exact fields:
- aadhaar_number: 12 digits (remove all spaces)
- full_name: Complete name in English
- date_of_birth: In DD/MM/YYYY format
- gender: Either "Male" or "Female"

OCR Text:
{text}

Return ONLY valid JSON, no markdown, no explanation."""

    else:  # back side
        prompt = f"""Extract information from this Aadhaar card BACK side OCR text.

    Return ONLY a JSON object with these exact fields:
    - address: Complete address
    - pincode: 6-digit postal code
    - district: District name in English (e.g., Mumbai, Pune, Nagpur, Thane, Nashik, Ahmednagar, Satara, Kolhapur, etc.)
    - state: State name in English (e.g., Maharashtra, Madhya Pradesh)

    RULES:
    - If the address or district clearly indicates a state, return that state.
    - If the district/city is present, infer the state from the city/district when possible.
    - If a 6-digit pincode is present, use it to determine the state (pincode mapping has priority).
    - If address mentions Mumbai suburbs (Andheri, Borivali, Bandra, Dadar, Kurla, etc.) → district "Mumbai", state "Maharashtra".
    - If address mentions Pune areas (Hinjewadi, Kharadi, Hadapsar, Wakad, etc.) → district "Pune", state "Maharashtra".
    - If you cannot be certain of the state, return the best guess but include the district.

    OCR Text:
    {text}

    Return ONLY valid JSON, no markdown, no explanation."""

    try:
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured data from Aadhaar cards. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        result = result.replace("```json", "").replace("```", "").strip()
        
        parsed = json.loads(result)
        print(f"✅ AI successfully parsed {document_side} side")
        return parsed
        
    except Exception as e:
        print(f"⚠️ AI parsing failed: {e}")
        return None


def extract_aadhaar_front_details(ocr_text: str) -> Dict[str, Any]:
    # Try AI parsing first
    ai_result = parse_with_ai(ocr_text, "front")
    
    if ai_result:
        # Convert AI result to expected format
        data = {
            "AadhaarNo": ai_result.get("aadhaar_number"),
            "FullName": ai_result.get("full_name"),
            "DateOfBirth": None,
            "Gender": ai_result.get("gender")
        }
        
        # Parse date from AI result
        dob_str = ai_result.get("date_of_birth")
        if dob_str:
            try:
                data["DateOfBirth"] = datetime.strptime(dob_str, "%d/%m/%Y").date()
            except ValueError:
                pass
        
        return data
    
    # Fallback to regex-based extraction
    data = {
        "AadhaarNo": None,
        "FullName": None,
        "DateOfBirth": None,
        "Gender": None
    }

    # Extract 12-digit Aadhaar number
    aadhaar = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', ocr_text)
    if aadhaar:
        data["AadhaarNo"] = aadhaar.group().replace(" ", "")

    # Extract name (assumes "Name:" or "नाम:" label)
    name = re.search(r'(?:Name|नाम)[:\s]+([A-Za-z\s]+)', ocr_text, re.IGNORECASE)
    if name:
        data["FullName"] = name.group(1).strip()

    # Extract date of birth (DD/MM/YYYY format)
    dob = re.search(r'\b\d{2}/\d{2}/\d{4}\b', ocr_text)
    if dob:
        try:
            data["DateOfBirth"] = datetime.strptime(dob.group(), "%d/%m/%Y").date()
        except ValueError:
            data["DateOfBirth"] = None

    # Extract gender
    if re.search(r'\b(?:Female|महिला)\b', ocr_text, re.IGNORECASE):
        data["Gender"] = "Female"
    elif re.search(r'\b(?:Male|पुरुष)\b', ocr_text, re.IGNORECASE):
        data["Gender"] = "Male"

    return data


def extract_aadhaar_back_details(ocr_text: str) -> Dict[str, Any]:
    # Try AI parsing first
    ai_result = parse_with_ai(ocr_text, "back")
    
    if ai_result:
        # Normalize AI result and ensure State is present (fallbacks applied below)
        ai_addr = ai_result.get("address")
        ai_pin = ai_result.get("pincode")
        ai_district = ai_result.get("district")
        ai_state = ai_result.get("state")

        # If AI returned state, use it directly. Otherwise infer below.
        data = {
            "Address": ai_addr,
            "Pincode": ai_pin,
            "District": ai_district,
            "State": ai_state if ai_state else None,
            "Country": "India"
        }

        # Fallbacks: infer state from pincode prefix, district name, or address keywords
        if not data["State"]:
            inferred = None

            # small district -> state map (add more as needed)
            district_map = {
                "balaghat": "Madhya Pradesh",
                "nagpur": "Maharashtra",
                "pune": "Maharashtra",
                "mumbai": "Maharashtra",
                "thane": "Maharashtra",
                "nashik": "Maharashtra"
            }

            if data["Pincode"] and re.match(r"\d{6}", str(data["Pincode"])):
                pin = str(data["Pincode"]).strip()
                # Quick heuristic: 48xxxx -> Madhya Pradesh (covers Balaghat example)
                if pin.startswith("48"):
                    inferred = "Madhya Pradesh"

            if not inferred and data["District"]:
                key = data["District"].lower()
                inferred = district_map.get(key)

            if not inferred and data["Address"]:
                addr_lower = data["Address"].lower()
                if any(k in addr_lower for k in ["madhya pradesh", "मध्य प्रदेश", "m.p.", "m.p"]):
                    inferred = "Madhya Pradesh"
                elif any(k in addr_lower for k in ["maharashtra", "महाराष्ट्र"]):
                    inferred = "Maharashtra"

            data["State"] = inferred if inferred else data["State"] or "Maharashtra"
        return data
    
    # Fallback to regex-based extraction
    data = {
        "Address": None,
        "Pincode": None,
        "District": None,
        "State": "Maharashtra",
        "Country": "India"
    }

    # Extract 6-digit pincode
    pin = re.search(r'\b\d{6}\b', ocr_text)
    if pin:
        data["Pincode"] = pin.group()

    # Extract address (look for text after "Address:" label)
    address = re.search(r'(?:Address|पत्ता|पता)[:\s]+([\s\S]{30,200})', ocr_text, re.IGNORECASE)
    if address:
        addr = re.sub(r'\s+', ' ', address.group(1)).strip()
        data["Address"] = addr
        
        # Simple regex-based district detection for fallback
        if re.search(r'Mumbai|मुंबई|Andheri|Borivali|Bandra|Dadar|Kurla|Malad|Goregaon', addr, re.IGNORECASE):
            data["District"] = "Mumbai"
        elif re.search(r'Pune|पुणे|Hinjewadi|Kharadi|Hadapsar|Wakad', addr, re.IGNORECASE):
            data["District"] = "Pune"
        elif re.search(r'Nagpur|नागपूर', addr, re.IGNORECASE):
            data["District"] = "Nagpur"
        elif re.search(r'Thane|ठाणे', addr, re.IGNORECASE):
            data["District"] = "Thane"
        elif re.search(r'Nashik|नाशिक', addr, re.IGNORECASE):
            data["District"] = "Nashik"

    return data

def validate_aadhaar_number(aadhaar_number: str) -> bool:

    return bool(re.fullmatch(r'\d{12}', aadhaar_number))


def merge_aadhaar(front: Dict[str, Any], back: Dict[str, Any]) -> Dict[str, Any]:

    # Convert date object to string if it exists
    dob = front.get("DateOfBirth")
    if dob and hasattr(dob, 'isoformat'):
        dob = dob.isoformat()  # Converts date to 'YYYY-MM-DD' string

    # Calculate age from date of birth
    age = None
    if dob:
        try:
            dob_date = datetime.fromisoformat(dob)
            today = datetime.now()
            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
        except:
            pass
    
    return {
        "aadhaar_number": front.get("AadhaarNo"),
        "full_name": front.get("FullName"),
        "date_of_birth": dob,
        "age": age,
        "gender": front.get("Gender"),
        "address": back.get("Address"),
        "pincode": back.get("Pincode"),
        "district": back.get("District"),
        "state": back.get("State", "Maharashtra"),
        "country": back.get("Country", "India")
    }

def clean_text(text: str) -> str:

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep necessary punctuation
    text = re.sub(r'[^\w\s\/:,.-]', '', text)
    return text.strip()


# ============== Helper Functions ==============

def initialize_session(session_id: str):
    """Initialize session state for Aadhaar processing"""
    if session_id not in sessions:
        sessions[session_id] = {
            "aadhaar": {
                "front": None,
                "back": None,
                "merged": None,
                "source": None,
                "saved": False
            },
            "beneficiary_id": None,
            "language": None  # ✅ Add language field
        }
    return sessions[session_id]

def format_aadhaar_confirmation(aadhaar_data: dict, language: str = "marathi") -> str:
    """Format Aadhaar extracted data confirmation message"""
    

    age = aadhaar_data.get("age", "N/A")
    name = aadhaar_data.get("full_name", "N/A")
    district = aadhaar_data.get("district", "N/A")
    pincode = aadhaar_data.get("pincode", "N/A")
    state = aadhaar_data.get("state", "Maharashtra")
    
    if language == "marathi":
        return (
            f"आधार सत्यापित झाले!\n\n"
            f"आधारानुसार\n"
            f"तुमचे नाव {name} आहे,\n"
            f"तुमचे वय {age} वर्षे आहे,\n"
            f"तुमचा पत्ता {state} राज्यातील {district} जिल्ह्यात आहे.\n"
            f"पिनकोड: {pincode}\n\n"
            f"ही माहिती बरोबर आहे का? कृपया कळवा."
        )
    elif language == "hindi":
        return (
            f"आधार सत्यापित हो गया!\n\n"
            f"आधार के अनुसार\n"
            f"आपका नाम {name} है,\n"
            f"आपकी उम्र {age} वर्ष है,\n"
            f"आपका पता {state} राज्य के {district} जिले में है।\n"
            f"पिनकोड: {pincode}\n\n"
            f"क्या यह जानकारी सही है? कृपया बता दें।"
        )
    else:  # english
        return (
            f"Aadhaar verified!\n\n"
            f"According to Aadhaar\n"
            f"your name is {name},\n"
            f"your age is {age} years,\n"
            f"your address is in {district} district of {state}.\n"
            f"Pincode: {pincode}\n\n"
            f"Is this information correct? Please confirm."
        )
# ============== Helper Function for main.py ==============

async def process_aadhaar_details(
    message: str,
    session_id: str,
    prev_res: Optional[str] = None,
    doc_type: Optional[str] = None,
    file: Optional[UploadFile] = None,
    prev_res_mode: str = "eligibility"
) -> Dict[str, Any]:
    """Helper function to process Aadhaar details - to be called from main.py"""
    try:
        # Initialize session
        session = initialize_session(session_id)
        aadhaar_state = session["aadhaar"]

        # Get language from session if available, or detect from message
        # Don't detect language from file upload notifications
        if session.get("language"):
            user_language = session["language"]
            print(f"📝 Using stored language: {user_language}")
        elif message and not message.lower().startswith("uploaded:"):
            user_language = detect_language(message)
            session["language"] = user_language
            print(f"🔍 Detected and stored language: {user_language}")
        else:
            # Default language if no language stored and message is file upload
            user_language = "marathi"
            session["language"] = user_language
            print(f"📝 Using default language: {user_language}")
        
        # -------- OPTION A: Aadhaar Number Verification --------
        aadhaar_match = re.fullmatch(r"\d{12}", message.strip())
        if aadhaar_match:
            aadhaar_no = aadhaar_match.group()
            
            # Validate Aadhaar number
            if not validate_aadhaar_number(aadhaar_no):
                return {
                    "success": False,
                    "message": get_multilingual_message("invalid_aadhaar", user_language),
                    "data": None,
                    "both_sides_complete": False
                }
            
            # Check if Aadhaar exists in database
            data = db_manager.get_aadhaar_details(aadhaar_no)
            
            if data:
                # Calculate age from date of birth
                age = None
                if data.get("DateOfBirth"):
                    try:
                        dob = data["DateOfBirth"]
                        if isinstance(dob, str):
                            dob = datetime.fromisoformat(dob)
                        today = datetime.now()
                        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    except:
                        pass
                
                # Update session with found data
                aadhaar_state.update({
                    "source": "number",
                    "merged": {
                        "aadhaar_number": data["AadhaarNo"],
                        "full_name": data["FullName"],
                        "date_of_birth": str(data["DateOfBirth"]),
                        "age": age,
                        "gender": data["Gender"],
                        "address": data.get("Address"),
                        "district": data.get("District", data.get("City")),
                        "state": data.get("State", "Maharashtra"),
                        "pincode": data.get("Pincode")
                    },
                    "saved": True
                })
                
                return {
                    "success": True,
                    "message": get_multilingual_message("aadhaar_verified", user_language),
                    "data": aadhaar_state["merged"],
                    "both_sides_complete": True
                }
            else:
                return {
                    "success": False,
                    "message": get_multilingual_message("aadhaar_not_in_db", user_language),
                    "data": None,
                    "both_sides_complete": False
                }
        
        # -------- OPTION B: Aadhaar Card Upload --------
        if file and doc_type == "aadhaar" and aadhaar_state["source"] != "number":
            # Validate file type
            allowed_extensions = {".jpg", ".jpeg", ".png", ".pdf"}
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                return {
                    "success": False,
                    "message": get_multilingual_message(
                        "invalid_file_type", 
                        user_language, 
                        extensions=', '.join(allowed_extensions)
                    ),
                    "data": None,
                    "both_sides_complete": False
                }
            
            # Read file
            file_bytes = await file.read()
            
            # Extract text using OCR
            ocr_text = extract_text_from_bytes(file_bytes, file_extension)
            
            if not ocr_text.strip():
                return {
                    "success": False,
                    "message": get_multilingual_message("ocr_extraction_failed", user_language),
                    "data": None,
                    "both_sides_complete": False
                }
            
            # Detect side (front or back)
            side = detect_aadhaar_side(ocr_text)
            
            if side == "unknown":
                return {
                    "success": False,
                    "message": get_multilingual_message("side_detection_failed", user_language),
                    "data": None,
                    "both_sides_complete": False
                }
            
            # Set source as upload
            aadhaar_state["source"] = "upload"
            
            # Extract details based on detected side
            if side == "front":
                extracted_data = extract_aadhaar_front_details(ocr_text)
                aadhaar_state["front"] = extracted_data
                
                # Convert date to string for JSON serialization
                if extracted_data.get("DateOfBirth") and hasattr(extracted_data["DateOfBirth"], 'isoformat'):
                    extracted_data["DateOfBirth"] = extracted_data["DateOfBirth"].isoformat()
                
                return {
                    "success": True,
                    "message": get_multilingual_message("front_side_uploaded", user_language),
                    "data": extracted_data,
                    "side_detected": "front",
                    "both_sides_complete": False
                }
                
            elif side == "back":
                extracted_data = extract_aadhaar_back_details(ocr_text)
                aadhaar_state["back"] = extracted_data
                
                # Check if both sides are now available
                if aadhaar_state["front"] and aadhaar_state["back"]:
                    # Merge data from both sides
                    merged_data = merge_aadhaar(
                        aadhaar_state["front"],
                        aadhaar_state["back"]
                    )
                    aadhaar_state["merged"] = merged_data
                    
                    # Save to database if not already saved
                    if not aadhaar_state["saved"]:
                        beneficiary_id = db_manager.save_beneficiary_from_aadhaar(merged_data)
                        aadhaar_state["saved"] = True
                        session["beneficiary_id"] = beneficiary_id
                    
                    return {
                        "success": True,
                        "message": get_multilingual_message("both_complete", user_language),
                        "data": merged_data,
                        "side_detected": "back",
                        "both_sides_complete": True
                    }
                else:
                    return {
                        "success": True,
                        "message": get_multilingual_message("back_side_uploaded", user_language),
                        "data": extracted_data,
                        "side_detected": "back",
                        "both_sides_complete": False
                    }
            
            return {
                "success": True,
                "message": "File processed",
                "data": None,
                "both_sides_complete": False
            }
        
        # -------- No Aadhaar Processing Needed --------
        if not aadhaar_match and not (file and doc_type == "aadhaar"):
            # Check current status
            if aadhaar_state["merged"]:
                return {
                    "success": True,
                    "message": get_multilingual_message("aadhaar_already_verified", user_language),
                    "data": aadhaar_state["merged"],
                    "both_sides_complete": True
                }
            else:
                return {
                    "success": False,
                    "message": get_multilingual_message("aadhaar_input_request", user_language),
                    "data": None,
                    "both_sides_complete": False
                }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing request: {str(e)}",
            "data": None,
            "both_sides_complete": False
        }


# ============== API Endpoint ==============

@router.post("/aadhaar-details")
async def aadhaar_details(
    message: str = Form(...),
    session_id: str = Form(...),
    prev_res: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    prev_res_mode: str = Form("eligibility")
) -> Dict[str, Any]:
    """API endpoint that calls the helper function"""
    return await process_aadhaar_details(message, session_id, prev_res, doc_type, file, prev_res_mode)
