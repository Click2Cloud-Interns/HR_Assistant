"""
Ladki Bahin Yojana - Complete Application System (Language Selection: English/Hindi/Marathi)
FastAPI Backend with OCR, AI Parsing, Azure Blob Storage, and Database Integration
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import uuid
import re
import json
import logging
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Azure Blob Storage
from azure.storage.blob import BlobServiceClient, ContentSettings, BlobSasPermissions, generate_blob_sas

# Tesseract OCR
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# Azure OpenAI for intelligent parsing
from openai import AzureOpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database
try:
    from database import db_manager
    logger.info("✅ Database module imported")
except Exception as e:
    logger.error(f"❌ Failed to import database module: {e}")


# ============================================
# BASE ENGLISH TEMPLATES (for LLM Translation)
# ============================================

MESSAGE_TEMPLATES = {
    "aadhaar_prefilled": """Aadhaar details verified successfully.
════════════════════════════════════════════════════
Retrieved Information:
- Name: {name}
- Age: {age} years
- Date of Birth: {dob}
- Address: {address}
- District: {district}
════════════════════════════════════════════════════""",

"document_info": """you will need to upload the following documents:
════════════════════════════════════════════════════
1. Aadhaar Card
2. Domicile Proof (any ONE of: Domicile Certificate / Ration Card / Voter ID / Birth Certificate / School LC)
3. Income Certificate (compulsory, unless Yellow/Orange Ration Card selected as domicile proof)
4. Bank Passbook
5. Photograph
════════════════════════════════════════════════════
""",

    "mobile_prompt": """Enter your Mobile Number (10 digits):""",
    
    "mobile_confirmed": """Mobile: {mobile}

Enter your Email ID 
(or type 'skip' to skip):""",

    "email_confirmed": """Email: {email}

What is your Marital Status?

1. Married
2. Unmarried
3. Widow
4. Divorced

Please enter the number (1-4):""",

    "marital_confirmed": """Marital Status: {status}

════════════════════════════════════════════════════
Upload Aadhaar Card
════════════════════════════════════════════════════

Please upload your Aadhaar Card (PDF/JPG/PNG)

Use the plus button and select 'Aadhaar Card' from the dropdown.""",

"aadhaar_success": """Aadhaar Card uploaded successfully.

Extracted Data:
- Aadhaar: {aadhaar}""",

    "domicile_prompt": """
════════════════════════════════════════════════════
DOMICILE PROOF
════════════════════════════════════════════════════

Please select ONE document to prove Maharashtra domicile:

1. Domicile Certificate
2. Ration Card
3. Voter ID
4. Birth Certificate
5. School Leaving Certificate

Enter the number (1-5):""",

    "domicile_selected": """Selected: {doc_name}

Please upload your {doc_name} (PDF/JPG/PNG)

Use the plus button and select '{doc_name}' from the dropdown.""",

    "domicile_success": """{doc_name} uploaded successfully.

Extracted Data:
- Document: {doc_type}
- Name: {name}
{extra_info}

Please upload your Income Certificate (PDF/JPG/PNG)

Use the plus button and select 'Income Certificate' from the dropdown.""",

    "ration_color": """Ration Card uploaded successfully.

Extracted Data:
- Card Number: {card_number}
- Name: {name}

What is the color of your Ration Card?

1. Yellow
2. Orange
3. White

Enter the number (1-3):""",

    "ration_white": """Ration Card Type: {color}

Note: White Ration Card requires Income Certificate

Please upload your Income Certificate (PDF/JPG/PNG)

Use the plus button and select 'Income Certificate' from the dropdown.""",

    "ration_yellow_orange": """Ration Card Type: {color}

Income Certificate NOT required for {color} Ration Card.
Income will be extracted from Ration Card.

Please upload your Bank Passbook (first page) (PDF/JPG/PNG)

Use the plus button and select 'Bank Passbook' from the dropdown.""",

    "income_success": """Income Certificate uploaded successfully.

Extracted Data:
- Certificate No: {cert_no}
- Name: {name}
- Annual Income: Rs. {income}

Please upload your Bank Passbook (first page) (PDF/JPG/PNG)

Use the plus button and select 'Bank Passbook' from the dropdown.""",

    "bank_success": """Bank Passbook uploaded successfully.

Extracted Data:
- Account Holder: {holder}
- Account No: {account}
- IFSC: {ifsc}
- Bank: {bank}

Please upload your Passport Size Photograph (JPG, PNG, PDF)

Use the plus button and select 'Photograph' from the dropdown.""",

    "photo_success": """Photograph uploaded successfully.

════════════════════════════════════════════════════
APPLICATION FORM - FINAL REVIEW
════════════════════════════════════════════════════

Personal Details:
- Name: {name}
- Date of Birth: {dob} (Age: {age})
- Aadhaar: {aadhaar}
- Marital Status: {marital}

Contact Details:
- Mobile: {mobile}
- Email: {email}
- Address: {address}

Bank Details:
- Account: {account}
- IFSC: {ifsc}
- Bank: {bank}
- Aadhaar-Bank Link: Verified

Income Details:
- Annual Income: Rs. {income}
- Ration Card: {ration}

Documents Attached ({count}):
{doc_list}

════════════════════════════════════════════════════

Is all the above information correct?

Type 'YES' to proceed to declaration or 'NO' to make changes:""",

    "declaration": """
════════════════════════════════════════════════════
FINAL SECTION - DECLARATION
════════════════════════════════════════════════════

I hereby declare that:

- All information provided is true and correct
- I understand providing false information may lead to cancellation of benefits
- I have read and agree to the terms and conditions

════════════════════════════════════════════════════

Type 'I AGREE' to confirm:""",

    "declaration_accepted": """Declaration accepted.

Type 'SUBMIT' to submit your application:""",

    "aadhaar_exists": """APPLICATION FAILED

An application with this Aadhaar number already exists in our system.

If you believe this is an error, please contact the helpline.""",

    "success": """Processing your application...

- Validating all details
- Checking document integrity
- Uploading documents to secure storage
- Registering for tracking system

════════════════════════════════════════════════════
APPLICATION SUBMITTED SUCCESSFULLY
════════════════════════════════════════════════════

Congratulations, {name}!

Your Unique Application ID:
{app_id}

IMPORTANT: Save this ID for tracking

Monthly Benefit: Rs. 1,500
Payment Mode: Direct Bank Transfer (DBT)

════════════════════════════════════════════════════

Next Steps:

1. You will receive SMS/Email confirmation within 24 hours
2. Documents will be verified within 3-5 working days
3. If approved, first payment within 15 days
4. Track status using Application ID

SMS updates will be sent to {mobile}

Thank You!""",

    "invalid_date": """Invalid format. Please enter date in DD/MM/YYYY format:""",

    "invalid_mobile": """Invalid mobile number. Please enter 10 digits:""",

    "invalid_email": """Invalid email format. Please enter a valid email:""",
    
    "income_exceeds": """Registration not eligible: Annual income exceeds Rs. 2,50,000.""",

    "income_retry_prompt": """Please upload a corrected Income Certificate or type 'EXIT' to quit.""",

    "income_not_found": """Annual income value was not found in the uploaded Income Certificate. Please upload a clear certificate where the income amount is visible.""",

    "invalid_option": """Please enter a valid option:""",

    "upload_prompt": """Please upload your {doc}:""",

    "exit_prompt": """Your application has been submitted!""",

    "error": """An error occurred while processing your request. Please try again.""",
    
    "invalid_aadhaar": """Invalid Document! Please upload a valid Aadhaar Card.""",
    
    "invalid_bank_passbook": """Invalid Document! Please upload a valid Bank Passbook.""",
    
    "invalid_income_certificate": """Invalid Document! Please upload a valid Income Certificate.""",
    
    "invalid_ration_card": """Invalid Document! Please upload a valid Ration Card.""",
    
    "invalid_voter_id": """Invalid Document! Please upload a valid Voter ID Card.""",
    
    "invalid_domicile_certificate": """Invalid Document! Please upload a valid Domicile Certificate.""",
    
    "invalid_birth_certificate": """Invalid Document! Please upload a valid Birth Certificate.""",

    "invalid_school_leaving": """Invalid Document! Please upload a valid School Leaving Certificate.""",

    "invalid_name_mismatch": """Name Mismatch! The name on the document ('{extracted}') does not match the name you provided ('{expected}').""",

    "confirmation_yes_no": """Please type 'YES' or 'NO':""",

    "confirmation_i_agree": """Please type 'I AGREE' to accept the declaration:""",

    "confirmation_submit": """Type 'SUBMIT' to submit your application:""",

    "edit_request": """Please type 'EXIT' and select the section you want to proceed with """,

    "restart_prompt": """Your application has been submitted successfully!""",

   "mobile_number_prompt": """📱 Enter your **Mobile Number** (10 digits):""",

    "dpip_consent": """To verify your identity, we need to collect your personal information (Aadhaar number, PAN number, income details, bank information, etc.).

This information will be kept secure according to Digital Personal Data Protection Act (DPDP) rules.

Do you consent to share your information for verification?

Please answer: होय / नाही""",

    "dpip_declined": """Thank you for your response.

You can come back anytime.

Ladki Bahin Yojana – For your empowerment.

════════════════════════════════════════════════════
Application process ended.
════════════════════════════════════════════════════""",

    "dpip_accepted": """Thank you. Please upload a SINGLE IMAGE that contains BOTH the FRONT and BACK sides of your Aadhaar Card.

════════════════════════════════════════════════════
IMPORTANT:
• Front and Back side must be in the SAME photo
• Both sides must be clearly visible
• Text should be readable
════════════════════════════════════════════════════

Use the plus button and select 'Aadhaar Card' from the dropdown.""",

    "aadhaar_verification_started": """Your information verification has started...""",

    "aadhaar_details_retrieved": """Aadhaar details retrieved:

Name: {name}
Date of Birth: {dob}
Address: {address}

Is this information correct?

Type YES if correct or CORRECTION if you need to make changes:""",

    "aadhaar_correction_prompt": """Please specify what needs to be corrected and we will assist you.""",

    "pan_upload_prompt": """Please upload your PAN Card.

════════════════════════════════════════════════════
IMPORTANT: Upload image of PAN Card
════════════════════════════════════════════════════

Use the plus button and select 'PAN Card' from the dropdown.""",

    "pan_verification_started": """Your PAN number verification is in progress...""",

    "pan_aadhaar_linked": """Your Aadhaar and PAN are successfully linked.

Proceeding to next step...""",

    "pan_aadhaar_not_linked": """Your Aadhaar and PAN are not linked.

Please link them first before proceeding with the application.

Visit: https://www.incometax.gov.in/iec/foportal/

════════════════════════════════════════════════════
Application process ended.
════════════════════════════════════════════════════""",

    "invalid_pan": """Invalid Document! Please upload a valid PAN Card.""",

    "document_uploaded_success": """{doc_name} uploaded successfully.

Document is being verified...""",

    "confirmation_yes_correction": """Please type 'YES' or 'CORRECTION':"""
}

# ============================================
# DYNAMIC TRANSLATION FUNCTION
# ============================================

def get_translated_message(message_key: str, language: str, **kwargs) -> str:
    """
    Use Azure OpenAI to translate messages dynamically.
    
    Args:
        message_key: Key from MESSAGE_TEMPLATES
        language: Target language (marathi/hindi/english)
        **kwargs: Template variables to format (e.g., name, mobile, etc.)
    
    Returns:
        Translated message with formatted variables
    """
    # If English, return directly
    if language == "english" or language not in ["marathi", "hindi"]:
        base_message = MESSAGE_TEMPLATES.get(message_key, MESSAGE_TEMPLATES["error"])
        return base_message.format(**kwargs) if kwargs else base_message
    
    # Check if OpenAI client is available
    if not openai_client:
        logger.warning("OpenAI client not available, falling back to English")
        base_message = MESSAGE_TEMPLATES.get(message_key, MESSAGE_TEMPLATES["error"])
        return base_message.format(**kwargs) if kwargs else base_message
    
    try:
        base_message = MESSAGE_TEMPLATES.get(message_key)
        
        # If message key doesn't exist, return error
        if base_message is None:
            logger.error(f"Message key '{message_key}' not found in MESSAGE_TEMPLATES")
            return "An error occurred. Please try again."
        
        # Build system prompt for translation
        system_prompt = f"""You are a professional translator for Maharashtra Government's Ladki Bahin Yojana scheme.

    Translate the following English text to {language.title()} in a formal, respectful, government-appropriate tone.

    CRITICAL RULES:
    1. Maintain ALL formatting exactly (line breaks, dashes, separators like ════)
    2. Keep ALL placeholders EXACTLY as-is: {{name}}, {{mobile}}, {{app_id}}, etc.
    3. Do NOT translate placeholder variable names inside curly braces
    4. Keep numbers, currency symbols (Rs.), and special characters unchanged
    5. Maintain the same indentation and spacing - this is very important
    6. Use formal/respectful tone appropriate for government communication
    7. Preserve all line breaks and empty lines for proper formatting
    8. Keep section headers aligned and properly indented

    Return ONLY the translated text, no explanations or notes."""

        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_message}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        translated = response.choices[0].message.content.strip()
        
        # Format with provided kwargs
        formatted_message = translated.format(**kwargs) if kwargs else translated
        
        return formatted_message
        
    except Exception as e:
        logger.error(f"Translation error for key '{message_key}' to {language}: {e}")
        logger.error(f"Full error details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Fallback to English with formatting
    try:
        base_message = MESSAGE_TEMPLATES.get(message_key, MESSAGE_TEMPLATES.get("error"))
        if base_message is None:
            return "An error occurred. Please try again."
        
        # Try formatting with kwargs
        if kwargs:
            try:
                return base_message.format(**kwargs)
            except KeyError as ke:
                logger.error(f"Missing key in template: {ke}")
                return base_message
        return base_message
    except Exception as format_error:
        logger.error(f"Formatting error in fallback: {format_error}")
        return "An error occurred. Please try again."

# ============================================
# AZURE BLOB STORAGE CONFIGURATION
# ============================================

AZURE_SA_NAME = os.getenv("AZURE_SA_NAME", "")
AZURE_SA_ACCESSKEY = os.getenv("AZURE_SA_ACCESSKEY", "")
AZURE_STORAGE_CONTAINER_NAME = "ladki-bahin-documents"

blob_service_client = None
container_client = None

def initialize_blob_storage():
    """Initialize Azure Blob Storage with private container"""
    global blob_service_client, container_client
    
    if not AZURE_SA_NAME or not AZURE_SA_ACCESSKEY:
        logger.error("❌ AZURE_SA_NAME or AZURE_SA_ACCESSKEY not found in .env")
        return False
    
    try:
        account_url = f"https://{AZURE_SA_NAME}.blob.core.windows.net"
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=AZURE_SA_ACCESSKEY
        )
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        
        try:
            container_client.create_container()
            logger.info(f"✅ Created private blob container: {AZURE_STORAGE_CONTAINER_NAME}")
        except Exception:
            logger.info(f"✅ Blob container already exists: {AZURE_STORAGE_CONTAINER_NAME}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize blob storage: {e}")
        return False


def upload_to_blob(file_content: bytes, application_id: str, document_type: str, file_extension: str) -> str:
    """Upload document to Azure Blob Storage (PRIVATE) and return SAS URL"""
    try:
        if not container_client:
            logger.error("❌ Blob storage not initialized")
            raise Exception("Blob storage not initialized")
        
        blob_name = f"{application_id}/{document_type}{file_extension}"
        logger.info(f"📤 Uploading: {blob_name}")
        
        blob_client = container_client.get_blob_client(blob_name)
        
        content_settings = ContentSettings(
            content_type='application/pdf' if file_extension == '.pdf' else f'image/{file_extension[1:]}'
        )
        
        blob_client.upload_blob(file_content, overwrite=True, content_settings=content_settings)
        
        sas_token = generate_blob_sas(
            account_name=AZURE_SA_NAME,
            container_name=AZURE_STORAGE_CONTAINER_NAME,
            blob_name=blob_name,
            account_key=AZURE_SA_ACCESSKEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=10000)
        )
        
        blob_url = f"https://{AZURE_SA_NAME}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER_NAME}/{blob_name}?{sas_token}"
        
        logger.info(f"✅ Document uploaded: {blob_name}")
        logger.info(f"🔗 SAS URL generated (length: {len(blob_url)} chars)")
        
        return blob_url
    
    except Exception as e:
        logger.error(f"❌ Blob upload failed: {e}")
        raise


def download_from_blob(blob_url: str) -> bytes:
    """Download document from Azure Blob Storage for OCR processing"""
    try:
        blob_name = blob_url.split(f"{AZURE_STORAGE_CONTAINER_NAME}/")[1].split("?")[0]
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob()
        return blob_data.readall()
    except Exception as e:
        logger.error(f"❌ Blob download failed: {e}")
        return b""


# ============================================
# CROSS-PLATFORM TESSERACT SETUP
# ============================================

def setup_tesseract():
    """Configure Tesseract OCR for both Windows and Linux"""
    try:
        if os.name == "nt":
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"✅ Windows: Tesseract found at {tesseract_path}")
            else:
                logger.warning("⚠️ Windows: Tesseract not found at default path.")
        else:
            result = shutil.which("tesseract")
            if result:
                pytesseract.pytesseract.tesseract_cmd = result
                logger.info(f"✅ Linux: Tesseract found at {result}")
            else:
                logger.warning("⚠️ Linux: Tesseract not found. Attempting to install...")
                try:
                    subprocess.run(["apt-get", "update"], check=False, stdout=subprocess.DEVNULL)
                    subprocess.run(["apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-eng", "tesseract-ocr-hin"], 
                                 check=True, stdout=subprocess.DEVNULL)
                    result = shutil.which("tesseract")
                    if result:
                        pytesseract.pytesseract.tesseract_cmd = result
                        logger.info(f"✅ Successfully installed Tesseract at {result}")
                except Exception as e:
                    logger.error(f"❌ Failed to install Tesseract: {e}")
    except Exception as e:
        logger.error(f"❌ Error configuring Tesseract: {e}")

setup_tesseract()


# ============================================
# FASTAPI APP SETUP
# ============================================

app = FastAPI(title="Ladki Bahin Yojana API")

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Starting application initialization...")
        
        if not initialize_blob_storage():
            logger.warning("⚠️ Blob storage initialization failed. Document uploads will not work.")
        
        logger.info("✅ Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR during startup: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

openai_client = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY:
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    logger.info("✅ Azure OpenAI client initialized")

sessions = {}


# ============================================
# DOCUMENT CONFIGURATION
# ============================================

DOCUMENT_TYPES = {
    "aadhaar": "Aadhaar Card",
    "pan_card": "PAN Card",
    "domicile_certificate": "Domicile Certificate",
    "ration_card": "Ration Card",
    "voter_id": "Voter ID",
    "birth_certificate": "Birth Certificate",
    "school_leaving": "School Leaving Certificate",
    "income_certificate": "Income Certificate",
    "bank_passbook": "Bank Passbook",
    "photograph": "Photograph",
}

DOMICILE_PROOF_OPTIONS = ["domicile_certificate", "ration_card", "voter_id", "birth_certificate", "school_leaving"]


# ============================================
# CUSTOM DOCUMENT INTELLIGENCE
# ============================================

class DocumentIntelligence:
    """Custom OCR + AI parsing system with document validation"""
    
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6'
        self.tesseract_lang = 'eng+hin'
        
        self.validation_keywords = {
            "aadhaar": [
                "aadhaar", "आधार", "aadhar", "unique identification", 
                "uidai", "government of india", "भारत सरकार"
            ],
            "pan_card": [
                "income tax", "आयकर", "permanent account number", "pan",
                "पॅन", "income tax department", "आयकर विभाग"
            ],
            "bank_passbook": [
                "bank", "बैंक", "बँक", "account", "passbook", "savings", "current",
                "ifsc", "branch", "balance"
            ],
            "income_certificate": [
                "income", "आय", "उत्पन्न", "certificate", "प्रमाणपत्र", "annual income",
                "वार्षिक आय", "tehsildar", "तहसीलदार","प्रमाणणपत्र"
            ],
            "ration_card": [
                "ration", "राशन", "शिधापत्रिका", "पुरवठापत्रिका", "food",
                "अन्न", "civil supplies", "नागरी पुरवठा"
            ],
            "voter_id": [
                "election", "निर्वाचन", "voter", "मतदाता", "epic",
                "election commission of india", "भारत निर्वाचन आयोग"
            ],
            "domicile_certificate": [
                "domicile", "अधिवास", "निवास", "residence", "certificate",
                "प्रमाणपत्र", "maharashtra", "महाराष्ट्र"
            ],
            "birth_certificate": [
                "birth", "जन्म", "certificate", "प्रमाणपत्र", "registration",
                "पंजीकरण", "नोंदणी"
            ],
            "school_leaving": [
                "school", "शाळा", "स्कूल", "leaving", "certificate", "प्रमाणपत्र",
                "education", "शिक्षण", "student"
            ],
            "photograph": []
        }
    
    def extract_text_from_bytes(self, file_content: bytes, file_extension: str, document_type: str = None) -> str:
        """Extract raw text from file bytes using Tesseract OCR"""
        try:
            if file_extension.lower() == '.pdf':
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                
                images = convert_from_path(tmp_path, dpi=300)
                text = ""
                for img in images:
                    text += pytesseract.image_to_string(img, lang=self.tesseract_lang, config=self.tesseract_config)
                    if document_type == "income_certificate":
                        text += "\n" + self._ocr_with_preprocessing(img)
                
                os.unlink(tmp_path)
                return text
            else:
                img = Image.open(io.BytesIO(file_content))
                base_text = pytesseract.image_to_string(img, lang=self.tesseract_lang, config=self.tesseract_config)
                if document_type == "income_certificate":
                    return base_text + "\n" + self._ocr_with_preprocessing(img)
                return base_text
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return f"OCR Error: {str(e)}"

    def _ocr_with_preprocessing(self, img: Image.Image) -> str:
        """Extra OCR pass with preprocessing to improve numeric capture."""
        try:
            gray = img.convert("L")
            # Try multiple thresholds and OCR modes to capture faint digits
            texts = []
            for thresh in (140, 160, 180, 200):
                enhanced = gray.point(lambda x, t=thresh: 0 if x < t else 255, mode="1")
                for psm in (6, 7, 11, 13):
                    cfg = f"--oem 3 --psm {psm}"
                    texts.append(pytesseract.image_to_string(enhanced, lang=self.tesseract_lang, config=cfg))
            return "\n".join(texts)
        except Exception as e:
            logger.error(f"OCR preprocessing error: {e}")
            return ""
    
    def validate_document_type(self, raw_text: str, document_type: str, user_language: str = "english") -> tuple:
        """Validate if the uploaded document matches the expected document type"""
        if document_type == "photograph":
            return True, None
        
        keywords = self.validation_keywords.get(document_type, [])
        if not keywords:
            return True, None
        
        text_lower = raw_text.lower()
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return True, None
        
        error_messages = {
            "aadhaar": get_translated_message("invalid_aadhaar", user_language),
            "bank_passbook": get_translated_message("invalid_bank_passbook", user_language),
            "income_certificate": get_translated_message("invalid_income_certificate", user_language),
            "ration_card": get_translated_message("invalid_ration_card", user_language),
            "voter_id": get_translated_message("invalid_voter_id", user_language),
            "domicile_certificate": get_translated_message("invalid_domicile_certificate", user_language),
            "birth_certificate": get_translated_message("invalid_birth_certificate", user_language),
            "school_leaving": get_translated_message("invalid_school_leaving", user_language)
        }
        return False, error_messages.get(document_type, "Invalid Document!")
    
    def validate_name(self, extracted_name: str, expected_name: str, user_language: str = "english") -> tuple:
        """Validate if the name extracted from document matches the expected name"""
        if not extracted_name or not expected_name:
            return True, None
        
        def normalize_name(name):
            return ' '.join(name.lower().strip().split())
        
        extracted_normalized = normalize_name(extracted_name)
        expected_normalized = normalize_name(expected_name)
        
        extracted_words = set(extracted_normalized.split())
        expected_words = set(expected_normalized.split())
        
        if extracted_words == expected_words:
            return True, None
        
        common_words = extracted_words.intersection(expected_words)
        if len(common_words) >= max(1, len(expected_words) * 0.7):
            return True, None
        
        error_msg = get_translated_message("invalid_name_mismatch", user_language, extracted=extracted_name, expected=expected_name)
        return False, error_msg
    
    def parse_with_ai(self, text: str, document_type: str) -> Dict[str, Any]:
        """Use AI to intelligently extract fields from OCR text"""
        
        if not openai_client:
            logger.warning("⚠️ Azure OpenAI not configured. Using basic extraction.")
            return self.basic_extract(text, document_type)
        
        prompts = {
            "aadhaar": f"""Extract from Aadhaar card:
- aadhaar_number (12 digits, remove spaces)
- name (full name)
- dob (DD/MM/YYYY)
- gender (M/F)
- address (complete address)

OCR Text:
{text}

Return JSON only with these exact keys.""",

            "pan_card": f"""Extract from PAN card:
- pan_number (10 characters: 5 letters + 4 digits + 1 letter, e.g., ABCDE1234F)
- name (full name as on card)
- father_name (father's name)
- date_of_birth (DD/MM/YYYY)

OCR Text:
{text}

Return JSON only with these exact keys.""",
"income_certificate": f"""Extract from income certificate:
- annual_income (extract the numeric value only, remove rupee symbol, commas, and spaces. For example: "₹ 4,20,000" should be extracted as "420000")
- certificate_number (format: MH-IC-XXXXXXX or similar)
- issuing_authority (Tahsildar Office or issuing authority name)
- issue_date (DD/MM/YYYY format)
- holder_name (full name of the person)

CRITICAL RULES:
1. For annual_income: Remove ₹, Rs., commas, spaces. Return only digits.
2. Examples: "₹ 4,20,000" → "420000", "Rs. 2,50,000" → "250000"
3. Do NOT return barcode numbers, certificate numbers (like MH-IC-...), or other long digit sequences as `annual_income`.
    If the OCR text contains a barcode or certificate id, avoid using that value as income.
4. Prefer numbers that directly follow the label "Annual Income" (or its Marathi/Hindi equivalent) in the document.
5. If no clear income value is found near the label, return an empty string or 0 for `annual_income` rather than guessing.
3. Look for income in table format with "Annual Income" header
4. Certificate number usually starts with state code (MH, GJ, etc.)

OCR Text:
{text}

Return ONLY valid JSON with these exact keys. No explanations.""",
            
            "domicile_certificate": f"""Extract from domicile certificate:
- certificate_number
- holder_name (name of person)
- state
- district
- taluka
- village
- issue_date (DD/MM/YYYY)

OCR Text:
{text}

Return JSON only.""",
            
            "ration_card": f"""Extract from ration card:
- card_number
- card_type (Yellow/Orange/White - determine from color mentioned or category)
- holder_name (head of family name)
- family_members (count)
- issue_date
- annual_income (if mentioned, number only)

OCR Text:
{text}

Return JSON only.""",

            "voter_id": f"""Extract from Voter ID Card:
- voter_id_number (EPIC number - format like ABC1234567)
- holder_name (elector name)
- father_name
- address
- date_of_birth

OCR Text:
{text}

Return JSON only.""",

            "birth_certificate": f"""Extract from birth certificate:
- certificate_number
- name (child name)
- date_of_birth (DD/MM/YYYY)
- place_of_birth
- father_name
- mother_name

OCR Text:
{text}

Return JSON only.""",

            "school_leaving": f"""Extract from school leaving certificate:
- certificate_number
- student_name (name of student)
- date_of_birth
- school_name
- district
- issue_date

OCR Text:
{text}

Return JSON only."""
        }
        
        prompt = prompts.get(document_type, f"Extract key information from:\n{text}\n\nReturn JSON.")
        
        try:
            response = openai_client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "Extract structured data from OCR text. Return ONLY valid JSON with no markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            return json.loads(result_text)
            
        except Exception as e:
            logger.error(f"AI parsing error: {e}")
            return self.basic_extract(text, document_type)
    
    def basic_extract(self, text: str, document_type: str) -> Dict[str, Any]:
        """Basic regex extraction as fallback"""
        result = {}
        
        if document_type == "aadhaar":
            aadhaar_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)
            if aadhaar_match:
                result['aadhaar_number'] = aadhaar_match.group(0).replace(' ', '')
        
        elif document_type == "pan_card":
            pan_match = re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text)
            if pan_match:
                result['pan_number'] = pan_match.group(0)
        
        elif document_type == "bank_passbook":
            ifsc_match = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', text)
            if ifsc_match:
                result['ifsc_code'] = ifsc_match.group(0)
            
            account_matches = re.findall(r'\b\d{9,18}\b', text)
            if account_matches:
                result['account_number'] = account_matches[0]
        
        elif document_type == "ration_card":
            card_match = re.search(r'\b(MH\d{10,}|\d{10,})\b', text)
            if card_match:
                result['card_number'] = card_match.group(0)
        
        elif document_type == "voter_id":
            voter_match = re.search(r'\b[A-Z]{3}\d{7}\b', text)
            if voter_match:
                result['voter_id_number'] = voter_match.group(0)
        
        return result
    
    def get_name_field(self, fields: Dict, document_type: str) -> str:
        """Get the name field from extracted data based on document type"""
        name_keys = {
            "aadhaar": "name",
            "pan_card": "name",
            "bank_passbook": "account_holder_name",
            "income_certificate": "holder_name",
            "domicile_certificate": "holder_name",
            "ration_card": "holder_name",
            "voter_id": "holder_name",
            "birth_certificate": "name",
            "school_leaving": "student_name"
        }
        key = name_keys.get(document_type, "name")
        return fields.get(key, "") or fields.get("name", "")
    
    def analyze_document(self, file_content: bytes, file_extension: str, document_type: str, 
                blob_url: str, expected_name: str = None, user_language: str = "english") -> Dict[str, Any]:
        """Complete document analysis with validation"""
        raw_text = self.extract_text_from_bytes(file_content, file_extension, document_type)
        
        is_valid_type, type_error = self.validate_document_type(raw_text, document_type, user_language)
        if not is_valid_type:
            return {
                "document_type": document_type,
                "raw_text": raw_text if document_type == "income_certificate" else raw_text[:500],
                "fields": {},
                "is_valid": False,
                "validation_error": type_error,
                "blob_url": blob_url
            }
        
        structured_data = self.parse_with_ai(raw_text, document_type)
        
        if expected_name and document_type != "photograph":
            extracted_name = self.get_name_field(structured_data, document_type)
            is_valid_name, name_error = self.validate_name(extracted_name, expected_name, user_language)
            if not is_valid_name:
                return {
                    "document_type": document_type,
                    "raw_text": raw_text if document_type == "income_certificate" else raw_text[:500],
                    "fields": structured_data,
                    "is_valid": False,
                    "validation_error": name_error,
                    "blob_url": blob_url
                }
        
        return {
            "document_type": document_type,
            "raw_text": raw_text if document_type == "income_certificate" else raw_text[:500],
            "fields": structured_data,
            "is_valid": True,
            "validation_error": None,
            "blob_url": blob_url
        }


doc_intelligence = DocumentIntelligence()


# ============================================
# HELPER FUNCTIONS
# ============================================

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date from DD/MM/YYYY format"""
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except:
        return None

def calculate_age(dob_str: str) -> int:
    """Calculate age from date of birth string"""
    try:
        dob = datetime.strptime(dob_str, "%d/%m/%Y")
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except:
        return 0

def mask_aadhaar(aadhaar: str) -> str:
    """Mask Aadhaar number for display"""
    if aadhaar and len(aadhaar) >= 4:
        return f"XXXX-XXXX-{aadhaar[-4:]}"
    return "Not provided"

def mask_account(account: str) -> str:
    """Mask account number for display"""
    if account and len(account) >= 4:
        return f"XXXX-XXXX-{account[-4:]}"
    return "Not provided"


def parse_numeric(value: Any) -> float:
    """Parse a numeric amount from various formats (commas, ₹, Rs., spaces).

    Returns float value or 0.0 if parsing fails.
    """
    try:
        if value is None:
            return 0.0
        s = str(value)
        # Quick reject: if the string contains letters mixed with digits in ways that look like IDs (e.g., MH-IC-12345), return 0
        if re.search(r'[A-Za-z]{2,}[- ]?\d', s):
            # allow if there's also a rupee sign nearby
            if '₹' not in s and 'Rs' not in s and 'रु' not in s:
                return 0.0

        # Remove common currency symbols and whitespace but keep commas and digits
        cleaned = re.sub(r'[₹Rs\.\s]', '', s)

        # Find all numeric groups (allow commas)
        candidates = re.findall(r'[0-9,]{1,15}(?:\.\d+)?', cleaned)
        nums = []
        for c in candidates:
            # strip commas
            c_clean = c.replace(',', '')
            # skip excessively long numbers (likely barcodes)
            if len(c_clean) > 9:
                continue
            try:
                val = float(c_clean)
                nums.append(val)
            except Exception:
                continue

        if not nums:
            return 0.0

        # Prefer values that are <= 10 million (sensible cap) and > 0
        sensible = [n for n in nums if 0 < n <= 10000000]
        if sensible:
            # If multiple, choose the smallest sensible (more likely income)
            return float(sorted(sensible)[0])

        # Otherwise choose smallest numeric candidate under very large cap
        nums = [n for n in nums if n > 0 and n < 1e12]
        if nums:
            return float(sorted(nums)[0])

    except Exception:
        pass
    return 0.0

def parse_income_amount(raw_income: Any, raw_text: str = "") -> float:
    """Parse income with strong preference for values near income labels."""
    candidates = []

    if raw_income:
        parsed = parse_numeric(raw_income)
        if parsed:
            candidates.append(parsed)

    text = raw_text or ""
    if text:
        label_pattern = re.compile(
            r'(annual\s*income|annual_income|वार्षिक\s*आय|वार्षिक\s*उत्पन्न|आय|उत्पन्न)',
            re.IGNORECASE
        )
        for match in label_pattern.finditer(text):
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 120)
            window = text[start:end]
            window = re.sub(r'(?<=\d)\s+(?=\d)', '', window)
            for num in re.findall(r'[₹Rs\.\s]*([0-9][0-9,]{2,9})', window):
                val = parse_numeric(num)
                if val:
                    candidates.append(val)

        for match in re.finditer(r'(?:₹|Rs\.?|रु\.?)\s*([0-9][0-9,]{2,9})', text):
            val = parse_numeric(match.group(1))
            if val:
                candidates.append(val)

        compact = re.sub(r'\s+', ' ', text)
        for match in label_pattern.finditer(compact):
            start = max(0, match.start() - 20)
            end = min(len(compact), match.end() + 120)
            window = compact[start:end]
            window = re.sub(r'(?<=\d)\s+(?=\d)', '', window)
            for num in re.findall(r'([0-9][0-9,]{2,9})', window):
                val = parse_numeric(num)
                if val:
                    candidates.append(val)

    candidates = [c for c in candidates if 1000 <= c <= 10000000]
    if candidates:
        non_years = [c for c in candidates if not (1900 <= c <= 2099)]
        if non_years:
            candidates = non_years
    if not candidates:
        return 0.0
    return float(sorted(candidates)[-1])


def format_currency(num: float) -> str:
    """Format number to Indian-style currency string with ₹ and commas.

    Example: 420000 -> '₹ 4,20,000'
    """
    try:
        n = int(round(num))
        s = f"{n:,}"
        # Convert to Indian grouping (replace default commas with Indian grouping)
        parts = s.split(',')
        if len(parts) <= 3:
            # standard formatting already acceptable for smaller numbers
            return f"₹ {s}"
        # For Indian grouping, join last 3 then preceding groups of 2
        last3 = parts[-3:]
        prefix = ''.join(parts[:-3])
        if prefix:
            # insert commas every two digits from right in prefix
            prefix = prefix[::-1]
            grouped = ','.join([prefix[i:i+2] for i in range(0, len(prefix), 2)])
            grouped = grouped[::-1]
            formatted = grouped + ',' + ','.join(last3)
        else:
            formatted = ','.join(last3)
        return f"₹ {formatted}"
    except Exception:
        return str(num)


# ============================================
# CHATBOT LOGIC
# ============================================

def get_bot_response(session_id: str, user_message: str = "", file_uploaded: dict = None):
    """Main chatbot conversation logic with Aadhaar pre-fill and dynamic translation"""
    
    # ✅ GET LANGUAGE AND AADHAAR DATA FROM MAIN.PY SESSION
    from main import SESSION_DATA
    
    session = sessions.get(session_id)
    if not session:
        session = {
            "step": "dpip_consent",  # ✅ Start with DPIP consent
            "documents": {},
            "extracted_data": {},
            "personal_info": {},
            "contact_info": {},
            "bank_info": {},
            "income_info": {},
            "domicile_info": {},
            "uploaded_docs": [],
            "conversation": [],
            "ration_card_color": None,
            "domicile_proof_type": None,
            "beneficiary_id": None,
            "application_id": None,
            "language": None,
            "aadhaar_prefilled": False
        }
        sessions[session_id] = session
    
    user_language = session.get("language") or "english"  # Default fallback
    
    # ✅ Get language and Aadhaar data from main.py session
    if session_id in SESSION_DATA and not session.get("language_locked"):
        detected_language = SESSION_DATA[session_id].get("language")
        if detected_language:
            user_language = detected_language
            session["language"] = user_language
        
        # ✅ PRE-POPULATE AADHAAR DATA IF NOT ALREADY DONE
        aadhaar_data = SESSION_DATA[session_id].get("aadhaar_data")
        if aadhaar_data and not session.get("aadhaar_prefilled"):
            session["personal_info"]["name"] = aadhaar_data.get("full_name", "")
            session["personal_info"]["dob"] = aadhaar_data.get("date_of_birth", "")
            session["personal_info"]["age"] = aadhaar_data.get("age", "")
            session["contact_info"]["address"] = aadhaar_data.get("address", "")
            session["extracted_data"]["aadhaar_number"] = aadhaar_data.get("aadhaar_number", "")
            session["extracted_data"]["name_from_aadhaar"] = aadhaar_data.get("full_name", "")
            session["domicile_info"]["district"] = aadhaar_data.get("district", "")
            session["aadhaar_prefilled"] = True
            
            logger.info(f"✅ Pre-filled Aadhaar data for session {session_id}")
    
    if not user_language and "language" in session:
        user_language = session["language"]
    
    logger.info(f"📝 Using language: {user_language} for session {session_id}")
    
    if user_message:
        session["conversation"].append({"role": "user", "message": user_message})
    
    # Handle RESTART command
    if user_message and user_message.lower() in ["restart", "start over", "exit"]:
        sessions[session_id] = {
            "step": "dpip_consent",
            "documents": {},
            "extracted_data": {},
            "personal_info": {},
            "contact_info": {},
            "bank_info": {},
            "income_info": {},
            "domicile_info": {},
            "uploaded_docs": [],
            "conversation": [],
            "ration_card_color": None,
            "domicile_proof_type": None,
            "beneficiary_id": None,
            "application_id": None,
            "language": session.get("language"),
            "aadhaar_prefilled": False,
            "language_locked": False
        }
        session = sessions[session_id]

    
    current_step = session["step"]
    response = {}
    
    # ✅ STEP 0: DPIP CONSENT
    if current_step == "dpip_consent":
        if not user_message:
            # Show consent message
            response = {
                "response": get_translated_message("dpip_consent", user_language),
                "type": "info",
                "waiting_for": "dpip_response"
            }
        else:
            user_input = user_message.strip().lower()

            yes_words = ["yes", "y", "ho", "होय", "हां", "haan"]
            no_words = ["no", "n", "nahi", "नाही", "नहीं"]

            if user_input in yes_words:
                session["dpip_accepted"] = True
                session["language_locked"] = True  # 🔒 Lock language
                session["step"] = "upload_aadhaar_initial"
                response = {
                    "response": get_translated_message("dpip_accepted", user_language),
                    "type": "success",
                    "waiting_for": "aadhaar_upload_initial"
                }

            elif user_input in no_words:
                session["dpip_accepted"] = False
                session["step"] = "declined"
                response = {
                    "response": get_translated_message("dpip_declined", user_language),
                    "type": "info",
                    "waiting_for": "none"
                }

            else:
                response = {
                    "response": get_translated_message("confirmation_yes_no", user_language),
                    "type": "error",
                    "waiting_for": "dpip_response"
                }
    
    # ✅ STEP 0.5: UPLOAD AADHAAR INITIAL (with OCR)
    elif current_step == "upload_aadhaar_initial":
        if file_uploaded and file_uploaded.get("doc_type") == "aadhaar":
            application_id = session.get("application_id")
            if not application_id:
                application_id = db_manager.generate_application_id() if db_manager else f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
                session["application_id"] = application_id
            
            # Show verification started message
            session["step"] = "verifying_aadhaar"
            response = {
                "response": get_translated_message("aadhaar_verification_started", user_language),
                "type": "info",
                "waiting_for": "processing"
            }
            
            # Perform OCR and extract data
            result = doc_intelligence.analyze_document(
                file_uploaded["content"],
                file_uploaded["extension"],
                "aadhaar",
                "",  # Blob URL will be added after confirmation
                None,
                user_language
            )
            
            if not result.get("is_valid"):
                session["step"] = "upload_aadhaar_initial"
                response = {
                    "response": result.get("validation_error", get_translated_message("invalid_aadhaar", user_language)),
                    "type": "error",
                    "waiting_for": "aadhaar_upload_initial"
                }
            else:
                fields = result.get("fields", {})
                
                # Store extracted data temporarily
                session["temp_aadhaar_data"] = {
                    "file_content": file_uploaded["content"],
                    "file_extension": file_uploaded["extension"],
                    "fields": fields,
                    "name": fields.get("name", ""),
                    "dob": fields.get("dob", ""),
                    "address": fields.get("address", ""),
                    "aadhaar_number": fields.get("aadhaar_number", "")
                }
                
                session["step"] = "confirm_aadhaar_details"
                
                # Dynamic confirmation text based on language
                if user_language == "marathi":
                    confirmation_text = "योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:"
                elif user_language == "hindi":
                    confirmation_text = "यदि जानकारी सही है तो 'हाँ' टाइप करें या बदलाव के लिए 'सुधार' टाइप करें:"
                else:
                    confirmation_text = "Type YES if correct or CORRECTION if you need to make changes:"

                response = {
                    "response": f"""Aadhaar details retrieved:

                Name: {fields.get("name", "N/A")}
                Date of Birth: {fields.get("dob", "N/A")}
                Address: {fields.get("address", "N/A")}

                {confirmation_text}""",
                    "type": "success",
                    "waiting_for": "aadhaar_confirmation"
                }
        else:
            response = {
                "response": get_translated_message("dpip_accepted", user_language),
                "type": "text",
                "waiting_for": "aadhaar_upload_initial"
            }
    
    # ✅ STEP 0.6: CONFIRM AADHAAR DETAILS
    elif current_step == "confirm_aadhaar_details":

        user_input = user_message.strip().lower()

        yes_words = ["yes", "होय", "हां", "y"]
        correction_words = ["correction", "दुरुस्ती", "सुधार"]

        if user_input in yes_words:
            # User confirmed details are correct
            temp_data = session.get("temp_aadhaar_data", {})
            application_id = session.get("application_id")
            
            # Now upload to blob storage using user's name as folder
            user_name = temp_data.get("name", "unknown").replace(" ", "_")
            
            blob_url = upload_to_blob(
                temp_data.get("file_content"),
                user_name,  # Use name as folder instead of application_id
                "aadhaar",
                temp_data.get("file_extension")
            )
            
            # Store in session
            fields = temp_data.get("fields", {})
            session["personal_info"]["name"] = fields.get("name", "")
            session["personal_info"]["dob"] = fields.get("dob", "")
            session["personal_info"]["age"] = calculate_age(fields.get("dob", ""))
            session["contact_info"]["address"] = fields.get("address", "")
            session["extracted_data"]["aadhaar_number"] = fields.get("aadhaar_number", "")
            session["aadhaar_prefilled"] = True
            
            session["documents"]["aadhaar"] = {
                "is_valid": True,
                "fields": fields,
                "blob_url": blob_url
            }
            session["uploaded_docs"].append("aadhaar")
            
            # Clear temp data
            session.pop("temp_aadhaar_data", None)
            
            # Move to PAN upload
            session["step"] = "upload_pan_card"
            response = {
                "response": get_translated_message("pan_upload_prompt", user_language),
                "type": "success",
                "waiting_for": "pan_card_upload"
            }
            
        elif user_input in correction_words:
            session["step"] = "aadhaar_correction"
            response = {
                "response": get_translated_message("aadhaar_correction_prompt", user_language),
                "type": "info",
                "waiting_for": "correction_input"
            }
        else:
            response = {
                "response": get_translated_message("confirmation_yes_correction", user_language),
                "type": "error",
                "waiting_for": "aadhaar_confirmation"
            }
    
    # ✅ STEP 0.7: HANDLE AADHAAR CORRECTION
    elif current_step == "aadhaar_correction":

        temp_data = session.get("temp_aadhaar_data", {})
        fields = temp_data.get("fields", {})

            # If user is telling which field to correct
        if not session.get("correction_field"):

                msg = user_message.lower()

                if "name" in msg:
                    session["correction_field"] = "name"
                elif "dob" in msg or "birth" in msg:
                    session["correction_field"] = "dob"
                elif "address" in msg:
                    session["correction_field"] = "address"
                else:
                    response = {
                        "response": "Please specify what to correct: name / dob / address",
                        "type": "error",
                        "waiting_for": "correction_field"
                    }
                    session["conversation"].append({"role": "assistant", "message": response["response"]})
                    return response

                response = {
                    "response": f"Please enter corrected {session['correction_field']}:",
                    "type": "info",
                    "waiting_for": "correction_value"
                }

        else:
                # User has entered new value
                field = session["correction_field"]
                fields[field] = user_message.strip()

                # Update temp data
                session["temp_aadhaar_data"]["fields"] = fields

                # Clear correction field
                session.pop("correction_field", None)

                # Show updated data again
                if user_language == "marathi":
                    confirmation_text = "योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:"
                elif user_language == "hindi":
                    confirmation_text = "यदि जानकारी सही है तो 'हाँ' टाइप करें या बदलाव के लिए 'सुधार' टाइप करें:"
                else:
                    confirmation_text = "Type YES if correct or CORRECTION if you need to make changes:"

                response = {
                    "response": f"""Aadhaar details retrieved:

                Name: {fields.get("name", "N/A")}
                Date of Birth: {fields.get("dob", "N/A")}
                Address: {fields.get("address", "N/A")}

                {confirmation_text}""",
                    "type": "success",
                    "waiting_for": "aadhaar_confirmation"
                }

                session["step"] = "confirm_aadhaar_details"

    # ✅ STEP 0.8: UPLOAD PAN CARD
    elif current_step == "upload_pan_card":

        if file_uploaded and file_uploaded.get("doc_type") == "pan_card":

            user_name = session["personal_info"].get("name", "unknown").replace(" ", "_")

            # Show verification message immediately
            response = {
                "response": get_translated_message("pan_verification_started", user_language),
                "type": "info",
                "waiting_for": "processing"
            }

            # 🔍 Perform OCR + AI extraction
            result = doc_intelligence.analyze_document(
                file_uploaded["content"],
                file_uploaded["extension"],
                "pan_card",
                "",
                session["personal_info"].get("name", ""),
                user_language
            )

            # ❌ Document validation failed
            if not result.get("is_valid"):
                session["step"] = "upload_pan_card"
                return {
                    "response": result.get(
                        "validation_error",
                        get_translated_message("invalid_pan", user_language)
                    ),
                    "type": "error",
                    "waiting_for": "pan_card_upload"
                }

            # ✅ Extract PAN fields
            fields = result.get("fields", {})
            pan_number = fields.get("pan_number", "")

            # 🔒 Basic PAN format validation (extra security layer)
            if not pan_number or not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan_number):
                session["step"] = "upload_pan_card"
                return {
                    "response": get_translated_message("invalid_pan", user_language),
                    "type": "error",
                    "waiting_for": "pan_card_upload"
                }

            # 📤 Upload PAN to blob storage
            blob_url = upload_to_blob(
                file_uploaded["content"],
                user_name,
                "pan_card",
                file_uploaded["extension"]
            )

            # Store in session
            session["documents"]["pan_card"] = {
                "is_valid": True,
                "fields": fields,
                "blob_url": blob_url
            }
            session["uploaded_docs"].append("pan_card")
            session["extracted_data"]["pan_number"] = pan_number

            # 🔗 Aadhaar-PAN Linking Check
            aadhaar_number = session["extracted_data"].get("aadhaar_number", "")
            is_linked = False
            db_result = None

            if db_manager and aadhaar_number and pan_number:
                db_result = db_manager.verify_pan_aadhaar_link(aadhaar_number, pan_number)
                if db_result:
                    is_linked = True

            # 🚨 If NOT linked
            if not is_linked:
                session["step"] = "pan_not_linked"
                return {
                    "response": get_translated_message("pan_aadhaar_not_linked", user_language),
                    "type": "error",
                    "waiting_for": "none"
                }

            # 💰 Income Eligibility Check from DB
            annual_income_db = None
            if db_result:
                annual_income_db = db_result.get("AnnualIncome")

            try:
                if annual_income_db and float(annual_income_db) > 250000:
                    session["step"] = "completed"
                    return {
                        "response": get_translated_message("income_exceeds", user_language),
                        "type": "error",
                        "waiting_for": "restart"
                    }
            except Exception:
                pass  # If DB value malformed, ignore and continue

            # ✅ All checks passed → Proceed
            session["step"] = "collect_mobile"

            pan_linked_msg = get_translated_message("pan_aadhaar_linked", user_language)

            aadhaar_msg = get_translated_message(
                "aadhaar_prefilled",
                user_language,
                name=session["personal_info"].get("name", ""),
                age=session["personal_info"].get("age", ""),
                dob=session["personal_info"].get("dob", ""),
                address=session["contact_info"].get("address", ""),
                district=session.get("domicile_info", {}).get("district", "")
            )

            doc_msg = get_translated_message("document_info", user_language)
            mobile_msg = get_translated_message("mobile_number_prompt", user_language)

            return {
                "response": f"{pan_linked_msg}\n\n{aadhaar_msg}\n\n{doc_msg}\n\n{mobile_msg}",
                "type": "success",
                "waiting_for": "mobile_input"
            }

        else:
            return {
                "response": get_translated_message("pan_upload_prompt", user_language),
                "type": "text",
                "waiting_for": "pan_card_upload"
            }

    # ✅ STEP 1: COLLECT MOBILE (First step)
    elif  current_step == "collect_mobile":
        # Generate application ID if not exists
        if not session.get("application_id"):
            application_id = db_manager.generate_application_id() if db_manager else f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            session["application_id"] = application_id
        
        # If no user message yet, show initial greeting with Aadhaar confirmation
        if not user_message:
            aadhaar_msg = get_translated_message(
                "aadhaar_prefilled", 
                user_language,
                name=session["personal_info"].get("name", ""),
                age=session["personal_info"].get("age", ""),
                dob=session["personal_info"].get("dob", ""),
                address=session["contact_info"].get("address", ""),
                district=session["domicile_info"].get("district", "")
            )
            
            doc_msg = get_translated_message(
                "document_info",
                user_language,
                app_id=session.get("application_id")
            )
            
            # ✅ CREATE PROPER MOBILE PROMPT (not using mobile_confirmed template)
            mobile_msg = get_translated_message("mobile_number_prompt", user_language)
            
            response = {
                "response": f"{aadhaar_msg}\n\n{doc_msg}\n\n{mobile_msg}",
                "type": "info",
                "waiting_for": "mobile_input"
            }
        else:
            # Process mobile number input
            if re.match(r'^[6-9]\d{9}$', user_message):
                session["contact_info"]["mobile"] = user_message
                session["step"] = "collect_email"
                response = {
                    "response": get_translated_message("mobile_confirmed", user_language, mobile=user_message),
                    "type": "success",
                    "waiting_for": "email_input"
                }
            else:
                response = {
                    "response": get_translated_message("invalid_mobile", user_language),
                    "type": "error",
                    "waiting_for": "mobile_input"
                }
    
    # ✅ STEP 2: COLLECT EMAIL
    elif current_step == "collect_email":
        # Allow user to explicitly skip
        if user_message.lower() == "skip":
            session["contact_info"]["email"] = ""
            email_display = "Skipped"
            session["step"] = "collect_marital_status"
            response = {
                "response": get_translated_message("email_confirmed", user_language, email=email_display),
                "type": "success",
                "waiting_for": "marital_status"
            }
        else:
            # Validate email format before accepting
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
            if user_message and re.match(email_pattern, user_message):
                session["contact_info"]["email"] = user_message
                email_display = user_message
                session["step"] = "collect_marital_status"
                response = {
                    "response": get_translated_message("email_confirmed", user_language, email=email_display),
                    "type": "success",
                    "waiting_for": "marital_status"
                }
            else:
                # Invalid email — prompt retry
                response = {
                    "response": get_translated_message("invalid_email", user_language),
                    "type": "error",
                    "waiting_for": "email_input"
                }
    
    # ✅ STEP 3: COLLECT MARITAL STATUS
    elif current_step == "collect_marital_status":
        status_map = {"1": "Married", "2": "Unmarried", "3": "Widow", "4": "Divorced"}
        status = status_map.get(user_message, user_message)
        
        if status in ["Married", "Unmarried", "Widow", "Divorced"]:
            session["personal_info"]["marital_status"] = status
            session["step"] = "upload_aadhaar"
            response = {
                "response": get_translated_message("marital_confirmed", user_language, status=status),
                "type": "success",
                "waiting_for": "aadhaar_upload"
            }
        else:
            response = {
                "response": get_translated_message("invalid_option", user_language) + "\n\nWhat is your Marital Status?\n\n1. Married\n2. Unmarried\n3. Widow\n4. Divorced\n\nPlease enter the number (1-4):",
                "type": "error",
                "waiting_for": "marital_status"
            }
    
    # ✅ STEP 4: UPLOAD AADHAAR (For blob storage only - data already extracted)
    elif current_step == "upload_aadhaar":
        if file_uploaded and file_uploaded.get("doc_type") == "aadhaar":
            application_id = session.get("application_id")
            
            # Upload to blob storage
            blob_url = upload_to_blob(
                file_uploaded["content"],
                application_id,
                "aadhaar",
                file_uploaded["extension"]
            )
            
            if not blob_url:
                response = {
                    "response": "❌ Upload failed. Please try again.",
                    "type": "error",
                    "waiting_for": "aadhaar_upload"
                }
            else:
                # Store blob URL (data already extracted from main.py)
                if "aadhaar" not in session["uploaded_docs"]:
                    session["uploaded_docs"].append("aadhaar")
                
                session["documents"]["aadhaar"] = {
                    "is_valid": True,
                    "fields": {
                        "aadhaar_number": session["extracted_data"].get("aadhaar_number", ""),
                        "name": session["personal_info"].get("name", "")
                    },
                    "blob_url": blob_url
                }
                
                session["step"] = "select_domicile_proof"
                
                aadhaar_display = mask_aadhaar(session["extracted_data"].get("aadhaar_number", ""))
                
                response = {
                    "response": get_translated_message("aadhaar_success", user_language, aadhaar=aadhaar_display) + "\n\n" + 
                            get_translated_message("domicile_prompt", user_language),
                    "type": "success",
                    "waiting_for": "domicile_selection"
                }
        else:
            response = {
                "response": get_translated_message("marital_confirmed", user_language),
                "type": "text",
                "waiting_for": "aadhaar_upload"
            }
    
    # ✅ STEP 5: SELECT DOMICILE PROOF
    elif current_step == "select_domicile_proof":
        proof_map = {
            "1": ("domicile_certificate", "Domicile Certificate"),
            "2": ("ration_card", "Ration Card"),
            "3": ("voter_id", "Voter ID"),
            "4": ("birth_certificate", "Birth Certificate"),
            "5": ("school_leaving", "School Leaving Certificate")
        }
        
        if user_message in proof_map:
            doc_type, doc_name = proof_map[user_message]
            session["domicile_proof_type"] = doc_type
            session["step"] = "upload_domicile_proof"
            response = {
                "response": get_translated_message("domicile_selected", user_language, doc_name=doc_name),
                "type": "info",
                "waiting_for": f"{doc_type}_upload"
            }
        else:
            response = {
                "response": get_translated_message("invalid_option", user_language) + "\n\n" + 
                          get_translated_message("domicile_prompt", user_language),
                "type": "error",
                "waiting_for": "domicile_selection"
            }
    
    # ✅ STEP 6: UPLOAD DOMICILE PROOF
    elif current_step == "upload_domicile_proof":
        domicile_type = session.get("domicile_proof_type")
        
        if file_uploaded:
            doc_type = file_uploaded.get("doc_type")
            expected_name = session["personal_info"].get("name", "")
            application_id = session.get("application_id")
            
            blob_url = upload_to_blob(
                file_uploaded["content"],
                application_id,
                doc_type,
                file_uploaded["extension"]
            )
            
            if not blob_url:
                response = {
                    "response": "❌ Upload failed",
                    "type": "error",
                    "waiting_for": f"{domicile_type}_upload"
                }
            else:
                result = doc_intelligence.analyze_document(
                file_uploaded["content"],
                file_uploaded["extension"],
                doc_type,
                blob_url,
                expected_name,
                user_language
            )
                
                if not result.get("is_valid"):
                    response = {
                        "response": result.get("validation_error", "❌ Invalid document"),
                        "type": "error",
                        "waiting_for": f"{domicile_type}_upload"
                    }
                else:
                    session["documents"][doc_type] = result
                    session["uploaded_docs"].append(doc_type)
                    
                    fields = result.get("fields", {})
                    
                    session["domicile_info"] = {
                        "type": doc_type,
                        "certificate_number": fields.get("certificate_number") or fields.get("card_number") or fields.get("voter_id_number"),
                        "district": fields.get("district", ""),
                        "taluka": fields.get("taluka", ""),
                        "village": fields.get("village", "")
                    }
                    
                    if doc_type == "ration_card":
                        session["step"] = "ask_ration_color"
                        response = {
                            "response": get_translated_message(
                                "ration_color",
                                user_language,
                                card_number=fields.get("card_number", "Extracted"),
                                name=fields.get("holder_name", "Extracted")
                            ),
                            "type": "success",
                            "waiting_for": "ration_color"
                        }
                    else:
                        session["step"] = "upload_income_certificate"
                        extra_info = ""
                        if doc_type == "voter_id" and fields.get("voter_id_number"):
                            extra_info = f"• Voter ID: {fields.get('voter_id_number')}\n"
                        if fields.get("district"):
                            extra_info += f"• District: {fields.get('district')}"
                        
                        response = {
                            "response": get_translated_message(
                                "domicile_success",
                                user_language,
                                doc_name=DOCUMENT_TYPES.get(doc_type),
                                doc_type=DOCUMENT_TYPES.get(doc_type),
                                name=doc_intelligence.get_name_field(fields, doc_type) or "Extracted",
                                extra_info=extra_info
                            ),
                            "type": "success",
                            "waiting_for": "income_certificate_upload"
                        }
        else:
            doc_name = DOCUMENT_TYPES.get(domicile_type, "Document")
            response = {
                "response": get_translated_message("upload_prompt", user_language, doc=doc_name),
                "type": "text",
                "waiting_for": f"{domicile_type}_upload"
            }
    
    # ✅ STEP 7: ASK RATION COLOR
    elif current_step == "ask_ration_color":
        color_map = {"1": "Yellow", "2": "Orange", "3": "White"}
        color = color_map.get(user_message, user_message.capitalize())
        
        if color in ["Yellow", "Orange", "White"]:
            session["ration_card_color"] = color
            session["income_info"]["ration_card_type"] = color
            
            ration_fields = session["documents"].get("ration_card", {}).get("fields", {})
            
            if color == "White":
                session["step"] = "upload_income_certificate"
                response = {
                    "response": get_translated_message("ration_white", user_language, color=color),
                    "type": "info",
                    "waiting_for": "income_certificate_upload"
                }
            else:
                extracted_income = ration_fields.get("annual_income", "")
                if extracted_income:
                    parsed_income = parse_numeric(extracted_income)
                    session["income_info"]["annual_income"] = parsed_income
                    session["income_info"]["annual_income_display"] = extracted_income
                else:
                    session["income_info"]["annual_income"] = 0
                    session["income_info"]["annual_income_display"] = "As per Ration Card"
                session["income_info"]["source"] = "ration_card"
                
                session["step"] = "upload_bank_passbook"
                response = {
                    "response": get_translated_message("ration_yellow_orange", user_language, color=color),
                    "type": "success",
                    "waiting_for": "bank_passbook_upload"
                }
        else:
            response = {
                "response": get_translated_message("invalid_option", user_language),
                "type": "error",
                "waiting_for": "ration_color"
            }
    
    # ✅ STEP 8: UPLOAD INCOME CERTIFICATE
    elif current_step == "upload_income_certificate":
        # Allow user to type 'exit' to quit after a rejection
        if user_message and isinstance(user_message, str) and user_message.strip().lower() in ["exit", "quit", "cancel"]:
            session["step"] = "exit"
            return {
                "response": get_translated_message("exit_prompt", user_language),
                "type": "info",
                "waiting_for": "none"
            }

        if file_uploaded and file_uploaded.get("doc_type") == "income_certificate":
            expected_name = session["personal_info"].get("name", "")
            application_id = session.get("application_id")
            doc_type = "income_certificate"
            
            blob_url = upload_to_blob(
                file_uploaded["content"],
                application_id,
                "income_certificate",
                file_uploaded["extension"]
            )
            
            if not blob_url:
                response = {
                    "response": "❌ Upload failed",
                    "type": "error",
                    "waiting_for": "income_certificate_upload"
                }
            else:
                result = doc_intelligence.analyze_document(
                    file_uploaded["content"],
                    file_uploaded["extension"],
                    doc_type,
                    blob_url,
                    expected_name,
                    user_language
                )
                
                # ✅ ADD THIS ENTIRE SECTION
                if not result.get("is_valid"):
                    response = {
                        "response": result.get("validation_error"),
                        "type": "error",
                        "waiting_for": "income_certificate_upload"
                    }
                else:
                    fields = result.get("fields", {})
                    session["income_info"]["certificate_number"] = fields.get("certificate_number", "")

                    # Try several common field keys that might contain the income
                    raw_income = ""
                    for key in [
                        "annual_income", "annual_income_amount", "annual_income_rupees",
                        "income", "amount", "amount_in_words"
                    ]:
                        val = fields.get(key)
                        if val:
                            raw_income = str(val)
                            break

                    # If still empty, inspect all field values and pick the best candidate
                    if not raw_income and isinstance(fields, dict):
                        income_key_words = ['income', 'amount', 'annual', 'salary', 'rupee', 'rupees', 'रु', 'वार्षिक']
                        reject_key_words = ['certificate', 'cert', 'number', 'id', 'card', 'date', 'year', 'authority', 'issuer']

                        for k, v in fields.items():
                            try:
                                if not isinstance(v, str):
                                    continue
                                s = v.strip()
                                if not s:
                                    continue

                                lname = str(k).lower()

                                # Prefer keys that explicitly mention income/amount
                                if any(word in lname for word in income_key_words):
                                    raw_income = s
                                    break

                                # Prefer values that contain currency symbols or rupee words
                                if re.search(r'₹|\bRs\b|\bRs\.|rupee|रु', s, flags=re.IGNORECASE):
                                    raw_income = s
                                    break

                                # If value is mostly numeric (digits, commas, spaces) and key is not a reject-type, accept it
                                if re.fullmatch(r'[₹Rs.\s,\d]+', s, flags=re.IGNORECASE):
                                    if not any(word in lname for word in reject_key_words):
                                        raw_income = s
                                        break

                                # Otherwise skip values like 'MH-IC-1234567' (contains letters/hyphens)
                            except Exception:
                                continue

                    # Normalize numeric income using context-aware parsing
                    parsed_income = parse_income_amount(raw_income, result.get("raw_text", ""))

                    # If income could not be extracted, ask user to re-upload
                    if not parsed_income:
                        response = {
                            "response": get_translated_message("income_not_found", user_language),
                            "type": "error",
                            "waiting_for": "income_certificate_upload"
                        }
                        session["step"] = "upload_income_certificate"
                        return response

                    # Final display value should match numeric income when available
                    display_income = format_currency(parsed_income)

                    # Detailed debug log for inspection
                    logger.info(f"[IncomeParse] session={session_id} fields_keys={list(fields.keys()) if fields else []} raw_income='{raw_income}' parsed={parsed_income}")
                    logger.debug(f"[IncomeParse] raw_text_snippet={result.get('raw_text','')[:500]}")

                    # Also print to stdout so user running server sees immediate values
                    try:
                        print('\n[IncomeParse DEBUG] session=', session_id)
                        print('[IncomeParse DEBUG] fields keys =', list(fields.keys()) if fields else [])
                        print('[IncomeParse DEBUG] raw_income =', repr(raw_income))
                        print('[IncomeParse DEBUG] parsed_income =', parsed_income)
                        print('[IncomeParse DEBUG] raw_text_snippet =', result.get('raw_text','')[:500])
                        print('\n')
                    except Exception as _:
                        logger.exception('Failed to print debug income parse info')

                    # If income exceeds eligibility cap, reject upload and prompt retry or exit
                    if parsed_income and parsed_income > 250000:
                        # Debug: Log the parsed income and raw data to diagnose parsing issues
                        logger.warning(f"⚠️ Income exceeds limit - Session: {session_id}, Parsed: {parsed_income}, Raw: '{raw_income}', fields={fields}")
                        logger.warning(f"⚠️ Raw OCR Text (first 500 chars): {result.get('raw_text', '')[:500]}")

                        # Mark that we are awaiting user's decision (re-upload or exit)
                        session["awaiting_income_action"] = True
                        session["income_info"]["source"] = "income_certificate"
                        session["income_info"]["annual_income"] = parsed_income
                        session["income_info"]["annual_income_display"] = display_income

                        response = {
                            "response": (
                                get_translated_message("income_exceeds", user_language)
                                + "\n\n"
                                + get_translated_message("income_retry_prompt", user_language)
                            ),
                            "type": "error",
                            "waiting_for": "income_certificate_upload"
                        }
                        # ensure we stay on same step so user can retry
                        session["step"] = "upload_income_certificate"
                    else:
                        # Accept and store
                        session["documents"]["income_certificate"] = result
                        if "income_certificate" not in session["uploaded_docs"]:
                            session["uploaded_docs"].append("income_certificate")
                        session["income_info"]["annual_income"] = parsed_income
                        session["income_info"]["annual_income_display"] = display_income
                        session["income_info"]["issue_date"] = fields.get("issue_date", "")
                        session["income_info"]["source"] = "income_certificate"
                        # Clear awaiting flag on successful accept
                        session.pop("awaiting_income_action", None)

                        session["step"] = "upload_bank_passbook"
                        response = {
                            "response": get_translated_message(
                                "income_success",
                                user_language,
                                cert_no=fields.get("certificate_number", "Extracted"),
                                name=fields.get("holder_name", "Extracted"),
                                income=session["income_info"].get("annual_income_display", "Extracted")
                            ),
                            "type": "success",
                            "waiting_for": "bank_passbook_upload"
                        }
        else:
            # ✅ ADD THIS ELSE BLOCK TOO
            response = {
                "response": get_translated_message("upload_prompt", user_language, doc="Income Certificate"),
                "type": "text",
                "waiting_for": "income_certificate_upload"
            }
    
    # ✅ STEP 9: UPLOAD BANK PASSBOOK
    elif current_step == "upload_bank_passbook":
        if file_uploaded and file_uploaded.get("doc_type") == "bank_passbook":
            expected_name = session["personal_info"].get("name", "")
            application_id = session.get("application_id")
            doc_type = "bank_passbook"  # ✅ ADD THIS LINE
            
            blob_url = upload_to_blob(
                file_uploaded["content"],
                application_id,
                "bank_passbook",
                file_uploaded["extension"]
            )
            
            if not blob_url:
                response = {
                    "response": "❌ Upload failed",
                    "type": "error",
                    "waiting_for": "bank_passbook_upload"
                }
            else:
                result = doc_intelligence.analyze_document(
                file_uploaded["content"],
                file_uploaded["extension"],
                doc_type,
                blob_url,
                expected_name,
                user_language
            )
                
                if not result.get("is_valid"):
                    response = {
                        "response": result.get("validation_error"),
                        "type": "error",
                        "waiting_for": "bank_passbook_upload"
                    }
                else:
                    session["documents"]["bank_passbook"] = result
                    session["uploaded_docs"].append("bank_passbook")
                    
                    fields = result.get("fields", {})
                    session["bank_info"]["account_number"] = fields.get("account_number", "")
                    session["bank_info"]["ifsc"] = fields.get("ifsc_code", "")
                    session["bank_info"]["bank_name"] = fields.get("bank_name", "")
                    
                    session["step"] = "upload_photograph"
                    
                    account_display = mask_account(fields.get("account_number", ""))
                    
                    response = {
                        "response": get_translated_message(
                            "bank_success",
                            user_language,
                            holder=fields.get("account_holder_name", "Extracted"),
                            account=account_display,
                            ifsc=fields.get("ifsc_code", "Extracted"),
                            bank=fields.get("bank_name", "Extracted")
                        ),
                        "type": "success",
                        "waiting_for": "photograph_upload"
                    }
        else:
            response = {
                "response": get_translated_message("upload_prompt", user_language, doc="Bank Passbook"),
                "type": "text",
                "waiting_for": "bank_passbook_upload"
            }
    
    # ✅ STEP 10: UPLOAD PHOTOGRAPH
    elif current_step == "upload_photograph":
        if file_uploaded and file_uploaded.get("doc_type") == "photograph":
            application_id = session.get("application_id")
            
            blob_url = upload_to_blob(
                file_uploaded["content"],
                application_id,
                "photograph",
                file_uploaded["extension"]
            )
            
            session["documents"]["photograph"] = {"file_path": blob_url, "is_valid": True, "blob_url": blob_url}
            session["uploaded_docs"].append("photograph")
            
            session["step"] = "final_review"
            
            name = session["personal_info"].get("name", "N/A")
            dob = session["personal_info"].get("dob", "N/A")
            age = session["personal_info"].get("age", "N/A")
            marital = session["personal_info"].get("marital_status", "N/A")
            mobile = session["contact_info"].get("mobile", "N/A")
            email = session["contact_info"].get("email", "Not provided")
            address = session["contact_info"].get("address", "N/A")
            
            aadhaar_masked = mask_aadhaar(session["extracted_data"].get("aadhaar_number", ""))
            pan_masked = session["extracted_data"].get("pan_number", "XXXXXXXXXX")
            account_masked = mask_account(session["bank_info"].get("account_number", ""))
            
            ifsc = session["bank_info"].get("ifsc", "N/A")
            bank_name = session["bank_info"].get("bank_name", "N/A")
            
            annual_income = session["income_info"].get("annual_income_display", session["income_info"].get("annual_income", "N/A"))
            ration_type = session["income_info"].get("ration_card_type", "N/A")
            
            doc_count = len(session["uploaded_docs"])
            
            doc_list_items = []
            doc_list_items.append("- Aadhaar Card")
            doc_list_items.append(f"- {DOCUMENT_TYPES.get(session.get('domicile_proof_type', ''), 'Domicile Proof')}")
            if "income_certificate" in session["uploaded_docs"]:
                doc_list_items.append("- Income Certificate")
            doc_list_items.append("- Bank Passbook")
            doc_list_items.append("- Photograph")
            doc_list = "\n".join(doc_list_items)
            
            response = {
                "response": get_translated_message(
                    "photo_success",
                    user_language,
                    name=name,
                    dob=dob,
                    age=age,
                    aadhaar=aadhaar_masked,
                    pan=pan_masked,
                    marital=marital,
                    email=email,
                    address=address,
                    account=account_masked,
                    ifsc=ifsc,
                    bank=bank_name,
                    income=annual_income,
                    ration=ration_type,
                    count=doc_count,
                    doc_list=doc_list
                ),
                "type": "success",
                "waiting_for": "review_confirmation"
            }
        else:
            response = {
                "response": get_translated_message("upload_prompt", user_language, doc="Photograph"),
                "type": "text",
                "waiting_for": "photograph_upload"
            }
    
    # ✅ STEP 11: FINAL REVIEW
    elif current_step == "final_review":
        if user_message.upper() == "YES":
            session["step"] = "declaration"
            response = {
                "response": get_translated_message("declaration", user_language),
                "type": "info",
                "waiting_for": "declaration_confirm"
            }
        elif user_message.upper() == "NO":
            response = {
                "response": get_translated_message("edit_request", user_language),
                "type": "info",
                "waiting_for": "edit_request"
            }
        else:
            response = {
                "response": get_translated_message("confirmation_yes_no", user_language),
                "type": "text",
                "waiting_for": "review_confirmation"
            }
    
    # ✅ STEP 12: DECLARATION
    elif current_step == "declaration":
        if user_message.upper() == "I AGREE":
            session["step"] = "submit_application"
            session["declaration_accepted"] = True
            response = {
                "response": get_translated_message("declaration_accepted", user_language),
                "type": "success",
                "waiting_for": "final_submit"
            }
        else:
            response = {
                "response": get_translated_message("confirmation_i_agree", user_language),
                "type": "error",
                "waiting_for": "declaration_confirm"
            }
    
    # ✅ STEP 13: SUBMIT APPLICATION
    elif current_step == "submit_application":
        if user_message.upper() == "SUBMIT":
            session["step"] = "processing"
            
            aadhaar_number = session["extracted_data"].get("aadhaar_number", "")
            if aadhaar_number and db_manager and db_manager.check_aadhaar_exists(aadhaar_number):
                session["step"] = "completed"
                response = {
                    "response": get_translated_message("aadhaar_exists", user_language),
                    "type": "error",
                    "waiting_for": "restart"
                }
            else:
                application_id = session.get("application_id")
                
                if db_manager:
                    try:
                        dob_str = session["personal_info"].get("dob", "")
                        dob_date = parse_date(dob_str)
                        
                        # Normalize annual income (stored as numeric or display string)
                        raw_income_candidate = session["income_info"].get("annual_income", session["income_info"].get("annual_income_display", 0))
                        if isinstance(raw_income_candidate, (int, float)):
                            annual_income = float(raw_income_candidate)
                        else:
                            annual_income = parse_numeric(raw_income_candidate)

                        # Eligibility check: income must be <= 250000
                        if annual_income > 250000:
                            session["step"] = "completed"
                            response = {
                                "response": get_translated_message("income_exceeds", user_language),
                                "type": "error",
                                "waiting_for": "restart"
                            }
                            return response
                        
                        beneficiary_data = {
                            "username": session["contact_info"].get("mobile", ""),
                            "password_hash": "",
                            "aadhaar_number": aadhaar_number,
                            "full_name": session["personal_info"].get("name", ""),
                            "date_of_birth": dob_date,
                            "gender": "F",
                            "mobile_number": session["contact_info"].get("mobile", ""),
                            "email": session["contact_info"].get("email", ""),
                            "address": session["contact_info"].get("address", ""),
                            "district": session["domicile_info"].get("district", ""),
                            "taluka": session["domicile_info"].get("taluka", ""),
                            "village": session["domicile_info"].get("village", ""),
                            "annual_income": annual_income,
                            "bank_account_no": session["bank_info"].get("account_number", ""),
                            "bank_ifsc": session["bank_info"].get("ifsc", "")
                        }
                        
                        beneficiary_id = db_manager.save_beneficiary_application(beneficiary_data, application_id)
                        session["beneficiary_id"] = beneficiary_id
                        
                        if beneficiary_id:
                            for doc_type in session["uploaded_docs"]:
                                doc_data = session["documents"].get(doc_type, {})
                                fields = doc_data.get("fields", {})
                                blob_url = doc_data.get("blob_url", "")
                                
                                document_entry = {
                                    "beneficiary_id": application_id,
                                    "mobile_number": session["contact_info"].get("mobile", ""),
                                    "aadhaar_number": aadhaar_number,
                                    "document_type": doc_type,
                                    "document_url": blob_url,
                                    "full_name": session["personal_info"].get("name", ""),
                                }
                                
                                if doc_type == "income_certificate":
                                    document_entry["income_certificate_number"] = fields.get("certificate_number")
                                    document_entry["income_cert_issue_date"] = parse_date(fields.get("issue_date", ""))
                                    try:
                                        document_entry["annual_income_amount"] = parse_numeric(fields.get("annual_income", 0) or 0)
                                    except:
                                        document_entry["annual_income_amount"] = 0
                                
                                elif doc_type == "bank_passbook":
                                    document_entry["bank_account_number"] = fields.get("account_number")
                                    document_entry["bank_ifsc"] = fields.get("ifsc_code")
                                    document_entry["bank_name"] = fields.get("bank_name")
                                
                                elif doc_type == "domicile_certificate":
                                    document_entry["domicile_certificate_number"] = fields.get("certificate_number")
                                    document_entry["domicile_issuing_district"] = fields.get("district")
                                    document_entry["domicile_issue_date"] = parse_date(fields.get("issue_date", ""))
                                    document_entry["residence_district"] = fields.get("district")
                                    document_entry["residence_taluka"] = fields.get("taluka")
                                    document_entry["residence_village"] = fields.get("village")
                                
                                elif doc_type == "ration_card":
                                    document_entry["ration_card_number"] = fields.get("card_number")
                                    document_entry["ration_card_type"] = session["income_info"].get("ration_card_type")
                                    document_entry["ration_card_issue_date"] = parse_date(fields.get("issue_date", ""))
                                
                                elif doc_type == "voter_id":
                                    document_entry["voter_id_number"] = fields.get("voter_id_number")
                                
                                db_manager.save_document(document_entry)
                    
                    except Exception as e:
                        logger.error(f"Database save error: {e}")
                
                session["step"] = "completed"
                session["status"] = "SUBMITTED"
                session["submitted_at"] = datetime.now().isoformat()
                
                name = session["personal_info"].get("name", "Applicant")
                mobile = session["contact_info"].get("mobile", "XXXXXXXXXX")
                
                response = {
                    "response": get_translated_message("success", user_language, name=name, app_id=application_id, mobile=mobile),
                    "type": "success",
                    "waiting_for": "none",
                    "application_id": application_id
                }
        else:
            response = {
                "response": get_translated_message("confirmation_submit", user_language),
                "type": "text",
                "waiting_for": "final_submit"
            }
    
    # ✅ STEP 14: COMPLETED
    elif current_step == "completed":
        response = {
            "response": get_translated_message("restart_prompt", user_language, app_id=session.get('application_id', 'N/A')),
            "type": "success",
            "waiting_for": "restart"
        }
    
    else:
        response = {
            "response": get_translated_message("error", user_language),
            "type": "error",
            "waiting_for": "restart"
        }

    if not response:
        response = {
            "response": get_translated_message("error", user_language),
            "type": "error",
            "waiting_for": "restart"
        }

    session["conversation"].append({"role": "assistant", "message": response.get("response", "")})
    return response


# ============================================
# ROUTES
# ============================================

@app.post("/api/chat")
async def chat_endpoint(
    session_id: str = Form(...),
    message: str = Form(""),
    file: Optional[UploadFile] = File(None),
    doc_type: str = Form("aadhaar")
):
    try:
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
        
        response = get_bot_response(session_id, message, file_uploaded)
        return response
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse({"error": str(e), "message": "An error occurred. Please try again."}, status_code=500)


@app.post("/api/start-session")
async def start_session():
    session_id = str(uuid.uuid4())
    response = get_bot_response(session_id)
    return {"session_id": session_id, "message": response["response"]}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "blob_storage": "connected" if container_client else "disconnected",
        "azure_openai": "connected" if openai_client else "disconnected"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

