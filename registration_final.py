"""
Ladki Bahin Yojana - Complete Application System (Language Selection: English/Hindi/Marathi)
FastAPI Backend with OCR, AI Parsing, Azure Blob Storage, and Database Integration
"""

from attrs import fields
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
DOMICILE PROOF
════════════════════════════════════════════════════

Please select ONE document to prove Maharashtra domicile:

1. Domicile Certificate
2. Ration Card
3. Voter ID
4. Birth Certificate
5. School Leaving Certificate

Enter the number (1-5):""",

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

════════════════════════════════════════════════════
INCOME DETAILS
════════════════════════════════════════════════════

You may:

1. Upload Income Certificate (PDF/JPG/PNG)
OR
2. Select Income Manually:

   1. Less than ₹1 Lakh
   2. ₹1-2 Lakh
   3. ₹2-2.5 Lakh
   4. More than ₹2.5 Lakh

Enter 1-4 OR upload the certificate using the plus button.""",

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
- Annual Income: {income}

════════════════════════════════════════════════════
BANK DETAILS
════════════════════════════════════════════════════

You may:

1. Upload Bank Passbook (PDF/JPG/PNG)
OR
2. Enter Bank Details Manually:

   Bank Name, Account Number, IFSC Code

Example:
State Bank of India, 12345678901234, SBIN0001234

Use the plus button OR type details directly.""",

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

    "dpip_accepted": """Please upload a SINGLE IMAGE that contains BOTH the FRONT and BACK sides of your Aadhaar Card, OR type your 12-digit Aadhaar number directly.

════════════════════════════════════════════════════
IMPORTANT:
• Front and Back side must be in the SAME photo
• Both sides must be clearly visible
• Text should be readable
• OR simply type your 12-digit Aadhaar number
════════════════════════════════════════════════════

Use the plus button and select 'Aadhaar Card' from the dropdown.""",

    "aadhaar_verification_started": """Your information verification has started...""",

    "aadhaar_details_retrieved": {
        "english": """Aadhaar details retrieved:

    Name: {name}
    Date of Birth: {dob}
    Address: {address}

    Is this information correct?

    Type YES if correct or CORRECTION if you need to make changes:""",

        "marathi": """आधार तपशील प्राप्त झाले:

    नाव: {name}
    जन्मतारीख: {dob}
    पत्ता: {address}

    ही माहिती योग्य आहे का?

    योग्य असल्यास 'होय' टाइप करा किंवा बदल आवश्यक असल्यास 'दुरुस्ती' टाइप करा:""",

        "hindi": """आधार विवरण प्राप्त हुआ:

    नाम: {name}
    जन्म तिथि: {dob}
    पता: {address}

    क्या यह जानकारी सही है?

    यदि सही है तो 'हाँ' टाइप करें या बदलाव के लिए 'सुधार' टाइप करें:"""
    },

    "aadhaar_correction_prompt": """Please specify what needs to be corrected and we will assist you.""",

    "pan_upload_prompt": """Please upload your PAN Card, OR type your 10-character PAN number directly (e.g., ABCDE1234F).

══════════════════════════════════════════════════════
IMPORTANT: Upload image of PAN Card OR type PAN number 
══════════════════════════════════════════════════════

Use the plus button and select 'PAN Card' from the dropdown.""",

    "pan_verification_started": """Your PAN number verification is in progress...""",

    "pan_aadhaar_linked": """आपला आधार आणि पॅन क्रमांक यशस्वीरीत्या लिंक आहेत.

पुढील टप्प्याकडे जात आहोत...""",

    "pan_aadhaar_not_linked": """Your Aadhaar and PAN are not linked.

Please link them first before proceeding with the application.

Visit: https://www.incometax.gov.in/iec/foportal/

════════════════════════════════════════════════════
Application process ended.
════════════════════════════════════════════════════""",

    "invalid_pan": """Invalid Document! Please upload a valid PAN Card.""",

    "document_uploaded_success": """{doc_name} uploaded successfully.

Document is being verified...""",

    "confirmation_yes_correction": """Please type 'YES' or 'CORRECTION':""",

    "income_selection": """Please select your annual family income:

1. Less than ₹1 Lakh
2. ₹1-2 Lakh
3. ₹2-2.5 Lakh
4. More than ₹2.5 Lakh

Please enter the number (1-4):""",

"income_manual_success": """Income recorded successfully.

Now provide Bank Details.

You may:
• Upload Bank Passbook
OR
• Enter bank details manually:
Bank Name, Account Number, IFSC Code

Example:
State Bank of India, 12345678901234, SBIN0001234""",

"bank_manual_success": """Bank details recorded successfully.

Bank: {bank}
Account: {account}
IFSC: {ifsc}

Please upload your Photograph.""",


    "eligibility_success": """Congratulations! 🎉
You are eligible for Ladki Bahin Yojana.

Please provide your bank details in the following format:

Bank Name, Account Number, IFSC Code

Example:
State Bank of India, 12345678901234, SBIN0001234""",

    "eligibility_failure": """Sorry.

Based on the information provided, you are currently not eligible for this scheme.

Reason: {reason}

You can contact the helpline for more information.

════════════════════════════════════════════════════
Thank you.""",

    "bank_details_invalid_format": """Invalid bank details. Please enter in correct format:

Bank Name, Account Number, IFSC Code

Example:
State Bank of India, 12345678901234, SBIN0001234

Note:
- Account number must be 9-18 digits
- IFSC code must be 11 characters (e.g. SBIN0001234)""",

    "bank_details_confirmed": """Bank details received:
- Bank: {bank}
- Account: {account}
- IFSC: {ifsc}

Verifying details...""",

    "final_confirmation": """════════════════════════════════════════════════════
APPLICATION SUMMARY - Please verify
════════════════════════════════════════════════════

Personal Details:
- Name: {name}
- Date of Birth: {dob} (Age: {age})
- Aadhaar: {aadhaar}
- Pan : {pan_card}

Contact Details:
- Address: {address}

Bank Details:
- Bank: {bank}
- Account: {account}
- IFSC: {ifsc}

Income Details:
- Annual Income: {income}

════════════════════════════════════════════════════

Do you want to submit this information?

Type होय to submit or नाही to cancel:""",

    "final_submitted": """Your application has been submitted successfully!

════════════════════════════════════════════════════
Your Application Number: {app_id}
════════════════════════════════════════════════════

You can check your application status using:
- Application Number
- Aadhaar Number

Thank you.
Ladki Bahin Yojana – For empowered women.""",

"final_cancelled": """Application cancelled.

What would you like to do?

1. Re-enter bank details
2. Re-select income bracket
3. Exit

Please enter the number (1-3):""",

    "correction_menu": """What would you like to correct?

1. Name
2. Date of Birth
3. Address
4. Mobile
5. Email
6. Bank Details (Bank Name, Account, IFSC)
7. Income

Enter the number (1-7):""",

    "correction_invalid_choice": """Please enter a number between 1 and 7:""",

    "correction_prompt_name":    """Enter corrected Name:""",
    "correction_prompt_dob":     """Enter corrected Date of Birth (DD/MM/YYYY):""",
    "correction_prompt_address": """Enter corrected Address:""",
    "correction_prompt_mobile":  """Enter corrected Mobile Number (10 digits):""",
    "correction_prompt_email":   """Enter corrected Email (or type 'skip'):""",
    "correction_prompt_bank":    """Enter corrected Bank Details:
Bank Name, Account Number, IFSC Code

Example: State Bank of India, 12345678901234, SBIN0001234""",

    "correction_prompt_income":  """Select corrected Income:
1. Less than ₹1 Lakh
2. ₹1-2 Lakh
3. ₹2-2.5 Lakh
4. More than ₹2.5 Lakh""",

    "correction_invalid_income_choice": """Please enter 1, 2, 3, or 4:""",

        "post_registration_options": """
════════════════════════════════════════════════════
आपण पुढे काय करू इच्छिता?

1. पात्रता तपासा (Eligibility)
2. अर्ज स्थिती तपासा (Track Application)
3. बाहेर पडा

कृपया 1-3 पैकी एक निवडा:
""",


    "aadhaar_incomplete": """Incomplete Aadhaar Card!

Please upload a SINGLE IMAGE containing BOTH the FRONT and BACK sides of your Aadhaar Card.

════════════════════════════════════════════════════
IMPORTANT:
- Both sides must be in the SAME photo
- Front side: Contains Name, DOB, Aadhaar Number
- Back side: Contains Address
- Both sides must be clearly visible
════════════════════════════════════════════════════

Please re-upload with both sides visible.""",

    "pan_name_mismatch_upload": """Name Mismatch between Aadhaar and PAN Card!

- Name on Aadhaar: {aadhaar_name}
- Name on PAN Card: {pan_name}

Please ensure both documents belong to the same person and re-upload your PAN Card.""",

    "pan_name_mismatch_typed": """Name Mismatch between Aadhaar and PAN records!

- Name on Aadhaar: {aadhaar_name}
- Name on PAN record: {pan_name}

Please verify your documents and try again."""

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
    base_message = MESSAGE_TEMPLATES.get(message_key, MESSAGE_TEMPLATES["error"])
    
    # ✅ Handle dict-type templates (hardcoded per-language strings)
    if isinstance(base_message, dict):
        base_message = base_message.get(language) or base_message.get("english", "")
        return base_message.format(**kwargs) if kwargs else base_message

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
        template = MESSAGE_TEMPLATES.get(message_key)

        # If template is language-specific dict
        if isinstance(template, dict):
            base_message = template.get(language, template.get("english"))
        else:
            base_message = template
        
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
                logger.error(f"Missing key in template: {ke} — kwargs provided: {list(kwargs.keys())}")
                # Partial format: replace only keys we have, leave others as-is
                result = base_message
                for k, v in kwargs.items():
                    result = result.replace("{" + k + "}", str(v))
                return result
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

        text_lower = raw_text.lower()

        # ✅ Special validation for PAN using PAN format instead of keywords
        if document_type == "pan_card":
            print("RAW OCR PAN TEXT:", raw_text)
            # Normalize OCR text (remove spaces, newlines, tabs)
            cleaned_text = re.sub(r'\s+', '', raw_text.upper())

            # Strict PAN pattern
            pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', cleaned_text)
            if pan_match:
                return True, None

            # Relaxed fallback (handles minor OCR distortions)
            possible_matches = re.findall(r'[A-Z0-9]{10}', cleaned_text)
            for candidate in possible_matches:
                if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', candidate):
                    return True, None

            return False, get_translated_message("invalid_pan", user_language)

                # ✅ Strict Aadhaar validation using Aadhaar number format
        if document_type == "aadhaar":
            print("RAW OCR AADHAAR TEXT:", raw_text)

            # Normalize text (remove spaces, newlines)
            cleaned_text = re.sub(r'\s+', '', raw_text)

            # Strict Aadhaar pattern: 12 digits
            aadhaar_match = re.search(r'\b\d{12}\b', cleaned_text)

            if aadhaar_match:
                return True, None

            # Relaxed fallback: 4-4-4 format
            aadhaar_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', raw_text)
            if aadhaar_match:
                return True, None

            return False, get_translated_message("invalid_aadhaar", user_language)


        # ✅ For other documents, keep keyword validation
        keywords = self.validation_keywords.get(document_type, [])
        if not keywords:
            return True, None

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
        """Validate if the name extracted from document matches the expected name.
        
        Handles real-world mismatches:
        - Spelling variants:  Rajeshree vs RAJASHREE
        - Middle name on PAN: RAJASHREE RAJ MAHAJAN vs Rajeshree Mahajan
        - Case differences:   MAHAJAN vs Mahajan
        """
        if not extracted_name or not expected_name:
            return True, None

        def normalize_name(name):
            """Lowercase, strip, collapse spaces."""
            return ' '.join(name.lower().strip().split())

        def name_words(name):
            """Return list of words in normalized name."""
            return normalize_name(name).split()

        def fuzzy_word_match(w1: str, w2: str) -> bool:
            """True if two words are close enough (handles Rajeshree/Rajashree)."""
            if w1 == w2:
                return True
            # Allow 1-character difference for words longer than 4 chars
            if len(w1) > 4 and len(w2) > 4:
                # Count character differences
                longer  = max(w1, w2, key=len)
                shorter = min(w1, w2, key=len)
                if len(longer) - len(shorter) <= 1:
                    # Check edit distance is just 1 (simple substitution/insertion)
                    diffs = sum(a != b for a, b in zip(shorter.ljust(len(longer)), longer))
                    if diffs <= 1:
                        return True
            return False

        extracted_words = name_words(extracted_name)
        expected_words  = name_words(expected_name)

        # --- Rule 1: All expected words must appear (fuzzy) in extracted words ---
        # This handles PAN having an extra middle name
        matched = 0
        for exp_w in expected_words:
            for ext_w in extracted_words:
                if fuzzy_word_match(exp_w, ext_w):
                    matched += 1
                    break

        if matched == len(expected_words):
            return True, None

        # --- Rule 2: Relaxed — at least 50% of expected words match (fuzzy) ---
        # Catches cases where even the Aadhaar name has a typo vs PAN
        if matched >= max(1, len(expected_words) * 0.5):
            return True, None

        # --- Rule 3: Last name must match (minimum bar) ---
        # If the last word of both names matches, accept it
        if expected_words and extracted_words:
            if fuzzy_word_match(expected_words[-1], extracted_words[-1]):
                return True, None

        error_msg = get_translated_message(
            "invalid_name_mismatch",
            user_language,
            extracted=extracted_name,
            expected=expected_name
        )
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

Return JSON only.""",

"bank_passbook": f"""Extract from bank passbook:
- account_holder_name
- account_number
- ifsc_code
- bank_name

OCR Text:
{text}

Return JSON only.""",
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

            # Extract IFSC
            ifsc_match = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', text.upper())
            if ifsc_match:
                result['ifsc_code'] = ifsc_match.group(0)

            # Extract Account Number — look near account labels first
            acct_label_match = re.search(
                r'(?:A/?C\s*No|Account\s*No|Ac\s*No|खाते\s*क्र|खाता\s*[सं]*ख्या)[^\d]*(\d{9,18})',
                text, re.IGNORECASE
            )
            if acct_label_match:
                result['account_number'] = acct_label_match.group(1)
            else:
                # Fallback: find all 9-18 digit numbers, exclude Aadhaar (12-digit) and phone (10-digit)
                account_matches = re.findall(r'\b\d{9,18}\b', text)
                filtered = [a for a in account_matches if len(a) not in (10, 12)]
                if filtered:
                    result['account_number'] = max(filtered, key=len)
                elif account_matches:
                    result['account_number'] = max(account_matches, key=len)

            # Extract Bank Name
            bank_match = re.search(r'(State Bank of India|SBI|HDFC Bank|ICICI Bank|Bank of Baroda|Punjab National Bank)', text, re.IGNORECASE)
            if bank_match:
                result['bank_name'] = bank_match.group(0)

            # Try to extract account holder name (line above Account No usually)
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if re.search(r'account', line, re.IGNORECASE) and i > 0:
                    possible_name = lines[i-1].strip()
                    if len(possible_name) > 3 and not re.search(r'\d', possible_name):
                        result['account_holder_name'] = possible_name
                        break

        
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
        
        # Skip name validation for photograph and PAN card
        # PAN often contains middle/father's name not present on Aadhaar
        SKIP_NAME_VALIDATION = {"photograph", "pan_card"}
        
        if expected_name and document_type not in SKIP_NAME_VALIDATION:
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
# Aadhaar Front/Back Detection 
# ============================================


def _has_both_sides(text: str) -> bool:
    """Return True only if OCR text contains indicators from BOTH front AND back."""
    t = text.lower()
    front = [
        bool(re.search(r'\d{4}\s?\d{4}\s?\d{4}', text)),
        bool(re.search(r'\b(dob|date of birth|जन्म|born|year of birth|yob)\b', t)),
        bool(re.search(r'\b(male|female|पुरुष|महिला|transgender)\b', t)),
    ]
    back = [
        bool(re.search(r'\b(address|पत्ता)\b', t)),
        bool(re.search(r'\b(s/o|d/o|w/o|c/o|son of|daughter of|wife of)\b', t)),
        bool(re.search(r'\b(pin|pincode|dist|district|state|taluka|village|po )\b', t)),
        bool(re.search(r'\b(uidai|unique identification|माझी ओळख|qr|barcode)\b', t)),
    ]
    has_front = sum(front) >= 2
    has_back  = sum(back)  >= 1
    print(f"  front={sum(front)}/3  back={sum(back)}/4")
    return has_front and has_back


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
    try:
        # Handle both formats
        if "-" in dob_str:
            dob = datetime.strptime(dob_str, "%d-%m-%Y")
        else:
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

        # ✅ Handle typed Aadhaar number (12 digits)
        if user_message and not file_uploaded:
            cleaned_input = re.sub(r'\s+', '', user_message.strip())
            if re.fullmatch(r'\d{12}', cleaned_input):
                # Fetch from DB
                aadhaar_db_data = None
                if db_manager:
                    aadhaar_db_data = db_manager.get_aadhaar_details(cleaned_input)

                if not aadhaar_db_data:
                    return {
                        "response": get_translated_message("invalid_aadhaar", user_language) +
                                    "\n\nAadhaar number not found in records. Please upload your Aadhaar Card image instead.",
                        "type": "error",
                        "waiting_for": "aadhaar_upload_initial"
                    }

                # Generate application ID
                if not session.get("application_id"):
                    application_id = (
                        db_manager.generate_application_id()
                        if db_manager
                        else f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    )
                    session["application_id"] = application_id

                # Map DB fields to session fields
                name    = aadhaar_db_data.get("FullName", "")
                dob     = aadhaar_db_data.get("DateOfBirth", "")
                address = aadhaar_db_data.get("Address", "")
                district = aadhaar_db_data.get("City", "")
                session["domicile_info"]["district"] = district

                # Normalize dob to DD/MM/YYYY if it's a datetime object
                if hasattr(dob, 'strftime'):
                    dob = dob.strftime("%d/%m/%Y")
                else:
                    dob = str(dob) if dob else ""

                # Store as temp aadhaar data (no file content since number was typed)
                session["temp_aadhaar_data"] = {
                    "file_content": None,
                    "file_extension": None,
                    "source": "manual_number",
                    "fields": {
                        "name": name,
                        "dob": dob,
                        "address": address,
                        "aadhaar_number": cleaned_input
                    },
                    "name": name,
                    "dob": dob,
                    "address": address,
                    "aadhaar_number": cleaned_input
                }

                session["step"] = "confirm_aadhaar_details"

                return {
                    "response": get_translated_message(
                        "aadhaar_details_retrieved",
                        user_language,
                        name=name or "N/A",
                        dob=dob or "N/A",
                        address=address or "N/A"
                    ),
                    "type": "success",
                    "waiting_for": "aadhaar_confirmation"
                }
            else:
                # Not a valid 12-digit number, re-prompt
                return {
                    "response": get_translated_message("dpip_accepted", user_language),
                    "type": "error",
                    "waiting_for": "aadhaar_upload_initial"
                }

        if file_uploaded:

            expected_doc = "aadhaar"

            # 🔐 Strict dropdown validation
            if file_uploaded.get("doc_type") != expected_doc:
                return {
                    "response": get_translated_message("invalid_aadhaar", user_language),
                    "type": "error",
                    "waiting_for": "aadhaar_upload_initial"
                }

            # 🔎 Single OCR validation
            result = doc_intelligence.analyze_document(
                file_uploaded["content"],
                file_uploaded["extension"],
                expected_doc,
                "",
                None,
                user_language
            )

            # ❌ If OCR does NOT confirm Aadhaar → reject
            if not result.get("is_valid"):
                return {
                    "response": result.get(
                        "validation_error",
                        get_translated_message("invalid_aadhaar", user_language)
                    ),
                    "type": "error",
                    "waiting_for": "aadhaar_upload_initial"
                }

            # ✅ Check both sides present: need address (back side) AND aadhaar number (front side)
            fields = result.get("fields", {})
            raw_text = result.get("raw_text", "")
            raw_text = result.get("raw_text", "")
            if not _has_both_sides(raw_text):
                return {
                    "response": get_translated_message("aadhaar_incomplete", user_language),
                    "type": "error",
                    "waiting_for": "aadhaar_upload_initial"
                }

            if not result.get("is_valid"):
                return {
                    "response": result.get(
                        "validation_error",
                        get_translated_message("invalid_aadhaar", user_language)
                    ),
                    "type": "error",
                    "waiting_for": "aadhaar_upload_initial"
                }

            # ✅ Generate application ID once
            if not session.get("application_id"):
                application_id = (
                    db_manager.generate_application_id()
                    if db_manager
                    else f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
                session["application_id"] = application_id

            fields = result.get("fields", {})

            # Store temporarily until confirmation
            session["temp_aadhaar_data"] = {
                "file_content": file_uploaded["content"],
                "file_extension": file_uploaded["extension"],
                "source": "ocr",
                "fields": fields,
                "name": fields.get("name", ""),
                "dob": fields.get("dob", ""),
                "address": fields.get("address", ""),
                "aadhaar_number": fields.get("aadhaar_number", "")
            }

            session["step"] = "confirm_aadhaar_details"

            return {
                "response": get_translated_message(
                    "aadhaar_details_retrieved",
                    user_language,
                    name=fields.get("name", "N/A"),
                    dob=fields.get("dob", "N/A"),
                    address=fields.get("address", "N/A")
                ),
                "type": "success",
                "waiting_for": "aadhaar_confirmation"
            }

        else:
            return {
                "response": get_translated_message("dpip_accepted", user_language),
                "type": "text",
                "waiting_for": "aadhaar_upload_initial"
            }
        
    # ✅ STEP 0.6: CONFIRM AADHAAR DETAILS
    elif current_step == "confirm_aadhaar_details":

        user_input = user_message.strip().lower()

        yes_words = ["yes", "ho" , "y", "होय", "हो", "हां", "haan"]
        correction_words = ["correction", "durusti", "Durusti","दुरुस्ती", "सुधार"]

        if user_input in yes_words:
            # User confirmed details are correct
            temp_data = session.get("temp_aadhaar_data", {})
            application_id = session.get("application_id")
            
            # Now upload to blob storage using user's name as folder
            user_name = temp_data.get("name", "unknown").replace(" ", "_")
            
            # Only upload to blob if Aadhaar was uploaded as image (not typed manually)
            blob_url = ""
            if temp_data.get("source") != "manual_number" and temp_data.get("file_content"):
                blob_url = upload_to_blob(
                    temp_data.get("file_content"),
                    user_name,
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
            
            # ✅ FLOW CHANGE: Move to PAN upload
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

            msg = user_message.strip().lower()

            # Normalize Marathi/Hindi/English (remove punctuation)
            msg = re.sub(r'[^\w\u0900-\u097F]', '', msg)

            name_words = ["name", "Name", "नाव", "naam"]
            dob_words = ["dob", "DOB", "dateofbirth", "date of birth", "birth", "जन्मतारीख", "जन्म", "जन्मतिथि"]
            address_words = ["address", "Address", "पत्ता", "पता"]

            if any(word in msg for word in name_words):
                session["correction_field"] = "name"

            elif any(word in msg for word in dob_words):
                session["correction_field"] = "dob"

            elif any(word in msg for word in address_words):
                session["correction_field"] = "address"

            else:
                return {
                    "response": get_translated_message("aadhaar_correction_prompt", user_language),
                    "type": "error",
                    "waiting_for": "correction_field"
                }

            # Ask for corrected value (language-safe)
            if user_language == "marathi":
                field_label = {
                    "name": "नाव",
                    "dob": "जन्मतारीख",
                    "address": "पत्ता"
                }.get(session["correction_field"], "माहिती")

                return {
                    "response": f"कृपया योग्य {field_label} टाइप करा:",
                    "type": "info",
                    "waiting_for": "correction_value"
                }

            elif user_language == "hindi":
                field_label = {
                    "name": "नाम",
                    "dob": "जन्म तिथि",
                    "address": "पता"
                }.get(session["correction_field"], "जानकारी")

                return {
                    "response": f"कृपया सही {field_label} दर्ज करें:",
                    "type": "info",
                    "waiting_for": "correction_value"
                }

            else:
                return {
                    "response": f"Please enter corrected {session['correction_field']}:",
                    "type": "info",
                    "waiting_for": "correction_value"
                }

        # If user entered corrected value
        else:
            field = session["correction_field"]
            fields[field] = user_message.strip()

            session["temp_aadhaar_data"]["fields"] = fields
            session.pop("correction_field", None)

            session["step"] = "confirm_aadhaar_details"

            return {
                "response": get_translated_message(
                    "aadhaar_details_retrieved",
                    user_language,
                    name=fields.get("name", "N/A"),
                    dob=fields.get("dob", "N/A"),
                    address=fields.get("address", "N/A")
                ),
                "type": "success",
                "waiting_for": "aadhaar_confirmation"
            }

    # ✅ STEP 0.8: UPLOAD PAN CARD
    elif current_step == "upload_pan_card":

        # ✅ Handle typed PAN number
        if user_message and not file_uploaded:
            pan_input = user_message.strip().upper().replace(" ", "")
            if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan_input):
                aadhaar_number = re.sub(r'\D', '', session["extracted_data"].get("aadhaar_number", ""))

                is_linked = False
                db_result = None

                if db_manager and aadhaar_number and pan_input:
                    db_result = db_manager.verify_pan_aadhaar_link(aadhaar_number, pan_input)
                    if db_result:
                        is_linked = True

                if not is_linked:
                    session["step"] = "pan_not_linked"
                    return {
                        "response": get_translated_message("pan_aadhaar_not_linked", user_language),
                        "type": "error",
                        "waiting_for": "none"
                    }
                # ✅ For typed PAN, db_result may contain name — check if available
                if db_result and isinstance(db_result, dict):
                    pan_name_db  = db_result.get("FullName", "") or db_result.get("name", "")
                    aadhaar_name = session["personal_info"].get("name", "")
                    if pan_name_db and aadhaar_name:
                        is_name_ok, _ = doc_intelligence.validate_name(pan_name_db, aadhaar_name, user_language)
                        if not is_name_ok:
                            return {
                                "response": get_translated_message("pan_name_mismatch_typed", user_language,
                                                                aadhaar_name=aadhaar_name, pan_name=pan_name_db),
                                "type": "error",
                                "waiting_for": "pan_card_upload"
                            }

                # Store PAN (no blob since no file uploaded)
                session["documents"]["pan_card"] = {
                    "is_valid": True,
                    "fields": {"pan_number": pan_input},
                    "blob_url": ""
                }
                session["uploaded_docs"].append("pan_card")
                session["extracted_data"]["pan_number"] = pan_input

                # ✅ FLOW CHANGE: Go to collect_mobile instead of income_selection
                session["step"] = "collect_mobile"
                pan_linked_msg = get_translated_message("pan_aadhaar_linked", user_language)
                mobile_msg = get_translated_message("mobile_number_prompt", user_language)

                return {
                    "response": f"{pan_linked_msg}\n\n{mobile_msg}",
                    "type": "success",
                    "waiting_for": "mobile_input"
                }
            else:
                # Invalid PAN format typed
                return {
                    "response": get_translated_message("invalid_pan", user_language) +
                                "\n\nPlease enter a valid 10-character PAN (e.g., ABCDE1234F) or upload your PAN Card image.",
                    "type": "error",
                    "waiting_for": "pan_card_upload"
                }

        if file_uploaded:

            expected_doc = "pan_card"

            # 🔐 Strict dropdown validation
            if file_uploaded.get("doc_type") != expected_doc:
                return {
                    "response": get_translated_message("invalid_pan", user_language),
                    "type": "error",
                    "waiting_for": "pan_card_upload"
                }

            user_name = session["personal_info"].get("name", "unknown").replace(" ", "_")

            # 🔍 Single OCR + AI extraction
            result = doc_intelligence.analyze_document(
                file_uploaded["content"],
                file_uploaded["extension"],
                expected_doc,
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

            # ✅ Fallback: regex extraction if AI failed
            if not pan_number:
                raw_text = result.get("raw_text", "")
                pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', raw_text.upper())
                if pan_match:
                    pan_number = pan_match.group(0)
                    fields["pan_number"] = pan_number
                    print("⚡ PAN extracted via regex fallback:", pan_number)

            # 🔒 Final PAN format validation
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
            aadhaar_number = re.sub(r'\D', '', session["extracted_data"].get("aadhaar_number", ""))
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

            # ✅ Details match check: PAN name vs Aadhaar name
            pan_name     = fields.get("name", "")
            aadhaar_name = session["personal_info"].get("name", "")
            if pan_name and aadhaar_name:
                is_name_ok, name_error = doc_intelligence.validate_name(pan_name, aadhaar_name, user_language)
                if not is_name_ok:
                    return {
                        "response": get_translated_message("pan_name_mismatch_upload", user_language,
                                                           aadhaar_name=aadhaar_name, pan_name=pan_name),
                        "type": "error",
                        "waiting_for": "pan_card_upload"
                    }

            # ✅ FLOW CHANGE: Linked → Go to collect_mobile instead of income_selection
            session["step"] = "collect_mobile"
            pan_linked_msg = get_translated_message("pan_aadhaar_linked", user_language)
            mobile_msg = get_translated_message("mobile_number_prompt", user_language)

            return {
                "response": f"{pan_linked_msg}\n\n{mobile_msg}",
                "type": "success",
                "waiting_for": "mobile_input"
            }

        else:
            return {
                "response": get_translated_message("pan_upload_prompt", user_language),
                "type": "text",
                "waiting_for": "pan_card_upload"
            }

    # ✅ STEP 1: COLLECT MOBILE
    elif current_step == "collect_mobile":
        # Generate application ID if not exists
        if not session.get("application_id"):
            application_id = db_manager.generate_application_id() if db_manager else f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            session["application_id"] = application_id

        if not user_message:
            response = {
                "response": get_translated_message("mobile_number_prompt", user_language),
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
            # ✅ FLOW CHANGE: Go directly to domicile proof selection (skip separate aadhaar upload step)
            session["step"] = "select_domicile_proof"
            response = {
                "response": get_translated_message("marital_confirmed", user_language, status=status),
                "type": "success",
                "waiting_for": "domicile_selection"
            }
        else:
            response = {
                "response": get_translated_message("invalid_option", user_language) + "\n\nWhat is your Marital Status?\n\n1. Married\n2. Unmarried\n3. Widow\n4. Divorced\n\nPlease enter the number (1-4):",
                "type": "error",
                "waiting_for": "marital_status"
            }
    
    # ✅ STEP 4: SELECT DOMICILE PROOF
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
    
    # ✅ STEP 5: UPLOAD DOMICILE PROOF
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
    
    # ✅ STEP 6: ASK RATION COLOR
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
    
    # ✅ STEP 7: UPLOAD INCOME CERTIFICATE
    elif current_step == "upload_income_certificate":

        # 🔹 Allow user to type 'exit' to quit after a rejection
        if user_message and isinstance(user_message, str) and user_message.strip().lower() in ["exit", "quit", "cancel"]:
            session["step"] = "exit"
            return {
                "response": get_translated_message("exit_prompt", user_language),
                "type": "info",
                "waiting_for": "none"
            }

        # =========================================================
        # 🔹 OPTION 7.1: Manual Income Selection (CHECKED FIRST)
        # =========================================================
        if user_message and not file_uploaded:

            income_map = {
                "1": {"label": "Less than ₹1 Lakh",  "max": 100000,        "eligible": True},
                "2": {"label": "₹1-2 Lakh",           "max": 200000,        "eligible": True},
                "3": {"label": "₹2-2.5 Lakh",         "max": 250000,        "eligible": True},
                "4": {"label": "More than ₹2.5 Lakh", "max": float('inf'), "eligible": False},
            }

            selected = income_map.get(user_message.strip())

            if selected:
                session["income_info"]["annual_income_display"] = selected["label"]
                session["income_info"]["annual_income"] = selected["max"] if selected["eligible"] else 300000
                session["income_info"]["source"] = "user_selected"

                # ❌ Not eligible
                if not selected["eligible"]:
                    session["step"] = "completed"
                    return {
                        "response": get_translated_message(
                            "eligibility_failure",
                            user_language,
                            reason="Annual income exceeds ₹2.50 Lakh"
                        ),
                        "type": "error",
                        "waiting_for": "none"
                    }

                # ✅ Eligible → Move to bank step
                session["step"] = "upload_bank_passbook"

                return {
                    "response": get_translated_message(
                        "income_manual_success",
                        user_language
                    ),
                    "type": "success",
                    "waiting_for": "bank_passbook_upload"
                }


        # =========================================================
        # 🔹 OPTION 7.2: Upload Income Certificate 
        # =========================================================
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
                return {
                    "response": "❌ Upload failed",
                    "type": "error",
                    "waiting_for": "income_certificate_upload"
                }

            result = doc_intelligence.analyze_document(
                file_uploaded["content"],
                file_uploaded["extension"],
                doc_type,
                blob_url,
                expected_name,
                user_language
            )

            if not result.get("is_valid"):
                return {
                    "response": result.get("validation_error"),
                    "type": "error",
                    "waiting_for": "income_certificate_upload"
                }

            # ------------------------------
            # 🔹 YOUR EXISTING INCOME LOGIC
            # ------------------------------

            fields = result.get("fields", {})
            session["income_info"]["certificate_number"] = fields.get("certificate_number", "")

            raw_income = ""
            for key in [
                "annual_income", "annual_income_amount", "annual_income_rupees",
                "income", "amount", "amount_in_words"
            ]:
                val = fields.get(key)
                if val:
                    raw_income = str(val)
                    break

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

                        if any(word in lname for word in income_key_words):
                            raw_income = s
                            break

                        if re.search(r'₹|\bRs\b|\bRs\.|rupee|रु', s, flags=re.IGNORECASE):
                            raw_income = s
                            break

                        if re.fullmatch(r'[₹Rs.\s,\d]+', s, flags=re.IGNORECASE):
                            if not any(word in lname for word in reject_key_words):
                                raw_income = s
                                break
                    except Exception:
                        continue

            parsed_income = parse_income_amount(raw_income, result.get("raw_text", ""))

            if not parsed_income:
                return {
                    "response": get_translated_message("income_not_found", user_language),
                    "type": "error",
                    "waiting_for": "income_certificate_upload"
                }

            display_income = format_currency(parsed_income)

            if parsed_income > 250000:
                session["income_info"]["annual_income"] = parsed_income
                session["income_info"]["annual_income_display"] = display_income
                session["income_info"]["source"] = "income_certificate"

                return {
                    "response": (
                        get_translated_message("income_exceeds", user_language)
                        + "\n\n"
                        + get_translated_message("income_retry_prompt", user_language)
                    ),
                    "type": "error",
                    "waiting_for": "income_certificate_upload"
                }

            session["documents"]["income_certificate"] = result
            if "income_certificate" not in session["uploaded_docs"]:
                session["uploaded_docs"].append("income_certificate")

            session["income_info"]["annual_income"] = parsed_income
            session["income_info"]["annual_income_display"] = display_income
            session["income_info"]["issue_date"] = fields.get("issue_date", "")
            session["income_info"]["source"] = "income_certificate"

            session["step"] = "upload_bank_passbook"

            return {
                "response": get_translated_message(
                    "income_success",
                    user_language,
                    cert_no=fields.get("certificate_number", "Extracted"),
                    name=fields.get("holder_name", "Extracted"),
                    income=display_income
                ),
                "type": "success",
                "waiting_for": "bank_passbook_upload"
            }

        # =========================================================
        # 🔹 DEFAULT PROMPT (Clear Dual Option)
        # =========================================================
        return {
            "response": get_translated_message("income_selection", user_language),
            "type": "text",
            "waiting_for": "income_certificate_upload"
        }


    # ✅ STEP 8: UPLOAD BANK PASSBOOK
    elif current_step == "upload_bank_passbook":

            # =========================================================
            # 🔹 OPTION 1: Manual Bank Entry (CHECKED FIRST)
            # =========================================================
            if user_message and not file_uploaded:

                parts = [p.strip() for p in user_message.split(',')]

                if len(parts) >= 3:

                    bank_name      = parts[0].strip()
                    account_number = re.sub(r'\s+', '', parts[1])
                    ifsc_code      = parts[2].strip().upper().replace(" ", "")

                    account_valid = account_number.isdigit() and 9 <= len(account_number) <= 18
                    ifsc_valid    = bool(re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', ifsc_code))
                    bank_valid    = len(bank_name) >= 3

                    if account_valid and ifsc_valid and bank_valid:

                        session["bank_info"] = {
                            "bank_name":      bank_name,
                            "account_number": account_number,
                            "ifsc":           ifsc_code,
                            "source":         "manual_input"
                        }

                        session["documents"]["bank_passbook"] = {
                            "is_valid": True,
                            "fields": {
                                "bank_name":      bank_name,
                                "account_number": account_number,
                                "ifsc_code":      ifsc_code
                            },
                            "blob_url": ""
                        }

                        if "bank_passbook" not in session["uploaded_docs"]:
                            session["uploaded_docs"].append("bank_passbook")

                        session["step"] = "upload_photograph"

                        return {
                            "response": get_translated_message(
                                "bank_manual_success",
                                user_language,
                                bank=bank_name,
                                account=mask_account(account_number),
                                ifsc=ifsc_code
                            ),
                            "type": "success",
                            "waiting_for": "photograph_upload"
                        }


                    # ❌ Validation errors
                    errors = []
                    if not bank_valid:
                        errors.append("- Bank name is too short")
                    if not account_valid:
                        errors.append(f"- Account number must be 9-18 digits (entered: {len(account_number)} chars)")
                    if not ifsc_valid:
                        errors.append(f"- IFSC code must match format XXXX0XXXXXX (entered: {ifsc_code})")

                    return {
                        "response": get_translated_message("bank_details_invalid_format", user_language) +
                                    "\n\nErrors:\n" + "\n".join(errors),
                        "type": "error",
                        "waiting_for": "bank_passbook_upload"
                    }

            # =========================================================
            # 🔹 OPTION 2: Upload Bank Passbook 
            # =========================================================
            if file_uploaded and file_uploaded.get("doc_type") == "bank_passbook":

                expected_name  = session["personal_info"].get("name", "")
                application_id = session.get("application_id")
                doc_type       = "bank_passbook"

                blob_url = upload_to_blob(
                    file_uploaded["content"],
                    application_id,
                    "bank_passbook",
                    file_uploaded["extension"]
                )

                if not blob_url:
                    return {
                        "response": "❌ Upload failed",
                        "type": "error",
                        "waiting_for": "bank_passbook_upload"
                    }

                result = doc_intelligence.analyze_document(
                    file_uploaded["content"],
                    file_uploaded["extension"],
                    doc_type,
                    blob_url,
                    expected_name,
                    user_language
                )

                if not result.get("is_valid"):
                    return {
                        "response": result.get("validation_error"),
                        "type": "error",
                        "waiting_for": "bank_passbook_upload"
                    }

                session["documents"]["bank_passbook"] = result
                if "bank_passbook" not in session["uploaded_docs"]:
                    session["uploaded_docs"].append("bank_passbook")

                fields = result.get("fields", {})

                session["bank_info"]["account_number"] = fields.get("account_number", "")
                session["bank_info"]["ifsc"]           = fields.get("ifsc_code", "")
                session["bank_info"]["bank_name"]      = fields.get("bank_name", "")

                session["step"] = "upload_photograph"

                return {
                    "response": get_translated_message(
                        "bank_success",
                        user_language,
                        holder=fields.get("account_holder_name", "Extracted"),
                        account=mask_account(fields.get("account_number", "")),
                        ifsc=fields.get("ifsc_code", "Extracted"),
                        bank=fields.get("bank_name", "Extracted")
                    ),
                    "type": "success",
                    "waiting_for": "photograph_upload"
                }

            # =========================================================
            # 🔹 DEFAULT PROMPT (Clear Dual Option)
            # =========================================================
            return {
                "response": get_translated_message("upload_prompt", user_language, doc="Bank Passbook"),
                "type": "text",
                "waiting_for": "bank_passbook_upload"
            }


    # ✅ STEP 9: UPLOAD PHOTOGRAPH
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
                    mobile=mobile,
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
    
        # ✅ STEP 10: FINAL REVIEW
    elif current_step == "final_review":

        if user_message.upper() in ["SUBMIT", "YES", "HO", "होय", "हो"]:

            # Direct submit
            session["step"] = "submit_application"

            application_id = session.get("application_id")

            response = {
                "response": f"""आपला अर्ज यशस्वीरीत्या सबमिट झाला आहे.

    आपला अर्ज क्रमांक:
    {application_id}

    आपण अर्ज स्थिती खालील पर्यायाने तपासू शकता:
    • अर्ज क्रमांक
    • आधार क्रमांक

    धन्यवाद.
    लाडकी बहिण योजना – सशक्त महिलांसाठी.""",
                "type": "submit",
                "waiting_for": "none",
                "application_id": application_id
            }

        elif user_message.upper() == "NO":

            # NO → correction flow
            session["step"] = "inline_correction"

            response = {
                "response": get_translated_message("correction_menu", user_language),
                "type": "info",
                "waiting_for": "correction_choice"
            }

        else:
            response = {
                "response": get_translated_message("confirmation_yes_no", user_language),
                "type": "text",
                "waiting_for": "review_confirmation"
            }

        return response
    
    # ✅ STEP 10.5: INLINE CORRECTION
    elif current_step == "inline_correction":

        # Sub-step: waiting for which field to correct
        if not session.get("correction_choice"):
            choice_map = {
                "1": "name", "2": "dob", "3": "address",
                "4": "mobile", "5": "email", "6": "bank", "7": "income"
            }
            choice = choice_map.get(user_message.strip())

            if not choice:
                return {
                    "response": get_translated_message("correction_invalid_choice", user_language),
                    "type": "error",
                    "waiting_for": "correction_choice"
                }

            session["correction_choice"] = choice

            prompt_key_map = {
                "name":    "correction_prompt_name",
                "dob":     "correction_prompt_dob",
                "address": "correction_prompt_address",
                "mobile":  "correction_prompt_mobile",
                "email":   "correction_prompt_email",
                "bank":    "correction_prompt_bank",
                "income":  "correction_prompt_income"
            }

            return {
                "response": get_translated_message(prompt_key_map[choice], user_language),
                "type": "info",
                "waiting_for": "correction_value"
            }

        # Sub-step: user entered the corrected value
        else:
            choice = session["correction_choice"]

            if choice == "name":
                session["personal_info"]["name"] = user_message.strip()

            elif choice == "dob":
                if re.match(r'^\d{2}/\d{2}/\d{4}$', user_message.strip()):
                    session["personal_info"]["dob"] = user_message.strip()
                    session["personal_info"]["age"] = calculate_age(user_message.strip())
                else:
                    return {
                        "response": get_translated_message("invalid_date", user_language),
                        "type": "error",
                        "waiting_for": "correction_value"
                    }

            elif choice == "address":
                session["contact_info"]["address"] = user_message.strip()

            elif choice == "mobile":
                if re.match(r'^[6-9]\d{9}$', user_message.strip()):
                    session["contact_info"]["mobile"] = user_message.strip()
                else:
                    return {
                        "response": get_translated_message("invalid_mobile", user_language),
                        "type": "error",
                        "waiting_for": "correction_value"
                    }

            elif choice == "email":
                if user_message.strip().lower() == "skip":
                    session["contact_info"]["email"] = ""
                elif re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', user_message.strip()):
                    session["contact_info"]["email"] = user_message.strip()
                else:
                    return {
                        "response": get_translated_message("invalid_email", user_language),
                        "type": "error",
                        "waiting_for": "correction_value"
                    }

            elif choice == "bank":
                parts = [p.strip() for p in user_message.split(',')]
                if len(parts) >= 3:
                    bank_name      = parts[0].strip()
                    account_number = re.sub(r'\s+', '', parts[1])
                    ifsc_code      = parts[2].strip().upper()
                    account_valid  = account_number.isdigit() and 9 <= len(account_number) <= 18
                    ifsc_valid     = bool(re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', ifsc_code))
                    if account_valid and ifsc_valid and len(bank_name) >= 3:
                        session["bank_info"]["bank_name"]      = bank_name
                        session["bank_info"]["account_number"] = account_number
                        session["bank_info"]["ifsc"]           = ifsc_code
                    else:
                        return {
                            "response": get_translated_message("bank_details_invalid_format", user_language),
                            "type": "error",
                            "waiting_for": "correction_value"
                        }
                else:
                    return {
                        "response": get_translated_message("bank_details_invalid_format", user_language),
                        "type": "error",
                        "waiting_for": "correction_value"
                    }

            elif choice == "income":
                income_map = {
                    "1": ("Less than ₹1 Lakh", 100000),
                    "2": ("₹1-2 Lakh", 200000),
                    "3": ("₹2-2.5 Lakh", 250000),
                    "4": ("More than ₹2.5 Lakh", 300000)
                }
                sel = income_map.get(user_message.strip())
                if sel:
                    session["income_info"]["annual_income_display"] = sel[0]
                    session["income_info"]["annual_income"] = sel[1]
                else:
                    return {
                        "response": get_translated_message("correction_invalid_income_choice", user_language),
                        "type": "error",
                        "waiting_for": "correction_value"
                    }

            # ✅ Clear correction state and return to final review
            session.pop("correction_choice", None)
            session["step"] = "upload_photograph"  # Re-trigger photo_success summary

            # Rebuild summary inline (same as upload_photograph success)
            name   = session["personal_info"].get("name", "N/A")
            dob    = session["personal_info"].get("dob", "N/A")
            age    = session["personal_info"].get("age", "N/A")
            marital = session["personal_info"].get("marital_status", "N/A")
            mobile = session["contact_info"].get("mobile", "N/A")
            email  = session["contact_info"].get("email", "Not provided")
            address = session["contact_info"].get("address", "N/A")
            aadhaar_masked = mask_aadhaar(session["extracted_data"].get("aadhaar_number", ""))
            pan_masked     = session["extracted_data"].get("pan_number", "XXXXXXXXXX")
            account_masked = mask_account(session["bank_info"].get("account_number", ""))
            ifsc       = session["bank_info"].get("ifsc", "N/A")
            bank_name  = session["bank_info"].get("bank_name", "N/A")
            annual_income = session["income_info"].get("annual_income_display", session["income_info"].get("annual_income", "N/A"))
            ration_type   = session["income_info"].get("ration_card_type", "N/A")
            doc_count = len(session["uploaded_docs"])
            doc_list_items = ["- Aadhaar Card",
                              f"- {DOCUMENT_TYPES.get(session.get('domicile_proof_type', ''), 'Domicile Proof')}"]
            if "income_certificate" in session["uploaded_docs"]:
                doc_list_items.append("- Income Certificate")
            doc_list_items += ["- Bank Passbook", "- Photograph"]
            doc_list = "\n".join(doc_list_items)

            session["step"] = "final_review"

            return {
                "response": get_translated_message(
                    "photo_success", user_language,
                    name=name, dob=dob, age=age, aadhaar=aadhaar_masked,
                    pan=pan_masked, marital=marital, mobile=mobile, email=email,
                    address=address, account=account_masked, ifsc=ifsc, bank=bank_name,
                    income=annual_income, ration=ration_type, count=doc_count, doc_list=doc_list
                ),
                "type": "success",
                "waiting_for": "review_confirmation"
            }
        
        # ✅ STEP 10.9: POST REGISTRATION MENU
    elif current_step == "post_registration_menu":

        if user_message == "1":
            session["step"] = "completed"

            return {
                "response": "redirect_to_eligibility",
                "type": "redirect",
                "target": "eligibility"
            }

        elif user_message == "2":
            session["step"] = "completed"

            return {
                "response": "redirect_to_post_application",
                "type": "redirect",
                "target": "post_application"
            }

        elif user_message == "3":
            session["step"] = "completed"

            return {
                "response": "लाडकी बहिण योजना वापरल्याबद्दल धन्यवाद.",
                "type": "info",
                "waiting_for": "none"
            }

        else:
            return {
                "response": get_translated_message(
                    "post_registration_options",
                    user_language
                ),
                "type": "error",
                "waiting_for": "post_registration_choice"
            }
        

    # declaration and submit_application section 

    # ✅ STEP 13: COMPLETED
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


