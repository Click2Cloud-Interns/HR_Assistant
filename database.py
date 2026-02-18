"""
Database Connection Module for Ladki Bahin Yojana
Unified module - Handles all SQL Server connectivity and operations
Uses pymssql driver
"""

import os
import pymssql
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================
# Database Configuration
# ============================================
DB_CONFIG = {
    "server": os.getenv("DB_HOST", "20.40.43.136"),
    "port": int(os.getenv("DB_PORT", "1642")),
    "database": os.getenv("DB_NAME", "DBLadliBehan"),
    "user": os.getenv("DB_USER", "sa"),
    "password": os.getenv("DB_PASSWORD", ""),
}


# ============================================
# Connection Helpers
# ============================================
def get_db_connection(as_dict: bool = True):
    """Get a new database connection"""
    try:
        return pymssql.connect(
            server=DB_CONFIG["server"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            as_dict=as_dict
        )
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        raise


def row_to_dict(cursor, row) -> Optional[Dict]:
    """Convert pymssql row to dictionary (for non-dict cursors)"""
    if not row or not cursor.description:
        return None
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


# ============================================
# Standalone Query Functions (used by eligibility.py, main.py)
# ============================================
def get_user_by_phone(phone_number: str) -> Optional[Dict]:
    """
    Get beneficiary details by mobile number
    Used by: eligibility.py (voice chatbot)
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Take last 10 digits
        processed_phone = phone_number[-10:] if len(phone_number) >= 10 else phone_number

        query = """
            SELECT
                BeneficiaryId, Username, FullName, DateOfBirth, Gender,
                MobileNumber, Email, Address, District, Taluka, Village,
                AnnualIncome, BankAccountNo, BankIFSC, SchemeCode,
                ApplicationDate, ApplicationStatus, ApprovedBy, ApprovedOn, RejectionReason
            FROM BeneficiaryApplication
            WHERE RIGHT(REPLACE(MobileNumber, ' ', ''), 10) = %s
        """
        cursor.execute(query, (processed_phone,))
        return cursor.fetchone()

    except Exception as e:
        logger.error(f"❌ Error in get_user_by_phone: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# def get_beneficiary_by_aadhaar_last4(aadhaar_last4: str) -> Optional[int]:
#     """
#     Get BeneficiaryId by last 4 digits of Aadhaar
#     Used by: main.py (post-application queries)
#     """
#     conn = None
#     cursor = None
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute(
#             "SELECT BeneficiaryId FROM BeneficiaryApplication WHERE AadhaarNumber LIKE %s",
#             ('%' + aadhaar_last4,)
#         )
#         row = cursor.fetchone()
#         return row['BeneficiaryId'] if row else None
#     except Exception as e:
#         logger.error(f"❌ Error in get_beneficiary_by_aadhaar_last4: {e}")
#         return None
#     finally:
#         if cursor:
#             cursor.close()
#         if conn:
#             conn.close()


def get_beneficiary_by_aadhaar(aadhaar_number: str) -> Optional[int]:
    """
    Get BeneficiaryId by full Aadhaar number
    Used by: post_registration.py (post-application queries)
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT BeneficiaryId FROM BeneficiaryApplication WHERE AadhaarNumber = %s",
            (aadhaar_number,)
        )
        row = cursor.fetchone()
        return row['BeneficiaryId'] if row else None
    except Exception as e:
        logger.error(f"❌ Error in get_beneficiary_by_aadhaar: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_beneficiary_details(beneficiary_id: int) -> Optional[Dict]:
    """
    Get full beneficiary details by ID
    Used by: main.py
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM BeneficiaryApplication WHERE BeneficiaryId = %s", (beneficiary_id,))
        result = cursor.fetchone()
        if result:
            result.pop("PasswordHash", None)  # Remove sensitive data
        return result
    except Exception as e:
        logger.error(f"❌ Error in get_beneficiary_details: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_beneficiary_transactions(beneficiary_id: int) -> List[Dict]:
    """
    Get all transactions for a beneficiary
    Used by: main.py
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM BeneficiaryTransactions WHERE BeneficiaryId = %s ORDER BY TransactionDate",
            (beneficiary_id,)
        )
        return cursor.fetchall() or []
    except Exception as e:
        logger.error(f"❌ Error in get_beneficiary_transactions: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ============================================
# DatabaseManager Class (used by app.py)
# ============================================
class DatabaseManager:
    """Manages database connections and CRUD operations for application submission"""

    def __init__(self):
        self.connection = None

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = get_db_connection()
            logger.info("✅ Database connected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database disconnected")

    def _get_cursor(self):
        """Get cursor, reconnecting if needed"""
        if not self.connection:
            self.connect()
        return self.connection.cursor()

    def generate_application_id(self) -> int:
        """Generate unique 14-digit application ID: YYYYMMDD + 6 random digits"""
        import random

        max_attempts = 10
        for _ in range(max_attempts):
            date_part = datetime.now().strftime("%Y%m%d")
            random_part = ''.join([str(random.randint(0, 9)) for _ in range(6)])
            application_id = int(date_part + random_part)

            if not self.check_beneficiary_exists(application_id):
                return application_id

        # Fallback: timestamp-based
        import time
        return int(datetime.now().strftime("%Y%m%d") + str(int(time.time() * 1000))[-6:])
    
    def get_aadhaar_details(self, aadhaar_no: str) -> Optional[Dict]:
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT TOP 1
                    AadhaarNo,
                    FullName,
                    DateOfBirth,
                    Gender,
                    Address,
                    City,
                    State,
                    Pincode,
                    MobileNo,
                    Email
                FROM AadhaarCardDetails
                WHERE AadhaarNo = %s
            """, (aadhaar_no,))
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error fetching Aadhaar details: {e}")
            return None

    def check_beneficiary_exists(self, beneficiary_id: int) -> bool:
        """Check if BeneficiaryId exists"""
        try:
            cursor = self._get_cursor()
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM BeneficiaryApplication WHERE BeneficiaryId = %s",
                (beneficiary_id,)
            )
            result = cursor.fetchone()
            return result['cnt'] > 0 if result else False
        except Exception as e:
            logger.error(f"Error checking beneficiary existence: {e}")
            return False

    def check_aadhaar_exists(self, aadhaar_number: str) -> bool:
        """Check if Aadhaar already registered"""
        try:
            cursor = self._get_cursor()
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM BeneficiaryApplication WHERE AadhaarNumber = %s",
                (aadhaar_number,)
            )
            result = cursor.fetchone()
            return result['cnt'] > 0 if result else False
        except Exception as e:
            logger.error(f"Error checking aadhaar existence: {e}")
            return False

    def save_beneficiary_application(self, data: Dict[str, Any], beneficiary_id: int) -> Optional[int]:
        """Save new beneficiary application"""
        try:
            cursor = self._get_cursor()

            # Enable IDENTITY_INSERT
            cursor.execute("SET IDENTITY_INSERT BeneficiaryApplication ON")

            query = """
            INSERT INTO BeneficiaryApplication (
                BeneficiaryId, Username, PasswordHash, LastLogin,
                AadhaarNumber, FullName, DateOfBirth, Gender,
                MobileNumber, Email, Address, District, Taluka, Village,
                AnnualIncome, BankAccountNo, BankIFSC,
                SchemeCode, ApplicationDate, ApplicationStatus,
                ApprovedBy, ApprovedOn, RejectionReason,
                CreatedOn, UpdatedOn
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """

            now = datetime.now()
            aadhaar = data.get('aadhaar_number', '')
            aadhaar_clean = aadhaar.replace(' ', '')[:12] if aadhaar else None

            values = (
                beneficiary_id,
                data.get('username', ''),
                data.get('password_hash', ''),
                None,  # LastLogin
                aadhaar_clean,
                data.get('full_name', ''),
                data.get('date_of_birth'),
                data.get('gender', 'F'),
                data.get('mobile_number', ''),
                data.get('email', ''),
                data.get('address', ''),
                data.get('district', ''),
                data.get('taluka', ''),
                data.get('village', ''),
                data.get('annual_income', 0),
                data.get('bank_account_no', ''),
                data.get('bank_ifsc', ''),
                'LADLI_BEHNA',
                now,
                'UNDER_REVIEW',
                None, None, None,
                now, now
            )

            cursor.execute(query, values)
            cursor.execute("SET IDENTITY_INSERT BeneficiaryApplication OFF")
            self.connection.commit()

            logger.info(f"✅ Beneficiary saved with ID: {beneficiary_id}")
            return beneficiary_id

        except Exception as e:
            logger.error(f"❌ Error saving beneficiary: {e}")
            if self.connection:
                try:
                    cursor.execute("SET IDENTITY_INSERT BeneficiaryApplication OFF")
                except:
                    pass
                self.connection.rollback()
            return None

    def save_document(self, data: Dict[str, Any]) -> Optional[int]:
        """Save document record"""
        try:
            cursor = self._get_cursor()

            query = """
            INSERT INTO documents (
                BeneficiaryId, MobileNumber, AadhaarNumber,
                DocumentType, DocumentUrl, UploadedOn, FullName,
                IncomeCertificateNumber, IncomeCertIssueDate, AnnualIncomeAmount,
                BankAccountNumber, BankIFSC, BankName,
                DomicileCertificateNumber, DomicileIssuingDistrict, DomicileIssueDate,
                ResidenceDistrict, ResidenceTaluka, ResidenceVillage,
                RationCardNumber, RationCardType, RationCardIssueDate,
                VoterIDNumber
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """

            aadhaar = data.get('aadhaar_number', '')
            aadhaar_clean = aadhaar.replace(' ', '')[:12] if aadhaar else None

            values = (
                data.get('beneficiary_id'),
                data.get('mobile_number', ''),
                aadhaar_clean,
                data.get('document_type', ''),
                data.get('document_url', ''),
                datetime.now(),
                data.get('full_name', ''),
                data.get('income_certificate_number'),
                data.get('income_cert_issue_date'),
                data.get('annual_income_amount'),
                data.get('bank_account_number'),
                data.get('bank_ifsc'),
                data.get('bank_name'),
                data.get('domicile_certificate_number'),
                data.get('domicile_issuing_district'),
                data.get('domicile_issue_date'),
                data.get('residence_district'),
                data.get('residence_taluka'),
                data.get('residence_village'),
                data.get('ration_card_number'),
                data.get('ration_card_type'),
                data.get('ration_card_issue_date'),
                data.get('voter_id_number')
            )

            cursor.execute(query, values)

            cursor.execute("SELECT SCOPE_IDENTITY() AS DocumentId")
            result = cursor.fetchone()
            document_id = result['DocumentId'] if result else None

            self.connection.commit()
            logger.info(f"✅ Document saved with ID: {document_id}")
            return document_id

        except Exception as e:
            logger.error(f"❌ Error saving document: {e}")
            if self.connection:
                self.connection.rollback()
            return None

    def update_beneficiary_status(self, beneficiary_id: int, status: str) -> bool:
        """Update application status"""
        try:
            cursor = self._get_cursor()
            cursor.execute(
                "UPDATE BeneficiaryApplication SET ApplicationStatus = %s, UpdatedOn = %s WHERE BeneficiaryId = %s",
                (status, datetime.now(), beneficiary_id)
            )
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return False

    def get_application_by_id(self, application_id: str) -> Optional[Dict]:
        """Retrieve application by ID"""
        try:
            cursor = self._get_cursor()
            cursor.execute(
                "SELECT * FROM BeneficiaryApplication WHERE BeneficiaryId = %s",
                (application_id,)
            )
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error retrieving application: {e}")
            return None

    def save_beneficiary_from_aadhaar(self, aadhaar: Dict[str, Any]) -> Optional[int]:
        try:
            cursor = self._get_cursor()

            beneficiary_id = self.generate_application_id()
            now = datetime.now()

            cursor.execute("""
                INSERT INTO BeneficiaryApplication (
                    BeneficiaryId,
                    AadhaarNumber,
                    FullName,
                    DateOfBirth,
                    Gender,
                    Address,
                    District,
                    SchemeCode,
                    ApplicationDate,
                    ApplicationStatus,
                    CreatedOn,
                    UpdatedOn
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,
                    'LADLI_BEHNA',
                    %s,
                    'UNDER_REVIEW',
                    %s,%s
                )
            """, (
                beneficiary_id,
                aadhaar["aadhaar_number"],
                aadhaar["full_name"],
                aadhaar["date_of_birth"],
                aadhaar["gender"],
                aadhaar["address"],
                aadhaar["district"],
                now,
                now,
                now
            ))

            self.connection.commit()
            return beneficiary_id

        except Exception as e:
            logger.error(f"❌ Error saving Aadhaar beneficiary: {e}")
            self.connection.rollback()
            return None

    def verify_pan_aadhaar_link(self, aadhaar_no: str, pan_no: str) -> Optional[Dict]:
        """
        Verify if PAN is linked with Aadhaar
        """
        try:
            cursor = self._get_cursor()

            # Normalize before DB check
            aadhaar_no = str(aadhaar_no).strip()
            pan_no = str(pan_no).strip().upper()

            print("🔎 DB CHECK -> Aadhaar:", aadhaar_no)
            print("🔎 DB CHECK -> PAN:", pan_no)

            cursor.execute("""
                SELECT TOP 1 *
                FROM AadhaarCardDetails
                WHERE AadhaarNo = %s
                AND PANNo = %s
                AND IsPANNoAttached = 1
            """, (aadhaar_no, pan_no))

            result = cursor.fetchone()

            print("🔎 DB RESULT:", result)

            return result

        except Exception as e:
            logger.error(f"Error verifying PAN-Aadhaar link: {e}")
            return None

# ============================================
# Singleton Instance
# ============================================
db_manager = DatabaseManager()
