import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# Base directory of the backend folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Directories
UPLOAD_DIR = BASE_DIR / "uploads"
DATABASE_DIR = BASE_DIR / "database"
CHROMA_DB_DIR = DATABASE_DIR / "chroma_db"
AUDIT_DIR = BASE_DIR / "audit"
INDEX_DIR = BASE_DIR / "uploads" / "_indexes"

# Automatically Create Directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Audit Log
AUDIT_LOG_PATH = AUDIT_DIR / "audit_log.jsonl"

# Application Information
APP_NAME = os.getenv(
    "APP_NAME",
    "Sensitive Data Detection & Compliance Assistant"
)

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# LLM API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Embedding Model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# RAG Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K_RETRIEVAL = 4

# Upload Configuration
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv"}
MAX_FILE_SIZE_MB = 20

# Risk Scoring
# Higher weight = more severe if found in the document.
RISK_WEIGHTS = {
    "aadhaar_number": 10,
    "pan_number": 10,
    "credit_card_number": 10,
    "bank_account_number": 8,
    "ifsc_code": 4,
    "api_key": 10,
    "password": 9,
    "email_address": 3,
    "phone_number": 3,
    "employee_id": 2,
    "confidential_business_info": 2
}

# High Risk Categories
HIGH_RISK_TYPES = {
    "aadhaar_number", "pan_number", "credit_card_number",
    "bank_account_number", "api_key", "password"
}

# Score Risk Thresholds -> risk level
RISK_THRESHOLD_LOW = 0
RISK_THRESHOLD_MEDIUM = 8
RISK_THRESHOLD_HIGH = 25