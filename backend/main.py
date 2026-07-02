"""
FastAPI Backend

Sensitive Data Detection & Compliance Assistant

This file exposes REST APIs for:

1. Uploading documents
2. Detecting sensitive data
3. Risk Classification
4. Compliance Summary
5. Question Answering (RAG)
6. Audit Logs

Run:

uvicorn backend.main:app --reload
"""

import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configuration
from backend.config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

# Utilities
from backend.utils.file_parser import extract_text
from backend.utils.masking import build_masked_document, mask_value

# Detection
from backend.detectors.pii_detector import detect

# Classification
from backend.classifiers.risk_classifier import classify

# Summary
from backend.classifiers.summary_generator import generate_summary

# RAG
from backend.rag.qa_engine import build_index, answer_question

# Models
from backend.models.schemas import UploadResponse, DetectionSummary, DetectionMatch, RiskClassification, ComplianceSummary, AskRequest, AskResponse, AuditEntry

# Audit Logs
from backend.audit_log import log_event, read_log

app = FastAPI(
    title="Sensitive Data Detection & Compliance Assistant",
    description="""
    Upload a PDF/TXT/CSV document.
    Detect sensitive information.
    Classify document risk.
    Generate compliance summaries.
    Ask questions using RAG.
    """,

    version="1.0.0",
)

# CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory Document Registry
DOCUMENT_STORE: Dict[str, Dict[str, Any]] = {}

# Health Check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "Sensitive Data Detection API",
        "version": "1.0.0",
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and analyze a document."""
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed types: {sorted(ALLOWED_EXTENSIONS)}")

    # Read uploaded file
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400,detail=f"Maximum upload size is {MAX_FILE_SIZE_MB} MB.")

    # Save uploaded file
    document_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{document_id}{suffix}"
    save_path.write_bytes(contents)

    # Extract text
    try:
        raw_text = extract_text(save_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Unable to parse document: {e}")
    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No readable text found inside document.")

    # 1. Detect Sensitive Data
    detections = detect(raw_text)

    # 2. Mask Sensitive Data (used for preview, RAG index, and LLM summary — raw
    # PII is never sent to any external LLM call in this app).
    masked_text = build_masked_document(raw_text, detections)

    # 3. Risk Classification
    risk = classify(detections)

    # 4.  Build FAISS Index over the MASKED text
    build_index(document_id, masked_text)

    # 5. AI Summary
    summary = generate_summary(risk["breakdown"], risk["risk_level"], masked_text)

    # Build response detection summaries (with masked examples only)
    # Convert detections into response schema
    detection_results = []
    for category, matches in detections.items():
        if not matches:
            continue
        examples = []
        for match in matches[:5]:
            masked_value = mask_value(category, match["value"])
            # Mask the raw value inside its own context snippet too, so no
            # example surfaced to the client ever contains real PII.
            masked_context = match["context"].replace(match["value"], masked_value)
            examples.append(DetectionMatch(
                type=category,
                masked_value=masked_value,
                context=masked_context,
            ))
        detection_results.append(DetectionSummary(
            type=category,
            count=len(matches),
            risk_weight=risk["breakdown"].get(category, 0),
            examples=examples,
        ))

    # Store in memory
    DOCUMENT_STORE[document_id] = {
        "filename": file.filename,
        "masked_text": masked_text,
        "char_count": len(raw_text),
        "risk": risk,
        "detections": {key: len(value) for key, value in detections.items() if value},
    }

    log_event("upload", document_id=document_id, filename=file.filename,
              detail=f"risk={risk['risk_level']} score={risk['risk_score']}")
    log_event("detection_run", document_id=document_id, filename=file.filename,
              detail=str(DOCUMENT_STORE[document_id]["detections"]))

    # Return Response
    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        char_count=len(raw_text),
        detections=detection_results,
        risk=RiskClassification(**risk),
        summary=ComplianceSummary(**summary),
        masked_preview=masked_text[:2000],
    )

@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    """Ask a question about a previously uploaded document."""
    # Verify document exists
    if payload.document_id not in DOCUMENT_STORE:
        raise HTTPException(404, "Unknown document_id. Upload the document first.")

    # Generate answer using RAG
    answer, sources = answer_question(payload.document_id, payload.question)

    # Audit logging
    log_event("question_asked", document_id=payload.document_id,
              filename=DOCUMENT_STORE[payload.document_id]["filename"],
        detail=payload.question)

    # Return response
    return AskResponse(
        document_id=payload.document_id,
        question=payload.question,
        answer=answer,
        sources=sources,
    )

@app.get("/documents")
def list_documents():
    """List all uploaded documents."""
    return [
        {
            "document_id": document_id,
            "filename": info["filename"],
            "char_count": info["char_count"],
            "risk_level": info["risk"]["risk_level"],
            "risk_score": info["risk"]["risk_score"],
            "detections": info["detections"],
        }
        for document_id, info in DOCUMENT_STORE.items()
    ]

@app.get("/audit-log", response_model=list[AuditEntry])
def audit_log(limit: int = 200):
    return read_log(limit)