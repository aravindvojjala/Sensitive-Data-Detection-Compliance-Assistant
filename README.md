# Sensitive Data Detection & Compliance Assistant

An AI-powered application that uploads a document (PDF/TXT/CSV), detects
sensitive/confidential information, classifies risk, generates a
compliance summary, and answers questions about the document using RAG.

**Stack:** FastAPI (backend/API) + Streamlit (frontend UI) + regex/rule-based
PII detection + sentence-transformers & FAISS (RAG) + Groq LLM (optional,
for summaries and answers) + Docker.

---

## 1. Architecture Overview

```
┌─────────────────┐        HTTP (JSON)         ┌──────────────────────┐
│   Streamlit UI    │ ───────────────────────▶ │     FastAPI Backend     │
│   (frontend/app.py)│ ◀─────────────────────── │      (backend/main.py)   │
└─────────────────┘                            └──────────┬───────────┘
                                                            │
                     ┌──────────────────────────────────────┼──────────────────────────────┐
                     ▼                                      ▼                              ▼
           ┌───────────────────┐                ┌────────────────────┐          ┌──────────────────┐
           │ file_parser.py       │                │ pii_detector.py         │          │ risk_classifier.py │
           │ (PDF/TXT/CSV -> text) │                │ (regex + Luhn detection)│          │ (Low/Med/High score)│
           └───────────────────┘                └────────────────────┘          └──────────────────┘
                                                            │
                                                            ▼
                                                  ┌────────────────────┐
                                                  │   masking.py          │  -> masked document (never raw PII downstream)
                                                  └──────────┬─────────┘
                                                             │
                              ┌──────────────────────────────┼───────────────────────────┐
                              ▼                              ▼                           ▼
                  ┌────────────────────┐         ┌─────────────────────┐      ┌──────────────────┐
                  │ summary_generator.py │         │ qa_engine.py (RAG)     │      │  audit_log.py        │
                  │ (compliance summary) │         │ FAISS + embeddings +   │      │ (JSONL audit trail)  │
                  │                       │         │ Groq LLM (optional)    │      │                       │
                  └────────────────────┘         └─────────────────────┘      └──────────────────┘
```

**Key design decision:** detection is done with deterministic regex + Luhn
validation (not an LLM) so results are explainable, reproducible, and fast
— critical for a compliance tool where you must be able to say *exactly*
why something was flagged. The LLM (Groq, optional) is used only for the
"soft" tasks: turning the detection breakdown into a readable narrative,
and answering free-form questions over the (already masked) document.
If no LLM key is configured, the app still fully works using rule-based
summaries and extractive Q&A.

---

## 2. Project Structure

```
sensitive-data-compliance-assistant/
├── backend/
│   ├── main.py                       # FastAPI app & endpoints
│   ├── config.py                     # central configuration
│   ├── audit_log.py                  # JSONL audit trail
│   ├── detectors/
│   │   └── pii_detector.py           # regex + Luhn-based detection engine
│   ├── classifiers/
│   │   ├── risk_classifier.py        # Low/Medium/High scoring
│   │   └── summary_generator.py      # compliance summary (LLM or rule-based)
│   ├── rag/
│   │   └── qa_engine.py              # chunking, FAISS index, Q&A
│   ├── utils/
│   │   ├── file_parser.py            # PDF/TXT/CSV -> text
│   │   └── masking.py                # redaction helpers
│   ├── models/
│   │   └── schemas.py                # Pydantic request/response models
│   └── requirements.txt
├── frontend/
│   ├── app.py                        # Streamlit UI
│   └── requirements.txt
├── data/
│   └── sample_document.txt           # test file with fake sensitive data
├── uploads/                          # uploaded files + FAISS indexes (runtime)
├── audit/                            # audit_log.jsonl (runtime)
├── .env.example
└── README.md
```

---

## 3. Step-by-Step: Running Locally (no Docker)

### Step 1 — Prerequisites
- Python 3.10+ installed
- (Optional but recommended) a free Groq API key from
  https://console.groq.com/keys — enables LLM-generated summaries and
  natural-language answers. Without it, the app uses rule-based/extractive
  fallbacks and still works end-to-end.

### Step 2 — Get a copy of the project
Unzip the project you downloaded, or clone it if you pushed it to GitHub:
```bash
cd sensitive-data-compliance-assistant
```

### Step 3 — Configure environment variables
```bash
cp .env.example .env
# then open .env and paste your GROQ_API_KEY (optional)
```

### Step 4 — Set up and run the backend (Terminal 1)
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
Backend is now live at http://localhost:8000 — visit
http://localhost:8000/docs for the interactive Swagger UI.

### Step 5 — Set up and run the frontend (Terminal 2, new window)
```bash
cd frontend
python -m venv venv
venv\Scripts\activate        # or source venv/bin/activate
pip install -r requirements.txt
streamlit run frontend/app.py
```
Streamlit opens at http://localhost:8501.

### — Access the app
- Backend/API: http://localhost:8000/docs
- Frontend UI: http://localhost:8501

### Step 6 — Try it out
1. In the sidebar, upload `data/sample_document.txt` (included in this
   project — it contains realistic-looking but fake sensitive data).
2. Click **Analyze Document**.
3. Explore the **Overview**, **Detections**, and **Compliance Summary** tabs.
4. Go to **Ask Questions** and try:
   - "What sensitive data exists in the document?"
   - "How many email addresses are present?"
   - "Summarize this document."
   - "What compliance risks are identified?"

---

## 4. API Reference (FastAPI)

| Method | Endpoint       | Description                                         |
|--------|----------------|------------------------------------------------------|
| POST   | `/upload`      | Upload a PDF/TXT/CSV, run detection + classification + summary, build RAG index |
| POST   | `/ask`         | Ask a question about a previously uploaded document (`{document_id, question}`) |
| GET    | `/documents`   | List all documents processed in this session          |
| GET    | `/audit-log`   | View the audit trail (uploads, detections, questions) |
| GET    | `/health`      | Liveness check                                         |

Full interactive docs (with request/response schemas) are auto-generated
by FastAPI at `/docs` once the backend is running.

---

## 5. How Each Functional Requirement Is Met

1. **Document Upload** — `POST /upload` accepts PDF/TXT/CSV; parsed by
   `utils/file_parser.py` (`pdfplumber` for PDF, `pandas` for CSV).
2. **Sensitive Data Detection** — `detectors/pii_detector.py` uses
   category-specific regex plus validation rules (e.g. Luhn checksum for
   credit cards, contextual keyword proximity for bank account numbers)
   to detect Aadhaar, PAN, email, phone, credit card, bank details/IFSC,
   API keys/passwords, employee IDs, and confidential-business language.
3. **Risk Classification** — `classifiers/risk_classifier.py` computes a
   weighted score across detected categories and maps it to Low/Medium/High,
   with a floor rule so a single instance of a high-severity type (e.g. a
   credit card number) is never classified as Low.
4. **AI-Generated Summary** — `classifiers/summary_generator.py` produces
   compliance observations, security risks, and remediation steps, either
   via the Groq LLM (grounded in the masked document + detection
   breakdown) or a deterministic rule-based generator if no LLM key is set.
5. **Question Answering** — `rag/qa_engine.py` chunks the *masked* document,
   embeds chunks locally with `sentence-transformers`, indexes them in
   FAISS, retrieves top-K relevant chunks per question, and (if configured)
   asks Groq to answer using only those excerpts.
6. **Interface** — FastAPI backend + Streamlit frontend, communicating over
   HTTP/JSON, as requested.

### Bonus features implemented
- **Data masking/redaction** — every detected value is masked before it
  is stored, indexed, sent to any LLM, or shown in example results.
- **RAG implementation** — full FAISS + embeddings + LLM pipeline (with
  extractive fallback).
- **Dockerization** — `docker/Dockerfile.backend`, `Dockerfile.frontend`,
  and `docker-compose.yml`.
- **Audit logging** — every upload, detection run, and question is
  appended to `audit/audit_log.jsonl` and viewable from the sidebar.

### Bonus features not implemented (documented as future work)
- OCR for scanned/image-only PDFs (would add `pytesseract` + `pdf2image`
  to `utils/file_parser.py`).
- Multi-document cross-referencing / comparison in a single chat session.
- Persistent storage via a real database instead of the in-memory
  `DOCUMENT_STORE` dict in `main.py` (swap in SQLite/Postgres for production).

---

## 6. Security & Compliance Notes

- Raw sensitive values are **never** persisted beyond the uploaded file
  itself; all downstream artifacts (RAG index, LLM prompts, API
  responses, UI displays) use masked values only.
- The `/upload` endpoint enforces file type and size limits.
- `audit_log.py` provides a basic, append-only trail of who accessed what
  and when. For real production use, pair this with authentication
  (currently out of scope — this is a local/demo app with no auth layer)
  and a proper database-backed audit store.
- Regex-based detection can produce false positives/negatives — this tool
  is a decision-support aid, not a substitute for a full DLP/compliance
  review process.

---

## 7. Extending the Project

- **Add a new sensitive-data type:** add a pattern to `PATTERNS` in
  `detectors/pii_detector.py`, a weight in `config.RISK_WEIGHTS`, and a
  label/remediation tip in `classifiers/summary_generator.py`.
- **Swap the LLM provider:** replace the Groq client calls in
  `rag/qa_engine.py` / `classifiers/summary_generator.py` with
  OpenAI/Gemini/HuggingFace equivalents — the rest of the pipeline is
  provider-agnostic.
- **Add OCR:** extend `utils/file_parser.py`'s `_read_pdf` to fall back to
  `pytesseract` when `page.extract_text()` returns empty text.
