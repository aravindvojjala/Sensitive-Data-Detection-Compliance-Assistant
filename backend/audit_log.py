"""
Provides simple JSONL-based audit logging.

Each event is written as one JSON object per line.

Example:

{"timestamp":"2026-07-02T10:21:31",
 "event":"upload",
 "document_id":"123",
 "filename":"employees.pdf",
 "detail":"risk=High score=32"}
"""

import json
from datetime import datetime, timezone
from typing import Optional

from backend.config import AUDIT_LOG_PATH

# Ensure the audit log file exists
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

if not AUDIT_LOG_PATH.exists():
    AUDIT_LOG_PATH.touch()

# Write an audit event
def log_event(action: str, document_id: Optional[str] = None,
              filename: Optional[str] = None, detail: Optional[str] = None) -> None:
    """Append a single audit event to the JSONL log file.
    Parameters
    ----------
    event : str
        Type of event (upload, detection_run, question_asked, etc.)
    document_id : str
        Unique ID assigned to the uploaded document.
    filename : str
        Original uploaded filename.
    detail : str
        Additional information about the event.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        #"event": event,
        "action": action,
        "document_id": document_id,
        "filename": filename,
        "detail": detail,
    }

    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# Read audit log
def read_log(limit: int = 200):
    """Read the audit log and return the newest entries first.
    Parameters
    ----------
    limit : int
        Maximum number of log entries to return.
    Returns
    -------
    List[Dict[str, Any]]
        List of audit log records.
    """
    if not AUDIT_LOG_PATH.exists():
        return []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entries = [json.loads(line) for line in lines[-limit:]]
    entries.reverse()  # most recent first
    return entries


    # if not AUDIT_LOG_PATH.exists():
    #     return []
    # records: List[Dict[str, Any]] = []
    # with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as file:
    #     for line in file:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         try:
    #             record = json.loads(line)
    #             records.append(record)
    #         except json.JSONDecodeError:
    #             # Skip malformed log entries instead of crashing
    #             continue
    # # Return newest entries first
    # records.reverse()
    # return records[:limit]