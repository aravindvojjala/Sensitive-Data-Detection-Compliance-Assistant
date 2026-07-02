"""
Masking / redaction helpers.

Utility functions for masking sensitive information
This module provides:
1. mask_value() - Masks an individual sensitive value.
2. build_masked_document() - Produces a document with all detected
   sensitive values replaced by their masked versions.
"""

from typing import Dict, List, Any


def mask_value(category: str, value: str) -> str:
    """Return a redacted representation of a single detected value."""
    digits_only = "".join(ch for ch in value if ch.isalnum())

    if category in ("aadhaar_number", "credit_card_number", "bank_account_number"):
        if len(digits_only) <= 4:
            return "*" * len(digits_only)
        return "*" * (len(digits_only) - 4) + digits_only[-4:]

    if category == "pan_number":
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    if category == "email_address":
        try:
            local, domain = value.split("@", 1)
            visible = local[:2]
            return f"{visible}{'*' * max(1, len(local) - 2)}@{domain}"
        except ValueError:
            return "***@***"

    if category == "phone_number":
        return "*" * max(0, len(digits_only) - 4) + digits_only[-4:]

    if category in ("api_key", "password"):
        return value[:6] + "*" * max(0, len(value) - 6)

    if category == "ifsc_code":
        return value[:4] + "*" * (len(value) - 4)

    if category == "employee_id":
        if len(value) <= 3:
            return "*" * len(value)
        return "*" * (len(value) - 3) + value[-3:]

    return value[:2] + "*" * max(0, len(value) - 2)


def build_masked_document(text: str, detections: Dict[str, List[Dict[str, Any]]]) -> str:
    """Replace every detected span in `text` with its masked form."""
    spans = []
    for category, matches in detections.items():
        for m in matches:
            spans.append((m["start"], m["end"], category, m["value"]))

    # Apply replacements from the end of the string backwards so earlier
    # offsets stay valid as we mutate the string.
    spans.sort(key=lambda s: s[0], reverse=True)

    masked = text
    for start, end, category, value in spans:
        masked = masked[:start] + mask_value(category, value) + masked[end:]
    return masked