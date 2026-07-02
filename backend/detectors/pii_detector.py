"""
Sensitive data detection engine.

Approach: deterministic regex + validation-rule based detection (no LLM
in this stage, on purpose) so results are reproducible, auditable, and
fast — important properties for a compliance tool. Each detector returns
raw matches with their character span so callers can mask / redact them
later without re-scanning the text.

Detected categories:
    - Aadhaar numbers
    - PAN numbers
    - Email addresses
    - Phone numbers (Indian mobile format, with/without +91)
    - Credit card numbers (Luhn-validated)
    - Bank account numbers + IFSC codes
    - API keys / secrets / passwords
    - Employee IDs
    - Confidential business information (keyword heuristic)
"""

import re
from typing import List, Dict, Any
import spacy

nlp = spacy.load("en_core_web_sm")

# Regex patterns

PATTERNS = {
    "aadhaar_number": re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
    "pan_number": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "email_address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_number": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "credit_card_number": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ifsc_code": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "bank_account_number": re.compile(r"\b\d{9,18}\b"),
    # Literal key formats (the match itself IS the secret, mask the whole thing)
    "api_key": re.compile(
        r"AKIA[0-9A-Z]{16}"          # AWS access key
        r"|sk-[A-Za-z0-9]{20,}"       # OpenAI-style secret key
        r"|gsk_[A-Za-z0-9]{20,}"      # Groq-style key
        r"|ghp_[A-Za-z0-9]{30,}",     # GitHub PAT
    ),
    # Labeled formats ("api_key: xyz") — only the value after the label is
    # the secret, so it's captured in a named group and masked on its own,
    # keeping the human-readable label intact.
    "api_key_labeled": re.compile(
        r"(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?(?P<secret>[A-Za-z0-9\-_]{12,})['\"]?",
        re.IGNORECASE,
    ),
    "password": re.compile(
        r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?(?P<secret>\S{4,}?)['\"]?(?=\s|$|[,.;])",
        re.IGNORECASE,
    ),
    "employee_id": re.compile(r"\b(?:EMP|EMPID|E)[-_]?\d{3,7}\b", re.IGNORECASE),
}

CONFIDENTIAL_KEYWORDS = [
    "confidential", "strictly private", "internal use only",
    "do not distribute", "proprietary", "trade secret",
    "not for external distribution", "company confidential",
]

# Words that, if seen near a raw digit string, boost confidence it's a bank
# account number rather than some other arbitrary long number.
BANK_CONTEXT_WORDS = ["account no", "account number", "a/c no", "a/c", "bank account", "acct"]


def _luhn_valid(number: str) -> bool:
    """Validate a numeric string against the Luhn checksum (credit cards)."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _context_window(text: str, start: int, end: int, radius: int = 40) -> str:
    """
    Return a small slice of text around the detected match.
    Used for displaying context in reports.
    """
    return text[max(0, start - radius):min(len(text), end + radius)]


def detect(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all detectors over `text`.

    Returns a dict: { category: [ {value, start, end, context}, ... ] }
    """
    # Output categories exposed to callers. "api_key_labeled" and "password"
    # both feed the labeled-secret categories below but merge into the
    # simpler public keys.
    OUTPUT_CATEGORY = {"api_key_labeled": "api_key"}

    results: Dict[str, List[Dict[str, Any]]] = {
        OUTPUT_CATEGORY.get(k, k): [] for k in PATTERNS
    }
    results["confidential_business_info"] = []

    # Track spans already claimed by a higher-priority category so, e.g.,
    # a 16-digit credit card isn't *also* reported as a generic bank account.
    claimed_spans: List[tuple] = []

    def is_claimed(start, end):
        return any(s <= start < e or s < end <= e for s, e in claimed_spans)

    # Priority order matters: more specific patterns first.
    priority_order = [
        "credit_card_number", "aadhaar_number", "pan_number", "email_address",
        "api_key", "api_key_labeled", "password", "ifsc_code", "phone_number",
        "employee_id", "bank_account_number",
    ]

    for category in priority_order:
        pattern = PATTERNS[category]
        output_category = OUTPUT_CATEGORY.get(category, category)

        for m in pattern.finditer(text):
            # For labeled patterns, only the "secret" group is the actual
            # sensitive value — mask/report that span, not the label text.
            if "secret" in pattern.groupindex:
                raw_value = m.group("secret")
                start, end = m.span("secret")
            else:
                raw_value = m.group(0)
                start, end = m.start(), m.end()

            if is_claimed(start, end):
                continue

            if category == "credit_card_number":
                digits_only = re.sub(r"[ -]", "", raw_value)
                if not (13 <= len(digits_only) <= 16) or not _luhn_valid(digits_only):
                    continue

            if category == "bank_account_number":
                digits_only = re.sub(r"\D", "", raw_value)
                if not (9 <= len(digits_only) <= 18):
                    continue
                window = _context_window(text, start, end, radius=30).lower()
                if not any(w in window for w in BANK_CONTEXT_WORDS):
                    # Too risky to call an arbitrary long number a bank account
                    # without contextual evidence -> skip to avoid false positives.
                    continue

            if category == "phone_number":
                digits_only = re.sub(r"\D", "", raw_value)
                if len(digits_only) not in (10, 12):
                    continue

            claimed_spans.append((start, end))
            results[output_category].append({
                "value": raw_value,
                "start": start,
                "end": end,
                "context": _context_window(text, start, end),
            })

    # Keyword-based detection for confidential business language.
    lower_text = text.lower()
    for kw in CONFIDENTIAL_KEYWORDS:
        for m in re.finditer(re.escape(kw), lower_text):
            start, end = m.start(), m.end()
            results["confidential_business_info"].append({
                "value": text[start:end],
                "start": start,
                "end": end,
                "context": _context_window(text, start, end),
            })

    return results