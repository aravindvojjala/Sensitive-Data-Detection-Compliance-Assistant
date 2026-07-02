"""
Generates the compliance/security summary shown after upload.

Uses the Groq LLM (if configured) to turn the raw detection breakdown +
risk level into a readable narrative, compliance observations, security
risks, and remediation steps. Falls back to a deterministic rule-based
summary if no LLM key is configured, so the app is fully usable offline.
"""
from typing import Dict, Any, List

from backend.config import GROQ_API_KEY, GROQ_MODEL

CATEGORY_LABELS = {
    "aadhaar_number": "Aadhaar Numbers",
    "pan_number": "PAN Numbers",
    "email_address": "Email Addresses",
    "phone_number": "Phone Numbers",
    "credit_card_number": "Credit Card Numbers",
    "bank_account_number": "Bank Account Numbers",
    "ifsc_code": "IFSC Codes",
    "api_key": "API Keys / Secrets",
    "password": "Passwords",
    "employee_id": "Employee IDs",
    "confidential_business_info": "Confidential Business Language",
}

REMEDIATION_LIBRARY = {
    "aadhaar_number": "Mask or tokenize Aadhaar numbers before storage; restrict access under DPDP Act requirements.",
    "pan_number": "Redact PAN numbers in shared copies; store only where legally required, with access controls.",
    "credit_card_number": "Remove or tokenize card numbers immediately (PCI-DSS); never store CVV.",
    "bank_account_number": "Mask account numbers in logs and shared documents; encrypt at rest.",
    "ifsc_code": "Low sensitivity alone, but combined with account numbers increases fraud risk — mask together.",
    "api_key": "Rotate any exposed API keys immediately and move secrets to a vault / environment variables.",
    "password": "Rotate exposed credentials immediately; never store plaintext passwords in documents.",
    "email_address": "Minimize sharing of email lists; apply consent/purpose checks under applicable privacy law.",
    "phone_number": "Mask phone numbers in externally shared versions of the document.",
    "employee_id": "Low sensitivity; ensure document isn't combined with other identifiers externally.",
    "confidential_business_info": "Apply document classification labels and restrict distribution to a need-to-know basis.",
}


def _rule_based_summary(breakdown: Dict[str, int], risk_level: str) -> Dict[str, Any]:
    observations: List[str] = []
    risks: List[str] = []
    remediation: List[str] = []

    for cat, count in breakdown.items():
        label = CATEGORY_LABELS.get(cat, cat)
        observations.append(f"{count} instance(s) of {label} detected.")
        if cat in ("aadhaar_number", "pan_number", "credit_card_number",
                    "bank_account_number", "api_key", "password"):
            risks.append(f"Exposure of {label} may violate data protection / financial regulations.")
        if cat in REMEDIATION_LIBRARY:
            remediation.append(REMEDIATION_LIBRARY[cat])

    if not breakdown:
        observations.append("No sensitive data patterns were detected in this document.")
        risks.append("No significant security risks identified from pattern-based scanning.")
        remediation.append("No immediate action required; periodic re-scanning is still recommended.")

    narrative = (
        f"This document was classified as {risk_level} risk based on {sum(breakdown.values())} "
        f"sensitive data instance(s) across {len(breakdown)} categor{'y' if len(breakdown)==1 else 'ies'}. "
        "Review the detection breakdown and apply the suggested remediation steps before sharing "
        "this document outside its intended audience."
    )

    return {
        "compliance_observations": observations,
        "security_risks": list(dict.fromkeys(risks)),       # de-dup, keep order
        "remediation_steps": list(dict.fromkeys(remediation)),
        "narrative": narrative,
    }


def _llm_summary(breakdown: Dict[str, int], risk_level: str, masked_preview: str) -> Dict[str, Any]:
    import json
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate

    breakdown_text = "\n".join(f"- {CATEGORY_LABELS.get(c, c)}: {n}" for c, n in breakdown.items()) or "None detected"

    system_prompt = (
        "You are a data compliance and security analyst. Given a sensitive-data "
        "detection breakdown and a masked document preview, produce a JSON object "
        "with EXACTLY these keys: compliance_observations (list of strings), "
        "security_risks (list of strings), remediation_steps (list of strings), "
        "narrative (a short paragraph). Respond with ONLY the JSON object — no "
        "markdown fences, no text outside the JSON. Never speculate about "
        "unmasked values."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Risk level: {risk_level}\n\n"
         "Detected data breakdown:\n{breakdown_text}\n\n"
         "Masked document preview (first 1500 chars):\n{preview}"),
    ])

    llm = ChatGroq(
        api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.3, max_tokens=800,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    chain = prompt | llm
    result = chain.invoke({
        "risk_level": risk_level,
        "breakdown_text": breakdown_text,
        "preview": masked_preview[:1500],
    })

    raw = result.content.strip()
    # Defensive cleanup in case the model wraps the JSON in ```json fences anyway.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw.lower().startswith("json") else raw
    return json.loads(raw)


def generate_summary(breakdown: Dict[str, int], risk_level: str, masked_preview: str) -> Dict[str, Any]:
    if GROQ_API_KEY:
        try:
            result = _llm_summary(breakdown, risk_level, masked_preview)
            # Basic shape validation; fall back if the LLM returned something odd.
            required = {"compliance_observations", "security_risks", "remediation_steps", "narrative"}
            if required.issubset(result.keys()):
                return result
        except Exception:
            pass  # fall through to rule-based summary

    return _rule_based_summary(breakdown, risk_level)
