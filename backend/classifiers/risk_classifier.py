"""
Turns raw detection counts into a Low / Medium / High risk classification.

Scoring model (simple, explainable — important for a compliance tool
where "why did you flag this as High risk" needs a clear answer):

    score = sum( count(type) * weight(type) )

Additional rule: presence of ANY high-severity type (Aadhaar, PAN, card,
bank account, API key, password) automatically floors the result at
"Medium", even if the numeric score alone would land on "Low" — a single
leaked credit card number is never a "Low risk" finding.
"""

from typing import Dict, List, Any
from backend.config import RISK_WEIGHTS, HIGH_RISK_TYPES, RISK_THRESHOLD_MEDIUM, RISK_THRESHOLD_HIGH


def classify(detections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    breakdown = {cat: len(matches) for cat, matches in detections.items() if matches}
    score = sum(count * RISK_WEIGHTS.get(cat, 1) for cat, count in breakdown.items())

    has_high_severity_type = any(cat in HIGH_RISK_TYPES for cat in breakdown)

    if score >= RISK_THRESHOLD_HIGH:
        level = "High"
    elif score >= RISK_THRESHOLD_MEDIUM or has_high_severity_type:
        level = "Medium" if score < RISK_THRESHOLD_HIGH else "High"
    else:
        level = "Low"

    # Floor rule: any high-severity type present -> at least Medium.
    if has_high_severity_type and level == "Low":
        level = "Medium"

    return {
        "risk_level": level,
        "risk_score": score,
        "breakdown": breakdown,
    }