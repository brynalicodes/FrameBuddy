# src/safety.py

import re


DISALLOWED_PATTERNS = [
    r"\bworth\b",
    r"\bvalue\b",
    r"\bvaluation\b",
    r"\bappraisal\b",
    r"\bmarket price\b",
    r"\bprice\b",
    r"\binvestment\b",
    r"\bprovenance\b",
    r"\bauthentic\b",
    r"\bauthenticate\b",
    r"\bfake\b",
    r"\bforgery\b",
]

REPLACEMENT_MESSAGE = (
    "I cannot provide valuation, appraisal, investment advice, or authentication "
    "claims. However, I can describe the artwork based on visible features and "
    "retrieved metadata."
)


def remove_disallowed_claims(text: str) -> str:
    """
    Remove or replace valuation/authentication/provenance claims.
    """
    cleaned = text

    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = REPLACEMENT_MESSAGE + "\n\n" + cleaned
            break

    return cleaned


def enforce_grounding(text: str) -> str:
    """
    If the model hallucinates unsupported facts, encourage fallback.
    This is a simple heuristic: if the model uses strong certainty language
    without metadata references, we soften it.
    """

    hallucination_markers = [
        "definitely",
        "certainly",
        "without a doubt",
        "this is unquestionably",
        "this artwork is by",
    ]

    if any(marker in text.lower() for marker in hallucination_markers):
        text = (
            "Some statements may not be fully supported by the retrieved metadata. "
            "I will provide only grounded information.\n\n" + text
        )

    return text


def enforce_insufficient_info_rule(text: str, metadata_count: int) -> str:
    """
    If retrieval returned no metadata, force the fallback message.
    """
    if metadata_count == 0:
        return (
            "I don’t have enough information to provide a confident answer. "
            "No matching metadata was retrieved."
        )
    return text


def apply_safety_filter(model_output: str, metadata_count: int = 1) -> str:
    """
    Apply all safety layers in order.
    """

    text = model_output

    text = remove_disallowed_claims(text)
    text = enforce_grounding(text)
    text = enforce_insufficient_info_rule(text, metadata_count)

    return text.strip()
