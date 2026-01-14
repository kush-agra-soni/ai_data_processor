# ai/decision_schema.py

"""
This file defines the ONLY valid decision format that the LLM
(Gemini or any future model) is allowed to return.

If the response does not validate against this schema,
it MUST be rejected.
"""

from typing import Dict, List, Any
from jsonschema import Draft7Validator


# ==========================================================
# Allowed pipeline steps
# ==========================================================
ALLOWED_STEPS = [
    "detect",
    "clean",
    "standardize",
    "handle_outliers",
    "impute",
    "extract_dates",
    "enforce_types",
    "auto_convert_types",
    "encode",
    "select_features",
    "feature_importance",
    "scale",
    "resample",
    "validate",
]


# ==========================================================
# JSON Schema definition
# ==========================================================
DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["confidence", "steps"],
    "additionalProperties": False,

    "properties": {
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },

        "steps": {
            "type": "array",
            "minItems": 1,

            "items": {
                "type": "object",
                "required": ["name"],
                "additionalProperties": False,

                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ALLOWED_STEPS
                    },

                    "params": {
                        "type": "object",
                        "additionalProperties": True
                    },

                    "reason": {
                        "type": "string",
                        "maxLength": 500
                    },

                    "risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    }
                }
            }
        }
    }
}


# ==========================================================
# Validator utilities
# ==========================================================
_validator = Draft7Validator(DECISION_SCHEMA)


def validate_decision(decision: Dict[str, Any]) -> None:
    """
    Validate LLM decision against the strict schema.

    Raises:
        ValueError if invalid.
    """
    errors = sorted(_validator.iter_errors(decision), key=lambda e: e.path)

    if errors:
        messages = []
        for err in errors:
            loc = " → ".join(map(str, err.path))
            messages.append(f"{loc}: {err.message}")

        raise ValueError(
            "Invalid AI decision schema:\n" + "\n".join(messages)
        )


def is_high_risk(decision: Dict[str, Any]) -> bool:
    """
    Check if the decision contains any high-risk steps.
    """
    for step in decision.get("steps", []):
        if step.get("risk") == "high":
            return True
    return False
