import json
import re


REQUIRED_JSON_KEYS = {"action", "target", "target_type", "parameters"}


def extract_entities(text):
    pats = [
        r"ge-\d+/\d+/\d+",
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        r"\bvlan\s*\d+\b",
        r"\b(?:ospf|bgp|dhcp|stp)\b",
    ]
    out = set()
    low = text.lower()
    for p in pats:
        out.update(re.findall(p, low))
    return out


def validate_json_structure(prediction_json):
    errors = []
    parsed = prediction_json

    if isinstance(prediction_json, str):
        try:
            parsed = json.loads(prediction_json)
        except Exception:
            return False, ["invalid_json"], None

    if not isinstance(parsed, dict):
        return False, ["json_not_object"], None

    missing = REQUIRED_JSON_KEYS - set(parsed.keys())
    if missing:
        errors.append("missing_keys:" + ",".join(sorted(missing)))
    if not isinstance(parsed.get("parameters"), dict):
        errors.append("parameters_not_object")

    return len(errors) == 0, errors, parsed


def validate_prediction(prediction, ground_truth, intent=""):
    errors = []
    if not prediction.strip():
        errors.append("empty_output")
    if prediction.strip() != ground_truth.strip():
        errors.append("final_mismatch")
    if any(x in prediction.lower() for x in ["because", "here is", "explanation"]):
        errors.append("contains_explanation")

    missing = [e for e in extract_entities(intent) if e not in prediction.lower()]
    if missing:
        errors.append("missing_entities:" + ",".join(missing))

    return len(errors) == 0, errors


def validate(record):
    """Validate both intermediate JSON and final assembled CLI prediction."""
    errors = []

    json_ok, json_errors, _ = validate_json_structure(record.get("prediction_json") or record.get("prediction_json_raw", ""))
    if not json_ok:
        errors.extend(json_errors)

    final_ok, final_errors = validate_prediction(
        record.get("prediction", ""),
        record.get("target_command", ""),
        record.get("intent", ""),
    )
    if not final_ok:
        errors.extend(final_errors)

    return {"is_valid": len(errors) == 0, "errors": errors}
