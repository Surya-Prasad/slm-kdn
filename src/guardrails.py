import re
from typing import List, Tuple


OPERATIONAL_ACTIONS = {"show", "ping", "traceroute", "monitor", "clear", "request"}


def _strip_noise(command: str) -> str:
    text = str(command or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(Command:|Output:)\s*", "", text, flags=re.I)
    return text.strip()


def _split_commit(command: str) -> tuple[str, int]:
    normalized = command.replace("\\n", "\n").strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    commit_count = 0
    while lines and lines[-1].lower() == "commit":
        commit_count += 1
        lines.pop()
    body = " ".join(lines).strip()
    return body, commit_count


def apply_command_guardrails(command: str, parsed_json: dict, context: dict) -> Tuple[str, List[str]]:
    applied: List[str] = []
    cleaned = _strip_noise(command)
    if cleaned != command:
        applied.append("stripped_model_noise")

    body, commit_count = _split_commit(cleaned)
    action = body.split(maxsplit=1)[0].lower() if body else str(parsed_json.get("action", "")).lower()
    mode = str(context.get("mode", "unknown")).lower()
    requires_commit = bool(context.get("requires_commit", False))
    should_have_commit = action in {"set", "delete", "load"} and mode == "configuration" and requires_commit

    if action in OPERATIONAL_ACTIONS or mode == "operational":
        if commit_count:
            applied.append("commit_removed_operational")
        commit_count = 0
    elif should_have_commit:
        if commit_count != 1:
            applied.append("commit_normalized")
        commit_count = 1
    else:
        if commit_count > 1:
            applied.append("duplicate_commit_removed")
        if not should_have_commit:
            commit_count = 0

    final = body
    if commit_count:
        final = f"{body}\\ncommit"
    return final.strip(), applied
