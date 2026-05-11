import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rag_store  # noqa: E402
from command_context import retrieve_command_context  # noqa: E402
from guardrails import apply_command_guardrails  # noqa: E402
from infer import parse_semantic_json  # noqa: E402
from rag_store import TemplateRecord  # noqa: E402


def seed_store():
    rag_store._TEMPLATE_STORE.clear()
    rag_store._TEMPLATE_STORE.update(
        {
            ("show", "chassis", "led"): TemplateRecord(
                action="show",
                domain="chassis",
                sub_domain="led",
                template="show chassis led",
                mode="operational",
                requires_commit=False,
            ),
            ("clear", "ethernet-switching", "table"): TemplateRecord(
                action="clear",
                domain="ethernet-switching",
                sub_domain="table",
                template="clear ethernet-switching-table",
                mode="operational",
                requires_commit=False,
            ),
            ("set", "protocols", "ospf"): TemplateRecord(
                action="set",
                domain="protocols",
                sub_domain="ospf",
                template="set protocols ospf disable",
                mode="configuration",
                requires_commit=True,
            ),
        }
    )


def test_show_chassis_led_must_not_contain_commit():
    seed_store()
    parsed = {"action": "show", "domain": "chassis", "sub_domain": "led", "parameters": {}}
    context = retrieve_command_context(parsed)
    command, _ = apply_command_guardrails("show chassis led\ncommit", parsed, context)
    assert command == "show chassis led"


def test_clear_ethernet_switching_table_must_not_contain_commit():
    seed_store()
    parsed = {"action": "clear", "domain": "ethernet-switching", "sub_domain": "table", "parameters": {}}
    context = retrieve_command_context(parsed)
    command, _ = apply_command_guardrails("clear ethernet-switching-table\ncommit", parsed, context)
    assert command == "clear ethernet-switching-table"


def test_set_protocols_ospf_disable_gets_exactly_one_commit():
    seed_store()
    parsed = {"action": "set", "domain": "protocols", "sub_domain": "ospf", "parameters": {}}
    context = retrieve_command_context(parsed)
    command, _ = apply_command_guardrails("set protocols ospf disable", parsed, context)
    assert command == "set protocols ospf disable\\ncommit"
    assert command.count("commit") == 1


def test_duplicate_commit_is_collapsed():
    seed_store()
    parsed = {"action": "set", "domain": "protocols", "sub_domain": "ospf", "parameters": {}}
    context = retrieve_command_context(parsed)
    command, _ = apply_command_guardrails("set protocols ospf disable\ncommit\ncommit", parsed, context)
    assert command == "set protocols ospf disable\\ncommit"


def test_missing_template_returns_template_not_found():
    seed_store()
    parsed = {"action": "show", "domain": "protocols", "sub_domain": "bgp", "parameters": {}}
    context = retrieve_command_context(parsed)
    assert context["found"] is False
    assert context["reason"] == "template_not_found"


def test_parse_semantic_json_rejects_missing_required_keys():
    parsed, error = parse_semantic_json('{"action":"show","parameters":{}}')
    assert parsed is None
    assert "missing_keys" in error


def test_numeric_vlan_id_converts_to_int():
    parsed, error = parse_semantic_json(
        '{"action":"show","domain":"interfaces","sub_domain":"vlan","parameters":{"vlan_id":"100"}}'
    )
    assert error is None
    assert parsed["parameters"]["vlan_id"] == 100


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("semantic RAG smoke tests passed")
