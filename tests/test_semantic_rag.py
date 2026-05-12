import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rag_store  # noqa: E402
from command_context import retrieve_command_context  # noqa: E402
from guardrails import apply_command_guardrails  # noqa: E402
from infer import assemble_command_from_context, parse_semantic_json  # noqa: E402
from rag_store import TemplateRecord  # noqa: E402
from evaluate_semantic_rag import evaluate_rows, expected_frame, failure_stage  # noqa: E402


def seed_store():
    rag_store._TEMPLATE_STORE.clear()
    rag_store._TEMPLATE_STORE.update(
        {
            ("show", "chassis", "led", "show_chassis_led", "plain"): TemplateRecord(
                action="show",
                domain="chassis",
                sub_domain="led",
                template="show chassis led",
                operation="show_chassis_led",
                mode="operational",
                requires_commit=False,
            ),
            ("clear", "ethernet-switching", "table", "clear_table", "plain"): TemplateRecord(
                action="clear",
                domain="ethernet-switching",
                sub_domain="table",
                template="clear ethernet-switching-table",
                operation="clear_table",
                mode="operational",
                requires_commit=False,
            ),
            ("set", "protocols", "ospf", "disable", "plain"): TemplateRecord(
                action="set",
                domain="protocols",
                sub_domain="ospf",
                template="set protocols ospf disable",
                operation="disable",
                mode="configuration",
                requires_commit=True,
            ),
            ("set", "protocols", "sflow", "traceoptions_flag_enable", "plain"): TemplateRecord(
                action="set",
                domain="protocols",
                sub_domain="sflow",
                operation="traceoptions_flag_enable",
                template="set protocols sflow traceoptions flag {flag}",
                mode="configuration",
                requires_commit=True,
                required_params=["flag"],
                positive_cues=["traceoptions", "flag"],
                negative_cues=["disable"],
            ),
            ("set", "protocols", "sflow", "traceoptions_flag_disable", "plain"): TemplateRecord(
                action="set",
                domain="protocols",
                sub_domain="sflow",
                operation="traceoptions_flag_disable",
                template="set protocols sflow traceoptions flag {flag} disable",
                mode="configuration",
                requires_commit=True,
                required_params=["flag"],
                positive_cues=["disable", "traceoptions", "flag"],
            ),
            ("set", "protocols", "sflow", "interface_enable", "plain"): TemplateRecord(
                action="set",
                domain="protocols",
                sub_domain="sflow",
                operation="interface_enable",
                template="set protocols sflow interface {interface}",
                mode="configuration",
                requires_commit=True,
                required_params=["interface"],
                positive_cues=["interface"],
                negative_cues=["flag"],
            ),
            ("set", "protocols", "sflow", "sample_rate_egress", "plain"): TemplateRecord(
                action="set",
                domain="protocols",
                sub_domain="sflow",
                operation="sample_rate_egress",
                template="set protocols sflow sample-rate egress <rate>",
                mode="configuration",
                requires_commit=True,
                required_params=["rate"],
            ),
            ("set", "ethernet-switching-options", "secure-access-port", "mac_move_limit", "plain"): TemplateRecord(
                action="set",
                domain="ethernet-switching-options",
                sub_domain="secure-access-port",
                operation="mac_move_limit",
                template="set ethernet-switching-options secure-access-port vlan <vlan-name-or-id> mac-move-limit <limit>",
                mode="configuration",
                requires_commit=True,
                required_params=["vlan_id_or_name", "limit"],
            ),
            ("set", "ethernet-switching-options", "secure-access-port", "mac_limit_action_log", "plain"): TemplateRecord(
                action="set",
                domain="ethernet-switching-options",
                sub_domain="secure-access-port",
                operation="mac_limit_action_log",
                template="set ethernet-switching-options secure-access-port interface <interface-name> mac-limit <limit> action log",
                mode="configuration",
                requires_commit=True,
                required_params=["interface", "limit"],
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


def test_full_cli_action_is_repaired_and_trailing_commit_ignored():
    parsed, error = parse_semantic_json(
        '{"action":"show chassis led", "domain":"chassis", "sub_domain":"led", "parameters":{}}\ncommit'
    )
    assert error is None
    assert parsed["action"] == "show"
    assert parsed["domain"] == "chassis"
    assert parsed["sub_domain"] == "led"
    assert "repaired_full_command_action" in parsed["_parse_warnings"]
    assert "stripped_trailing_text_after_json" in parsed["_parse_warnings"]


def test_full_set_command_repairs_frame_and_extracts_parameters():
    parsed, error = parse_semantic_json(
        '{"action":"set ethernet-switching-options secure-access-port vlan HR mac-move-limit 2", '
        '"domain":"set ethernet-switching-options secure-access-port vlan HR mac-move-limit 2", '
        '"sub_domain":"set ethernet-switching-options secure-access-port vlan HR mac-move-limit 2", '
        '"parameters":{}}\ncommit'
    )
    assert error is None
    assert parsed["action"] == "set"
    assert parsed["domain"] == "ethernet-switching-options"
    assert parsed["sub_domain"] == "secure-access-port"
    assert parsed["parameters"]["vlan_name"] == "HR"
    assert parsed["parameters"]["limit"] == 2
    assert "repaired_full_command_action" in parsed["_parse_warnings"]
    assert "inferred_domain_sub_domain" in parsed["_parse_warnings"]


def test_sflow_traceoptions_bad_domain_repairs_to_protocol_key():
    parsed, error = parse_semantic_json(
        '{"action":"set protocols sflow traceoptions flag all", "domain":"set", "sub_domain":"set protocols sflow traceoptions flag all", "parameters":{}}'
    )
    assert error is None
    assert f"{parsed['action']}/{parsed['domain']}/{parsed['sub_domain']}" == "set/protocols/sflow"
    assert parsed["operation"] == "traceoptions_flag_enable"
    assert parsed["parameters"]["flag"] == "all"


def test_virtual_chassis_traceoptions_bad_domain_repairs_key():
    parsed, error = parse_semantic_json(
        '{"action":"set virtual-chassis traceoptions flag csn", "domain":"set", "sub_domain":"set virtual-chassis traceoptions flag csn", "parameters":{}}'
    )
    assert error is None
    assert f"{parsed['action']}/{parsed['domain']}/{parsed['sub_domain']}" == "set/virtual-chassis/traceoptions"
    assert parsed["operation"] == "traceoptions_flag_enable"
    assert parsed["parameters"]["flag"] == "csn"


def test_show_configuration_protocols_sflow_repairs_key_and_protocol_param():
    parsed, error = parse_semantic_json(
        '{"action":"show configuration protocols sflow", "domain":"show configuration protocols sflow", "sub_domain":"show configuration protocols sflow", "parameters":{}}'
    )
    assert error is None
    assert f"{parsed['action']}/{parsed['domain']}/{parsed['sub_domain']}" == "show/configuration/protocols"
    assert parsed["parameters"]["protocol"] == "sflow"


def test_clear_ethernet_switching_table_repairs_key():
    parsed, error = parse_semantic_json(
        '{"action":"clear ethernet-switching-table", "domain":"clear ethernet-switching-table", "sub_domain":"clear ethernet-switching-table", "parameters":{}}'
    )
    assert error is None
    assert f"{parsed['action']}/{parsed['domain']}/{parsed['sub_domain']}" == "clear/ethernet-switching/table"
    assert parsed["operation"] == "clear_table"


def test_operation_selects_sflow_flag_enable_not_disable():
    seed_store()
    parsed = {
        "action": "set",
        "domain": "protocols",
        "sub_domain": "sflow",
        "operation": "traceoptions_flag_enable",
        "parameters": {"flag": "all"},
        "_intent_context": "trace all events for sflow",
    }
    context = retrieve_command_context(parsed)
    assert context["operation"] == "traceoptions_flag_enable"
    assert context["template"] == "set protocols sflow traceoptions flag {flag}"


def test_operation_selects_sflow_interface_over_flag():
    seed_store()
    parsed = {
        "action": "set",
        "domain": "protocols",
        "sub_domain": "sflow",
        "operation": "interface_enable",
        "parameters": {"interface": "ge-0/0/1"},
        "_intent_context": "enable sflow on interface ge-0/0/1",
    }
    context = retrieve_command_context(parsed)
    assert context["operation"] == "interface_enable"
    assert context["template"] == "set protocols sflow interface {interface}"


def test_angle_rate_placeholder_binds_from_intent():
    seed_store()
    parsed = {
        "action": "set",
        "domain": "protocols",
        "sub_domain": "sflow",
        "operation": "sample_rate_egress",
        "parameters": {},
        "_intent_context": "set the sflow egress sampling rate to 1000",
    }
    context = retrieve_command_context(parsed)
    command, error, commit_added, _ = assemble_command_from_context(parsed, context)
    assert error is None
    assert commit_added is True
    assert command == "set protocols sflow sample-rate egress 1000\\ncommit"


def test_vlan_name_or_id_and_limit_placeholders_bind_from_intent():
    seed_store()
    parsed = {
        "action": "set",
        "domain": "ethernet-switching-options",
        "sub_domain": "secure-access-port",
        "operation": "mac_move_limit",
        "parameters": {},
        "_intent_context": "set a mac moving limit of 2 on vlan HR",
    }
    context = retrieve_command_context(parsed)
    command, error, _, _ = assemble_command_from_context(parsed, context)
    assert error is None
    assert command == "set ethernet-switching-options secure-access-port vlan HR mac-move-limit 2\\ncommit"


def test_interface_and_limit_placeholders_bind_from_intent():
    seed_store()
    parsed = {
        "action": "set",
        "domain": "ethernet-switching-options",
        "sub_domain": "secure-access-port",
        "operation": "mac_limit_action_log",
        "parameters": {},
        "_intent_context": "put a mac limit of 1 on interface ge-0/0/15 and log the violation",
    }
    context = retrieve_command_context(parsed)
    command, error, _, _ = assemble_command_from_context(parsed, context)
    assert error is None
    assert command == "set ethernet-switching-options secure-access-port interface ge-0/0/15 mac-limit 1 action log\\ncommit"


def test_semantic_evaluator_normalizes_literal_newline_exact_match():
    row = {
        "intent": "disable ospf",
        "target_command": "set protocols ospf disable\ncommit",
        "prediction": "set protocols ospf disable\\ncommit",
        "semantic_json": {"action": "set", "domain": "protocols", "sub_domain": "ospf", "operation": "disable", "parameters": {}},
        "semantic_parse_error": None,
        "command_context": {
            "found": True,
            "reason": "",
            "template_key": "set/protocols/ospf",
            "template_variant_key": "set/protocols/ospf/disable/plain",
            "mode": "configuration",
            "requires_commit": True,
        },
        "template_key": "set/protocols/ospf",
        "assembly_error": None,
        "guardrails_applied": [],
        "context_warnings": [],
    }
    assert failure_stage(row, expected_frame(row)) == "ok"
    metrics, failures = evaluate_rows([row])
    assert metrics["raw_exact_match"] == 0.0
    assert metrics["normalized_exact_match"] == 1.0
    assert metrics["exact_match"] == 1.0
    assert failures == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("semantic RAG smoke tests passed")
