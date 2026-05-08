import re

def extract_entities(text):
    pats=[r"ge-\d+/\d+/\d+",r"\b(?:\d{1,3}\.){3}\d{1,3}\b",r"\bvlan\s*\d+\b",r"\b(?:ospf|bgp|dhcp|stp)\b"]
    out=set(); low=text.lower()
    for p in pats: out.update(re.findall(p,low))
    return out

def validate(prediction, intent=""):
    errors=[]
    if not prediction.strip(): errors.append("empty_output")
    if any(x in prediction.lower() for x in ["because","here is","explanation"]): errors.append("contains_explanation")
    if prediction.count('"')%2!=0 or prediction.count("(")!=prediction.count(")"): errors.append("unbalanced_symbols")
    if not re.search(r"\b(set|delete|add|remove|show|interface|ip|vlan)\b", prediction.lower()): errors.append("not_command_like")
    missing=[e for e in extract_entities(intent) if e not in prediction.lower()]
    if missing: errors.append("missing_entities:"+",".join(missing))
    return {"is_valid": len(errors)==0, "errors": errors}
