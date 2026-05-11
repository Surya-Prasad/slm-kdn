import json
import re
from pathlib import Path

def migrate_target_json(target_command: str, old_json_str: str) -> str:
    """Derives domain and sub_domain directly from the target command."""
    old_json = json.loads(old_json_str)
    
    # Clean the command and split it into parts
    clean_command = re.sub(r"\s+", " ", target_command.strip().split("\\n")[0])
    parts = clean_command.split()
    
    action = parts[0].lower() if len(parts) > 0 else "unknown"
    domain = parts[1].lower().replace("-", "_") if len(parts) > 1 else "unknown"
    sub_domain = parts[2].lower().replace("-", "_") if len(parts) > 2 else "general"

    # Filter out empty or null parameters from the old format
    clean_params = {
        k: v for k, v in old_json.get("parameters", {}).items() 
        if v not in [None, "", 0]
    }

    new_json = {
        "action": action,
        "domain": domain,
        "sub_domain": sub_domain,
        "parameters": clean_params
    }
    
    return json.dumps(new_json)

def migrate_file(filepath: str):
    path = Path(filepath)
    if not path.exists():
        return
        
    lines = path.read_text().strip().split("\n")
    migrated_lines = []
    
    for line in lines:
        if not line: continue
        record = json.loads(line)
        
        # Rewrite the target_json field using the actual command as the source of truth
        record["target_json"] = migrate_target_json(
            record.get("target_command", ""), 
            record.get("target_json", "{}")
        )
        migrated_lines.append(json.dumps(record))
        
    path.write_text("\n".join(migrated_lines) + "\n")
    print(f"Migrated {len(migrated_lines)} records in {filepath}")

if __name__ == "__main__":
    # Migrate all dataset splits
    for split in ["train", "val", "test", "clean_test"]:
        migrate_file(f"data/{split}.jsonl")