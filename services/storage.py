# services/storage.py
import json, time, os
from pathlib import Path

DATA = Path("data"); DATA.mkdir(exist_ok=True)
FILE = DATA / "plans.json"

def save_plan(plan_dict):
    data = []
    if FILE.exists():
        data = json.loads(FILE.read_text())
    plan_dict["saved_at"] = int(time.time())
    data.append(plan_dict)
    FILE.write_text(json.dumps(data, indent=2))

def load_plans():
    if not FILE.exists():
        return []
    return json.loads(FILE.read_text())
