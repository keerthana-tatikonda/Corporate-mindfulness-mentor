import json, time
from pathlib import Path

DATA = Path("data"); DATA.mkdir(exist_ok=True)
FILE = DATA / "checkins.json"

def save_checkin(checkin_dict, day_adjustment_dict):
    payload = {
        "checkin": checkin_dict,
        "day_adjustment": day_adjustment_dict,
        "saved_at": int(time.time()),
    }
    items = []
    if FILE.exists():
        items = json.loads(FILE.read_text())
    items.append(payload)
    FILE.write_text(json.dumps(items, indent=2))
