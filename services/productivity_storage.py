
# services/productivity_storage.py

import os
import json
import time
from typing import List, Dict, Any

# Directory + file where productivity entries are stored
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

PRODUCTIVITY_FILE = os.path.join(DATA_DIR, "productivity.json")


def _load_all() -> List[Dict[str, Any]]:
    """Internal helper: load all saved productivity entries from JSON."""
    if not os.path.exists(PRODUCTIVITY_FILE):
        return []
    try:
        with open(PRODUCTIVITY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # If file is corrupted or unreadable, start fresh
        return []


def _save_all(entries: List[Dict[str, Any]]) -> None:
    """Internal helper: write the full list back to JSON."""
    with open(PRODUCTIVITY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def save_productivity(date_iso: str, productivity: float, notes: str | None = None) -> None:
    """
    Append a new productivity record.

    Parameters
    ----------
    date_iso : str
        Date in ISO format (e.g. "2025-11-29").
    productivity : float
        User's productivity rating (0–10).
    notes : str | None
        Optional free-text notes about that day.
    """
    entries = _load_all()
    entries.append(
        {
            "date": date_iso,
            "productivity": float(productivity),
            "notes": notes or "",
            "saved_at": time.time(),  # epoch seconds, similar to checkin_storage
        }
    )
    _save_all(entries)


def load_productivity() -> List[Dict[str, Any]]:
    """
    Return the full list of saved productivity entries.

    Each entry dict has:
      - "date": str (ISO date)
      - "productivity": float
      - "notes": str
      - "saved_at": float (timestamp)
    """
    return _load_all()
