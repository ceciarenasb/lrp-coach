"""Simple JSON persistence for the athlete's profile, zones, plan, and history."""

import json
from pathlib import Path

_PATH = Path(__file__).parent.parent / "data" / "state.json"


def load() -> dict:
    if _PATH.exists():
        with open(_PATH) as f:
            return json.load(f)
    return {}


def save(data: dict) -> None:
    _PATH.parent.mkdir(exist_ok=True)
    if _PATH.exists():
        _PATH.replace(_PATH.with_suffix(".backup.json"))
    with open(_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def update(key: str, value) -> None:
    s = load()
    s[key] = value
    save(s)
