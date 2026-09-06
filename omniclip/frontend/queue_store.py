"""A batch queue that lives on disk.

Streamlit re-executes its whole script on every click, so anything held in a
Python variable is gone by the next interaction. A queue that survives has to be
a file, and every reader has to tolerate another process writing it at the same
moment.

Entries are never deleted on completion, only marked. A finished batch is a
record of what ran, which is the thing a person wants to look at afterwards.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..utils.config import resolve_path

QUEUE_FILE = "queue.json"

# Where an entry can be. "review" is its own resting place rather than a kind of
# failure: the job produced something, it just wants a person to look before the
# expensive half runs.
WAITING = "waiting"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
REVIEW = "review"
CANCELLED = "cancelled"
# Ran out of generation capacity. Not a failure: the work so far is kept and
# the queue asks again by itself, so nobody has to notice or intervene.
HOLDING = "holding"

OPEN_STATES = (WAITING, RUNNING, HOLDING)
STATE_LABEL = {
    WAITING: "Waiting",
    RUNNING: "Building",
    DONE: "Done",
    FAILED: "Failed",
    REVIEW: "Needs review",
    CANCELLED: "Cancelled",
    HOLDING: "Waiting for capacity",
}

# What each state means for the person looking at it, and what happens next.
# Written here rather than in the page so every screen says the same thing.
STATE_EXPLAINS = {
    WAITING: "Queued. It starts when the build ahead of it finishes.",
    RUNNING: "Building now.",
    DONE: "Finished.",
    REVIEW: "Some frames need your eye before the expensive stage runs. "
            "The queue has moved on to the next build.",
    HOLDING: "The generation service is out of capacity right now. Everything "
             "built so far is saved and this retries by itself.",
    FAILED: "Stopped and will not retry on its own.",
    CANCELLED: "Stopped by you.",
}


def queue_path() -> Path:
    from ..utils.config import load_settings

    root = resolve_path(load_settings()["output"]["dir"])
    root.mkdir(parents=True, exist_ok=True)
    return root / QUEUE_FILE


def _read() -> dict:
    path = queue_path()
    if not path.exists():
        return {"version": 1, "entries": []}
    lock_path = path.with_suffix(".lock")
    try:
        from filelock import FileLock
        with FileLock(str(lock_path), timeout=5):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("entries"), list):
                return data
    except Exception:
        pass
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("entries"), list):
            return data
    except Exception:
        return {"version": 1, "entries": []}
    return {"version": 1, "entries": []}


def _write(data: dict) -> None:
    path = queue_path()
    lock_path = path.with_suffix(".lock")
    try:
        from filelock import FileLock
        with FileLock(str(lock_path), timeout=5):
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            temp.replace(path)
    except Exception:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)


def entries(state: str | None = None) -> list[dict]:
    items = _read()["entries"]
    if state:
        items = [e for e in items if e.get("state") == state]
    return items


def add(url: str, name: str, options: dict) -> dict:
    """Put one video at the back of the queue."""
    data = _read()
    entry = {
        "id": uuid.uuid4().hex[:12],
        "url": url.strip(),
        "name": name,
        "options": options,
        "state": WAITING,
        "added": time.time(),
        "started": None,
        "finished": None,
        "message": "",
    }
    data["entries"].append(entry)
    _write(data)
    return entry


def update(entry_id: str, **fields) -> dict | None:
    data = _read()
    for entry in data["entries"]:
        if entry["id"] == entry_id:
            entry.update(fields)
            _write(data)
            return entry
    return None


def remove(entry_id: str) -> bool:
    data = _read()
    before = len(data["entries"])
    data["entries"] = [e for e in data["entries"] if e["id"] != entry_id]
    if len(data["entries"]) == before:
        return False
    _write(data)
    return True


def move(entry_id: str, delta: int) -> bool:
    """Shuffle a waiting entry up or down the order."""
    data = _read()
    items = data["entries"]
    index = next((i for i, e in enumerate(items) if e["id"] == entry_id), None)
    if index is None:
        return False
    target = max(0, min(len(items) - 1, index + delta))
    if target == index:
        return False
    items.insert(target, items.pop(index))
    _write(data)
    return True


def clear_finished() -> int:
    data = _read()
    before = len(data["entries"])
    keep = set(OPEN_STATES) | {REVIEW}
    data["entries"] = [e for e in data["entries"] if e.get("state") in keep]
    _write(data)
    return before - len(data["entries"])


def next_due_hold() -> dict | None:
    """A held entry whose retry time has arrived."""
    now = time.time()
    for entry in _read()["entries"]:
        if entry.get("state") == HOLDING and (entry.get("retry_at") or 0) <= now:
            return entry
    return None


def soonest_retry() -> float | None:
    """When the next held entry is due, so a page can count down to it."""
    times = [e.get("retry_at") or 0 for e in _read()["entries"]
             if e.get("state") == HOLDING]
    return min(times) if times else None


def next_waiting() -> dict | None:
    """The next job to build, or None when the queue is empty.

    A job parked for review is stepped over rather than waited on, so one video
    needing a person does not stall the ten behind it.
    """
    for entry in _read()["entries"]:
        if entry.get("state") == WAITING:
            return entry
    return None


def counts() -> dict:
    tally = {state: 0 for state in STATE_LABEL}
    for entry in _read()["entries"]:
        state = entry.get("state")
        if state in tally:
            tally[state] += 1
    return tally
