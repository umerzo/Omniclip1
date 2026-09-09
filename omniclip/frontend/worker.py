"""The process that works through the queue, one video at a time.

Streamlit cannot own this loop. It re-runs its script on every interaction and
stops entirely when the browser closes, so a batch driven from the page would
die the moment nobody was watching. The supervisor is a separate process: the
page starts it if it is not already running, and from then on the two only
communicate through files.

One at a time is deliberate, not a simplification. Clip generation is limited to
roughly one submission a minute for the whole pool, so two jobs at once do not
finish sooner, they just take turns badly and trip each other's rate limits.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..core.jobstore import Job
from ..utils.config import PROJECT_ROOT, resolve_path
from . import queue_store as q
from . import runner

WORKER_PID = "worker.pid"
WORKER_LOG = "worker.log"
POLL_SECONDS = 5


def _root() -> Path:
    from ..utils.config import load_settings

    root = resolve_path(load_settings()["output"]["dir"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def is_running() -> bool:
    """Whether a supervisor is alive, cleaning up after one that is not."""
    path = _root() / WORKER_PID
    if not path.exists():
        return False
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        path.unlink(missing_ok=True)
        return False
    alive = False
    try:
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                    capture_output=True, text=True,
                                    creationflags=creationflags)
            out_str = (result.stdout or "").lower()
            alive = "no tasks" not in out_str and any(part == str(pid) for part in (result.stdout or "").split())
        else:
            os.kill(pid, 0)
            alive = True
    except (OSError, subprocess.SubprocessError):
        alive = False
    if not alive:
        path.unlink(missing_ok=True)
    return alive


def ensure_running() -> bool:
    """Start the supervisor if it is not already up. Safe to call every rerun."""
    if is_running():
        return False
    # Held work needs the supervisor alive too, or a capacity wait never ends.
    if not q.next_waiting() and not q.entries(q.HOLDING):
        return False

    root = _root()
    log = (root / WORKER_LOG).open("a", encoding="utf-8", errors="replace")
    log.write(f"\n=== supervisor started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log.flush()
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    if getattr(sys, "frozen", False):
        worker_cmd = [sys.executable, "--worker"]
    else:
        worker_cmd = [runner.python_exe(), "-u", "-m", "omniclip.frontend.worker"]
    process = subprocess.Popen(
        worker_cmd,
        cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
        creationflags=creationflags, start_new_session=(os.name != "nt"),
    )
    (root / WORKER_PID).write_text(str(process.pid), encoding="utf-8")
    return True


def stop() -> bool:
    path = _root() / WORKER_PID
    if not path.exists():
        return False
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        return False
    try:
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, creationflags=creationflags)
        else:
            os.kill(pid, 15)
    except (OSError, subprocess.SubprocessError):
        return False
    path.unlink(missing_ok=True)
    return True


# How long to wait before asking for capacity again. Grows so a long outage is
# not hammered, but the first retry is soon because most limits are short.
HOLD_BACKOFF = (10 * 60, 20 * 60, 40 * 60, 60 * 60)


def outcome_for(out_dir: Path, held_before: int = 0) -> tuple[str, str, float]:
    """Read the finished job and decide where its queue entry belongs.

    Returns the state, a sentence for a person, and when to try again.

    The decision comes from the job file's stop record, which the run writes
    before it exits. It used to be read out of the log tail instead, and that
    was wrong in a way that only showed up under load: the marker sits at the
    end of a line more than a thousand characters long, so a run whose last
    line had not finished being flushed was filed as failed when it had only
    run out of capacity. A field cannot be half-true in that way.
    """
    job = Job.load(out_dir)
    if job is None:
        return q.FAILED, "no job file was written", 0.0

    review = job.stages.get("review") or {}
    if review.get("needs_review"):
        problems = review.get("problems") or []
        return q.REVIEW, f"{len(problems)} frame(s) need your eye", 0.0

    if job.done("render") and job.video_path and Path(job.video_path).exists():
        return q.DONE, "", 0.0

    done_clips = sum(1 for s in job.scenes if s.has_asset)

    def waiting_for_capacity() -> tuple[str, str, float]:
        """Out of capacity is a wait, not a failure: the clips already made are
        on disk and asking again later is the whole remedy."""
        wait = HOLD_BACKOFF[min(held_before, len(HOLD_BACKOFF) - 1)]
        return (q.HOLDING,
                f"out of generation capacity with {done_clips}/{len(job.scenes)} "
                f"clips saved; retrying automatically",
                time.time() + wait)

    reason = (job.stop or {}).get("reason")
    if reason == "quota":
        return waiting_for_capacity()
    if reason == "paused":
        stage = (job.stop or {}).get("stage") or job.resume_from()
        return q.REVIEW, f"paused after '{stage}'; waiting for you", 0.0
    if reason == "error":
        detail = (job.stop or {}).get("detail") or f"stopped at '{job.resume_from()}'"
        return q.FAILED, detail[:160], 0.0

    # No stop record: either a job written before this field existed, or a run
    # killed hard enough that it never got to write one. The log is all there
    # is, so it stays as the fallback -- but only as the fallback.
    tail = runner.tail_log(out_dir, 30)
    if "QuotaExhausted" in tail or "out of capacity" in tail:
        return waiting_for_capacity()

    lines = [l.strip() for l in tail.strip().splitlines() if l.strip()]
    fallback = lines[-1] if lines else f"stopped at '{job.resume_from()}'"
    return q.FAILED, fallback[:160], 0.0


def run_one(entry: dict) -> None:
    """Build one queued video from start to finish."""
    out_dir = _root() / entry["name"]
    q.update(entry["id"], state=q.RUNNING, started=time.time(), message="")
    print(f"[{time.strftime('%H:%M:%S')}] building {entry['name']} <- {entry['url']}",
          flush=True)
    try:
        runner.start(entry["url"], out_dir, entry.get("options") or {})
    except Exception as error:
        q.update(entry["id"], state=q.FAILED, finished=time.time(),
                 message=f"could not start: {error}"[:160])
        return

    while runner.is_running(out_dir):
        time.sleep(POLL_SECONDS)
        # A cancel from the page takes effect here rather than mid-stage.
        current = next((e for e in q.entries() if e["id"] == entry["id"]), None)
        if current and current.get("state") == q.CANCELLED:
            runner.stop(out_dir)
            print(f"[{time.strftime('%H:%M:%S')}] cancelled {entry['name']}", flush=True)
            return

    held = int(entry.get("held") or 0)
    state, message, retry_at = outcome_for(out_dir, held)
    fields = {"state": state, "finished": time.time(), "message": message}
    if state == q.HOLDING:
        fields["retry_at"] = retry_at
        fields["held"] = held + 1
    q.update(entry["id"], **fields)
    print(f"[{time.strftime('%H:%M:%S')}] {entry['name']} -> {state} {message}",
          flush=True)


def supervise() -> int:
    """Work through the queue until nothing is waiting, then exit."""
    root = _root()
    (root / WORKER_PID).write_text(str(os.getpid()), encoding="utf-8")
    try:
        while True:
            entry = q.next_waiting()
            if entry is not None:
                run_one(entry)
                continue

            # Nothing queued, but something may be waiting on capacity. Put it
            # back in the queue the moment its retry time arrives.
            due = q.next_due_hold()
            if due is not None:
                print(f"[{time.strftime('%H:%M:%S')}] retrying {due['name']}",
                      flush=True)
                q.update(due["id"], state=q.WAITING, message="")
                continue

            if q.entries(q.HOLDING):
                # Stay alive so the retry happens without anyone reopening the
                # page; sleeping here is the whole point of the supervisor.
                time.sleep(POLL_SECONDS * 12)
                continue

            print("queue empty, supervisor exiting", flush=True)
            return 0
    finally:
        (root / WORKER_PID).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(supervise())
