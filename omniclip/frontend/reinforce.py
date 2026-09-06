"""Pipeline Reinforcement & Diagnostic Engine.

Probes the active queue, supervisor daemon, and underlying worker processes.
Automatically detects and heals common pipeline stall conditions:
- Zombie worker (marked running in queue, but OS process is dead)
- Hung network sockets / silent API freezes (> 4m without log activity)
- Stale file locks (.lock files orphaned by abrupt termination)
- Sleeping queue supervisor with jobs waiting
- Capacity hold / rate limit lockouts (nudge immediate retry)

Provides crisp 1-2 line status diagnostics explaining current state.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..core.jobstore import Job
from . import queue_store as q
from . import runner
from . import worker


def _clean_stale_locks() -> int:
    """Remove orphaned file locks left behind by abrupt shutdowns."""
    cleaned = 0
    try:
        q_path = q.queue_path()
        lock_path = q_path.with_suffix(".lock")
        if lock_path.exists():
            age = time.time() - lock_path.stat().st_mtime
            if age > 10.0:
                lock_path.unlink(missing_ok=True)
                cleaned += 1
    except Exception:
        pass
    return cleaned


def reinforce_pipeline(force_restart: bool = False) -> dict[str, Any]:
    """Inspect the pipeline state and heal/unstick if stalled.

    Returns:
        dict with keys:
            - 'status': 'active' | 'resumed' | 'unstuck' | 'started' | 'idle' | 'failed'
            - 'message': 1-2 line concise human-readable summary
            - 'tone': 'success' | 'warn' | 'info' | 'error'
    """
    _clean_stale_locks()

    running_entries = q.entries(q.RUNNING)
    waiting_entries = q.entries(q.WAITING)
    holding_entries = q.entries(q.HOLDING)
    failed_entries = q.entries(q.FAILED)

    # -------------------------------------------------------------------------
    # CASE 1: A job is marked RUNNING in the queue
    # -------------------------------------------------------------------------
    if running_entries:
        entry = running_entries[0]
        out_dir = runner.output_root() / entry["name"]
        job = Job.load(out_dir)
        pid_file = out_dir / runner.PID_FILE
        log_file = out_dir / runner.LOG_FILE

        is_active = runner.is_running(out_dir)

        # 1A. Stale / Dead Worker Process (Zombie state)
        if not is_active:
            pid_file.unlink(missing_ok=True)
            # Reset state to WAITING so queue supervisor picks it up immediately
            q.update(entry["id"], state=q.WAITING, message="Resumed via Reinforce Pipeline")
            worker.ensure_running()

            resume_stage = job.resume_from() if job else "start"
            return {
                "status": "resumed",
                "message": (
                    f"⚡ Pipeline Resumed: Worker had exited. Cleared stale state and resumed "
                    f"'{entry['name']}' from '{resume_stage}' (supervisor kickstarted)."
                ),
                "tone": "warn",
            }

        # 1B. Worker IS alive: inspect recent activity
        pid = 0
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
            except Exception:
                pass

        idle_seconds = 0.0
        last_line = ""
        if log_file.exists():
            try:
                idle_seconds = max(0.0, time.time() - log_file.stat().st_mtime)
                tail = runner.tail_log(out_dir, 15)
                lines = [line.strip() for line in tail.splitlines() if line.strip()]
                if lines:
                    last_line = lines[-1]
            except Exception:
                pass

        current_stage = job.resume_from() if job else "running"

        # Check for hung socket (> 300 seconds without a single log line)
        if idle_seconds > 300.0 or force_restart:
            runner.stop(out_dir)
            pid_file.unlink(missing_ok=True)
            q.update(
                entry["id"],
                state=q.WAITING,
                message=f"Auto-unhung after {int(idle_seconds // 60)}m silence at '{last_line or current_stage}'",
            )
            worker.ensure_running()
            return {
                "status": "unstuck",
                "message": (
                    f"🔄 Pipeline Unstuck: Process PID {pid} had no activity for {int(idle_seconds // 60)}m "
                    f"at '{last_line or current_stage}'. Terminated hung worker and resumed cleanly."
                ),
                "tone": "warn",
            }

        # Worker is alive and actively progressing
        stage_desc = last_line if last_line else f"stage: {current_stage}"
        if "key spent" in stage_desc:
            parts = stage_desc.split("(")
            stage_desc = parts[0].strip()
        if len(stage_desc) > 80:
            stage_desc = stage_desc[:77] + "..."

        return {
            "status": "active",
            "message": (
                f"🟢 Actively Progressing: Worker is alive (PID {pid}). "
                f"Currently at '{stage_desc}' (last updated {int(idle_seconds)}s ago). Normal build in progress."
            ),
            "tone": "success",
        }

    # -------------------------------------------------------------------------
    # CASE 2: Jobs are in HOLDING (waiting on rate-limit / capacity)
    # -------------------------------------------------------------------------
    if holding_entries:
        entry = holding_entries[0]
        q.update(entry["id"], state=q.WAITING, message="Capacity wait bypassed via Reinforce")
        worker.ensure_running()
        return {
            "status": "resumed",
            "message": (
                f"⚡ Capacity Hold Bypassed: Reset '{entry['name']}' to active queue. "
                f"Supervisor restarted with 25-key rotation pool."
            ),
            "tone": "info",
        }

    # -------------------------------------------------------------------------
    # CASE 3: Jobs are WAITING in the queue
    # -------------------------------------------------------------------------
    if waiting_entries:
        if not worker.is_running():
            worker.ensure_running()
            return {
                "status": "started",
                "message": (
                    f"▶️ Queue Supervisor Started: Woke up supervisor process to begin "
                    f"{len(waiting_entries)} waiting build(s)."
                ),
                "tone": "success",
            }
        return {
            "status": "active",
            "message": (
                f"⏳ Queue Supervisor Active: Processing {len(waiting_entries)} waiting build(s). "
                f"Next build starting shortly."
            ),
            "tone": "info",
        }

    # -------------------------------------------------------------------------
    # CASE 4: Failed job needing retry
    # -------------------------------------------------------------------------
    if failed_entries:
        failed_job = failed_entries[0]
        err_msg = failed_job.get("message") or "Pipeline halted"
        return {
            "status": "failed",
            "message": (
                f"⚠️ Build Halted: '{failed_job['name']}' stopped with error: {err_msg[:120]}. "
                f"Click 'Try again' on the row to rebuild."
            ),
            "tone": "error",
        }

    # -------------------------------------------------------------------------
    # CASE 5: Queue is completely idle
    # -------------------------------------------------------------------------
    return {
        "status": "idle",
        "message": "✅ Pipeline Ready: All queues are clear. Add a new YouTube URL in 'New build' to start.",
        "tone": "info",
    }
