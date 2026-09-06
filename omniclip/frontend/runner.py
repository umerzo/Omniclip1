"""Job launcher for the web UI.

Streamlit re-executes its whole script on every interaction, so a twenty-minute
render cannot run inside it — one refresh would kill the job. Instead the UI
starts a detached process and watches the files it leaves behind: `job.json` for
stage and scene state, `run.log` for output, `run.pid` for liveness.

That means the browser can be closed and reopened without disturbing anything.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..core.jobstore import Job
from ..utils.config import PROJECT_ROOT, resolve_path

PID_FILE = "run.pid"
LOG_FILE = "run.log"


def python_exe() -> str:
    """The interpreter running us — the venv one when launched from it."""
    return sys.executable or "python"


def build_command(url: str, out_dir: Path, options: dict) -> list[str]:
    """The exact CLI the UI would have typed, so both paths behave alike."""
    command = [python_exe(), "-u", "-m", "omniclip", url, "--out", str(out_dir)]
    if options.get("aspect"):
        command += ["--aspect", options["aspect"]]
    if options.get("mode") and options["mode"] != "auto":
        command += ["--mode", options["mode"]]
    if options.get("no_stock"):
        command.append("--no-stock")
    if options.get("no_ai"):
        command.append("--no-ai")
    if options.get("translate"):
        command.append("--translate")
    if options.get("trim"):
        command += ["--trim", str(options["trim"])]
    if options.get("max_scenes"):
        command += ["--max-scenes", str(options["max_scenes"])]
    if options.get("fresh"):
        command.append("--fresh")
    if options.get("repair"):
        command += ["--repair", ",".join(str(i) for i in options["repair"])]
    if options.get("workers"):
        command += ["--workers", str(options["workers"])]
    if options.get("start"):
        command += ["--start", str(options["start"])]
    if options.get("audio") and options["audio"] != "auto":
        command += ["--audio", options["audio"]]
    if options.get("no_captions"):
        command.append("--no-captions")
    if options.get("caption_style"):
        command += ["--caption-style", options["caption_style"]]
    if options.get("voice"):
        command += ["--voice", options["voice"]]
    if options.get("kind"):
        command += ["--kind", options["kind"]]
    if options.get("language"):
        command += ["--language", options["language"]]
    if options.get("caption_language"):
        command += ["--caption-language", options["caption_language"]]
    if options.get("no_music"):
        command.append("--no-music")
    if options.get("no_review"):
        command.append("--no-review")
    if options.get("scene_cap"):
        command += ["--scene-cap", str(options["scene_cap"])]
    if options.get("video_model"):
        command += ["--video-model", options["video_model"]]
    if options.get("pause_after"):
        command += ["--pause-after", options["pause_after"]]
    return command


def is_running(out_dir: str | Path) -> bool:
    """Whether this job's process is still alive.

    A stale pid file is common — the machine reboots, the process is killed —
    so the pid is checked rather than trusted, and cleaned up when dead.
    """
    pid_path = Path(out_dir) / PID_FILE
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        pid_path.unlink(missing_ok=True)
        return False

    alive = False
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True,
            )
            out_str = (result.stdout or "").lower()
            alive = "no tasks" not in out_str and any(part == str(pid) for part in (result.stdout or "").split())
        else:
            os.kill(pid, 0)
            alive = True
    except (OSError, subprocess.SubprocessError):
        alive = False

    if not alive:
        pid_path.unlink(missing_ok=True)
    return alive


def start(url: str, out_dir: str | Path, options: dict) -> int:
    """Launch a job detached from the UI process. Returns its pid."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_running(out_dir):
        raise RuntimeError("A job is already running in this folder")

    log = (out_dir / LOG_FILE).open("a", encoding="utf-8", errors="replace")
    log.write(f"\n=== started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log.flush()

    creationflags = 0
    if os.name == "nt":
        # Detach so closing the Streamlit process does not take the job with it.
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    process = subprocess.Popen(
        build_command(url, out_dir, options),
        cwd=str(PROJECT_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        start_new_session=(os.name != "nt"),
    )
    (out_dir / PID_FILE).write_text(str(process.pid), encoding="utf-8")
    return process.pid


def stop(out_dir: str | Path) -> bool:
    """Ask a running job to stop. Progress already saved is kept."""
    pid_path = Path(out_dir) / PID_FILE
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        return False
    pid_path.unlink(missing_ok=True)
    return True


def tail_log(out_dir: str | Path, lines: int = 40) -> str:
    path = Path(out_dir) / LOG_FILE
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def output_root() -> Path:
    from ..utils.config import load_settings

    return resolve_path(load_settings()["output"]["dir"])


def list_jobs(root: str | Path | None = None) -> list[dict]:
    """Every job on disk, newest first, with enough state to draw a card."""
    root = Path(root) if root else output_root()
    if not root.exists():
        return []

    jobs = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        job = Job.load(directory)
        if not job:
            continue
        state = job.summary()
        state["directory"] = str(directory)
        state["name"] = directory.name
        state["running"] = is_running(directory)
        jobs.append(state)
    return sorted(jobs, key=lambda j: j["updated"], reverse=True)
