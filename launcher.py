"""Desktop Application Launcher for OmniClip AI Content Studio.

Supports:
- GUI Mode: Launches silent background Streamlit server + native desktop window
- Server Mode: `OmniClip.exe --server <port>`
- Worker Mode: `OmniClip.exe --worker`
- CLI Mode: `OmniClip.exe --cli [args]`
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.kernel32.AttachConsole(-1)
    except Exception:
        pass

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")

IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    APP_DIR = Path(sys._MEIPASS)
    EXE_DIR = Path(sys.executable).parent
else:
    APP_DIR = Path(__file__).resolve().parent
    EXE_DIR = APP_DIR

# Seed default environment variables from APPDATA or project .env
def setup_environment() -> None:
    appdata = Path(os.environ.get("APPDATA", "~")).expanduser() / "OmniClip"
    appdata.mkdir(parents=True, exist_ok=True)
    user_env = appdata / ".env"
    beside_env = EXE_DIR / ".env"
    bundled_env = APP_DIR / ".env"

    if not user_env.exists():
        if beside_env.exists():
            try:
                shutil.copy2(beside_env, user_env)
            except Exception:
                pass
        elif bundled_env.exists():
            try:
                shutil.copy2(bundled_env, user_env)
            except Exception:
                pass

    for cand in [bundled_env, user_env, beside_env]:
        if cand.exists():
            for line in cand.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def find_free_port(start: int = 8501) -> int:
    for p in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def run_server(port: int) -> None:
    """Internal server entry point called as child process."""
    setup_environment()
    app_script = APP_DIR / "omniclip" / "frontend" / "app.py"
    if not app_script.exists():
        app_script = APP_DIR / "app.py"

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_script),
        "--server.headless=true",
        f"--server.port={port}",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())


def open_desktop_window(url: str) -> None:
    """Open a standalone native desktop window pointing to the local studio."""
    try:
        import webview
        webview.create_window(
            title="OmniClip AI Content Studio",
            url=url,
            width=1440,
            height=920,
            min_size=(1024, 700),
            background_color="#0E1114",
        )
        webview.start()
        return
    except Exception as exc:
        print(f"pywebview window fallback: {exc}", flush=True)

    # Fallback to Edge App Mode (guaranteed on Windows 10 & 11)
    try:
        edge_paths = [
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        ]
        edge_exe = next((str(p) for p in edge_paths if p.exists()), "msedge")
        user_data = Path(os.environ.get("LOCALAPPDATA", "~")) / "OmniClip" / "webview_profile"
        user_data.mkdir(parents=True, exist_ok=True)

        edge_proc = subprocess.Popen([
            edge_exe,
            f"--app={url}",
            f"--user-data-dir={user_data}",
            "--window-size=1440,920",
        ])
        edge_proc.wait()
        return
    except Exception:
        pass

    # Ultimate fallback: default web browser
    import webbrowser
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main_gui() -> None:
    port = find_free_port(8501)

    # Spawn the headless Streamlit server
    if IS_FROZEN:
        server_cmd = [sys.executable, "--server", str(port)]
    else:
        server_cmd = [sys.executable, str(Path(__file__).resolve()), "--server", str(port)]

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    server_proc = subprocess.Popen(server_cmd, creationflags=creationflags)

    url = f"http://localhost:{port}"
    print(f"Starting OmniClip Studio at {url}...", flush=True)

    # Wait for server to become healthy (up to 30s)
    ready = False
    for _ in range(60):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass

    if not ready:
        print("Failed to contact local Streamlit server.", file=sys.stderr)
        server_proc.terminate()
        sys.exit(1)

    try:
        open_desktop_window(url)
    finally:
        # Clean shutdown of server
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(server_proc.pid), "/T", "/F"],
                               capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                server_proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    setup_environment()
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--worker":
        from omniclip.frontend.worker import supervise
        sys.exit(supervise())
    elif len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from omniclip.cli import main as cli_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        sys.exit(cli_main())
    else:
        main_gui()
