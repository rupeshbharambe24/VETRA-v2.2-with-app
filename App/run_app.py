#!/usr/bin/env python
"""
Run both the FastAPI backend and the Vite+React frontend with simple logs.

Usage:
    python run_app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import List, Optional


def start_process(cmd: List[str], cwd: Optional[str], name: str) -> subprocess.Popen:
    print(f"[runner] Starting {name} with command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd)
    print(f"[runner] {name} started with PID {proc.pid}")
    return proc


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_cwd = os.path.join(root_dir, "backend")
    frontend_cwd = os.path.join(root_dir, "frontend")

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ]

    frontend_cmd = ["npm", "run", "dev"]

    backend_proc = None
    frontend_proc = None

    try:
        backend_proc = start_process(backend_cmd, backend_cwd, "backend")
        frontend_proc = start_process(frontend_cmd, frontend_cwd, "frontend")

        print("[runner] Both backend and frontend started. Press Ctrl+C to stop.\n")

        while True:
            time.sleep(1)
            backend_code = backend_proc.poll()
            frontend_code = frontend_proc.poll()

            if backend_code is not None:
                print(f"[runner] Backend exited with code {backend_code}. Stopping frontend...")
                if frontend_proc and frontend_proc.poll() is None:
                    frontend_proc.terminate()
                break

            if frontend_code is not None:
                print(f"[runner] Frontend exited with code {frontend_code}. Stopping backend...")
                if backend_proc and backend_proc.poll() is None:
                    backend_proc.terminate()
                break

    except KeyboardInterrupt:
        print("\n[runner] Caught Ctrl+C, shutting down both services...")
        if backend_proc and backend_proc.poll() is None:
            print("[runner] Terminating backend...")
            backend_proc.terminate()
        if frontend_proc and frontend_proc.poll() is None:
            print("[runner] Terminating frontend...")
            frontend_proc.terminate()
    finally:
        exit_code = 0

        if backend_proc:
            backend_proc.wait()
            if backend_proc.returncode not in (0, None):
                print(f"[runner] Backend final exit code: {backend_proc.returncode}")
                exit_code = backend_proc.returncode or exit_code

        if frontend_proc:
            frontend_proc.wait()
            if frontend_proc.returncode not in (0, None):
                print(f"[runner] Frontend final exit code: {frontend_proc.returncode}")
                exit_code = frontend_proc.returncode or exit_code

        print("[runner] All services stopped. Bye!")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

