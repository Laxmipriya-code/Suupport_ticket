"""Start the assessment API and UI together; Ctrl+C stops both services."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType


def port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return number


def stop(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=port, default=8000)
    parser.add_argument("--ui-port", type=port, default=8501)
    args = parser.parse_args()
    if args.api_port == args.ui_port:
        parser.error("API and UI need different ports.")

    root = Path(__file__).resolve().parent
    api_url = f"http://127.0.0.1:{args.api_port}"
    commands = [
        [
            "uvicorn",
            "support_ticket_ai.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.api_port),
            "--log-config",
            "src/support_ticket_ai/logging.json",
        ],
        [
            "streamlit",
            "run",
            "ui/app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(args.ui_port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
    ]
    processes: list[subprocess.Popen] = []
    signal.signal(signal.SIGTERM, stop)
    try:
        for command in commands:
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", *command],
                    cwd=root,
                    env={**os.environ, "API_BASE_URL": api_url},
                )
            )
        print(
            f"Starting UI: http://127.0.0.1:{args.ui_port}\nAPI docs: {api_url}/docs\n"
            "Press Ctrl+C to stop both services.",
            flush=True,
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.2)
        print("A service exited; stopping both. Check the service logs above.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
