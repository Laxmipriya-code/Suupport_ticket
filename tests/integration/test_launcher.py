"""Exercise the real two-service launcher without using model quota."""

import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack

import httpx
import pytest

from tests.conftest import ROOT


@pytest.mark.parametrize("occupied_api_port", [False, True])
def test_launcher_readiness_shutdown_and_startup_failure(tmp_path, occupied_api_port):
    with ExitStack() as stack:
        api_socket = stack.enter_context(socket.socket())
        ui_socket = stack.enter_context(socket.socket())
        api_socket.bind(("127.0.0.1", 0))
        ui_socket.bind(("127.0.0.1", 0))
        api_port = api_socket.getsockname()[1]
        ui_port = ui_socket.getsockname()[1]
        ui_socket.close()
        if occupied_api_port:
            api_socket.listen()
        else:
            api_socket.close()
        with (tmp_path / "services.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "--api-port",
                    str(api_port),
                    "--ui-port",
                    str(ui_port),
                ],
                cwd=tmp_path,
                env={
                    **os.environ,
                    "GROQ_API_KEY": "",
                    "DATA_PATH": str(ROOT / "data/support_tickets.csv"),
                    "API_BASE_URL": "http://127.0.0.1:1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                if occupied_api_port:
                    assert process.wait(timeout=30) == 1
                else:
                    with httpx.Client(timeout=1, trust_env=False) as client:
                        deadline = time.monotonic() + 30
                        while time.monotonic() < deadline:
                            assert process.poll() is None, log.read()
                            try:
                                health = client.get(f"http://127.0.0.1:{api_port}/health")
                                ui = client.get(f"http://127.0.0.1:{ui_port}/_stcore/health")
                                if health.is_success and ui.is_success:
                                    break
                            except httpx.HTTPError:
                                pass
                            time.sleep(0.1)
                        else:
                            pytest.fail("Both services must become ready within 30 seconds.")
                        assert health.json()["dataset_loaded"]
                        assert health.json()["status"] == "degraded"
                        stats = client.get(f"http://127.0.0.1:{api_port}/api/v1/stats")
                        assert stats.json()["total_tickets"] == 500
                    process.terminate()
                    assert process.wait(timeout=15) == 0
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)
            for port in [ui_port] if occupied_api_port else [api_port, ui_port]:
                with socket.socket() as probe:
                    assert probe.connect_ex(("127.0.0.1", port)) != 0
