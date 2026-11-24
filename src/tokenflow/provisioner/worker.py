"""Small entry point shipped into the inference image. Only the device probe imports torch."""

import argparse
import hmac
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


class GatewayAuth:
    """ASGI middleware protecting every Slurm vLLM HTTP route, including diagnostics."""

    def __init__(self, app):
        self.app = app
        self.header = ("Bearer " + os.environ["VLLM_API_KEY"]).encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            value = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(self.header, value):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await send({"type": "http.response.start", "status": 401, "headers": []})
                    await send({"type": "http.response.body", "body": b""})
                return
        await self.app(scope, receive, send)


def allocated_gpus(count, gpu_type):
    # Resolve CUDA's actual mapping while honoring Slurm/cgroup visibility. Releasing
    # this subprocess before the engine starts also releases its CUDA context.
    probe = (
        "import json, torch; "
        "props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; "
        "print(json.dumps([[p.name, str(p.uuid)] for p in props]))"
    )
    devices = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    name_pattern = rf"(?<![A-Za-z0-9]){re.escape(gpu_type)}(?![A-Za-z0-9])"
    if len(devices) != count or any(not re.search(name_pattern, name, re.I) for name, _ in devices):
        raise RuntimeError(f"Requested {count} {gpu_type} devices, CUDA reported {devices!r}")
    return ["GPU-" + uuid.removeprefix("GPU-") for _, uuid in devices]


def metrics_handler(uuids, token):
    class GPUMetrics(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_error(404)
                return
            if token and not hmac.compare_digest(
                ("Bearer " + token).encode(), self.headers.get("Authorization", "").encode()
            ):
                self.send_error(401)
                return
            try:
                samples = subprocess.check_output(
                    ["nvidia-smi", "-i", ",".join(uuids), "--query-gpu=uuid,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=3,
                )
                lines = []
                for row in samples.splitlines():
                    uuid, busy = (field.strip() for field in row.split(","))
                    if uuid in uuids and busy.isdecimal():
                        lines.append(f'DCGM_FI_DEV_GPU_UTIL{{UUID="{uuid}"}} {busy}\n')
                if len(lines) != len(uuids):
                    raise ValueError("Incomplete GPU sample")
            except (ValueError, OSError, subprocess.SubprocessError):
                self.send_error(503)
                return
            body = "".join(lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return GPUMetrics


def claim_ports(requested, handler):
    # Cloud has one worker per instance. On shared Slurm nodes ask the OS for a
    # free port, reserving the adjacent telemetry listener before publishing it.
    for _ in range(32):
        with socket.socket() as engine:
            engine.bind(("", requested))
            port = engine.getsockname()[1]
            if port == 65535:
                continue
            try:
                return port, ThreadingHTTPServer(("", port + 1), handler)
            except OSError:
                if requested:
                    raise
    raise RuntimeError("Could not allocate adjacent engine and metrics ports")


def engine_command(args, port, key_file):
    if args.backend == "vllm":
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", args.checkpoint, "--tensor-parallel-size", str(args.gpus),
                   "--max-num-seqs", str(args.capacity)]
        if key_file:
            command.extend(["--middleware", "worker.GatewayAuth"])
        length_flag = "--max-model-len"
    else:
        command = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.checkpoint,
                   "--tp", str(args.gpus), "--max-running-requests", str(args.capacity),
                   "--enable-metrics"]
        if key_file:
            command.extend(["--config", str(key_file.with_name("sglang.json"))])
        length_flag = "--context-length"
    command.extend(["--served-model-name", args.model, "--host", "0.0.0.0",
                    "--port", str(port), "--dtype", args.dtype])
    if args.max_model_len:
        command.extend([length_flag, str(args.max_model_len)])
    return command


def publish_when_ready(worker, port_file, port, token, model):
    # A Slurm allocation can share a host with other engines. Publish this
    # host/port only after our child survives startup and serves its model.
    opener = build_opener(ProxyHandler({}))
    request = Request(f"http://127.0.0.1:{port}/v1/models",
                      headers={"Authorization": "Bearer " + token})
    while worker.poll() is None:
        try:
            with opener.open(request, timeout=2) as response:
                models = json.load(response).get("data", [])
            if any(item.get("id") == model for item in models) and worker.poll() is None:
                staging = port_file.with_suffix(".tmp")
                staging.write_text(str(port))
                staging.replace(port_file)
                return
        except (OSError, URLError, ValueError):
            pass
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--port-file", type=Path)
    args = parser.parse_args()
    token = args.key_file.read_text().strip() if args.key_file else None
    uuids = allocated_gpus(args.gpus, args.gpu_type)
    port, server = claim_ports(args.port, metrics_handler(uuids, token))
    Thread(target=server.serve_forever, daemon=True).start()
    env = os.environ.copy()
    if token and args.backend == "vllm":
        env["VLLM_API_KEY"] = token
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    worker = subprocess.Popen(engine_command(args, port, args.key_file), env=env)
    signal.signal(signal.SIGTERM, lambda *_: worker.terminate())
    try:
        if args.port_file:
            publish_when_ready(worker, args.port_file, port, token, args.model)
        sys.exit(worker.wait())
    finally:
        worker.terminate()
        worker.wait()
        server.server_close()


if __name__ == "__main__":
    main()
