"""
Launch all 3 vLLM model servers on separate ports.

Usage:
    python scripts/serve_models.py              # Start all models
    python scripts/serve_models.py phi3-mini    # Start a single model
    python scripts/serve_models.py --stop       # Stop all running servers
"""

import argparse
import subprocess
import sys
import os
import signal
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS, VLLM_COMMON_ARGS

PID_FILE = Path(__file__).resolve().parent.parent / ".model_pids.json"


def build_vllm_command(key: str, cfg: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["model_id"],
        "--port", str(cfg["port"]),
        "--quantization", cfg["quantization"],
        "--max-model-len", str(cfg["max_model_len"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--dtype", VLLM_COMMON_ARGS["dtype"],
        "--served-model-name", key,
        "--trust-remote-code",
    ]
    if VLLM_COMMON_ARGS.get("enforce_eager"):
        cmd.append("--enforce-eager")
    return cmd


def save_pids(pids: dict):
    PID_FILE.write_text(json.dumps(pids))


def load_pids() -> dict:
    if PID_FILE.exists():
        return json.loads(PID_FILE.read_text())
    return {}


def start_model(key: str):
    cfg = MODELS[key]
    cmd = build_vllm_command(key, cfg)
    print(f"[*] Starting {cfg['name']} on port {cfg['port']}...")
    print(f"    Command: {' '.join(cmd)}")

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{key}.log"

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    pids = load_pids()
    pids[key] = proc.pid
    save_pids(pids)
    print(f"    PID: {proc.pid} | Log: {log_file}")
    return proc.pid


def stop_all():
    pids = load_pids()
    if not pids:
        print("[*] No running servers found.")
        return

    for key, pid in pids.items():
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"[*] Stopped {key} (PID {pid})")
        except ProcessLookupError:
            print(f"[*] {key} (PID {pid}) was already stopped.")

    PID_FILE.unlink(missing_ok=True)
    print("[*] All servers stopped.")


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Poll the health endpoint until the server is ready."""
    import requests
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(3)
    return False


def main():
    parser = argparse.ArgumentParser(description="Manage vLLM model servers")
    parser.add_argument("model", nargs="?", choices=list(MODELS.keys()),
                        help="Specific model to start (default: all)")
    parser.add_argument("--stop", action="store_true", help="Stop all servers")
    args = parser.parse_args()

    if args.stop:
        stop_all()
        return

    targets = [args.model] if args.model else list(MODELS.keys())

    print(f"[*] Launching {len(targets)} model(s)...\n")
    for key in targets:
        start_model(key)
        print()

    print("[*] Waiting for servers to become healthy...")
    all_healthy = True
    for key in targets:
        port = MODELS[key]["port"]
        print(f"    Waiting for {key} (port {port})...", end=" ", flush=True)
        if wait_for_server(port):
            print("READY")
        else:
            print("TIMEOUT — check logs/")
            all_healthy = False

    if all_healthy:
        print("\n[✓] All models are up and serving!")
    else:
        print("\n[!] Some models failed to start. Check logs/ directory.")


if __name__ == "__main__":
    main()
