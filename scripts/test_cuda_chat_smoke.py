#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAT_MODEL = ROOT / "models" / "chat" / "Qwen2-0.5B-Instruct-Q8_0.gguf"


def run_cmd(args, *, input_text=None, timeout=300):
    return subprocess.run(
        args,
        cwd=ROOT,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_cuda_unavailable_result(result):
    combined = result.stdout + result.stderr
    require(result.returncode != 0, "CUDA-unavailable command should fail")
    require("Backend setup failed:" in combined, combined)
    require("CUDA backend was not built" in combined or "CUDA device" in combined, combined)


def assert_chat_cuda_header(stdout):
    stdout_lower = stdout.lower()
    require("backend: cuda" in stdout, stdout)
    require("uploaded weights:" in stdout, stdout)
    require("cuda f32 linear weights:" in stdout_lower, stdout)
    require("cuda q8_0 linear weights:" in stdout_lower, stdout)


def assert_cuda_compute_note(stdout):
    stdout_lower = stdout.lower()
    require("compute: cuda linear" in stdout_lower, stdout)
    require("CUDA attention over GPU KV cache" in stdout, stdout)
    require("CPU sampler" in stdout, stdout)


def test_chat_inspect(binary):
    result = run_cmd(
        [
            binary,
            "inspect",
            "models/chat",
            "--backend",
            "cuda",
        ]
    )
    if result.returncode != 0:
        assert_cuda_unavailable_result(result)
        return False

    assert_chat_cuda_header(result.stdout)
    require("GGUF model loaded successfully" in result.stdout, result.stdout)
    return True


def test_chat_bench(binary):
    result = run_cmd(
        [
            binary,
            "bench",
            "models/chat",
            "--backend",
            "cuda",
            "-p",
            "hello",
            "-n",
            "1",
            "--seed",
            "1",
        ]
    )
    if result.returncode != 0:
        assert_cuda_unavailable_result(result)
        return False

    assert_chat_cuda_header(result.stdout)
    assert_cuda_compute_note(result.stdout)
    require("Benchmark: models/chat" in result.stdout, result.stdout)
    require("tokens/s (total):" in result.stdout, result.stdout)
    require("cuda runtime:" in result.stdout, result.stdout)
    require("cuda attention calls:" in result.stdout, result.stdout)
    require("cpu attention fallback calls:" in result.stdout, result.stdout)
    return True


def main():
    if not CHAT_MODEL.exists():
        print(f"SKIP cuda_chat_smoke missing {CHAT_MODEL.relative_to(ROOT)}")
        return

    binary = os.environ.get("MINI_LLAMA_BIN")
    if not binary:
        binary = str(ROOT / "build" / "dqm-llama")
    binary = str(Path(binary).resolve())

    cuda_inspect_ok = test_chat_inspect(binary)
    print("PASS cuda_chat_inspect" if cuda_inspect_ok else "PASS cuda_chat_inspect_unavailable")
    cuda_bench_ok = test_chat_bench(binary)
    print("PASS cuda_chat_bench" if cuda_bench_ok else "PASS cuda_chat_bench_unavailable")


if __name__ == "__main__":
    main()
