#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args, *, check=True):
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(map(str, args))}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def test_generate_backend_cpu(binary):
    result = run_cmd([
        binary,
        "generate",
        "--model", "models/tiny/model.bin",
        "--config", "models/tiny/model.json",
        "--tokenizer", "models/tiny/vocab.json",
        "--prompt", "hello",
        "--n-predict", "1",
        "--backend", "cpu",
    ])
    assert "backend: cpu" in result.stdout
    assert "compute: cpu" in result.stdout
    assert "generated tokens:" in result.stdout
    assert "trace event: trace_id=req_" in result.stdout
    assert "trace summary: trace_id=req_" in result.stdout
    assert "mode=generate" in result.stdout
    assert "status=ok" in result.stdout
    assert "backend=cpu" in result.stdout
    assert "stage=tokenize" in result.stdout
    assert "stage=prefill" in result.stdout
    assert "prompt_tokens=" in result.stdout


def test_bench_backend_cpu(binary):
    result = run_cmd([
        binary,
        "bench",
        "models/tiny",
        "--tokenizer", "models/tiny/vocab.json",
        "--prompt", "hello",
        "--n-predict", "1",
        "--backend", "cpu",
    ])
    assert "backend: cpu" in result.stdout
    assert "compute: cpu" in result.stdout
    assert "tokens/s (total):" in result.stdout


def test_invalid_backend_rejected(binary):
    result = run_cmd([
        binary,
        "generate",
        "--backend", "metal",
    ], check=False)
    assert result.returncode != 0
    assert "Invalid --backend value" in result.stderr


def test_invalid_device_rejected(binary):
    result = run_cmd([
        binary,
        "generate",
        "--device", "-1",
    ], check=False)
    assert result.returncode != 0
    assert "Invalid --device value" in result.stderr


def test_cpu_backend_rejects_device(binary):
    result = run_cmd([
        binary,
        "generate",
        "--backend", "cpu",
        "--device", "0",
    ], check=False)
    assert result.returncode != 0
    assert "--device can only be used with --backend cuda" in result.stderr


def test_generate_context_window_failure_has_trace(binary):
    result = run_cmd([
        binary,
        "generate",
        "--model", "models/tiny/model.bin",
        "--config", "models/tiny/model.json",
        "--tokenizer", "models/tiny/vocab.json",
        "--prompt", "hello",
        "--n-predict", "1000000",
        "--backend", "cpu",
    ], check=False)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "trace summary: trace_id=req_" in combined
    assert "mode=generate" in combined
    assert "status=error" in combined
    assert "Requested tokens exceed context window." in combined


def test_cuda_backend_request_has_clear_outcome(binary):
    result = run_cmd([
        binary,
        "generate",
        "--model", "models/tiny/model.bin",
        "--config", "models/tiny/model.json",
        "--tokenizer", "models/tiny/vocab.json",
        "--prompt", "hello",
        "--n-predict", "1",
        "--backend", "cuda",
    ], check=False)
    combined = result.stdout + result.stderr
    if result.returncode == 0:
        stdout_lower = result.stdout.lower()
        assert "backend: cuda" in result.stdout
        assert "compute: cuda linear" in stdout_lower
        assert "CUDA attention over GPU KV cache" in result.stdout
        assert "CPU sampler" in result.stdout
        assert "uploaded weights:" in result.stdout
        assert "cuda linear calls:" in stdout_lower
        assert "cuda activation calls:" in result.stdout
        assert "cuda attention calls:" in result.stdout
        assert "cpu attention fallback calls:" in result.stdout
    else:
        assert "Backend setup failed:" in combined
        assert "CUDA backend was not built" in combined or "CUDA device" in combined


def test_cuda_info_has_clear_outcome(binary):
    result = run_cmd([
        binary,
        "--cuda-info",
    ], check=False)
    combined = result.stdout + result.stderr
    if result.returncode == 0:
        assert "CUDA devices:" in result.stdout
        assert "device name:" in result.stdout
        assert "compute capability:" in result.stdout
        assert "total memory:" in result.stdout
        assert "driver version:" in result.stdout
        assert "runtime version:" in result.stdout
    else:
        assert "CUDA info failed:" in combined
        assert "CUDA backend was not built" in combined


def main():
    binary = os.environ.get("MINI_LLAMA_BIN")
    if not binary:
        binary = str(ROOT / "build" / "dqm-llama")
    binary = str(Path(binary).resolve())

    test_generate_backend_cpu(binary)
    print("PASS generate_backend_cpu")
    test_bench_backend_cpu(binary)
    print("PASS bench_backend_cpu")
    test_invalid_backend_rejected(binary)
    print("PASS invalid_backend_rejected")
    test_invalid_device_rejected(binary)
    print("PASS invalid_device_rejected")
    test_cpu_backend_rejects_device(binary)
    print("PASS cpu_backend_rejects_device")
    test_generate_context_window_failure_has_trace(binary)
    print("PASS generate_context_window_failure_has_trace")
    test_cuda_backend_request_has_clear_outcome(binary)
    print("PASS cuda_backend_request_has_clear_outcome")
    test_cuda_info_has_clear_outcome(binary)
    print("PASS cuda_info_has_clear_outcome")


if __name__ == "__main__":
    main()
