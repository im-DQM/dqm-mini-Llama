#!/usr/bin/env python3
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args, *, check=False, input_text=None):
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=input_text,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(map(str, args))}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def parse_generated_tokens(output):
    for line in output.splitlines():
        if line.startswith("generated tokens:"):
            return ast.literal_eval(line.split(":", 1)[1].strip())
    raise AssertionError(f"missing generated tokens line:\n{output}")


def test_cpp_matches_python_reference(binary):
    cpp = run_cmd([
        binary,
        "--config", "models/tiny/model.json",
        "--model", "models/tiny/model.bin",
        "-p", "hello",
        "-n", "16",
        "--raw-prompt",
        "--tokenizer", "/nonexistent",
    ], check=True)
    py = run_cmd([
        sys.executable,
        "scripts/reference_forward.py",
        "--config", "models/tiny/model.json",
        "--weights", "models/tiny/model.bin",
        "--prompt", "hello",
        "-n", "16",
    ], check=True)
    assert parse_generated_tokens(cpp.stdout) == parse_generated_tokens(py.stdout)


def test_zero_predict(binary):
    result = run_cmd([
        binary,
        "--config", "models/tiny/model.json",
        "--model", "models/tiny/model.bin",
        "-p", "hello",
        "-n", "0",
    ], check=True)
    assert parse_generated_tokens(result.stdout) == []
    assert "decode loop skipped." in result.stdout


def test_context_window_rejected(binary):
    prompt = "a" * 127
    result = run_cmd([
        binary,
        "--config", "models/tiny/model.json",
        "--model", "models/tiny/model.bin",
        "-p", prompt,
        "-n", "1",
        "--raw-prompt",
        "--tokenizer", "/nonexistent",
    ])
    assert result.returncode != 0
    assert "Requested tokens exceed context window" in result.stderr


def test_truncated_weights_rejected(binary):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bad_weights = tmp_path / "tiny_llama_truncated.bin"
        data = (ROOT / "models/tiny/model.bin").read_bytes()
        bad_weights.write_bytes(data[:-4])
        result = run_cmd([
            binary,
            "--config", "models/tiny/model.json",
            "--model", str(bad_weights),
            "-p", "hello",
            "-n", "1",
        ])
        assert result.returncode != 0
        assert "weights file ended while reading tensor" in result.stderr


def test_bad_config_rejected(binary):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bad_config = tmp_path / "bad_config.json"
        config = json.loads((ROOT / "models/tiny/model.json").read_text())
        del config["config"]["n_heads"]
        bad_config.write_text(json.dumps(config), encoding="utf-8")
        result = run_cmd([
            binary,
            "--config", str(bad_config),
            "--model", "models/tiny/model.bin",
            "-p", "hello",
            "-n", "1",
        ])
        assert result.returncode != 0
        assert "missing config key: n_heads" in result.stderr


def test_generate_accepts_tokenizer_arg(binary):
    result = run_cmd([
        binary,
        "generate",
        "--config", "models/tiny/model.json",
        "--model", "models/tiny/model.bin",
        "--tokenizer", "models/tiny/vocab.json",
        "-p", "hello",
        "-n", "4",
    ], check=True)
    assert "generated tokens:" in result.stdout


def test_run_accepts_tokenizer_arg(binary):
    result = run_cmd([
        binary,
        "run",
        "models/tiny",
        "--tokenizer", "models/tiny/vocab.json",
    ], check=True, input_text="/exit\n")
    assert "dqm-mini-llama.cpp chat" in result.stdout
    assert "Goodbye." in result.stdout


def test_inspect_bad_config_rejected(binary):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bad_config = tmp_path / "model.json"
        config = json.loads((ROOT / "models/tiny/model.json").read_text())
        del config["config"]["n_heads"]
        bad_config.write_text(json.dumps(config), encoding="utf-8")
        result = run_cmd([
            binary,
            "inspect",
            str(tmp_path),
        ])
        assert result.returncode != 0
        assert "missing config key: n_heads" in result.stderr


def test_bench_smoke(binary):
    result = run_cmd([
        binary,
        "bench",
        "models/tiny",
        "--tokenizer", "models/tiny/vocab.json",
        "--prompt", "hello",
        "--n-predict", "4",
    ], check=True)
    assert "Results:" in result.stdout
    assert "prompt tokens:" in result.stdout
    assert "generated tokens:" in result.stdout
    assert "tokens/s" in result.stdout


def main():
    binary = os.environ.get("MINI_LLAMA_BIN")
    if not binary:
        binary = str(ROOT / "build" / "dqm-llama")
    binary = str(Path(binary).resolve())
    tests = [
        test_cpp_matches_python_reference,
        test_zero_predict,
        test_context_window_rejected,
        test_truncated_weights_rejected,
        test_bad_config_rejected,
        test_generate_accepts_tokenizer_arg,
        test_run_accepts_tokenizer_arg,
        test_inspect_bad_config_rejected,
        test_bench_smoke,
    ]
    for test in tests:
        test(binary)
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
