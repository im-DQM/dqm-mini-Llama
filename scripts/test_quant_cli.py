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


def test_generate_quant_smoke(binary, quant):
    result = run_cmd([
        binary,
        "generate",
        "--config", "models/tiny/model.json",
        "--model", "models/tiny/model.bin",
        "--tokenizer", "models/tiny/vocab.json",
        "--prompt", "hello",
        "--n-predict", "4",
        "--quant", quant,
    ])
    assert f"quant: {quant}" in result.stdout
    assert "generated tokens:" in result.stdout
    assert "generated text:" in result.stdout


def test_bench_quant_smoke(binary, quant):
    result = run_cmd([
        binary,
        "bench",
        "models/tiny",
        "--tokenizer", "models/tiny/vocab.json",
        "--prompt", "hello",
        "--n-predict", "4",
        "--quant", quant,
    ])
    assert f"quant: {quant}" in result.stdout
    assert "weight memory:" in result.stdout
    assert "tokens/s (total):" in result.stdout
    assert "logits error vs model-native:" in result.stdout
    assert "max:" in result.stdout
    assert "mean:" in result.stdout


def main():
    binary = os.environ.get("MINI_LLAMA_BIN")
    if not binary:
        binary = str(ROOT / "build" / "dqm-llama")
    binary = str(Path(binary).resolve())

    for quant in ("q8_0", "q4_0"):
        test_generate_quant_smoke(binary, quant)
        print(f"PASS generate_{quant}_smoke")
        test_bench_quant_smoke(binary, quant)
        print(f"PASS bench_{quant}_smoke")


if __name__ == "__main__":
    main()
