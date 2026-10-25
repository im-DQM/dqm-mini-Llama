#!/usr/bin/env python3
"""顶层帮助：`-h` / `--help` 应列出子命令，而不是转发给 generate。"""
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args):
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_help_lists_subcommands(binary, flag):
    result = run_cmd([binary, flag])
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{flag} exit code {result.returncode}"
    for command in ("run", "generate", "bench", "inspect", "inspect-gguf", "--cuda-info"):
        assert command in output, f"{flag} output missing {command}: {output}"
    # 顶层帮助不应展示子命令的参数细节
    assert "--n-predict" not in output, f"{flag} leaked generate options: {output}"


def test_no_args_prints_usage(binary):
    result = run_cmd([binary])
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "run" in output and "generate" in output


def test_generate_help_unchanged(binary):
    result = run_cmd([binary, "generate", "--help"])
    output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "--n-predict" in output
    assert "--prompt" in output


def main():
    binary = os.environ.get("MINI_LLAMA_BIN")
    if not binary:
        raise SystemExit("MINI_LLAMA_BIN not set")

    for flag in ("-h", "--help"):
        test_help_lists_subcommands(binary, flag)
        print(f"PASS help_lists_subcommands[{flag}]")
    test_no_args_prints_usage(binary)
    print("PASS no_args_prints_usage")
    test_generate_help_unchanged(binary)
    print("PASS generate_help_unchanged")


if __name__ == "__main__":
    main()
