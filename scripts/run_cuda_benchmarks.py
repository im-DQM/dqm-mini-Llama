#!/usr/bin/env python3
"""Run the CUDA v2 benchmark matrix and write Markdown/JSON/CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CommandResult:
    name: str
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(str(arg)) for arg in self.args)


@dataclass
class BenchCase:
    name: str
    model: str
    backend: str
    n_predict: int
    prompt: str = "hello"
    seed: int = 1
    quant: str = ""
    tokenizer: str = ""
    skipped: str = ""
    command_result: CommandResult | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def run_command(args: list[str], *, name: str, timeout: int, check: bool = False) -> CommandResult:
    started = datetime.now(timezone.utc)
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    finished = datetime.now(timezone.utc)
    command_result = CommandResult(
        name=name,
        args=args,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_sec=(finished - started).total_seconds(),
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed: {command_result.command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return command_result


def first_line(command: list[str], timeout: int = 30) -> str:
    try:
        result = run_command(command, name="probe", timeout=timeout)
    except Exception as exc:
        return f"unavailable: {exc}"
    text = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        return f"unavailable: {text.splitlines()[0] if text else result.returncode}"
    return text.splitlines()[0] if text else ""


def collect_environment(binary: Path, build_dir: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(ROOT),
        "build_dir": str(build_dir),
        "binary": str(binary),
        "commit": first_line(["git", "rev-parse", "--short", "HEAD"]),
        "branch": first_line(["git", "status", "--short", "--branch"]),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cmake": first_line(["cmake", "--version"]),
        "cxx": first_line(["c++", "--version"]),
        "nvidia_smi": first_line([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]),
        "nvcc": first_line(["bash", "-lc", "nvcc --version | tail -n 1"]),
        "cuda_info": first_line([str(binary), "--cuda-info"]) if binary.exists() else "binary missing",
    }


def parse_count_bytes(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)\s+\((\d+)\s+bytes\)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_float(label: str, stdout: str) -> float | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.+\-eE]+)", stdout,
                      re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_int(label: str, stdout: str) -> int | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", stdout,
                      re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_weight_line(label: str, stdout: str) -> dict[str, int] | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*(\d+)/(\d+)", stdout,
                      re.IGNORECASE)
    if not match:
        return None
    return {"uploaded": int(match.group(1)), "total": int(match.group(2))}


def parse_benchmark_stdout(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for key, label in [
        ("prompt_tokens", "prompt tokens"),
        ("generated_tokens", "generated tokens"),
        ("decode_tokens", "decode tokens"),
        ("uploaded_weights", "uploaded weights"),
        ("cuda_linear_calls", "cuda linear calls"),
        ("cuda_activation_calls", "cuda activation calls"),
        ("cuda_attention_calls", "cuda attention calls"),
        ("cpu_attention_fallback_calls", "cpu attention fallback calls"),
        ("cuda_kv_write_bytes", "cuda kv write bytes"),
        ("cuda_kv_read_bytes", "cuda kv read bytes"),
    ]:
        value = parse_int(label, stdout)
        if value is not None:
            metrics[key] = value

    for key, label in [
        ("prefill_ms", "prefill time"),
        ("decode_ms", "decode time"),
        ("total_ms", "total time"),
        ("tokens_per_sec_total", "tokens/s (total)"),
        ("tokens_per_sec_decode", "tokens/s (decode)"),
        ("logits_error_max", "max"),
        ("logits_error_mean", "mean"),
    ]:
        value = parse_float(label, stdout)
        if value is not None:
            metrics[key] = value

    gpu_mem = re.search(r"gpu memory used:\s*(\d+) bytes", stdout)
    if gpu_mem:
        metrics["gpu_memory_bytes"] = int(gpu_mem.group(1))

    actual = re.search(r"actual:\s*(\d+) bytes", stdout)
    f32_equiv = re.search(r"f32 equiv:\s*(\d+) bytes", stdout)
    savings = re.search(r"savings:\s*([0-9.]+)x compression", stdout)
    if actual:
        metrics["weight_actual_bytes"] = int(actual.group(1))
    if f32_equiv:
        metrics["weight_f32_equiv_bytes"] = int(f32_equiv.group(1))
    if savings:
        metrics["weight_compression"] = float(savings.group(1))

    for key, label in [
        ("cuda_f32_linear_weights", "cuda f32 linear weights"),
        ("cuda_q8_0_linear_weights", "cuda q8_0 linear weights"),
        ("cuda_q4_0_linear_weights", "cuda q4_0 linear weights"),
        ("cuda_q4_1_linear_weights", "cuda q4_1 linear weights"),
    ]:
        value = parse_weight_line(label, stdout)
        if value is not None:
            metrics[key] = value

    fallback = re.search(
        r"cuda fallback:.*\(Q8_0=(\d+), Q4_0=(\d+), Q4_1=(\d+)\)",
        stdout,
    )
    if fallback:
        metrics["fallback_q8_0"] = int(fallback.group(1))
        metrics["fallback_q4_0"] = int(fallback.group(2))
        metrics["fallback_q4_1"] = int(fallback.group(3))

    for prefix, label in [
        ("host_to_device", "host->device copies"),
        ("device_to_host", "device->host copies"),
    ]:
        line = re.search(rf"{re.escape(label)}:\s*(\d+)\s+\((\d+)\s+bytes\)", stdout)
        if line:
            metrics[f"{prefix}_copies"] = int(line.group(1))
            metrics[f"{prefix}_bytes"] = int(line.group(2))

    return metrics


def model_exists(model: str) -> bool:
    return (ROOT / model).exists()


def make_cases(args: argparse.Namespace) -> list[BenchCase]:
    tiny_n = 4 if args.fast else args.tiny_n_predict
    chat_n = 1 if args.fast else args.chat_n_predict
    q4_n = 4 if args.fast else args.q4_n_predict

    cases = [
        BenchCase("tiny-cpu", "models/tiny", "cpu", tiny_n, args.prompt, args.seed),
        BenchCase("tiny-cuda", "models/tiny", "cuda", tiny_n, args.prompt, args.seed),
        BenchCase("chat-q8_0-cpu", "models/chat", "cpu", chat_n, args.prompt, args.seed),
        BenchCase("chat-q8_0-cuda", "models/chat", "cuda", chat_n, args.prompt, args.seed),
        BenchCase("tiny-q4_0-cpu", "models/tiny", "cpu", q4_n, args.prompt, args.seed, quant="q4_0"),
        BenchCase("tiny-q4_0-cuda", "models/tiny", "cuda", q4_n, args.prompt, args.seed, quant="q4_0"),
    ]
    for index, q4_model in enumerate(args.q4_model or []):
        name = Path(q4_model).name or f"q4-model-{index}"
        cases.append(BenchCase(f"{name}-cpu", q4_model, "cpu", args.q4_n_predict, args.prompt, args.seed))
        cases.append(BenchCase(f"{name}-cuda", q4_model, "cuda", args.q4_n_predict, args.prompt, args.seed))
    return cases


def bench_command(binary: Path, case: BenchCase) -> list[str]:
    command = [
        str(binary),
        "bench",
        case.model,
        "--backend",
        case.backend,
        "-p",
        case.prompt,
        "-n",
        str(case.n_predict),
        "--seed",
        str(case.seed),
    ]
    if case.quant:
        command.extend(["--quant", case.quant])
    if case.tokenizer:
        command.extend(["--tokenizer", case.tokenizer])
    return command


def run_ctests(build_dir: Path, timeout: int, skip_tests: bool) -> list[CommandResult]:
    if skip_tests:
        return []
    commands = [
        ("targeted correctness", [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-R",
            "cuda_quant_q4|cuda_forward_q4|cuda_forward_quant|cuda_forward_device_resident|cuda_forward_attention|cuda-chat-smoke",
            "--output-on-failure",
        ]),
        ("full ctest", [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--output-on-failure",
        ]),
    ]
    results = []
    for name, command in commands:
        results.append(run_command(command, name=name, timeout=timeout, check=True))
    return results


def run_benchmarks(binary: Path, cases: list[BenchCase], timeout: int, skip_cuda: bool) -> None:
    for case in cases:
        if case.backend == "cuda" and skip_cuda:
            case.skipped = "cuda skipped by --skip-cuda"
            continue
        if not model_exists(case.model):
            case.skipped = f"missing model path: {case.model}"
            continue
        result = run_command(bench_command(binary, case), name=case.name, timeout=timeout, check=True)
        case.command_result = result
        case.metrics = parse_benchmark_stdout(result.stdout)


def load_m10_baseline() -> dict[str, float]:
    baseline_path = ROOT / "docs" / "benchmarks" / "2026-05-20-cuda-v2-m10.md"
    text = baseline_path.read_text(encoding="utf-8") if baseline_path.exists() else ""
    return {
        "tiny_cuda_decode_tok_s": 127.37,
        "chat_cuda_decode_tok_s": 0.53,
        "source": str(baseline_path.relative_to(ROOT)) if text else "missing",
    }


def ratio(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline == 0:
        return None
    return current / baseline


def metric(case: BenchCase, key: str) -> Any:
    return case.metrics.get(key)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def format_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(env: dict[str, Any], ctests: list[CommandResult], cases: list[BenchCase]) -> str:
    baseline = load_m10_baseline()
    case_by_name = {case.name: case for case in cases}
    tiny_cuda = case_by_name.get("tiny-cuda")
    chat_cuda = case_by_name.get("chat-q8_0-cuda")

    lines = [
        f"# CUDA v2 Benchmark Harness - {datetime.now().date().isoformat()}",
        "",
        "本报告由 `scripts/run_cuda_benchmarks.py` 自动生成。",
        "",
        "## 环境",
        "",
        markdown_table(
            ["项目", "值"],
            [
                ["timestamp_utc", env.get("timestamp_utc", "-")],
                ["commit", env.get("commit", "-")],
                ["branch", env.get("branch", "-")],
                ["OS", env.get("os", "-")],
                ["CMake", env.get("cmake", "-")],
                ["C++ compiler", env.get("cxx", "-")],
                ["GPU", env.get("nvidia_smi", "-")],
                ["NVCC", env.get("nvcc", "-")],
                ["CUDA info", env.get("cuda_info", "-")],
            ],
        ),
        "",
        "## 正确性",
        "",
    ]

    if ctests:
        rows = [
            [
                result.name,
                result.returncode,
                f"{result.duration_sec:.2f}s",
                result.command,
            ]
            for result in ctests
        ]
        lines.append(markdown_table(["命令", "退出码", "耗时", "command"], rows))
    else:
        lines.append("本次运行跳过 CTest。")

    lines.extend(["", "## 性能汇总", ""])
    rows = []
    for case in cases:
        if case.skipped:
            rows.append([case.name, case.model, case.backend, case.quant or "model-native", "skipped", "-", "-", "-", "-", "-", "-", case.skipped])
            continue
        rows.append([
            case.name,
            case.model,
            case.backend,
            case.quant or "model-native",
            "ok",
            metric(case, "prompt_tokens") or "-",
            metric(case, "generated_tokens") or "-",
            format_float(metric(case, "prefill_ms")),
            format_float(metric(case, "decode_ms")),
            format_float(metric(case, "tokens_per_sec_total")),
            format_float(metric(case, "tokens_per_sec_decode")),
            "",
        ])
    lines.append(markdown_table(
        ["case", "model", "backend", "quant", "status", "prompt tokens", "generated", "prefill ms", "decode ms", "total tok/s", "decode tok/s", "note"],
        rows,
    ))

    lines.extend(["", "## CUDA Runtime", ""])
    runtime_rows = []
    for case in cases:
        if case.backend != "cuda" or case.skipped:
            continue
        runtime_rows.append([
            case.name,
            metric(case, "uploaded_weights") or "-",
            metric(case, "gpu_memory_bytes") or "-",
            format_float(metric(case, "tokens_per_sec_decode")),
            metric(case, "cuda_linear_calls") or "-",
            metric(case, "cuda_activation_calls") or "-",
            metric(case, "cuda_attention_calls") or "-",
            metric(case, "cpu_attention_fallback_calls") if metric(case, "cpu_attention_fallback_calls") is not None else "-",
            metric(case, "host_to_device_bytes") if metric(case, "host_to_device_bytes") is not None else "-",
            metric(case, "device_to_host_bytes") if metric(case, "device_to_host_bytes") is not None else "-",
        ])
    lines.append(markdown_table(
        ["case", "uploaded", "gpu bytes", "decode tok/s", "linear", "activation", "attention", "cpu attn fallback", "H2D bytes", "D2H bytes"],
        runtime_rows,
    ) if runtime_rows else "本次运行跳过 CUDA benchmark。")

    lines.extend(["", "## M10 Baseline 对比", ""])
    compare_rows = []
    if tiny_cuda and not tiny_cuda.skipped:
        compare_rows.append([
            "tiny CUDA decode tok/s",
            baseline["tiny_cuda_decode_tok_s"],
            format_float(metric(tiny_cuda, "tokens_per_sec_decode")),
            format_float(ratio(metric(tiny_cuda, "tokens_per_sec_decode"), baseline["tiny_cuda_decode_tok_s"])),
        ])
    if chat_cuda and not chat_cuda.skipped:
        compare_rows.append([
            "chat CUDA decode tok/s",
            baseline["chat_cuda_decode_tok_s"],
            format_float(metric(chat_cuda, "tokens_per_sec_decode")),
            format_float(ratio(metric(chat_cuda, "tokens_per_sec_decode"), baseline["chat_cuda_decode_tok_s"])),
        ])
    lines.append(markdown_table(["指标", "M10", "current", "current/M10"], compare_rows) if compare_rows else "CUDA case 被跳过，M10 对比待远程运行。")
    lines.append(f"\nbaseline source: `{baseline['source']}`")

    lines.extend(["", "## 原始命令输出", ""])
    for result in ctests:
        lines.extend([
            f"### {result.name}",
            "",
            "```bash",
            result.command,
            "```",
            "",
            "```text",
            (result.stdout + result.stderr).strip(),
            "```",
            "",
        ])
    for case in cases:
        if case.skipped or case.command_result is None:
            continue
        lines.extend([
            f"### {case.name}",
            "",
            "```bash",
            case.command_result.command,
            "```",
            "",
            "```text",
            (case.command_result.stdout + case.command_result.stderr).strip(),
            "```",
            "",
        ])

    lines.extend([
        "## 结论",
        "",
        "- 本报告覆盖环境、正确性、CPU/CUDA benchmark、CUDA runtime 统计和 M10 baseline 对比。",
        "- 完整 CUDA 结果以远程 NVIDIA GPU 机器运行为准。",
    ])
    return "\n".join(lines) + "\n"


def write_json(path: Path, env: dict[str, Any], ctests: list[CommandResult], cases: list[BenchCase]) -> None:
    payload = {
        "environment": env,
        "ctests": [result.__dict__ | {"command": result.command} for result in ctests],
        "cases": [
            {
                "name": case.name,
                "model": case.model,
                "backend": case.backend,
                "quant": case.quant or "model-native",
                "prompt": case.prompt,
                "n_predict": case.n_predict,
                "seed": case.seed,
                "skipped": case.skipped,
                "command": case.command_result.command if case.command_result else "",
                "returncode": case.command_result.returncode if case.command_result else None,
                "metrics": case.metrics,
            }
            for case in cases
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, cases: list[BenchCase]) -> None:
    fields = [
        "name",
        "model",
        "backend",
        "quant",
        "skipped",
        "prompt_tokens",
        "generated_tokens",
        "decode_tokens",
        "prefill_ms",
        "decode_ms",
        "total_ms",
        "tokens_per_sec_total",
        "tokens_per_sec_decode",
        "gpu_memory_bytes",
        "host_to_device_bytes",
        "device_to_host_bytes",
        "cuda_linear_calls",
        "cuda_activation_calls",
        "cuda_attention_calls",
        "cpu_attention_fallback_calls",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {
                "name": case.name,
                "model": case.model,
                "backend": case.backend,
                "quant": case.quant or "model-native",
                "skipped": case.skipped,
            }
            for field_name in fields:
                row.setdefault(field_name, case.metrics.get(field_name, ""))
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mini-llama CUDA v2 benchmark matrix")
    parser.add_argument("--build-dir", default="build-cuda", help="CMake build directory containing mini-llama")
    parser.add_argument("--binary", default="", help="Optional path to mini-llama binary")
    parser.add_argument("--out", required=True, help="Markdown report path")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tiny-n-predict", type=int, default=128)
    parser.add_argument("--chat-n-predict", type=int, default=32)
    parser.add_argument("--q4-n-predict", type=int, default=32)
    parser.add_argument("--q4-model", action="append", default=[], help="Optional real Q4 model directory or file")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true", help="Run only CPU cases; useful on macOS")
    parser.add_argument("--fast", action="store_true", help="Use small n_predict values for script smoke tests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = (ROOT / args.build_dir).resolve() if not Path(args.build_dir).is_absolute() else Path(args.build_dir)
    binary = Path(args.binary) if args.binary else build_dir / "mini-llama"
    binary = binary.resolve()
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    if not binary.exists():
        raise FileNotFoundError(f"mini-llama binary missing: {binary}")

    env = collect_environment(binary, build_dir)
    ctests = run_ctests(build_dir, args.timeout, args.skip_tests)
    cases = make_cases(args)
    run_benchmarks(binary, cases, args.timeout, args.skip_cuda)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_markdown(env, ctests, cases), encoding="utf-8")
    write_json(out.with_suffix(".json"), env, ctests, cases)
    write_csv(out.with_suffix(".csv"), cases)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.json')}")
    print(f"Wrote {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
