#!/usr/bin/env python3
"""Analyze xLLM NPU profiling data from Ascend profiling output.

Parses step_trace_time.csv, op_statistic.csv, kernel_details.csv,
and analysis.db to produce the five-table triage report.
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class KernelEntry:
    name: str
    gpu_time_ms: float
    gpu_time_pct: float
    count: int
    avg_time_ms: float
    stage: str


@dataclass
class OverlapEntry:
    compute_kernel: str
    comm_kernel: str
    overlap_ms: float
    potential_ms: float
    stage: str


@dataclass
class DispatchEntry:
    step_id: int
    total_time_ms: float
    aicore_time_ms: float
    idle_time_ms: float
    aicore_utilization: float
    dispatch_latency_ms: float


@dataclass
class MemoryEntry:
    name: str
    allocated_mb: float
    used_mb: float
    fragmentation_pct: float
    stage: str


def parse_step_trace(path: str) -> list[dict]:
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def parse_op_statistic(path: str) -> list[dict]:
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def parse_kernel_details(path: str) -> list[dict]:
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def parse_analysis_db(path: str) -> dict:
    data = {"kernels": [], "operators": [], "steps": []}
    if not os.path.exists(path):
        return data
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        for (table_name,) in tables:
            if "kernel" in table_name.lower():
                rows = cursor.execute(f"SELECT * FROM {table_name}").fetchall()
                cols = [d[0] for d in cursor.description]
                data["kernels"] = [dict(zip(cols, r)) for r in rows]
            elif "operator" in table_name.lower() or "op" in table_name.lower():
                rows = cursor.execute(f"SELECT * FROM {table_name}").fetchall()
                cols = [d[0] for d in cursor.description]
                data["operators"] = [dict(zip(cols, r)) for r in rows]
        conn.close()
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
    return data


def compute_kernel_table(kernel_details: list[dict], cutoff_pct: float = 1.0) -> list[KernelEntry]:
    total_time = sum(float(k.get("duration", k.get("Duration", 0))) for k in kernel_details)
    if total_time == 0:
        total_time = 1.0

    kernel_map: dict[str, dict] = {}
    for k in kernel_details:
        name = k.get("kernel_name", k.get("Kernel Name", "unknown"))
        dur = float(k.get("duration", k.get("Duration", 0)))
        count = int(k.get("count", k.get("Count", 1)))
        stage = k.get("stage", "mixed")
        if name not in kernel_map:
            kernel_map[name] = {"total": 0, "count": 0, "stage": stage}
        kernel_map[name]["total"] += dur
        kernel_map[name]["count"] += count

    entries = []
    for name, info in kernel_map.items():
        pct = info["total"] / total_time * 100
        if pct >= cutoff_pct:
            entries.append(KernelEntry(
                name=name,
                gpu_time_ms=info["total"] / 1000,
                gpu_time_pct=pct,
                count=info["count"],
                avg_time_ms=info["total"] / info["count"] / 1000 if info["count"] > 0 else 0,
                stage=info["stage"],
            ))

    entries.sort(key=lambda e: -e.gpu_time_pct)
    return entries


def compute_dispatch_table(step_trace: list[dict]) -> list[DispatchEntry]:
    entries = []
    for step in step_trace:
        total = float(step.get("total_time", 0))
        aicore = float(step.get("aicore_time", 0))
        idle = float(step.get("idle_time", 0))
        entries.append(DispatchEntry(
            step_id=int(step.get("step_id", 0)),
            total_time_ms=total,
            aicore_time_ms=aicore,
            idle_time_ms=idle,
            aicore_utilization=aicore / total * 100 if total > 0 else 0,
            dispatch_latency_ms=float(step.get("dispatch_latency", 0)),
        ))
    return entries


def render_kernel_table_table(entries: list[KernelEntry]) -> str:
    lines = [
        "| Kernel | GPU Time (ms) | GPU % | Count | Avg (ms) | Stage |",
        "|--------|--------------|-------|-------|----------|-------|",
    ]
    for e in entries[:20]:
        lines.append(f"| {e.name[:60]} | {e.gpu_time_ms:.2f} | {e.gpu_time_pct:.1f}% | {e.count} | {e.avg_time_ms:.4f} | {e.stage} |")
    return "\n".join(lines)


def render_dispatch_table(entries: list[DispatchEntry]) -> str:
    if not entries:
        return "No step trace data available."
    lines = [
        "| Step | Total (ms) | AICore (ms) | Idle (ms) | Utilization | Dispatch (ms) |",
        "|------|-----------|-------------|-----------|-------------|---------------|",
    ]
    for e in entries[:20]:
        lines.append(f"| {e.step_id} | {e.total_time_ms:.2f} | {e.aicore_time_ms:.2f} | {e.idle_time_ms:.2f} | {e.aicore_utilization:.1f}% | {e.dispatch_latency_ms:.2f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze xLLM NPU profiling data")
    parser.add_argument("--input", required=True, help="Profiling results directory")
    parser.add_argument("--framework", default="xllm", choices=["xllm", "vllm-ascend"])
    parser.add_argument("--cutoff", type=float, default=1.0, help="Minimum GPU%% cutoff (default 1.0)")
    parser.add_argument("--output", help="Output JSON path for structured data")
    parser.add_argument("--url", help="Server URL for live capture mode")
    parser.add_argument("--output-dir", help="Output directory for live capture")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--profile-by-stage", action="store_true")
    parser.add_argument("--profile-workload", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--mapping-input", help="Mapping trace input (two-trace mode)")
    parser.add_argument("--formal-input", help="Formal trace input (two-trace mode)")
    args = parser.parse_args()

    input_dir = args.input

    step_trace = parse_step_trace(os.path.join(input_dir, "step_trace_time.csv"))
    op_stat = parse_op_statistic(os.path.join(input_dir, "op_statistic.csv"))
    kernel_details = parse_kernel_details(os.path.join(input_dir, "kernel_details.csv"))
    analysis_db = parse_analysis_db(os.path.join(input_dir, "analysis.db"))

    kernel_entries = compute_kernel_table(kernel_details, args.cutoff)
    dispatch_entries = compute_dispatch_table(step_trace)

    if args.output:
        report = {
            "framework": args.framework,
            "input_dir": input_dir,
            "kernel_table": [
                {
                    "name": e.name, "gpu_time_ms": e.gpu_time_ms,
                    "gpu_time_pct": e.gpu_time_pct, "count": e.count,
                    "avg_time_ms": e.avg_time_ms, "stage": e.stage,
                }
                for e in kernel_entries
            ],
            "dispatch_table": [
                {
                    "step_id": e.step_id, "total_time_ms": e.total_time_ms,
                    "aicore_time_ms": e.aicore_time_ms, "idle_time_ms": e.idle_time_ms,
                    "aicore_utilization": e.aicore_utilization,
                    "dispatch_latency_ms": e.dispatch_latency_ms,
                }
                for e in dispatch_entries
            ],
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}")

    print("## Kernel Table")
    print(render_kernel_table_table(kernel_entries))
    print()
    print("## Dispatch Efficiency Table")
    print(render_dispatch_table(dispatch_entries))


if __name__ == "__main__":
    main()
