---
name: xllm-npu-pipeline-analysis
description: NPU serving pipeline and layer-level analysis for xLLM, vLLM-Ascend, and SGLang. Use when the user asks about prefill/decode boundaries, layer timing, rank skew, graph replay gaps, host bubbles between decode steps, or mapping profiling timeline events back to framework pipeline stages.
---

# xLLM NPU Pipeline Analysis

Use this skill when a five-table profiling report is not enough and the task
needs stage, layer, rank, or decode-step boundary reasoning.

## Inputs

- Profiling artifact that follows
  [`../../references/profiling-artifact-schema.md`](../../references/profiling-artifact-schema.md).
- Workload shape: input tokens, output tokens, parallel, warmup.
- Framework startup command and commit.
- Optional source-map notes from `xllm-npu-profiler`.

## Workflow

1. Split the trace into prefill, decode, graph replay, communication, and
   postprocess regions.
2. Identify repeated decode-step boundaries. For xLLM decode, pay attention to
   intervals such as `replaceToken` end to next `GatherV2` start.
3. Build tables:

   | Table | Purpose |
   |---|---|
   | Stage table | Prefill/decode/postprocess latency by step |
   | Layer table | Representative layer latency and top kernels |
   | Rank skew table | Slow rank vs fast rank and HCCL wait |
   | Bubble table | Host gap, copy, setup, synchronization, graph replay |

4. Map top timeline events back to likely source areas.
5. Propose only optimizations that can be validated by a before/after
   non-profiling performance run plus a follow-up trace.

## Output

```text
profiling/<run_id>/
  pipeline-analysis.md
  stage-table.csv
  rank-skew-table.csv
  bubble-table.csv
```

The report must clearly separate:

- Device compute bottlenecks.
- Communication/rank skew bottlenecks.
- Hostbound dispatch bubbles.
- Postprocess or sampling overhead.

## References

- [`references/pipeline-boundaries.md`](references/pipeline-boundaries.md)
- [`../xllm-npu-profiler/references/source-map.md`](../xllm-npu-profiler/references/source-map.md)
- [`../../references/profiling-artifact-schema.md`](../../references/profiling-artifact-schema.md)
