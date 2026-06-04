---
name: xllm-npu-capacity-planner
description: xLLM / vLLM-Ascend / SGLang NPU serving capacity planning. Use when the user asks about HBM budget, KV cache capacity, max concurrency, block size, max model length, MTP/speculative memory reserve, OOM risk, or how startup logs explain available request capacity on Ascend NPU.
---

# xLLM NPU Capacity Planner

Use this skill to explain whether a serving configuration has enough NPU memory
for the target workload, and which parameter is limiting capacity.

## Inputs

Collect:

- Model name, dtype, hidden size, layers, attention heads, KV heads.
- Framework and commit.
- NPU model, card count, visible devices.
- Startup flags: TP/PP/EP, `block_size`, `max_model_len`,
  `max_memory_utilization`, `max_tokens_per_batch`, `max_seqs_per_batch`.
- MTP/speculative flags: draft model path, `num_speculative_tokens`, reserved
  linear/cache bytes if logged.
- Startup logs and metrics that mention HBM, KV blocks, xTensor, block manager,
  reserved memory, or OOM.

## Workflow

1. Create a run manifest using
   [`../../references/run-manifest-template.md`](../../references/run-manifest-template.md).
2. Parse startup logs for model memory, available HBM, KV blocks, block size,
   reserved linear bytes, and allocation failures.
3. Build a capacity table:

   | Bucket | Bytes / Blocks | Source | Notes |
   |---|---:|---|---|
   | Model weights | | startup log / estimate | per rank |
   | Runtime workspace | | startup log | ATB / graph / xTensor |
   | KV cache | | block manager | blocks and token capacity |
   | Spec/MTP reserve | | startup log | draft/verify overhead |
   | Free margin | | npu-smi / log | safety headroom |

4. Estimate request capacity under the target prompt/output/concurrency shape.
5. Classify the bottleneck: HBM hard OOM, KV blocks, scheduler budget,
   speculative reserve, graph/workspace reserve, or fragmentation.
6. Produce a short tuning plan with safe parameter changes and validation steps.

## Output

Write:

```text
runs/capacity/<run_id>/
  manifest.md
  startup-log-excerpt.txt
  capacity.json
  report.md
```

`report.md` must include:

- Capacity verdict: pass / risk / fail / inconclusive.
- Limiting resource and evidence.
- Before/after parameter suggestions.
- Whether the result is strong enough for benchmark use.

## References

- [`references/capacity-log-patterns.md`](references/capacity-log-patterns.md)
- [`../../references/run-manifest-template.md`](../../references/run-manifest-template.md)
- [`../../references/perf-artifact-schema.md`](../../references/perf-artifact-schema.md)
