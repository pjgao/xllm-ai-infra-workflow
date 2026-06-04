---
name: xllm-npu-compute-simulation
description: NPU compute simulation for LLM serving. Use when the user asks whether a kernel is near hardware limits, wants FLOPs/MFU estimates, wants prefill/decode compute cost, or needs TP/EP/MTP shape what-if analysis for xLLM, vLLM-Ascend, or SGLang on Ascend NPU.
---

# xLLM NPU Compute Simulation

Use this skill to estimate theoretical compute cost and compare profiling time
against a hardware-bound lower bound.

## Inputs

- Model config: layers, hidden size, intermediate size, attention heads, KV
  heads, vocab size, MoE settings if any.
- Serving shape: batch, input tokens, output tokens, TP/PP/EP, dtype.
- Profiling kernel table when available.
- NPU spec from [`../../references/npu-specs.json`](../../references/npu-specs.json).

## Workflow

1. Estimate prefill and decode FLOPs by operator family:
   attention, MLP, projection, embedding/lm_head, MoE, recurrent/state modules.
2. Adjust for parallelism: TP/PP/EP and communication overhead are separate.
3. Compute lower-bound time from NPU peak throughput and dtype.
4. Compare profiling device time with theoretical time:

   ```text
   MFU = estimated_flops / (elapsed_seconds * peak_flops)
   ```

5. Classify the gap:
   compute-bound, memory-bound, communication-bound, hostbound, or unknown.
6. Produce what-if estimates for safe parameter changes, such as TP, MTP depth,
   batch size, or output length.

## Output

```text
runs/compute/<run_id>/
  manifest.md
  compute-estimate.json
  mfu-table.md
  what-if.md
```

## Rules

- Treat estimates as directional unless validated against profiling.
- Do not compare models with different tokenizer/template output lengths without
  recording the actual token counts.
- If the model config is incomplete, mark missing terms explicitly rather than
  inventing values.

## References

- [`../../references/npu-specs.json`](../../references/npu-specs.json)
- [`../../references/model-config-index.json`](../../references/model-config-index.json)
- [`references/llm-flops-formulas.md`](references/llm-flops-formulas.md)
