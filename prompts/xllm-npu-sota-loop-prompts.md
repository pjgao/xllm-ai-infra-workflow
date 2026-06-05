# xLLM NPU SOTA Loop Prompts

## Full Optimization Loop

```text
Use xllm-npu-sota-loop to optimize <target_framework> on Ascend <A2_or_A3>.

Goal:
- Model: <model_name>
- Workload: <input_tokens> input / <output_tokens> output, parallel=<parallel>, number=<number>
- Metric target: improve <TPOT_or_TTFT_or_TPS> by <target_percent>% against the current baseline
- Baseline framework(s): <xllm>, optionally <vllm-ascend>, <sglang-npu>
- Artifact root: <run_root>

Rules:
1. Before changing code, collect a warmed-up baseline performance run and a
   decode/prefill-focused profiling run.
2. Do not compare unfair data. Keep model, tokenizer, dtype, workload, sampling,
   SLA, and warmup policy aligned.
3. Use model-pr-optimization-history before writing a patch.
4. Each RLCR round must change only one optimization idea, then run code review,
   accuracy validation, warmed-up performance validation, and profiling when
   needed.
5. Stop only when the target is reached, the result is within the declared stop
   condition, or the run is blocked by a reproducible external issue.
6. Record the final result in the run ledger and update reusable skill/reference
   knowledge if a new lesson was found.
```

## Decode Gap Deep Dive

```text
Use xllm-npu-profiler and xllm-npu-pipeline-analysis to explain a decode TPOT
gap for <model_name> on Ascend <A2_or_A3>.

Inputs:
- Baseline run: <baseline_run_path>
- Current run: <current_run_path>
- Profiling path: <profiling_path>
- Known timeline anchors: <for example replaceToken, GatherV2, graph replay,
  PagedAttention, HCCL>

Output:
1. A stage table for one decode step.
2. A hostbound bubble table between consecutive decode steps.
3. A list of candidate code changes ranked by expected gain, risk, and
   validation cost.
4. A validation plan with warmed-up perf, profiling, and accuracy gates.
```

## Acceptance Rate Validation

```text
Use xllm-npu-eval-runner and xllm-npu-accuracy-debug to validate speculative
decoding acceptance rate for <model_name>.

Requirements:
- Use the same fixed prompt set for baseline and current code.
- Capture server counters before and after each run:
  speculative_num_accepted_tokens_total
  speculative_num_draft_tokens_total
- Do not rely only on client-side accepted-token estimates.
- Compare output quality with deterministic prompts and, when relevant, the
  target accuracy dataset subset.

Deliver a table with baseline/current accepted tokens, draft tokens, true accept
rate, TPOT, and accuracy status.
```
