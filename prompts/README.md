# Prompts

These prompts are task-entry templates for agent runs. They are intentionally
shorter than skills: a prompt starts a concrete run, while a skill defines the
workflow, evidence contract, and validation gates.

Use them by replacing the placeholders, then paste the prompt into Codex,
Claude Code, opencode, or another compatible coding agent with this repository's
skills installed.

## Prompt Index

| Prompt | Use When |
|---|---|
| [`xllm-npu-sota-loop-prompts.md`](xllm-npu-sota-loop-prompts.md) | Start an end-to-end NPU performance optimization loop. |
| [`xllm-npu-pr-fix-prompts.md`](xllm-npu-pr-fix-prompts.md) | Fix an xLLM PR regression, accuracy issue, crash, or review finding. |
| [`xllm-npu-op-migration-prompts.md`](xllm-npu-op-migration-prompts.md) | Migrate or optimize an NPU operator path with evidence gates. |

## Rules

- Fill in model, hardware, workload, framework commits, and artifact root before
  starting.
- For performance work, require warmup and keep formal performance runs separate
  from profiling runs.
- For accuracy work, start from a small deterministic reproducer, then scale to
  dataset subsets and full tasks.
- Every prompt should end with a record step: update the run ledger, case study,
  model PR history, or relevant skill reference.
