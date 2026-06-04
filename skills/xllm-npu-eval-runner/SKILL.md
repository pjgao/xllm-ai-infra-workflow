---
name: xllm-npu-eval-runner
description: xLLM NPU evaluation runner. Use when the user wants to execute xLLM service startup, evalscope performance runs, accuracy runs, and artifact collection. This skill produces raw perf/accuracy artifacts; use xllm-npu-benchmark for fairness and baseline comparison, xllm-npu-accuracy-debug for accuracy root cause analysis, and xllm-npu-profiler for msprof analysis.
---

# xLLM NPU Evaluation Runner

This skill is the execution layer for xLLM NPU evaluation. It starts or reuses
an xLLM service, runs evalscope performance and accuracy workloads, and writes
reproducible artifacts.

It does not own final performance fairness conclusions, accuracy root-cause
analysis, or profiling interpretation:

| Need | Use |
|---|---|
| Start service and run evalscope | this skill |
| Compare baseline/current or framework A/B | `xllm-npu-benchmark` |
| Debug bad answers, CEval drops,乱码 | `xllm-npu-accuracy-debug` |
| Collect and analyze msprof traces | `xllm-npu-profiler` |
| Drive end-to-end optimization loop | `xllm-npu-sota-loop` |

## Workflow Overview

```
1. Parameter Alignment (ask user)
       |
2. Update 3 Scripts (run.sh, eval_perf.sh, eval_acc.sh)
       |
3. Check Dependencies (evalscope, evalscope[perf])
       |
4. Create Run Root and Manifest
       |
5. Start xLLM Service (skip if already running)
       |
6. Wait for Service Ready
       |
7. Run Performance Test (eval_perf.sh)
       |
8. Run Accuracy Test (eval_acc.sh)
       |
9. Write Metrics and Report
```

For formal conclusions, hand the artifacts to `xllm-npu-benchmark` or
`xllm-npu-accuracy-debug` after this runner finishes.

## Step 1: Parameter Alignment

Before doing anything, use the Question tool to confirm these parameters with the user:

| Parameter | Description | Affects |
|---|---|---|
| **API URL** | Service endpoint (e.g. `http://localhost:18050/v1`) | All 3 scripts |
| **Model Name** | Model identifier (e.g. `Qwen35-27B`) | All 3 scripts |
| **Model Path** | Main model weights path | `run.sh` `--model`, `eval_perf.sh` `--tokenizer-path` |
| **Draft Model Path** | Speculative decoding draft model path | `run.sh` `--draft_model` |
| **xLLM Binary Path** | Path to xllm server binary | `run.sh` `XLLM_BIN` variable |
| **TP (NNODES)** | Tensor parallelism degree | `run.sh` `NNODES` variable |
| **NPU Devices** | NPU device IDs to use (e.g. `0,1,2,3`) | `run.sh` `ASCEND_RT_VISIBLE_DEVICES` variable |
| **Test Mode** | Smoke test (quick validation) or Full test | `eval_perf.sh` (skip parallel=5), `eval_acc.sh` (subset datasets) |

Ask all parameters in one batch. Provide sensible defaults based on current script values:
- API URL: `http://localhost:18050/v1`
- Model Name: `Qwen35-27B`
- Model Path: `<model-root>/Qwen35-27B`
- Draft Model Path: `<model-root>/Qwen35-27B-mtp`
- xLLM Binary Path: `<project_root>/xllm/build/xllm/core/server/xllm`
- TP: `4`
- NPU Devices: `0,1,2,3`
- Test Mode: `smoke` (recommended for quick validation)

## Step 2: Update Scripts

After collecting parameters, update all 3 scripts **atomically** (all at once).

### Script Locations (relative to project root)

- **Startup**: `.opencode/skills/xllm-npu-eval-runner/scripts/run.sh`
- **Performance**: `.opencode/skills/xllm-npu-eval-runner/scripts/eval_perf.sh`
- **Accuracy**: `.opencode/skills/xllm-npu-eval-runner/scripts/eval_acc.sh`

### run.sh Updates

Use the Edit tool to update these fields in `scripts/run.sh`:

1. `MODEL_PATH="<model_path>"`
2. `DRAFT_MODEL_PATH="<draft_model_path>"`
3. `XLLM_BIN="<xllm_binary_path>"`
4. `NNODES=<tp>`
5. `ASCEND_RT_VISIBLE_DEVICES=<npu_devices>`
6. `START_PORT` should match the port from the API URL

### eval_perf.sh Updates

Update **both** `evalscope perf` command blocks (parallel=1 and parallel=5) in `scripts/eval_perf.sh`:

1. `--model <model_name>`
2. `--url <api_url>/chat/completions` — note: append `/chat/completions` to base URL
3. `--tokenizer-path <model_path>`
4. `SMOKE_MODE` variable at top of script:
   - **Smoke test**: set `SMOKE_MODE="true"` (skips parallel=5 test)
   - **Full test**: set `SMOKE_MODE="false"` (runs both parallel=1 and parallel=5)

### eval_acc.sh Updates

Update `scripts/eval_acc.sh`:

1. `--model <model_name>`
2. `--api-url <api_url>`
3. `--datasets` parameter (based on Test Mode):
   - **Smoke test**: `--datasets ceval --dataset-args '{"ceval": {"subset_list": ["computer_network", "operating_system", "marxism"]}}'`
   - **Full test**: `--datasets ceval`

## Step 3: Check Dependencies

Before starting the service, verify `evalscope` is installed:

```bash
pip show evalscope > /dev/null 2>&1 || pip install evalscope
python3 -c "import evalscope.perf" 2>/dev/null || pip install evalscope[perf]
```

If either check fails, auto-install the missing package. Only proceed to Step 4 after both are confirmed available.

## Step 4: Create Run Root and Manifest

Create a run root before service startup:

```bash
RUN_ROOT="${RUN_ROOT:-runs/eval/$(date +%Y%m%d_%H%M%S)_xllm_npu_eval}"
mkdir -p "$RUN_ROOT"/{env,service,perf,accuracy}
```

Write `manifest.md` using
[`references/run-manifest-template.md`](../../references/run-manifest-template.md).
At minimum record:

- xLLM branch, commit, and dirty diff state.
- Model path, optional draft model path, tokenizer path.
- Device ids, CANN/driver/torch_npu versions if available.
- Service startup command and API URL.
- Workload shape, sampling parameters, warmup count, parallel, and number.
- Whether this run is `smoke`, `quick`, or `full`.

Save pre-run environment snapshots:

```bash
npu-smi info > "$RUN_ROOT/env/npu-smi.before.txt"
pgrep -af 'xllm|vllm|sglang|python|evalscope|msprof' > "$RUN_ROOT/env/process.before.txt" || true
free -h > "$RUN_ROOT/env/mem.before.txt"
uptime > "$RUN_ROOT/env/load.before.txt"
```

## Step 5: Start xLLM Service

Before starting, check if the service is already running:

```bash
if curl -s <api_url>/models > /dev/null 2>&1; then
  echo "xLLM service already running, skipping startup."
else
  echo "Starting xLLM service..."
  bash .opencode/skills/xllm-npu-eval-runner/scripts/run.sh
fi
```

If the service is already available, skip to Step 7 (Run Performance Test).

**Important**: The service starts in background. After launching, wait for it to be ready.

## Step 6: Wait for Service Ready

Poll the service health endpoint until it responds:

```bash
for i in $(seq 1 60); do
  if curl -s <api_url>/models > /dev/null 2>&1; then
    echo "Service is ready!"
    break
  fi
  echo "Waiting for service... ($i/60)"
  sleep 10
done
```

If the service doesn't start within 10 minutes, check `log/node_0.log` for errors and report to the user.

## Step 7: Run Performance Test

```bash
bash .opencode/skills/xllm-npu-eval-runner/scripts/eval_perf.sh
```

The performance test behavior depends on Test Mode:
- **Smoke mode** (`SMOKE_MODE="true"`): Only runs parallel=1, number=4 (single-request latency baseline)
- **Full mode** (`SMOKE_MODE="false"`): Runs both rounds:
  1. **Parallel=1, Number=4**: Single-request latency baseline
  2. **Parallel=5, Number=20**: Concurrent throughput test

Results are output to `outputs/` by default. For formal runs, copy or configure
outputs under `$RUN_ROOT/perf/` and keep the raw evalscope directory intact.
Look for `benchmark_summary.json` files and mirror key fields into
`$RUN_ROOT/perf/metrics.json`.

Formal performance runs must use request-level warmup. For evalscope, set
`--warmup-num 1` or higher unless the user explicitly asks for cold-start
latency. Record the warmup value in `manifest.md` and `metrics.json`.

## Step 8: Run Accuracy Test

```bash
bash .opencode/skills/xllm-npu-eval-runner/scripts/eval_acc.sh
```

**Important**: Use Bash tool with `timeout: 3600000` (1 hour) when executing this command. Accuracy evaluation takes significantly longer than performance tests.

Accuracy results are printed to stdout. For formal runs, save raw predictions,
failed cases, score files, and a short `report.md` under `$RUN_ROOT/accuracy/`.
Use [`references/accuracy-artifact-schema.md`](../../references/accuracy-artifact-schema.md)
for the required artifact shape.

## Step 9: Write Metrics and Report

This runner should write a compact execution report:

```text
$RUN_ROOT/
  manifest.md
  env/
  service/
  perf/
  accuracy/
  report.md
```

The report should say what was executed, where raw artifacts are stored, and
whether the run is strong enough for a formal conclusion. If it is only a smoke
run, say so explicitly.

## Optional: Fetch Baseline from GitHub

Fetch the benchmark baseline data from the GitHub repository:

```
BENCHMARK_URL=https://raw.githubusercontent.com/jd-opensource/xllm/main/docs/benchmark/baseline.md
```

Use the WebFetch tool to retrieve this URL. If the URL returns a 404, inform the user that the baseline file hasn't been uploaded yet and skip the comparison step.

Parse the markdown tables to extract baseline values for the matching model and configuration.

Baseline comparison here is only a convenience check. Formal benchmark
comparison belongs in `xllm-npu-benchmark`, which validates fairness,
environment gates, warmup, and comparable startup parameters.

## Optional: Quick Compare Table

Build a comparison table and present it to the user:

### Performance Comparison Template

```
| Metric | Current | Baseline | Delta | Status |
|---|---|---|---|---|
| Output Throughput (tok/s) | XX.XX | XX.XX | +X.X% | PASS/FAIL |
| TTFT (ms) | XXXX | XXXX | -X.X% | PASS/FAIL |
| TPOT (ms) | XX.XX | XX.XX | +X.X% | PASS/FAIL |
| ITL (ms) | XX.XX | XX.XX | +X.X% | PASS/FAIL |
```

### Accuracy Comparison Template

```
| Dataset | Current | Baseline | Delta | Status |
|---|---|---|---|---|
| ceval (overall) | XX.X% | XX.X% | +X.X% | PASS/FAIL |
```

### Status Rules

- **Performance metrics** (throughput, tok/s): PASS if current >= baseline * 0.95 (within 5% tolerance)
- **Latency metrics** (TTFT, TPOT, ITL, ms): PASS if current <= baseline * 1.05 (within 5% tolerance)
- **Accuracy metrics**: PASS if current >= baseline - 0.02 (within 2 percentage points)

### Report Summary

After the tables, provide a one-line summary:
- All PASS: "All metrics within acceptable range of baseline."
- Any FAIL: "WARNING: X metrics below baseline. Check [specific metrics] for details."

## Troubleshooting

- **Service won't start**: Check `log/node_*.log` files for errors. Common issues: port conflicts, insufficient NPU memory, wrong model path.
- **Performance test fails**: Ensure the service is fully ready before running. Check if the URL is correct.
- **Accuracy test fails**: Verify evalscope is installed (`pip show evalscope`). Check API connectivity.
- **Baseline not found**: The GitHub baseline file may not exist yet. Ask the user to upload `benchmark_baseline.md` to the repo.
