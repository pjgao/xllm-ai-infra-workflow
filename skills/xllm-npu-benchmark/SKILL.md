---
name: xllm-npu-benchmark
description: 在华为昇腾 NPU 910B3 (A3) 上进行 xLLM 与 vLLM-Ascend 的公平推理基准测试。当用户需要对比两个框架在相同模型/工作负载/NPU/SLA 条件下的性能时使用。支持分层搜索、JSONL 数据集、SLA 验证、CSV/Markdown 导出。
---

# xLLM NPU 基准测试

当用户需要在华为昇腾 NPU 910B3 (A3) 上公平对比 xLLM 与 vLLM-Ascend 时使用。

工作流入口：
- [scripts/collect_evalscope_results.py](scripts/collect_evalscope_results.py) — 收集 evalscope 原始产物并归一化为 JSONL
- [scripts/compare_npu_benchmark.py](scripts/compare_npu_benchmark.py) — 结果对比
- [scripts/validate_framework_cli.py](scripts/validate_framework_cli.py) — CLI 验证

职责边界：
- `xllm-npu-eval-runner` 负责启动/复用服务、执行 evalscope 和收集原始 artifacts。
- 本 skill 负责判断这些 artifacts 是否公平、可比，并输出性能结论。
- 若需要定位 TPOT/TTFT 背后的 NPU timeline 根因，进入 `xllm-npu-profiler`。

## 前置条件

- xLLM 和 vLLM-Ascend 均可正常启动并服务目标模型
- 模型权重已下载到可见路径
- 目标明确：
  - 固定 QPS 基准测试，或
  - 搜索满足 `max_ttft_ms` / `max_tpot_ms` SLA 的最大 QPS

如果以上条件不满足，先解决后再启动大规模搜索。

环境一致性检查：
- 验证 `npu-smi info` 可见所有 NPU A3 设备且状态健康
- 验证 `ASCEND_RT_VISIBLE_DEVICES` 设置一致
- 确认 CANN >= 8.0.RC1，HDK Driver >= 25.2.0
- 验证两个框架 CLI `--help` 输出，确认关键参数可用
- 跑性能前必须先做 NPU 占用与环境波动门禁；不满足门禁时，只能作为 smoke/debug 数据，不能写入 PR 性能结论
- 所有性能测试默认必须配置请求级 warmup；evalscope 使用 `--warmup-num 1` 或更高，除非明确是在测冷启动/首请求性能。正式对比表必须记录 warmup 数。
- 正式 benchmark run 应保存可复查产物：启动 manifest、服务 PID/日志、`npu-smi` 前后快照、原始 benchmark 输出、结构化 `metrics.json` 或 `summary.json`、人工可读 `report.md`。缺少这些产物的结果只能作为 smoke/debug。
- 正式产物目录和字段参考 [`../../references/perf-artifact-schema.md`](../../references/perf-artifact-schema.md)；run 元信息参考 [`../../references/run-manifest-template.md`](../../references/run-manifest-template.md)。

## 分级测试参考

参考多框架推理工作台经验，性能和精度测试建议分级：

- 性能 `simple`：少量固定 shape，用于 smoke 和快速回归，不直接作为最终 SOTA 结论。
- 性能 `complex`：覆盖长输入、高输出、并发和目标业务 shape，用于正式对比。
- 精度 `sanity`：单 prompt，看输出是否是人话。
- 精度 `quick`：固定小样本数据集，保存原始预测和失败样本。
- 精度 `full`：完整 CEval/MMLU/GSM8K 等评测集，用于 PR 正确性结论。

报告必须写明当前属于哪个等级，不能把 `sanity/simple` 结论扩大成完整精度或性能结论。

历史 MTP benchmark 数值和复盘放在
[`references/mtp-benchmark-lessons.md`](references/mtp-benchmark-lessons.md)。
只有当任务涉及 MTP/speculative decoding 时再加载该 reference。

## 性能测试前环境门禁

任何 TTFT/TPOT/TPS 性能测试前，先判断目标 NPU 是否被其他任务、残留 context、CPU/内存波动污染。未通过门禁的结果不得用于 PR 描述、优化结论或 before/after 对比。

### 必采信息

在启动待测服务前创建 `RUN_ROOT/env/` 并保存：

```bash
mkdir -p "$RUN_ROOT/env"
date -Is | tee "$RUN_ROOT/env/precheck.time"
npu-smi info | tee "$RUN_ROOT/env/npu-smi.before.txt"
for id in 0 1 2 3; do
  npu-smi info -t usages -i "$id" | tee "$RUN_ROOT/env/npu-usages.before.npu${id}.txt"
done
pgrep -af 'xllm|vllm|sglang|python|evalscope|msprof' | tee "$RUN_ROOT/env/process.before.txt" || true
free -h | tee "$RUN_ROOT/env/mem.before.txt"
uptime | tee "$RUN_ROOT/env/load.before.txt"
```

启动服务后、正式压测前，再采 3 次空闲态样本，间隔 10 秒：

```bash
for round in 1 2 3; do
  date -Is | tee -a "$RUN_ROOT/env/idle-usages.after-service.txt"
  for id in 0 1 2 3; do
    npu-smi info -t usages -i "$id" | tee -a "$RUN_ROOT/env/idle-usages.after-service.txt"
  done
  sleep 10
done
pgrep -af 'xllm|vllm|sglang|python|evalscope|msprof' | tee "$RUN_ROOT/env/process.after-service.txt" || true
```

压测结束后保存同样的 `npu-smi info`、`npu-smi info -t usages`、`pgrep`、`free -h`、`uptime`，用于判断 run 中是否发生外部波动。

### 判定规则

- 目标卡的 `Health` 必须为 `OK`，`AICore/NPU Utilization` 在服务空闲时应稳定接近 0。
- 目标卡 HBM 占用只能来自本轮待测服务；启动服务前若已有大额 HBM 占用，必须确认来源，否则该卡不能用于正式性能结论。
- `npu-smi info` 进程表中若出现 `ps` 查不到的 PID，按残留/异常 NPU context 处理；先清理或换卡，不能直接做 before/after 结论。
- before/after 两组实验必须使用同一批设备、同一可见卡顺序、同一服务启动参数，并在每组之间重启服务或清状态。
- CPU load、swap、后台 profiling/编译任务必须记录；若压测期间 load 或 swap 明显波动，重跑并标记旧结果为污染样本。
- evalscope 客户端、服务端和 profiling 不要混跑；msprof 采集 run 只用于 profiling 分析，不和无 profiling 性能数直接对比。
- evalscope 正式性能测试必须使用请求级 warmup，例如 `--warmup-num 1` 或 `--warmup-num 2`。`--warmup-num 0` 只能用于冷启动/首请求分析，不能和带 warmup 的稳定性能数据直接比较。

### 2026-05-28 经验记录

Qwen35-27B TP=4 MTP=3 复用 `causal_conv1d` 验证前，目标逻辑卡 0-3 的服务空闲态 `AICore=0%`，但 HBM 仍在 76%-77%，且 `npu-smi info` 进程表中存在多个 `ps` 查不到的历史 PID。该状态只能说明服务当前未计算，不能证明环境干净；在这种环境下得到的 evalscope TPOT 不能作为 PR #1536 的最终性能证据。正确流程是先记录门禁结果，清理残留 context 或换用干净卡，再重启服务进行同参数 A/B 测试。

### MTP 启用门禁与 evalscope 指标陷阱

Qwen3.5 官方 checkpoint 内置原生 MTP 权重，但 xLLM serving 前必须先导出为独立 draft model。不要只看模型目录里有 `mtp.*` 权重，也不要只传 `--num_speculative_tokens`。

```bash
python3 tools/export_mtp.py \
  --input-dir <model-root>/Qwen35-27B \
  --output-dir <model-root>/Qwen35-27B-mtp \
  --model-type qwen3_5
```

导出后至少确认：
- draft 目录存在 `mtp_layer_parameters.safetensors`。
- draft config 的 `model_type` 为 `qwen3_5_mtp`。
- 启动参数同时包含 `--draft_model <model-root>/Qwen35-27B-mtp`、`--draft_devices="npu:<rank>"` 和 `--num_speculative_tokens N`。
- rank 日志出现 `draft_model_path: ...Qwen35-27B-mtp`、`Using draft devices: npu:<rank>`、`Speculative decode is enabled, algorithm: MTP`。缺少这些证据时，不能把该 run 记为外置 MTP draft run。

踩坑记录：
- 当前 xLLM 的外置 MTP draft 路径依赖 `draft_model_path`。`--num_speculative_tokens=3` 单独存在时，scheduler 可能按 speculative token budget 变化，但不会证明已经进入 `SpeculativeEngine` / `MTPWorkerImpl` 的 draft model 推理。
- evalscope 的 `Decoded Tok/Iter` 和 `Spec Accept Rate` 来自流式 chunk 数推导，近似公式是 `L=(completion_tokens-1)/(chunks-1)`、`p=1-1/L`，不是 xLLM 服务端真实 accepted token counter。
- 正式分析 MTP 接受率时，必须在 evalscope 前后抓 xLLM `/vars` 中的 `speculative_num_accepted_tokens_total` 和 `speculative_num_draft_tokens_total`，用 delta 计算服务端真实 `accepted/draft`。注意该值是“被接受 draft token / 下发 draft token”，与 evalscope 的 `1-1/L` 近似公式不是同一个口径；报告中要分别标注。
- 如果服务端把多个 token 聚合到一个 streaming chunk，evalscope 可能在没有外置 draft model 的情况下也显示 `Decoded Tok/Iter > 1` 和看似正常的接受率。对 `MTP=3`，若 `Decoded Tok/Iter` 明显超过 `num_speculative_tokens+1`，应优先怀疑 chunk 聚合或指标假象。
- 因此 MTP benchmark 报告必须把 evalscope 性能表、rank 启动日志、`/vars` counter delta 和必要的 profiling 证据一起保存；evalscope 接受率只能作为弱 sanity signal，不能单独作为 MTP 已启用或精度稳定的证据。

### Chunked prefill 注意事项

`--enable_chunked_prefill=true --max_tokens_per_chunk_for_prefill=256` 不一定意味着单请求 prefill 会被强制切成 256-token 多步执行。当前 xLLM chunk scheduler 在没有 decode 请求竞争且 `prompt_len < max_tokens_per_batch` 时，会在 `handle_remaining_budget` 中把剩余 token budget 继续补给该 prefill，最终仍可能一次处理完整 20k prompt。

验证方法：
- 使用 20k 输入 / 1 token 输出采集 prefill profiling，对比 no-chunk 与 chunk。
- 如果 `MatMulV3`、`FusedInferAttentionScore`、`hcom_allReduce_` 等关键算子 count 完全一致，说明 chunk 没真正发生。
- 要验证 chunk 的收益，应增加并发/混入 decode，或降低 `--max_tokens_per_batch` 到 4096/8192 以强制多步 prefill。

## 工作流

### Step 1: Preflight

```bash
npu-smi info
# 确认 NPU A3 设备数量、显存、驱动版本

python scripts/validate_framework_cli.py \
  --framework xllm \
  --model /path/to/model \
  --extra-flags "--tensor-parallel-size 4"

python scripts/validate_framework_cli.py \
  --framework vllm-ascend \
  --model /path/to/model \
  --extra-flags "--tensor-parallel-size 4 --enforce-eager"
```

保存环境信息到 `artifact/env.json`：
- NPU 型号、数量、驱动版本
- CANN 版本
- 框架 commit hash、Docker 镜像
- `ASCEND_RT_VISIBLE_DEVICES`

### Step 2: 规范化工作负载

统一 JSONL 格式，每行一个请求：

```json
{"prompt": "请总结这篇文章的要点。", "output_len": 256}
{"prompt": [{"role": "user", "content": "解释量子纠缠。"}], "output_len": 512}
```

可选字段：

```json
{
  "prompt": [{"role": "user", "content": "使用天气工具"}],
  "output_len": 256,
  "extra_request_body": {"temperature": 0.0, "top_p": 0.95},
  "timestamp": 1710000000,
  "metadata": {"source": "custom"}
}
```

### Step 2.5: evalscope 实测用法 (推荐)

使用 `evalscope` 客户端进行 OpenAI-compatible serving benchmark。完整命令模板
见 [`references/benchmark-runbook.md`](references/benchmark-runbook.md)。

必要要求：
- 数据集使用 line-by-line JSONL，每行是一个完整 OpenAI 请求 body。
- 正式性能对比必须设置 `--warmup-num 1` 或更高。
- 保存 `benchmark_summary.json`、`benchmark_percentile.json`、原始日志和 HTML 报告。
- 将 evalscope 结果归一化为 compare 输入：

```bash
python scripts/collect_evalscope_results.py \
  --root /path/to/evalscope/results \
  --framework xllm \
  --output-jsonl /path/to/xllm_results.jsonl \
  --output-summary /path/to/xllm_summary.md
```

该脚本会递归扫描 `benchmark_summary.json`，读取同目录的
`benchmark_percentile.json`，并输出 `compare_npu_benchmark.py` 可直接消费的
JSONL。可选 `--sla-ttft-ms` / `--sla-tpot-ms` 用于把 p99 延迟 SLA 写入
`sla_pass`。

默认场景：

| 场景 | input_len | output_len | 说明 |
|------|-----------|------------|------|
| chat | 1000 | 1000 | 常规对话 |
| summary | 8000 | 1000 | 长文总结 |
| long-context | 32000 | 1000 | 超长上下文 |

所有框架使用相同的：
- tokenizer 路径
- 精度（bf16 或指定量化）
- 量化方案（若启用）
- 采样参数（temperature=0 等）

### Step 3: 搜索层级

| 层级 | 说明 | 候选数上限 |
|------|------|-----------|
| Tier 1 (smoke) | 快速验证，少量候选 | ≤3 |
| Tier 2 (default) | 有界扫描，高优先级参数优先 | ≤10 |
| Tier 3 (exhaustive) | 穷举搜索 | ≤30 |

### Step 4: xLLM 调优

按同一 workload 和 SLA 扫描高优先级参数：

| 类别 | 例子 |
|------|------|
| 并行 | TP / PP / EP |
| 图模式 | eager / npugraph_ex / ge |
| KV Cache | block size / max model len / PA layout |
| 调度 | max seqs / chunk prefill / PD 分离 |
| 内存 | memory utilization / xTensor pool |
| 算法 | speculative decoding / EPLB |

记录每次运行的完整启动命令。保留失败候选及原因。

### Step 5: vLLM-Ascend 调优

相同工作负载和 SLA 下独立调优 vLLM-Ascend。不要把 xLLM 的最优参数直接套给
vLLM-Ascend，也不要拿 vLLM-Ascend default 和 tuned xLLM 比较。调优维度至少
覆盖 TP/PP、KV cache、graph/eager、prefix caching 和 chunked prefill。

### Step 6: 结果归一化

调用 `compare_npu_benchmark.py`：

```bash
python scripts/collect_evalscope_results.py \
  --root /path/to/xllm/evalscope/results \
  --framework xllm \
  --output-jsonl /path/to/xllm_results.jsonl

python scripts/collect_evalscope_results.py \
  --root /path/to/vllm/evalscope/results \
  --framework vllm-ascend \
  --output-jsonl /path/to/vllm_results.jsonl

python scripts/compare_npu_benchmark.py \
  --xllm-results /path/to/xllm_results.jsonl \
  --vllm-results /path/to/vllm_results.jsonl \
  --output-dir /path/to/comparison/
```

排序规则：SLA通过 > 请求吞吐 > 输出token吞吐 > p50 TTFT > p50 TPOT

输出：
- `comparison/summary.md` — Markdown 对比表
- `comparison/summary.csv` — CSV 数据
- `comparison/winning-commands.md` — 最优配置命令
- `comparison/candidates.jsonl` — 所有候选记录

## 公平性规则

详见 [references/npu-fairness-rules.md](references/npu-fairness-rules.md)。

核心原则：
- 相同 NPU 型号 (A3) / 卡数 / 模型权重 / tokenizer / 精度 / 量化 / 采样设置
- 记录 CANN 版本、HDK Driver 版本、框架 commit、Docker 镜像
- 候选之间重启服务或清除状态
- `ASCEND_RT_VISIBLE_DEVICES` 设置一致
- 保留失败候选及失败原因
- 永远不要将 tuned xLLM 与 vLLM-Ascend defaults 比较，或反之

## 配置和脚本模板

可选 YAML 配置和 evalscope shell 模板见
[`references/benchmark-runbook.md`](references/benchmark-runbook.md)。

## 中断与恢复

使用 `search.resume: true` 恢复中断的搜索：

```yaml
search:
  tier: 2
  resume: true
```

- 每次试验结果追加到 `live_results.jsonl`
- SIGINT/SIGTERM 会保存已有结果
- Resume 假设候选顺序和数据集未变

## 返回值

运行完成后返回：
- 使用的层级
- 数据集类型（合成/真实流量）
- xLLM 最优配置及性能
- vLLM-Ascend 最优配置及性能
- 是否满足 SLA
- 最佳 QPS
- 文件路径：prepared dataset JSONL、results.jsonl、results.csv、关键日志
