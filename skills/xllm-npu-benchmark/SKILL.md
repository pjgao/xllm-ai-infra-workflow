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

## 实测案例 (Qwen3.5-27B, 910B3 x2 TP=2, 2026-05-23)

**环境**: 192.168.13.154 / xllm-gpj 容器 / CANN 8.5.0 / torch_npu 2.7.1.post2  
**模型**: `/home/data/weights/Qwen35-27B` (64 层混合注意力) + MTP draft  
**Benchmark**: evalscope `line_by_line` plugin + `jd_openai_20k.jsonl` (20k 真实对话 + `max_tokens=2048, temperature=0.0, stream=true`)  
**结果**:

| 并发 | 总请求 | Output Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Avg Output Tokens | 成功 |
|-----|--------|--------------------------|-----------|-----------|------------------|------|
| 1   | 5      | 29.33                    | 3564      | 29.8      | 917              | 5/5  |
| 2   | 4      | 46.83                    | 4445      | 34.4      | 1308             | 4/4  |

结论：并发 2 相比并发 1，吞吐 +59.7% (46.83 vs 29.33 tok/s)。

详细结果：`benchmark/baseline/parallel_1_number_5/20260523_150528/` 和 `benchmark/baseline/parallel_2_number_4/20260523_151051/`。

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
  --input-dir /home/data/weights/Qwen35-27B \
  --output-dir /home/data/weights/Qwen35-27B-mtp \
  --model-type qwen3_5
```

导出后至少确认：
- draft 目录存在 `mtp_layer_parameters.safetensors`。
- draft config 的 `model_type` 为 `qwen3_5_mtp`。
- 启动参数同时包含 `--draft_model /home/data/weights/Qwen35-27B-mtp`、`--draft_devices="npu:<rank>"` 和 `--num_speculative_tokens N`。
- rank 日志出现 `draft_model_path: ...Qwen35-27B-mtp`、`Using draft devices: npu:<rank>`、`Speculative decode is enabled, algorithm: MTP`。缺少这些证据时，不能把该 run 记为外置 MTP draft run。

踩坑记录：
- 当前 xLLM 的外置 MTP draft 路径依赖 `draft_model_path`。`--num_speculative_tokens=3` 单独存在时，scheduler 可能按 speculative token budget 变化，但不会证明已经进入 `SpeculativeEngine` / `MTPWorkerImpl` 的 draft model 推理。
- evalscope 的 `Decoded Tok/Iter` 和 `Spec Accept Rate` 来自流式 chunk 数推导，近似公式是 `L=(completion_tokens-1)/(chunks-1)`、`p=1-1/L`，不是 xLLM 服务端真实 accepted token counter。
- 正式分析 MTP 接受率时，必须在 evalscope 前后抓 xLLM `/vars` 中的 `speculative_num_accepted_tokens_total` 和 `speculative_num_draft_tokens_total`，用 delta 计算服务端真实 `accepted/draft`。注意该值是“被接受 draft token / 下发 draft token”，与 evalscope 的 `1-1/L` 近似公式不是同一个口径；报告中要分别标注。
- 如果服务端把多个 token 聚合到一个 streaming chunk，evalscope 可能在没有外置 draft model 的情况下也显示 `Decoded Tok/Iter > 1` 和看似正常的接受率。对 `MTP=3`，若 `Decoded Tok/Iter` 明显超过 `num_speculative_tokens+1`，应优先怀疑 chunk 聚合或指标假象。
- 因此 MTP benchmark 报告必须把 evalscope 性能表、rank 启动日志、`/vars` counter delta 和必要的 profiling 证据一起保存；evalscope 接受率只能作为弱 sanity signal，不能单独作为 MTP 已启用或精度稳定的证据。

## 实测案例 (Qwen3.5-27B, 910B3 x4 TP=4, random 20k/1k, 2026-05-24)

**环境**: xLLM `/home/g00510989/xllm/xllm`, commit `f514ad94 回退tilelang`  
**模型**: `/home/data/weights/Qwen35-27B` + `/home/data/weights/Qwen35-27B-mtp`  
**Benchmark**: evalscope `random`, `min/max-prompt-length=20000`, `min/max-tokens=1000`, `parallel=1`, `number=5`, `temperature=0.0`, `stream=true`  
**Warmup 经验补充**: 该历史数据未显式记录 evalscope warmup；后续复现和新性能结论必须加 `--warmup-num 1` 或更高，避免首条请求、graph lazy capture、缓存状态污染平均 TPOT。
**启动共性**: TP=4, `--communication_backend=lccl`, `--enable_graph=true`, `--enable_schedule_overlap=true`, `--enable_prefix_cache=false`, `--enable_chunked_prefill=true`, `--max_tokens_per_chunk_for_prefill=256`, `--max_tokens_per_batch=32768`, `--max_seqs_per_batch=16`, `--block_size=128`

| 配置 | Avg Latency | TTFT (ms) | TPOT (ms) | Output TPS | Decode TPS | Decoded Tok/Iter | Accept Rate |
|------|-------------|-----------|-----------|------------|------------|------------------|-------------|
| no MTP | 21.155s | 2507.1 | 18.67 | 46.39 | 53.56 | 1.01 | 1.4% |
| MTP=1 | 19.724s | 2566.2 | 17.18 | 49.69 | 58.21 | 1.88 | 46.8% |
| MTP=2 | 17.667s | 2524.0 | 15.16 | 55.35 | 65.96 | 2.64 | 62.1% |
| MTP=3 | 14.564s | 2524.8 | 12.05 | 66.82 | 82.99 | 3.21 | 68.8% |
| MTP=4 | 15.861s | 2623.8 | 13.25 | 61.49 | 75.47 | 3.57 | 72.0% |
| MTP=5 | 17.195s | 2585.0 | 14.62 | 56.83 | 68.40 | 3.38 | 70.4% |

结论：
- 对该单并发 20k/1k random 场景，`num_speculative_tokens=3` 最优；MTP=4/5 虽然 accept rate 更高，但 draft/verify 开销反而拉低端到端吞吐。
- MTP 加深会线性增加主模型 linear cache 预留：MTP=1/2/3/4/5 的 `reserved_linear_bytes` 约为 2.28/3.41/4.54/5.68/6.81 GB；长上下文和高并发下要同时检查 KV blocks 与 HBM 余量。
- 该场景 TTFT 主要由 20k prefill 决定，MTP 主要改善 decode/TPOT；不要用 TTFT 判断 MTP 是否有效。

### PR #1541 最小化验证补充 (2026-05-25)

**环境**: xLLM `/home/g00510989/xllm/xllm_pr1541_minimal`, commit `eaff9517 perf: overlap mtp draft extend preparation`

**模型/工作负载**: `/home/data/weights/Qwen35-27B` + `/home/data/weights/Qwen35-27B-mtp`, TP=4, MTP=3, random 20k/1k, `parallel=1`, `number=5`, `temperature=0.0`, `stream=true`

**启动关键参数**: `--enable_chunked_prefill=true --max_tokens_per_chunk_for_prefill=256 --enable_schedule_overlap=true --enable_graph=true --enable_prefix_cache=false --num_speculative_tokens 3`

| 指标 | PR #1541 最小化 | 2026-05-24 MTP=3 基线 | 变化 |
|------|----------------|------------------------|------|
| Avg Latency | 15.41s | 14.564s | +5.8% |
| TTFT | 2350.46ms | 2524.8ms | -6.9% |
| TPOT | 13.07ms | 12.05ms | +8.5% |
| Output TPS | 63.26 | 66.82 | -5.3% |
| Decode TPS | 76.51 | 82.99 | -7.8% |
| Decoded Tok/Iter | 3.08 | 3.21 | -4.0% |
| Accept Rate | 67.5% | 68.8% | -1.3pp |

准确性验证：GSM8K `limit=10`，10/10 正确，`mean_acc=1.0`。

性能判断：
- 相比 no MTP 基线仍有明显收益：Output TPS 63.26 vs 46.39 (+36.4%)，TPOT 13.07ms vs 18.67ms (-30.0%)，Avg Latency 15.41s vs 21.155s (-27.2%)。
- 相比 2026-05-24 的 MTP=3 历史基线，decode 侧存在轻微回落：Output TPS -5.3%，TPOT +8.5%，Decode TPS -7.8%；TTFT 反而更好。
- 相比早期 async draft overlap 全量实验结果，最小化分支吞吐也偏低；但该对比不是严格 apples-to-apples，需注意二进制、OPP/header、TileLang kernel 状态、evalscope random 实际 token 数和单轮 `number=5` 方差。
- 结论写法建议：该改动的核心卖点是 MTP draft preparation 提前下发/调度 overlap。当前数据没有证明该特性带来正收益，反而在 decode throughput/TPOT 上出现回落信号。因此不应作为性能优化 PR 继续推进合入；应先 hold/draft，基于 profiling 找到没有收益的原因，并用同一环境同一二进制重跑 baseline 与 PR。只有 A/B 数据显示稳定收益后才恢复提交。
- 源码复核：当前提交没有真正提前下发 draft model compute；`draft_impl_->step_async(extend_input)` 仍在 target validate 完成后执行。它只提前了 `ForwardInput::to()`、CPU view/copy 和少量元数据准备，收益上限很薄，还可能把 D2H/H2D 或 stream 同步插入 target compute 期间。
- 追加短 decode smoke：random 32/200 `number=1` 下 PR #1541 minimal 为 Output TPS 67.99、TPOT 9.94ms、Decode TPS 100.60；低于 transpose-opt profiling 的 69.70/9.90/101.03，不能证明调度提交有收益。

详细记录：`docs/pr-1541-mtp-draft-overlap-minimal-validation.md`。

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

使用 `evalscope` 客户端 (容器内已安装) 进行端到端 benchmark。支持 OpenAI 兼容 API 的 xLLM/vLLM-Ascend 实例。

**数据集格式** (JSONL, 每行是一个完整的 OpenAI 请求 body):

```json
{"model": "Qwen35-27B", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 2048, "temperature": 0.0, "stream": true}
{"model": "Qwen35-27B", "messages": [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "总结..."}], "max_tokens": 2048, "temperature": 0.0, "stream": true}
```

**命令**:

```bash
evalscope perf \
  --model Qwen35-27B \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --api openai \
  --dataset line_by_line \
  --dataset-path /path/to/jd_openai_20k.jsonl \
  --parallel 1 \
  --number 5 \
  --warmup-num 1 \
  --connect-timeout 120 \
  --read-timeout 300 \
  --outputs-dir /path/to/results/
```

**输出**:
- `benchmark_summary.json` — 汇总指标 (throughput, TTFT, TPOT, etc.)
- `benchmark_percentile.json` — 延迟百分位(P50/P99)
- `benchmark_data.db` — SQLite 请求明细
- `benchmark.log` — 日志
- `perf_report.html` — HTML 报告

**归一化为 compare 输入**:

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

**参数说明**:
- `--dataset line_by_line`: 使用 line-by-line plugin，逐行发送 JSONL 中的完整请求 body
- `--parallel N`: 并发数
- `--number N`: 总请求数
- `--warmup-num N`: 请求级 warmup 次数；正式性能对比必须 `N >= 1`，并在报告中写明。`N=0` 只用于冷启动/首请求分析。
- `--connect-timeout 120`: 连接超时(秒), xLLM cold start 需要更长时间
- `--read-timeout 300`: 读取超时(秒), 长序列生成需要更长时间

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

调优维度：

**并行策略**：
- `--tensor-parallel-size`: TP 大小
- `--pipeline-parallel-size`: PP 大小
- `--expert-parallel-size`: EP 大小（MoE 模型）

**图模式**：
- `--graph-mode`: `eager` / `npugraph_ex` / `ge`
- Prefill 默认 eager，Decode 默认 npugraph_ex

**KV Cache**：
- `--block-size`: Block 大小（PA 模式）
- `--max-model-len`: 最大模型长度
- KV Cache NZ 格式

**调度**：
- `--max-num-seqs`: 最大并发序列数
- `--chunked-prefill-size`: 分块 prefill 大小
- PD 分离策略

**内存**：
- `--gpu-memory-utilization`: GPU 显存利用率
- xTensor 内存池参数

**算法**：
- `--speculative-model`: 投机解码 draft 模型
- `--eplb-strategy`: 动态 EPLB 策略

记录每次运行的完整启动命令。保留失败候选及原因（如 OOM：建议增加 NPU 卡数或使用更大显存）。

### Step 5: vLLM-Ascend 调优

相同工作负载和 SLA 下调优：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /path/to/model \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --max-model-len 8192 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

调优维度：
- TP/PP/KV cache 配置
- graph mode 开关
- prefix caching
- chunked prefill

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

## 配置文件

示例 `qwen3-32b-a3.yaml`：

```yaml
model:
  path: /models/Qwen3-32B
  tokenizer: /models/Qwen3-32B
  precision: bf16

npu:
  device: A3
  count: 4
  visible_devices: "0,1,2,3"

dataset:
  kind: random
  scenarios:
    - name: chat
      input_len: 1000
      output_len: 1000
    - name: summary
      input_len: 8000
      output_len: 1000

benchmark:
  num_prompts: 80
  qps:
    search: true
    max_rounds: 5
  sla:
    max_ttft_ms: 500
    max_tpot_ms: 50

search:
  tier: 2
  max_candidates: 8

frameworks:
  xllm:
    base_flags:
      tensor-parallel-size: 4
      graph-mode: npugraph_ex
      block-size: 128
    search_space:
      chunked-prefill-size: [256, 512, 1024]
      max-num-seqs: [64, 128, 256]
      speculative-model: ["", "/models/draft-2b"]
  vllm-ascend:
    base_flags:
      tensor-parallel-size: 4
      enforce-eager: true
      block-size: 128
      gpu-memory-utilization: 0.9
    search_space:
      enable-prefix-caching: [true, false]
```

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

## 实测脚本: bench.sh

封装的 benchmark 脚本，支持 baseline / MTP 模式切换：

```bash
#!/bin/bash
MODE=${1:-baseline}
PARALLEL=${2:-1}
NUMBER=${3:-5}
RUN_ROOT=/home/g00510989/runs/20260523_qwen35_27b_npu_sota
DATASET=$RUN_ROOT/datasets/jd_openai_20k.jsonl

if [ "$MODE" = "mtp" ]; then
  URL=http://127.0.0.1:18170/v1/chat/completions
else
  URL=http://127.0.0.1:18160/v1/chat/completions
fi

OUT_DIR=$RUN_ROOT/benchmark/${MODE}/parallel_${PARALLEL}_number_${NUMBER}
mkdir -p $OUT_DIR

evalscope perf \
  --model Qwen35-27B \
  --url $URL \
  --api openai \
  --dataset line_by_line \
  --dataset-path $DATASET \
  --parallel $PARALLEL \
  --number $NUMBER \
  --warmup-num 1 \
  --connect-timeout 120 \
  --read-timeout 300 \
  --outputs-dir $OUT_DIR
```

**用法**:

```bash
./bench.sh baseline 1 5    # baseline 模式, 并发 1, 5 请求
./bench.sh mtp 2 4         # MTP 模式, 并发 2, 4 请求
```
