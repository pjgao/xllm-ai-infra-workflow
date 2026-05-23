---
name: xllm-npu-benchmark
description: 在华为昇腾 NPU 910B3 (A3) 上进行 xLLM 与 vLLM-Ascend 的公平推理基准测试。当用户需要对比两个框架在相同模型/工作负载/NPU/SLA 条件下的性能时使用。支持分层搜索、JSONL 数据集、SLA 验证、CSV/Markdown 导出。
---

# xLLM NPU 基准测试

当用户需要在华为昇腾 NPU 910B3 (A3) 上公平对比 xLLM 与 vLLM-Ascend 时使用。

工作流入口：
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
