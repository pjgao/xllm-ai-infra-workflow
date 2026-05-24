# 有效优化台账

记录有性能提升证据（benchmark + profiling）的优化项。

## Qwen3.5-27B（2026-05-23，910B3 A3，TP=2）

### 累积吞吐演进

| 阶段 | Output Throughput (tok/s, p=1) | 累计 Δ | 关键改动 |
|------|------------------------------|--------|---------|
| Baseline | 29.88 | - | 标准 xLLM baseline |
| MTP nst=1 | 36.11 | +20.9% | `--num_speculative_tokens 1` |
| MTP + Transpose 消除 | 39.54 | +32.3% | MTP conv transpose 消除 |

### 有效优化 #1：MTP 最优 nst 调参
- 收益：+20.9%（baseline → MTP）
- 方法：扫描 `num_speculative_tokens` ∈ {1,2,4}，发现 `nst=1` 是 910B3 上 27B 模型的唯一有效区间（更高 nst 反而降低吞吐）
- 证据：benchmark/mtp/parallel_1_number_5/20260523_170658/benchmark_summary.json
- 文档参考：commit `0de6b41`

### 有效优化 #2：MTP Transpose 消除（P2-1）
- 收益：+9.5%（MTP baseline → MTP+Transpose 消除）
- 累积收益：+32.3%（从 raw baseline 起算）
- 方法：对 `qwen3_gated_delta_net_base.cpp/h` 做三处修改
  1. 缓存 `conv_weight.transpose(0,1).contiguous()` → 成员变量
  2. `run_spec_verify_conv` 改为 `[B,T,C]` → `[B,T,C]`
  3. `process_mixed_qkv` 增加格式检测，spec_verify 跳过多余 transpose
- 证据：
  - **Benchmark**：`benchmark_summary.json` (Output 36.11→39.54 tok/s)
  - **msprof Kernel**：`op_statistic_*.csv` Transpose 14,400→960 calls
- 关键指标：
  - TPOT: 23.41→21.92 ms (-6.4%)
  - TTFT: 3953→3553 ms (-10.1%)
  - Transpose device time: 211.9→21.1 ms (-190.8 ms)
- 文档参考：commit `38b4eb6`，Section 9 of `skills/xllm-npu-profiler/references/qwen35-27b-kernel-profile.md`
- 补丁归档：`patches/qwen3_gated_delta_net_base.{cpp,h}`

## Qwen3.5-27B（2026-05-25，910B3 A3，TP=4，20k/1k random）

### 有效优化 #3：MTP=3 Transpose 消除（TileLang 回退后）
- 收益：Output TPS 66.82→69.31 (+3.7%)，TPOT 12.05→11.57 ms (-4.0%)
- 方法：恢复 MTP spec-verify Transpose 消除逻辑；为当前 CANN op_api 补 `beam_search_rec` 的 `topK` 参数以完成构建。
- 证据：
  - Benchmark: `/home/g00510989/xllm/runs/20260525_mtp3_transpose_opt/benchmark/random_20k_1k_parallel_1_number_5/evalscope.log`
  - Profiling: `Transpose` 505.08 ms / 36,701 calls → 205.86 ms / 12,604 calls
  - Accuracy: GSM8K `limit=10`, `mean_acc=1.0`
- 结论：在当前 TP=4 + MTP=3 + chunk prefill 的长输入场景，Transpose 消除仍然有效，但剩余瓶颈已转向 MatMul、AllReduce/AllGather 和 MTP accept/verify 小算子。

### 反例记录：MTP accept mask Torch-level 改写
- 结论：不计入有效优化，代码不提交到 xLLM。
- 尝试：
  - `cumprod` mask：Output TPS 68.87，TPOT 11.81 ms。
  - bool-prefix mask：Output TPS 68.69，TPOT 11.71 ms。
- 对照：transpose-opt baseline Output TPS 69.31，TPOT 11.57 ms。
- 经验：局部替换 `build_accepted_mask()` 的 eager Torch 小算子不足以带来收益；后续要做 accept/verify，应优先融合为单 kernel 或扩展现有 `kernel::rejection_sample` 路径。
