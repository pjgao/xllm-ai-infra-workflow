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
