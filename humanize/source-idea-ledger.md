# 源码想法出处台账

记录每个源码想法的出处：PR / 论文 / 文档 / profiling 推断。

## Qwen3.5-27B

### 2026-05-23 — P2-1 Transpose 消除

- 出处：**profiling 推断**
- 推理链：
  1. 对比 MTP vs Baseline 的 `op_statistic_*.csv`，发现 MTP 专属新增 `Transpose_be83..._high_performance_13` kernel 占 MTP 总 device time 的 **7.2%**（207.8ms / 14,400 calls）
  2. 14,400 = 1,200 steps × 12 calls/step，定位到 `qwen3_gated_delta_net_base.cpp` 的 DeltaNet conv transpose round-trip
  3. 代码走读发现：每次 decode step 有 6 次 transpose，其中 4 次可以通过格式调整消除
- 实施：缓存 weight transpose + 修改 conv 输入输出格式 + 在 `process_mixed_qkv` 中增加格式检测
- 关联文档：`skills/xllm-npu-profiler/references/qwen35-27b-kernel-profile.md` Section 9
