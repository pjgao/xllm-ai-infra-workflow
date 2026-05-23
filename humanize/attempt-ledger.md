# Attempt 台账

## Qwen3.5-27B（模型：Qwen35-27B，TP=2，Phy 14/15）

### Attempt #1 — MTP 参数搜索（失败/非性能提升）
- 日期：2026-05-23
- 方向：MTP nst (num_speculative_tokens) 参数扫描
- 修改文件：scripts/mtp.sh 启动参数
- 结果：成功确定最优配置 `nst=1`，吞吐 +20-23% vs baseline
- benchmark 文件：benchmark/mtp/parallel_1_number_5/20260523_170658/
- 状态：merged（doc update commit `0de6b41`）

### Attempt #2 — `--enable_multi_stream_parallel` 多流并行（失败，功能缺失）
- 日期：2026-05-23
- 方向：P0 AllReduce-Compute Overlap 探索
- 修改文件：启动脚本新增 `--enable_multi_stream_parallel` flag
- 结果：失败 — xLLM 该版本 (`82a407db`) 未实现该 flag，服务启动直接 crash
- 状态：放弃

### Attempt #3 — P2-1 MTP Transpose 消除（成功，+9.5% 吞吐）
- 日期：2026-05-23
- 方向：P2-1 MTP 专属 Transpose kernel 消除
- 修改文件：
  - `xllm/xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.h`
  - `xllm/xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp`
- 关键改动：
  1. 缓存 `conv_weight.transpose(0,1).contiguous()` 为成员变量 `conv_weight_transposed_`
  2. `run_spec_verify_conv` 改为收发 `[B,T,C]` 格式，消除 round-trip transpose
  3. `process_mixed_qkv` 增加格式检测，spec_verify 路径跳过多余 transpose
- 验证：
  - 编译：99/99 steps 通过，生成新 binary
  - Smoke test：服务 25s 内 ready，单请求正常
  - **msprof kernel 验证**：Transpose kernel 14,400→960 calls（**-93.3%**），211.9→21.1ms（**-190.8ms**）
  - **Benchmark 验证**：36.11→39.54 tok/s（**+9.5%**），TPOT 23.41→21.92ms（**-6.4%**），TTFT 3953→3553ms（**-10.1%**）
- benchmark 文件：
  - baseline: benchmark/mtp/parallel_1_number_5/20260523_170658/
  - optimized: benchmark/mtp-transpose/parallel_1_number_5/20260523_212513/
- profiler 文件：profiles/mtp-transpose/PROF_000001_20260523220245172_EFHGHGGLONHMFIOA
- 状态：merged，commit `38b4eb6`
