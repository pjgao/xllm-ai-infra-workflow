# PR #1541 MTP Draft Extend Preparation Overlap 最小化验证记录

日期：2026-05-25

## 背景

本轮验证目标是在 Preview 分支最新代码已回退 TileLang 相关修改的前提下，把 PR #1541 的修改最小化到 MTP draft extend preparation overlap 相关代码，并验证：

- 是否能正常构建；
- 10 条精度数据是否无异常；
- random 20k 输入 / 1k 输出场景性能是否劣化；
- 构建和验证过程中暴露的环境问题如何规避。

本地验证分支：

- 仓库：`/home/g00510989/xllm/xllm_pr1541_minimal`
- 分支：`pr1541-minimal`
- 提交：`eaff9517 perf: overlap mtp draft extend preparation`
- 主要改动：`xllm/core/runtime/mtp_worker_impl.{cpp,h}`

## 构建问题与经验

正常预期构建命令应是：

```bash
python setup.py build --device npu
```

本轮构建时间异常偏长，根因不在本次 PR 业务代码，而是环境和缓存问题叠加。

### 1. OPP 头文件与源码不匹配

现象：编译 `beam_search` 相关代码时，系统 OPP 头文件中的 `aclnnBeamSearchGroup` 签名仍是旧版本，包含额外 `topK` 参数；源码侧期望的是新签名。

检查方式：

```bash
grep -n "aclnnBeamSearchGroup" /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/include/aclnn_beam_search_group.h
```

处理经验：

- 不直接覆盖系统 `/usr/local/Ascend/.../opp/vendors/xllm`。
- 本轮使用 `/tmp/xllm_ops_pr1541/vendors/xllm` 作为临时 OPP vendor，并通过 `CMAKE_ARGS` 指向匹配的 header/lib。
- 验证报告中必须标注这种规避方式，避免把环境问题误读成代码问题。

### 2. build cache 命中错误架构 libtorch

现象：链接时报错：

```text
_deps/libtorch-src/lib/libtorch.so: file in wrong format
```

原因：`build/cmake.linux-aarch64-cpython-311/_deps/libtorch-src` 中缓存的是 x86-64 libtorch，而当前机器需要 aarch64 libtorch。

检查方式：

```bash
file build/cmake.linux-aarch64-cpython-311/_deps/libtorch-src/lib/libtorch.so
file /usr/local/lib64/python3.11/site-packages/torch/lib/libtorch.so
```

处理经验：

- 只修正 build artifact，不改源码。
- 本轮将错误架构缓存备份为 `.x86_64_bak`，并把 `_deps/libtorch-src` 指向系统 Python 环境中的 aarch64 torch。

### 3. Python.h 缺失

现象：编译扩展时找不到 `Python.h`。

处理经验：

```bash
export CPLUS_INCLUDE_PATH=/usr/include/python3.11
```

### 4. TileLang 目标重新生成导致构建时间拉长

虽然本次 PR 不依赖 TileLang 修改，但构建过程中可能触发已有 TileLang 目标或小算子重新生成，导致单次 build 明显慢于预期。

经验判断：

- 如果只改 MTP runtime 逻辑，长时间重编 TileLang 目标通常是构建缓存问题；
- 需要优先检查 build cache、OPP vendor 和 libtorch 架构，而不是先怀疑 PR 本身引入了大规模编译依赖。

## 验证方法

### 构建与单测

- build：通过
- `sampler_test`：13 个用例中 11 passed，2 skipped（MLU-only）

### 服务启动

物理 NPU：8, 9, 10, 11

关键启动参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
--tensor-parallel-size 4
--communication_backend=lccl
--enable_chunked_prefill=true
--max_tokens_per_chunk_for_prefill=256
--enable_schedule_overlap=true
--enable_graph=true
--enable_prefix_cache=false
--num_speculative_tokens 3
--max_tokens_per_batch=32768
--max_seqs_per_batch=16
--block_size=128
```

模型：

- target：`/home/data/weights/Qwen35-27B`
- draft：`/home/data/weights/Qwen35-27B-mtp`

短请求 smoke：通过。

### 性能 workload

使用 evalscope random 20k/1k：

- `parallel=1`
- `number=5`
- `temperature=0.0`
- `stream=true`
- 输入 token 目标：20k
- 输出 token：1k

结果目录：

```text
/home/g00510989/xllm/runs/20260525_pr1541_minimal_eaff9517/perf/random20k_1k_p1_n5/20260525_152615/Qwen35-27B
```

### 精度 workload

使用 GSM8K `limit=10`：

```text
/home/g00510989/xllm/runs/20260525_pr1541_minimal_eaff9517/accuracy/gsm8k_limit10
```

报告：

```text
/home/g00510989/xllm/runs/20260525_pr1541_minimal_eaff9517/accuracy/gsm8k_limit10/reports/Qwen35-27B/gsm8k.json
```

## 验证结果

### 性能

| 指标 | 结果 |
|------|------|
| Success | 5/5 |
| Avg Latency | 15.41s |
| TTFT | 2350.46ms |
| TPOT | 13.07ms |
| Output Throughput | 63.26 tok/s |
| Total Throughput | 1255.16 tok/s |
| Decode TPS | 76.51 tok/s |
| Decoded Tok/Iter | 3.08 |
| Spec Accept Rate | 67.5% |
| Avg Input Tokens | 18840.00 |
| Avg Output Tokens | 1000 |
| P99 Latency | 16.75s |
| P99 TTFT | 3000.62ms |
| P99 TPOT | 14.06ms |

说明：evalscope random 虽设置 20k 输入，但经过 chat template/tokenizer 后，本轮实际平均输入 token 为 18840，p99 接近 20000。

### 精度

| Workload | Result |
|----------|--------|
| GSM8K limit=10 | 10/10 correct |
| mean_acc | 1.0 |

GSM8K 10 条数据未发现精度异常。

## 性能是否劣化

### 对比 2026-05-24 MTP=3 基线

基线来自 skill 中记录的 Qwen3.5-27B TP=4 random 20k/1k MTP 扫描结果。

| 指标 | PR #1541 最小化 | MTP=3 基线 | 变化 |
|------|----------------|------------|------|
| Avg Latency | 15.41s | 14.564s | +5.8% |
| TTFT | 2350.46ms | 2524.8ms | -6.9% |
| TPOT | 13.07ms | 12.05ms | +8.5% |
| Output TPS | 63.26 | 66.82 | -5.3% |
| Decode TPS | 76.51 | 82.99 | -7.8% |
| Decoded Tok/Iter | 3.08 | 3.21 | -4.0% |
| Accept Rate | 67.5% | 68.8% | -1.3pp |

结论：相比历史 MTP=3 基线，decode 侧存在轻微性能回落信号，主要体现在 TPOT 变差、Output TPS 和 Decode TPS 下降；TTFT 反而更好。当前结果不能描述为大幅劣化，但应标记为可疑轻微劣化。

### 对比 no MTP 基线

| 指标 | PR #1541 最小化 MTP=3 | no MTP 基线 | 变化 |
|------|----------------------|-------------|------|
| Avg Latency | 15.41s | 21.155s | -27.2% |
| TTFT | 2350.46ms | 2507.1ms | -6.3% |
| TPOT | 13.07ms | 18.67ms | -30.0% |
| Output TPS | 63.26 | 46.39 | +36.4% |

结论：MTP=3 仍显著优于 no MTP。

### 与早期 async draft overlap 全量实验的关系

早期全量实验中还包含非最小化修改，性能结果更好。当前最小化分支相对早期全量实验偏低，但该对比不是严格同环境：

- 本轮是最小化分支；
- 当前构建环境有 OPP header 和 libtorch cache 规避；
- 日志中仍可能出现 fused TileLang kernel 路径，需要确认 official Preview 分支的实际小算子状态；
- evalscope random 的实际输入 token 数有波动；
- `number=5` 单轮测试方差较大。

因此最终 PR 结论应采用谨慎表述：已通过精度验证；相比 no MTP 仍有收益；相比历史 MTP=3 基线有轻微性能回落信号，建议在干净环境用同一二进制重跑 baseline 与 PR 进行最终裁决。

## 后续建议

1. 在 Preview 干净环境重新执行 `python setup.py build --device npu`，避免临时 OPP 和 libtorch cache 规避影响判断。
2. 用同一服务二进制分别跑 baseline commit 与 PR commit，固定 random seed、输入 token 统计和 NPU 映射。
3. 对 MTP=3 采集 profiling，重点看 draft extend preparation overlap 是否减少 host/device 空泡，同时确认是否引入新的同步点。
4. 若官方 Preview 已完全回退 TileLang 算子，验证日志中不应再依赖 fused TileLang kernel；否则需要把小算子状态作为结果解释的一部分。
