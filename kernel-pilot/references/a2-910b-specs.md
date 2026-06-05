# 昇腾 910B (A2) 硬件规格

> 用于 kernel-pilot 在 Ascend 910B / A2 上做算子设计时的硬件约束参考。
> 不同整机、固件、CANN 版本可能存在差异；正式性能结论应以当前机器
> `npu-smi info`、CANN 文档和 profiling artifact 为准。

## 芯片规格

| 参数 | 规格 |
|------|------|
| 芯片型号 | Ascend 910B |
| 常用代号 | A2 |
| AICore 数量 | 约 32（以实际 `npu-smi info` / profiler 为准） |
| AICore 架构 | Da Vinci 系列 |
| FP16/BF16 算力 | 以当前 SKU 官方规格为准 |
| INT8 算力 | 以当前 SKU 官方规格为准 |
| HBM 容量 | 常见 64GB 级别，按实际卡型确认 |
| HBM 带宽 | 以当前 SKU 官方规格为准 |

## 存储层次

| 层级 | 说明 |
|------|------|
| UB | 算子 tiling 的核心约束；按当前编译目标和 profiler 校验可用容量 |
| L1 / L0 | Cube / Vector pipeline 的片上缓存层级 |
| HBM | 全局内存；长序列 prefill 和 KV cache 容量主要受其约束 |

## 与 A3 / 910B3 的差异处理

- 不直接复用 A3 上的 tile size 作为结论；先按 A2 实测 profiler 复核。
- 对 UB / block 数 / AICore 数敏感的 kernel，需要单独配置 A2 profile。
- 如果同一 kernel 同时支持 A2 和 A3，优先把硬件参数做成显式 profile，而不是在代码中散落判断。

## Kernel Pilot 建议

| 场景 | 建议 |
|------|------|
| PyTorch / torch_npu 替换 | 先验证已有融合算子在 A2 上是否可用且无精度差异 |
| Triton-Ascend | 确认当前环境具备对应 runtime 和二进制资产 |
| TileLang | 检查 tilelang-ascend 版本和目标 CANN 是否匹配 |
| AscendC | 对 UB、block、dtype、对齐约束做 A2 独立 benchmark |

## 必须记录的环境字段

- NPU 型号和 HBM 容量。
- Driver、CANN、torch_npu、Triton-Ascend / TileLang 版本。
- 目标模型、shape、dtype、TP/PP/EP/MTP 配置。
- 单算子 benchmark、端到端 perf、profiling artifact 路径。
