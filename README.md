# xLLM AI Infra Workflow

[English](README.en.md)

面向昇腾 NPU 大模型推理优化的 Agent-ready workflow 仓库。它以
[xLLM](https://github.com/jd-opensource/xllm) 为首个完整落地框架，同时把
[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) 和 SGLang NPU
作为公平对照、经验迁移和后续扩展目标。

本仓库的核心目标不是保存零散优化笔记，而是沉淀一套 **xLLM AI Infra
Workflow**：让 Agent 和工程师在做性能、精度、profiling、事故排障和 PR
优化时，都能按统一证据标准闭环执行。

## 愿景

在 NPU serving 场景里，很多优化失败不是因为没有想法，而是因为证据链断了：

- baseline 没有 warmup 或环境不干净；
- profiling 和性能数据混在一起比较；
- MTP 是否真正启用只看 evalscope 接受率；
- 精度只看 10 条 prompt，没有坏例、数据集和 A/B；
- PR 修复、review、编译、UT、性能、精度没有形成可复查 artifact。

这个仓库希望把这些经验固化成可复用 skill，使每轮优化都留下：

```text
目标 → 基线 → profiling → 修改 → 精度验证 → 性能验证 → 复盘沉淀
```

## 架构

![xLLM AI Infra Workflow](docs/assets/xllm-ai-infra-workflow.png)

仓库按“执行闭环 + 证据库”组织，而不是按单次优化笔记堆放：

| 层级 | 入口 | 职责 |
|---|---|---|
| Orchestrator 编排层 | `xllm-npu-sota-loop` | 串联 Research、Learn、Code、Review、Validate、Record，驱动端到端优化 |
| Execution & Collection 执行采集层 | `xllm-npu-eval-runner`、`xllm-npu-profiler`、`xllm-npu-incident-triage` | 启动服务、执行评测、采集 profiling、复现事故并收集原始产物 |
| Analysis & Decision 分析决策层 | benchmark / pipeline / capacity / compute / accuracy / code-review | 把性能、精度、容量、空泡、理论下界和 PR 风险拆成可验证结论 |
| Supporting Knowledge 知识层 | `model-pr-optimization-history`、`kernel-pilot`、`references/`、`humanize/` | 保存历史 PR、算子实验、artifact schema、优化账本和 lineage |

一轮正式优化必须沿着这条证据流闭环：

```text
Target → Baseline → Profiling → Patch → Accuracy → Performance → Record
```

其中 baseline/current 性能必须带 warmup；profiling run 只用于解释瓶颈，不直接当正式性能结论。

## 核心方案

### 1. 统一证据契约

正式结论必须能复查。仓库提供四类全局契约：

| 契约 | 作用 |
|---|---|
| [`run-manifest-template.md`](references/run-manifest-template.md) | 记录 commit、环境、模型、启动参数、workload、artifact 路径 |
| [`perf-artifact-schema.md`](references/perf-artifact-schema.md) | 固定 TTFT/TPOT/TPS、warmup、采样参数和服务端 counter 字段 |
| [`accuracy-artifact-schema.md`](references/accuracy-artifact-schema.md) | 固定 raw predictions、failed cases、score、验证等级 |
| [`profiling-artifact-schema.md`](references/profiling-artifact-schema.md) | 固定 msprof capture、五表、timeline notes 和 inconclusive 判定 |

### 2. 公平 benchmark

`xllm-npu-benchmark` 的原则是：不比较不公平数据。

- 同模型、同 tokenizer、同 dtype、同 workload、同 sampling、同 SLA。
- 每个框架独立调优，不用 tuned A 对比 default B。
- 正式性能必须有 warmup。
- 记录失败候选、启动命令、NPU 状态和原始 evalscope 结果。

### 3. Profiling 五表

`xllm-npu-profiler` 将昇腾 profiling 分成五张表：

| 表 | 关注点 |
|---|---|
| Kernel | AICore / AI CPU 热点 |
| Communication / Overlap | HCCL、AllReduce、AllGather、重叠机会 |
| Fuse Pattern | 已知融合模式和可替换算子 |
| Dispatch | hostbound、图下发、空泡、同步、copy |
| Memory | KV cache、xTensor、HBM、碎片和 capacity |

Profiling run 只用于解释瓶颈，不能直接和无 profiling 性能 run 比较。

### 4. 精度异常阶梯

`xllm-npu-accuracy-debug` 按成本递增验证：

```text
L1 单 prompt 是否是人话
L2 5-10 条确定性 prompt
L3 数据集前 N 条
L4 单个 task 全量
L5 全量评测集
```

它强调先找稳定坏例，再看日志和代码逻辑，必要时通过 commit bisect 定位引入点。

### 5. RLCR 优化循环

`xllm-npu-sota-loop` 使用 RLCR：

```text
Research → Learn → Code → Review → Validate → Record
```

- Research：读 benchmark、profiling、capacity、accuracy 证据。
- Learn：查 PR history 和历史失败实验。
- Code：每轮只做一个可验证 patch。
- Review：做 NPU 专项 code review。
- Validate：编译、UT、精度、性能、profiling。
- Record：写 humanize ledger 和 case study。

### 6. 算子迁移流程

当 profiling 证明瓶颈落在后处理、cache、attention、MoE 或 Mamba/SSM
算子上时，先用 `xllm-npu-op-migration` 做源算子盘点、接口与 shape
对齐、实现路径选择、xLLM 接入和验证闭环，再决定是否进入 `kernel-pilot`
做更底层的 kernel 调优。

![xLLM 算子迁移流程](docs/assets/xllm-op-migration-flow.png)

## Skills

| Skill | 何时使用 | 主要产物 |
|---|---|---|
| [`xllm-npu-eval-runner`](skills/xllm-npu-eval-runner/SKILL.md) | 启动/复用 xLLM 服务并执行性能或精度评测 | `runs/eval`、`runs/perf`、`runs/accuracy` |
| [`xllm-npu-benchmark`](skills/xllm-npu-benchmark/SKILL.md) | 比较 xLLM / vLLM-Ascend / SGLang NPU 性能 | `summary.md`、`candidates.jsonl`、`winning-commands.md` |
| [`xllm-npu-profiler`](skills/xllm-npu-profiler/SKILL.md) | 定位 TTFT/TPOT/TPS 瓶颈 | 五表报告、timeline notes、optimization candidates |
| [`xllm-npu-pipeline-analysis`](skills/xllm-npu-pipeline-analysis/SKILL.md) | 分析 prefill/decode、layer、rank skew 和 decode 空泡 | stage table、rank skew table、bubble table |
| [`xllm-npu-capacity-planner`](skills/xllm-npu-capacity-planner/SKILL.md) | 解释 HBM/KV cache/并发容量和 OOM 风险 | capacity table、capacity.json、report.md |
| [`xllm-npu-compute-simulation`](skills/xllm-npu-compute-simulation/SKILL.md) | 估算 FLOPs/MFU 和硬件理论下界 | compute estimate、MFU table、what-if |
| [`xllm-npu-accuracy-debug`](skills/xllm-npu-accuracy-debug/SKILL.md) | 输出乱码、CEval 掉分、GPU/NPU 不一致 | bad cases、A/B 表、bisect notes |
| [`xllm-npu-incident-triage`](skills/xllm-npu-incident-triage/SKILL.md) | crash、OOM、HCCL、graph、PagedAttention 事故 | incident bundle、replay report |
| [`xllm-npu-code-review`](skills/xllm-npu-code-review/SKILL.md) | 提交 NPU 相关代码前审查 | 分级 review findings |
| [`xllm-npu-sota-loop`](skills/xllm-npu-sota-loop/SKILL.md) | 持续优化直到达到目标收益 | run manifest、RLCR ledger、final summary |
| [`xllm-npu-op-migration`](skills/xllm-npu-op-migration/SKILL.md) | 迁移 PyTorch/torch_npu、Triton-Ascend、AscendC 或 ATB 自定义算子到 xLLM | migration report、接口契约、验证表 |
| [`model-pr-optimization-history`](model-pr-optimization-history/SKILL.md) | 开始新模型优化前查历史 | model dossier、risk notes |
| [`kernel-pilot`](kernel-pilot/SKILL.md) | 现有路径用尽后做 kernel 试验 | op benchmark、kernel notes |

## 典型使用

### 直接使用 Prompt

如果你希望让 agent 从一个完整任务开始，而不是手工指定每个 skill，可以先从
[`prompts/`](prompts/) 复制模板：

| Prompt | 场景 |
|---|---|
| [`xllm-npu-sota-loop-prompts.md`](prompts/xllm-npu-sota-loop-prompts.md) | 端到端性能优化、decode gap、投机接受率验证 |
| [`xllm-npu-pr-fix-prompts.md`](prompts/xllm-npu-pr-fix-prompts.md) | PR 回归修复、精度异常、crash、review 回复 |
| [`xllm-npu-op-migration-prompts.md`](prompts/xllm-npu-op-migration-prompts.md) | NPU 算子迁移、算子优化、kernel-pilot 准入 |

Prompt 负责启动任务；skill 负责执行细节和证据门禁。

### 选择入口

| 任务 | 先用哪个 skill | 何时补充 |
|---|---|---|
| 跑服务、性能或精度评测 | `xllm-npu-eval-runner` | 需要跨框架公平对比时再用 `xllm-npu-benchmark` |
| 优化 TPOT / TTFT / TPS | `xllm-npu-sota-loop` | Phase 3 必须接 `xllm-npu-profiler` |
| 分析 decode 空泡或 rank 差异 | `xllm-npu-pipeline-analysis` | 需要理论上限时补 `xllm-npu-compute-simulation` |
| 判断 OOM、KV cache、并发容量 | `xllm-npu-capacity-planner` | crash 时补 `xllm-npu-incident-triage` |
| 输出乱码、CEval 掉分 | `xllm-npu-accuracy-debug` | commit 范围不清楚时启动 bisect |
| 提交 xLLM NPU PR 前 | `xllm-npu-code-review` | 再查目标仓库自己的 `.agents/skills` |
| 迁移外部或实验性 NPU 算子 | `xllm-npu-op-migration` | 需要重新写 kernel 时再用 `kernel-pilot` |
| 现有路径用尽，需要算子实验 | `kernel-pilot` | 先确认 profiling 已证明 kernel 是瓶颈 |

### 跑一次 xLLM 性能和精度评测

```text
使用 xllm-npu-eval-runner，启动 Qwen3-32B xLLM 服务，
用 evalscope 跑 5k 输入 / 50 输出 / temperature=0 的性能评测，
保存 manifest、metrics.json、report.md 和原始 evalscope 结果。
```

### 做多框架公平对比

```text
在 A3 NPU 上比较 xLLM、vLLM-Ascend、SGLang NPU，
同模型、同 workload、同 sampling、同 SLA。
每个框架独立搜索最优配置，输出 summary、candidates 和 winning commands。
```

### 定位 TPOT 回退

```text
先跑带 warmup 的 baseline/current 性能；
再用 xllm-npu-profiler 采集 decode-focused profiling；
对比 replaceToken 到下一轮 GatherV2 之间的 hostbound 空泡，
给出可验证优化点。
```

### 定位精度乱码或 CEval 掉分

```text
使用 xllm-npu-accuracy-debug：
先跑单 prompt 和 5-10 条确定性样例，
再跑 CEval 目标 task 子集，
保存 failed_cases.jsonl；
如果 commit 范围不清楚，启动 git bisect。
```

### 执行端到端优化循环

```text
使用 xllm-npu-sota-loop：
Phase 0 记录环境和目标；
Phase 0.5 查询 model-pr-optimization-history；
Phase 1 建立公平基线；
Phase 3 采集 profiling；
Phase 5 每轮只提交一个 patch；
每轮都重新做精度、性能和必要 profiling；
最终沉淀到 humanize ledger 和 model PR history。
```

## 安装

### Codex / Claude Code / opencode

Codex 和 opencode 都会使用根目录 `AGENTS.md` 作为项目级指令入口；Claude
Code 还会读取 `CLAUDE.md`。因此本仓库把通用 agent 行为原则写入
`AGENTS.md`，并在 `CLAUDE.md` 保持同构副本。更新 agent 行为规则时，两者要同步。

推荐使用 symlink，这样 `git pull` 后 skill 自动生效：

```bash
for skill_dir in skills/xllm-npu-*; do
  ln -sfn "$(pwd)/$skill_dir" "$CODEX_HOME/skills/$(basename "$skill_dir")"
done
ln -sfn "$(pwd)/kernel-pilot" "$CODEX_HOME/skills/xllm-npu-kernel-pilot"
ln -sfn "$(pwd)/model-pr-optimization-history" "$CODEX_HOME/skills/model-pr-optimization-history"
```

如果目标 Agent 不支持 symlink，也可以复制对应 skill 目录。

## 目录

```text
AGENTS.md                       Codex / opencode / 通用 agent 项目级规则
skills/                         核心 Agent skills
prompts/                        可直接复制给 agent 的任务启动模板
references/                     全局 artifact schema 和代码风格
tests/                          仓库卫生和 schema 最小测试
docs/                           设计文档、case study、路线图
humanize/                       run-level 优化账本契约，具体台账写入每次 run root
model-pr-optimization-history/   模型 PR 历史知识库
kernel-pilot/                   NPU kernel 试验辅助
patches/                        最小 patch 或历史迁移说明，不保存整文件快照
CLAUDE.md                       Claude Code 行为约束，与 AGENTS.md 同步
```

## 环境

- Huawei Ascend 910B3 / A3 NPU，兼容扩展到 Ascend 910B / A2
- CANN / HDK Driver 与目标框架兼容
- 至少一个 OpenAI-compatible serving 框架：xLLM、vLLM-Ascend 或 SGLang NPU
- 推荐安装 evalscope、msprof / MindStudio profiling 工具

具体环境和 profiling 采集要求见：

- [`docs/npu-ai-infra-standard-workflow.md`](docs/npu-ai-infra-standard-workflow.md)
- [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md)
- [`skills/xllm-npu-profiler/SKILL.md`](skills/xllm-npu-profiler/SKILL.md)

## 贡献原则

- 新经验优先沉淀为 skill、reference、schema 或 case study。
- 不提交本机路径、内网 IP、私有数据集名、真实密钥或不可公开日志。
- 不把 smoke 结果扩大成正式性能或精度结论。
- 不删除失败实验；把它们记录成 future-agent 避坑信息。
- 新增框架能力时，尽量拆成通用 NPU 证据层和框架适配层。
