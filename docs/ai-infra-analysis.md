# AI-Infra-Auto-Driven-SKILLS 仓库设计分析

> 对标参考仓库：https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS
> 分析日期：2026-05-23
> 分析目的：理解仓库设计理念，为 xllm NPU 优化方案提供参考

---

## 1. 仓库概述

**名称**：AI-Infra-Auto-Driven-SKILLS
**定位**：Agent-ready playbooks for LLM serving benchmarks, torch-profiler triage, SGLang optimization, human code review, production incidents, and model PR intelligence.
**核心聚焦**：SGLang (GPU/NVIDIA) 推理优化，框架中立对比 vLLM 和 TensorRT-LLM

**设计理念**：
- **Agent-ready playbooks**，而非通用 prompts——给 agent 可操作的运营记忆
- **证据驱动**：所有优化决策必须有 benchmark 数据、profiler trace、PR 历史支持
- **框架中立比较**：SGLang vs vLLM vs TensorRT-LLM 在相同 benchmark schema 下公平竞争

---

## 2. 核心架构

```
                    ┌─────────────────────────────────┐
                    │   sglang-sota-humanize-loop      │ (顶层自治循环)
                    │   Humanize RLCR 驱动代码优化      │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              ▼                ▼                     ▼
 ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
 │llm-serving-auto │ │llm-torch-profiler│ │model-pr-optimization │
 │benchmark        │ │analysis         │ │history (58 dossiers) │
 │(公平基准测试)    │ │(三表profiler报告) │ │(PR驱动模型历史)       │
 └─────────────────┘ └─────────────────┘ └──────────────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
 ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
 │sglang-humanize  │ │sglang-prod     │ │KernelPilot           │
 │review           │ │incident-triage │ │(内核证据辅助)         │
 │(人工PR审查语料)  │ │(生产事故诊断)   │ │(NCU Report)          │
 └─────────────────┘ └─────────────────┘ └──────────────────────┘
```

### 2.1 六大核心 Skills

| Skill | 使用时机 |
|-------|---------|
| `llm-serving-auto-benchmark` | 需要在相同模型/工作负载/GPU/SLA下公平比较 SGLang、vLLM、TensorRT-LLM |
| `llm-torch-profiler-analysis` | 需要三表 profiler 报告（kernel/overlap/fusion），保持 prefill 和 decode 证据分离 |
| `sglang-humanize-review` | 需要基于 10,959 条人类审查线程的 SGLang 代码审查 |
| `sglang-sota-humanize-loop` | 将 SGLang SOTA 工作流作为一个 Humanize RLCR 模型级 patch 循环运行 |
| `sglang-prod-incident-triage` | 从队列增长、超时、错误输出、崩溃或分布式停顿中诊断生产事故 |
| `model-architecture-diagram` | 需要查找公开原始架构图 |

### 2.2 辅助层

| 组件 | 说明 |
|------|------|
| `model-pr-optimization-history` | 58 个 PR 驱动模型历史档案（29 SGLang + 29 vLLM） |
| `KernelPilot` | 独立内核知识和 NCU report 工作流的兄弟项目 |
| `scripts/query.py` | 本地模型/关键词查询助手 |

---

## 3. 关键 Skill 详细分析

### 3.1 llm-serving-auto-benchmark

**核心工作流**：
1. **Preflight**：验证所有框架 CLI `--help`，确认参数可用，保存帮助输出到 artifact 目录
2. **标准化工作负载**：统一 JSONL 格式，默认 chat(1000→1000) 和 summary(8000→1000)
3. **搜索层级选择**：Tier1(smoke)→Tier2(默认bounded sweep)→Tier3(exhaustive)，每框架≤10候选
4. **框架调优**：分别调优 SGLang/vLLM/TensorRT-LLM
5. **结果归一化**：JSONL → Markdown/CSV，按 SLA通过 > 吞吐 > TTFT > TPOT 排序

**公平性规则**：
- 相同 GPU 类型/数量/模型权重/tokenizer/精度/量化/采样设置
- 记录框架版本、git commit、容器镜像、CUDA/NCCL 版本
- 保留失败候选及原因
- TensorRT-LLM 后端固定 `pytorch`

### 3.2 llm-torch-profiler-analysis

**三表输出契约**：

| 表名 | 内容 |
|------|------|
| **kernel table** | GPU 内核占比排序 |
| **overlap-opportunity table** | 重叠/并行机会分析 |
| **fuse-pattern table** | 融合模式匹配（source-backed，确定性） |

**阶段分离采集**：
- 默认 `--profile-workload both`，自动分 prefill(input=4090→output=1) 和 decode(input=1→output=2048）
- 每轮 warmup 10 步 + 活跃采集 5 步
- 与 SOTA loop 联动时必须使用慢场景的实际 input/output 长度

**双 trace 分析**：graph-off(映射) + graph-on(正式) 两个 trace 联合分析

### 3.3 sglang-sota-humanize-loop

**6 Phase 工作流**：

| Phase | 说明 |
|-------|------|
| Phase 0 | 初始化：收集模型/精度/GPU/框架集，创建运行目录 |
| Phase 0.5 | Model PR History Knowledge Gate：查询 PR 历史 |
| Phase 1 | 固定公平基准测试：调用 llm-serving-auto-benchmark |
| Phase 2 | 差异判定：阈值 1%，超出才启动 RLCR |
| Phase 3 | RLCR 前的必要 Profiling：调用 llm-torch-profiler-analysis |
| Phase 4 | 构建 Humanize Plan |
| Phase 5 | 启动 RLCR 迭代 |

**内核证据准入条件**（全部满足才启用）：
1. SGLang 仍落后最佳框架 > 1%
2. 慢阶段某个 kernel 族占 GPU 时间 >= 1%
3. profiler 对比显示该 kernel 是差异的合理解释
4. 有明确的正确性参考和代表性 shapes

**循环台账文件**：
- `humanize/attempt-ledger.md` — 每次尝试记录
- `humanize/optimization-ledger.md` — 有效优化记录
- `humanize/source-idea-ledger.md` — 源码想法及出处
- `humanize/lineage.jsonl` — 血缘追踪

---

## 4. 证据标准

| 维度 | 标准 |
|------|------|
| Benchmark | 包括模型、框架、GPU数、工作负载、QPS、SLA状态、启动命令、benchmark命令、原始artifact |
| Profiler | Prefill/Decode 分离，三表报告（kernel/overlap/fusion） |
| SOTA Claims | 必须明确模型、硬件、框架commit、精度、工作负载、SLA |
| 内核优化 | 引用 KernelPilot 知识 + NCU 计数器证据 + 真实模型 benchmark 验证 |
| 事故诊断 | 从 replayable 证据出发，保留证据链 |
| PR 历史 | 指向 PR、文件、diff、风险面，而非模糊总结 |

---

## 5. 安装机制

仓库使用标准 `SKILL.md` 目录结构，兼容多种 Agent 运行时：
- **Claude Code**：symlink 到 `~/.claude/skills/`
- **Codex/Kimi/OpenCode**：copy/symlink 到对应 runtime 的 skill 目录
- **通用 Agent**：直接复制 SKILL.md 目录

---

## 6. 对 xllm NPU 优化的设计启示

| AI-Infra 设计 | xllm NPU 适配 |
|--------------|-------------|
| 框架中立 benchmark（SGLang/vLLM/TRT-LLM） | 框架对比 benchmark（xllm vs vLLM-Ascend） |
| torch.profiler 三表报告 | 昇腾 Profiling 五表报告（增加下发效率+内存效率） |
| Humanize RLCR 驱动 SOTA 循环 | 适配 xllm 的 RLCR 循环，整合昇腾优化路径 |
| 10,959 条审查线程语料库 | xllm C++/TileLang/AscendC 代码审查知识库 |
| Replay-first 事故诊断 | NPU 特化事故诊断（AICore timeout/HCCL/OOM 等） |
| 58 个 PR 驱动模型历史 | xllm 模型家族 PR 历史档案 |
| KernelPilot（NCU 内核证据） | kernel-pilot（TileLang/AscendC/Triton-Ascend + 昇腾 profiling） |
| 阶段分离（Prefill/Decode） | 继承，适配昇腾 Profiling 采集方式 |
| 证据门槛（NCU 计数器+真实模型验证） | 适配昇腾 Profiling 指标+AICore 利用率+真实模型验证 |
