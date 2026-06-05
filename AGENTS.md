# Agent 使用指南

本仓库是面向华为昇腾 NPU 910B3 (A3) / 910B (A2) 的大模型推理与 AI Infra 开发
skill 集合。当前最完整的落地对象是京东 xLLM，但标准流程应服务于
xLLM、vLLM-Ascend、SGLang NPU 后端等多框架。Agent 在协助任何 NPU
推理优化任务时，**必须遵循本仓库的 evidence-driven 闭环流程**。

Codex、opencode 和其他支持 `AGENTS.md` 的 agent 必须直接遵守本文。Claude
Code 还应读取 [`CLAUDE.md`](CLAUDE.md)，其内容与本文的通用行为原则保持一致。
如果你需要一个可直接复制的任务入口，先从 [`prompts/`](prompts/) 选择模板，再加载对应 skill。

## 仓库定位

- 仓库目标：沉淀 NPU 大模型推理和 AI Infra 开发的证据驱动标准流程
- 框架范围：xLLM、vLLM-Ascend、SGLang NPU 后端
- 对照原则：按任务选择对照框架；默认优先做 xLLM vs vLLM-Ascend，也允许扩展到 SGLang NPU
- 目标模型：NPU serving 上的主流推理模型（Qwen3 / DeepSeek-V3 / GLM-5 / Llama / Kimi 等）
- 目标硬件：昇腾 910B3 (A3) / 910B (A2)，HDK Driver 25.2.0+，CANN 8.0.RC1+
- 当前经验底座：xLLM + Qwen3.5-27B + MTP 的真实 benchmark、profiling、patch、事故记录

## 核心原则（必须遵守）

1. **unfair 数据不比较**：benchmark 阶段必须让每个参测框架各自独立搜索最优配置（严禁用一方的最优参数套另一方）
2. **没 profiling 不动手**：Phase 3 报告不存在前，不允许开始任何 patch
3. **每步决策有数据支撑**：所有判定必须基于五表中的可复现指标
4. **RLCR 闭环**：Research → Learn → Code → Review → Validate → Record，严禁"写完就合"
5. **经验不得丢失**：通用化时不得删除已有 xLLM/Qwen3.5/MTP 经验；失败实验、反例和环境信息也必须保留
6. **启动与采集分离**：服务启动、性能压测、profiling 采集、精度评测必须有独立 artifact；profiling 脚本只 attach 已启动服务，不隐式启动 xLLM。
7. **run 产物可复查**：build/deploy/perf/accuracy/profiling 每类 run 至少保存 manifest、raw log、结构化 metrics 和 report，避免只留下终端结论。
8. **修改保持外科手术式**：只改本次任务需要的文件；发现无关问题时记录或提示，不顺手重构。
9. **敏感信息不得入库**：不提交本机用户名、真实机器路径、内网 IP、私有数据集名、密钥、完整生产日志。

## 通用 Agent 行为原则

这些原则适用于 Codex、opencode、Claude Code 和其他 coding agent。它们来自
Andrej Karpathy 风格的 agent guardrail，并已收敛成本仓库的执行约束。

1. **先想清楚再动手**：实现前明确假设；如果多个解释会改变代码路径，先问清楚；发现更简单的做法要说明。
2. **简单优先**：只实现用户要求和验证目标需要的内容；不为单次用途新增抽象；不添加未被要求的配置、兼容层或 speculative feature。
3. **外科手术式修改**：只改与任务直接相关的行；匹配现有风格；不顺手重构、格式化或删除无关代码。
4. **目标驱动执行**：把模糊任务转成可验证目标，例如复现坏例、跑指定 benchmark、通过 UT、达到目标 TPOT；循环直到验证完成或阻塞条件清晰。
5. **暴露不确定性**：不要隐藏猜测；不能确认的数据、环境、提交、指标要标记为待验证。
6. **验证结果说话**：修 bug 要有复现和修复后验证；做性能要有 warmup、baseline/current、profiling 解释和原始 artifact。
7. **记录可复用经验**：每次踩坑、失败优化、review 修正和性能结论，都要沉淀到 run ledger、reference、case study 或 model PR history。

## 入口选择

| 入口 | 适合场景 | 下一步 |
|---|---|---|
| `AGENTS.md` | Codex / opencode / 通用 agent 项目级规则 | 自动或手动加载后必须遵守 |
| `CLAUDE.md` | Claude Code 项目级规则 | 与 `AGENTS.md` 的通用原则保持一致 |
| `prompts/` | 需要启动一轮标准 agent 任务 | 复制模板，补齐模型/硬件/workload/run root |
| `skills/*/SKILL.md` | 任务已明确属于某个能力 | 先读 skill，再按它的门禁执行 |
| `model-pr-optimization-history/` | 开始新模型优化或 PR 风险分析 | 查询历史 PR、文件、符号和已知风险 |
| `references/` | 需要 artifact schema、代码风格或硬件信息 | 只加载本次任务需要的 reference |
| `humanize/` | 需要沉淀优化账本格式 | 具体 ledger 写入 run root，不写回本目录 |

## Skills 总览

| Skill | 路径 | 何时加载 |
|-------|------|---------|
| xllm-npu-eval-runner | `skills/xllm-npu-eval-runner/SKILL.md` | 需要启动/复用 xLLM 服务并执行 evalscope 性能或精度评测时 |
| xllm-npu-benchmark | `skills/xllm-npu-benchmark/SKILL.md` | 需要对比 xLLM / vLLM-Ascend / SGLang NPU 性能时 |
| xllm-npu-profiler | `skills/xllm-npu-profiler/SKILL.md` | 需要定位 NPU 性能瓶颈、生成五表报告时 |
| xllm-npu-pipeline-analysis | `skills/xllm-npu-pipeline-analysis/SKILL.md` | 需要分析 prefill/decode 边界、layer、rank skew 或 decode 空泡时 |
| xllm-npu-capacity-planner | `skills/xllm-npu-capacity-planner/SKILL.md` | 需要解释 HBM、KV cache、MTP reserve、并发容量或 OOM 风险时 |
| xllm-npu-compute-simulation | `skills/xllm-npu-compute-simulation/SKILL.md` | 需要估算 FLOPs/MFU、理论下界或 TP/MTP what-if 时 |
| xllm-npu-sota-loop | `skills/xllm-npu-sota-loop/SKILL.md` | 端到端驱动 NPU SOTA 优化闭环、判断优化是否达标时 |
| xllm-npu-code-review | `skills/xllm-npu-code-review/SKILL.md` | 提交 NPU 特化代码前必须审查的 7 个维度 |
| xllm-npu-incident-triage | `skills/xllm-npu-incident-triage/SKILL.md` | xLLM 在 A3 上出现 crash / hang / OOM / 异常结果时 |
| kernel-pilot | `kernel-pilot/SKILL.md` | 所有现成优化路径用尽、需自研 NPU 算子时 |
| model-pr-optimization-history | `model-pr-optimization-history/SKILL.md` | 开始新模型优化前查询历史已做工作 |

## 标准工作流（6-Phase）

```
Phase 0     环境准备（NPU 健康、框架可用、模型就绪）
  ↓
Phase 0.5   查 PR 历史（model-pr-optimization-history）
  ↓
Phase 1     评测执行（xllm-npu-eval-runner）+ 公平基准测试（xllm-npu-benchmark）
  ↓
Phase 2     差异判定
  ├─ 目标框架胜出 / 平局 → 写 final_summary.md → 结束
  └─ 目标框架落后 > 1% ↓
Phase 3     性能诊断（xllm-npu-profiler → 五表报告）
  ↓
Phase 4     构建 Plan（基于五表定位 + 优化路径优先级）
  ↓
Phase 5     RLCR 迭代（xllm-npu-sota-loop）
  ├─ Research → Learn → Code → Review(code-review) → Validate → Record
  └─ 回到 Phase 2
```

## RLCR 对应 Skill

| 阶段 | 主要 skill | 产物 |
|---|---|---|
| Research | `xllm-npu-benchmark`、`xllm-npu-profiler`、`xllm-npu-pipeline-analysis`、`xllm-npu-capacity-planner`、`xllm-npu-accuracy-debug` | baseline、五表、空泡表、容量表、坏例 |
| Learn | `model-pr-optimization-history` | 历史 PR 风险、相关文件、已有方案 |
| Code | `xllm-npu-op-migration`、`kernel-pilot`、目标仓库本地 skills | 最小 patch、算子迁移契约、测试 |
| Review | `xllm-npu-code-review`、目标仓库 `code-review` | 分级 review findings |
| Validate | `xllm-npu-eval-runner`、`xllm-npu-benchmark`、`xllm-npu-profiler`、`xllm-npu-accuracy-debug`、`xllm-npu-incident-triage` | 编译/UT、性能、精度、profiling、事故复现 |
| Record | `xllm-npu-sota-loop`、`humanize/`、`model-pr-optimization-history` | attempt ledger、optimization ledger、case study、模型历史 |

## 关键路径

- xLLM 本地仓库：`<xllm-repo>`
- vLLM-Ascend 本地仓库：`<vllm-ascend-repo>`
- SGLang NPU 本地仓库：按实际机器记录到 `$RUN_ROOT/manifest.md`
- 模型权重目录：`/models/`（如 `/models/Qwen3.5-27B`）
- 每次 SOTA run 输出目录：`<run-root>/<date>_<model>_npu_sota/`

## 启动新模型优化前必做

1. 查对应框架的 `model-pr-optimization-history/<framework>/` 看是否已有该模型档案
2. 没有则查 PR 历史并新建档案；当前 xLLM 档案可按 `qwen3-core.md` 格式扩展
3. 新建 `$RUN_ROOT/manifest.md` 记录本次 run 的框架 commit hash / CANN 版本 / NPU 型号 / workload / SLA

## 运行产物约定

参考 `<ai-infra-workbench>` 的多框架工作台经验，新任务尽量按任务类型拆分产物：

- `runs/build/<run_id>/`：编译命令、submodule 状态、build log、server binary 校验。
- `runs/deploy/<run_id>/`：启动命令、`npu-smi` 快照、可见卡、PID、服务日志、healthcheck、smoke。
- `runs/perf/<run_id>/`：原始 evalscope/benchmark 输出、`metrics.json`、`report.md`、环境门禁。
- `runs/accuracy/<run_id>/`：原始预测、失败样本、score、case 配置、`report.md`。
- `profiling/<run_id>/`：`PROF_*`、`mindstudio_profiler_output/`、capture log、workload log、manifest。

如果历史脚本已有固定输出目录，可以继续使用，但报告中要补齐上述字段。缺少关键 artifact 的 run 只能作为 debug/smoke，不应用作 PR 性能或精度结论。

统一模板：
- `references/run-manifest-template.md`：所有正式 run 的 manifest 字段。
- `references/perf-artifact-schema.md`：性能评测产物目录和 `metrics.json` 字段。
- `references/accuracy-artifact-schema.md`：精度评测产物目录、验证等级和 score 字段。
- `references/profiling-artifact-schema.md`：profiling 采集产物、timeline notes 和 inconclusive 判定。
- `references/npu-specs.json`：NPU compute simulation 的硬件规格占位。
- `references/model-config-index.json`：常见模型 config 索引占位，正式估算前必须补齐。

## 常用脚本

```bash
# 基准对比
skills/xllm-npu-eval-runner/scripts/eval_perf.sh

python skills/xllm-npu-benchmark/scripts/collect_evalscope_results.py \
  --root /path/to/evalscope/results \
  --framework xllm \
  --output-jsonl xllm.jsonl \
  --output-summary xllm-summary.md

python skills/xllm-npu-benchmark/scripts/compare_npu_benchmark.py \
  --xllm-results xllm.jsonl --vllm-results vllm.jsonl --output-dir comparison/

# Profiling 分析
export PROFILING_MODE=dynamic
ps -ef | grep xllm
MODEL=Qwen35-27B TOKENIZER=<model-root>/Qwen35-27B PORT=8080 \
  skills/xllm-npu-profiler/scripts/run_profiling.sh <xllm_parent_pid> profiles/xllm full

python skills/xllm-npu-profiler/scripts/analyze_xllm_npu_profile.py \
  --input profiles/xllm_YYYYMMDD_HHMMSS/PROF_xxx --framework xllm --output profiles/xllm-analysis.json

# 五表报告渲染
python skills/xllm-npu-profiler/scripts/render_triage_npu.py \
  --analysis-root profiles/ --output analysis/root-cause.md

# Model PR 历史查询
python model-pr-optimization-history/scripts/query.py --model "Qwen"

# Triage 报告渲染
python skills/xllm-npu-incident-triage/scripts/render_triage_npu.py \
  --artifacts ./artifacts --output ./triage-report-<id>.md

# 算子 benchmark
python kernel-pilot/tools/npu-op-benchmark.py --op swiglu --shapes "128,4096" --dtype float16
```

## Prompt 模板

根目录 [`prompts/`](prompts/) 保存可直接交给 agent 的任务模板：

- `xllm-npu-sota-loop-prompts.md`：端到端性能优化、decode gap、投机接受率验证。
- `xllm-npu-pr-fix-prompts.md`：PR 精度/性能/事故回归修复、review 回复。
- `xllm-npu-op-migration-prompts.md`：torch_npu / Triton-Ascend / AscendC / ATB 算子迁移。

Prompt 只负责启动任务；真正的步骤和门禁以对应 skill 为准。

## xLLM 启动模板（A3）

```bash
# xLLM (baseline)
xllm serve /models/<MODEL> \
  --tensor-parallel-size 4 \
  --graph-mode npugraph_ex \
  --block-size 128 \
  --port 8080

# vLLM-Ascend (baseline)
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /models/<MODEL> \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

## 多框架 benchmark 模板（A3）

```bash
# xLLM
xllm serve /models/<MODEL> \
  --tensor-parallel-size 4 \
  --graph-mode npugraph_ex \
  --block-size 128 \
  --port 8080

# vLLM-Ascend
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /models/<MODEL> \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# SGLang NPU（示例，实际参数以后端支持为准）
python -m sglang.launch_server \
  --model-path /models/<MODEL> \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30000
```

## 编译与测试

```bash
# xLLM NPU 构建 + 单元测试提交门禁
python setup.py build test --device npu

# xLLM 端到端精度测试
python test/test_xllm_serve_generation.py --model /models/<MODEL> --device npu
```

提交 xLLM PR 前必须完成：

1. 在唯一权威 PR worktree 中确认 `git status --short` 无非预期修改。
2. 执行 `git submodule update --init --recursive`，确认依赖已更新。
3. 确认没有其他同仓库编译/测试进程在写同一 build 目录。
4. 执行 `python setup.py build test --device npu`，编译和 UT 全部通过后再提交。
5. push 后分别确认 fork 分支和 PR head 指向预期 commit；PR 描述或检视回复更新不代表代码已推到 PR。
6. 对关键 review 点，用远端 PR ref 的文件内容复查，避免本地 worktree 误判。

## lint / typecheck

当前仓库为文档+脚本集合，主要 lint 为：

- Python 脚本：`ruff check skills/ model-pr-optimization-history/`
- Markdown：无强制 lint，但保持 80 字宽折行

## 何时停下来问用户

- 优化 gap 判定落在 ±1% 平局区：询问是否继续扩大优势或结束
- kernel-pilot 准入条件不全满足：是否强制启动（通常不建议）
- 优化路径冲突（如两个候选都要改同一文件）：让用户拍板优先级
- benchmark 显示 SLA 违反：询问是否放宽 SLA 或更换配置

## 反模式（禁止）

- ❌ 拿任意框架 default 直接当 baseline，让另一框架单方面追赶
- ❌ 没看五表就开始写 patch
- ❌ 多个 patch 一起上、不单独验证
- ❌ 修了 bug 但不更新 incident 台账
- ❌ 为了通用化重写文档时删掉已有 xLLM/MTP 实测经验
- ❌ 把参考仓库、外部 prompt 或本地日志原文直接搬进本仓库
