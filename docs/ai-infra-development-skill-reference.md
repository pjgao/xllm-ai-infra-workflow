# ai-infra-development Skill 参考分析

> 来源：`<ai-infra-workbench>`
> 分析日期：2026-06-04

## 结论

该目录的 skills 更像一个多框架推理工作台，覆盖 build、deploy、perf、
accuracy、profiling、bugfix、review 和 framework routing。它的
`SKILL.md` 都很短，真正值得参考的是工程约束和运行产物规范，而不是直接把
`inference-*` skills 整体复制进本仓库。

本仓库已有更细的 xLLM NPU 优化经验，例如 MTP、profiling 五表、CEval 精度、
PR 事故复盘和 NPU 性能门禁。因此同步策略应是吸收通用流程，不替换现有
xLLM/NPU 专项 skill。

## 值得同步的设计

### 1. 单一启动入口

参考仓库要求 xLLM 启动只走 `scripts/launch_xllm.sh`。该脚本负责：

- 读取统一配置。
- 选择空闲 NPU。
- 设置 `ASCEND_RT_VISIBLE_DEVICES`、HCCL、Ascend/ATB 环境变量。
- 创建 `runs/deploy/<run_id>/`。
- 保存 `npu-smi.before.txt`、`visible_devices.txt`、`pids.txt`、节点日志、
  `models.json`、`healthcheck.log` 和 smoke 结果。

对本仓库的启发：正式 benchmark/profiling 前，服务启动本身也应有可复查
artifact。尤其是多次遇到 HBM 残留、PID 查不到、端口/HCCL 残留时，只有
`pids.txt`、`npu-smi.before/after` 和启动 manifest 能快速判断 run 是否污染。

### 2. run artifact contract

参考仓库把不同任务固定写入：

- `runs/build/<run_id>/`
- `runs/deploy/<run_id>/`
- `runs/perf/<run_id>/`
- `runs/accuracy/<run_id>/`
- `profiling/<run_id>/`

每类 run 都应至少包含 raw log、manifest、结构化 metrics 和 `report.md`。

对本仓库的启发：当前优化记录已经有大量路径，但命名和产物字段不完全统一。
后续新任务应尽量补齐：

- `manifest.json`：commit、branch、model、devices、CANN、启动参数、workload。
- `metrics.json` 或 `summary.json`：TTFT/TPOT/TPS、成功数、token 数、精度分数。
- `report.md`：结论、A/B 表、污染风险、后续方向。
- 原始日志：服务日志、evalscope 输出、profiling 导出日志。

### 3. profiling capture 不启动服务

参考仓库的 `inference-profiling` 和 `capture_xllm_profile.sh` 明确：

- profiling 脚本只 attach 到已启动服务。
- 服务必须先用正常 launcher 启动，并继承 `PROFILING_MODE=dynamic`。
- warmup 命令在 profiling 窗口外执行。
- measured workload 命令在 `start/stop` 窗口内执行。
- 若缺少 `PROF_*` 或 `mindstudio_profiler_output/`，结论必须标为
  `INCONCLUSIVE`。

这与本仓库 `xllm-npu-profiler` 的方向一致，但参考仓库把职责边界写得更清楚：
启动归 deploy，采集归 profiling，性能结论归 benchmark/optimization。

### 4. 性能和精度分级

参考仓库把性能分为 `simple` / `complex`，精度分为 `sanity` / `quick` / `full`：

- `sanity`：一个 prompt，看是否是人话。
- `quick`：少量公开题，几分钟内完成。
- `full`：完整数据集或接近正式评测。

这与本仓库的精度异常定位阶梯一致，可以继续沿用到每次修复报告中：

`单 prompt 人话 -> 小样本数据集 -> 全量 CEval/MMLU`

### 5. framework router

参考仓库有 `inference-framework-router`，通过 `development.yaml` 识别当前框架，
再路由到 build/deploy/perf/accuracy/review/bugfix/profiling。

本仓库已经在 README/AGENTS 中定位为 xLLM、vLLM-Ascend、SGLang NPU 的通用
NPU 优化 skill 集合，但当前最强经验仍是 xLLM。后续如果要扩展到
vLLM-Ascend/SGLang，可以参考该 router 思路：把框架差异放到
`frameworks/<framework>.md`，skill 只保留流程和证据标准。

## 不建议直接同步的内容

- 不直接复制 `development.yaml`：本仓库是 skill/知识库，不是单一工作台。
- 不直接复制所有 `inference-*` skill：会和现有 xLLM NPU 专项 skill 重叠，
  反而降低触发清晰度。
- 不直接复制脚本实现：参考仓库脚本绑定它自己的 `code/`、`runs/`、配置结构；
  本仓库应只吸收可复用的契约和检查项。

## 已同步到本仓库的规则

- benchmark skill：补充标准 run artifact、性能/精度分级参考。
- profiler skill：强调服务启动和 profiling capture 分离，`PROFILING_MODE=dynamic`
  是启动侧要求。
- AGENTS：补充单一启动入口、run artifact contract 和 framework adapter 思路。

