---
name: model-pr-optimization-history
description: 查询 xLLM 历史 PR 中的优化信息，辅助当前模型的优化决策。
---

# Model PR Optimization History

通过查询 xLLM 代码库的历史 PR，获取模型相关的优化信息，避免重复工作。

## 使用场景

1. 开始优化某模型前，先查询历史 PR 中的已知优化
2. 遇到特定性能问题时，搜索历史 PR 中的解决方案
3. 了解 xLLM 对该模型的支持状态

## 工作流

### Step 1: 查询历史 PR

使用 `scripts/query.py` 查询 xLLM 仓库的 PR：

```bash
cd /home/gaopengju/projects/xllm

# 按模型查询
python ../xllm-npu-optimization-skills/model-pr-optimization-history/scripts/query.py \
    --model "Qwen3" \
    --type optimization

# 按关键词查询
python ../xllm-npu-optimization-skills/model-pr-optimization-history/scripts/query.py \
    --keyword "flash attention" \
    --type feature

# 查询特定时间段
python ../xllm-npu-optimization-skills/model-pr-optimization-history/scripts/query.py \
    --since "2024-01-01" \
    --until "2024-06-01"
```

### Step 2: 整理优化历史

将查询结果整理为模型档案，存入 `xllm/<model>.md`：

```markdown
## 已知优化 (来自 PR 历史)

| PR # | 优化内容 | 效果 | 日期 |
|------|---------|------|------|
| #123 | 实现 NPU fused attention | +15% throughput | 2024-03-15 |
| #456 | 添加 KV Cache NZ 格式支持 | -20% memory | 2024-04-20 |
```

### Step 3: 识别未覆盖优化

对比当前工作负载与历史优化，找出：
1. **已实现的优化**：无需重复实现
2. **部分实现的优化**：需要进一步完善
3. **未实现的优化**：新的优化机会

## 查询输出字段

| 字段 | 说明 |
|------|------|
| pr_number | PR 编号 |
| title | PR 标题 |
| author | 作者 |
| merged_at | 合并时间 |
| labels | PR 标签 |
| files_changed | 涉及的文件 |
| related_model | 相关模型 |
| optimization_type | 优化类型 |
| performance_impact | 性能影响描述 |

## 模型档案目录

- `model-pr-optimization-history/xllm/deepseek-v3.md` — DeepSeek-V3 (MoE)
- `model-pr-optimization-history/xllm/qwen3-core.md` — Qwen3 系列 (Dense)
- `model-pr-optimization-history/xllm/glm-5.md` — GLM-5 系列

## 维护

模型档案应随新 PR 合并而更新。建议：
- 每次 xLLM 新 PR 合并后，检查是否影响已归档的模型
- 季度审查一次档案的准确性
