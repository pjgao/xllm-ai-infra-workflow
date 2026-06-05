# xLLM NPU Operator Migration Prompts

## Operator Migration

```text
Use xllm-npu-op-migration and kernel-pilot to migrate or optimize <operator>
for xLLM on Ascend <A2_or_A3>.

Inputs:
- Source implementation: <torch_npu | Triton-Ascend | AscendC | ATB | PyTorch>
- xLLM target path: <file_or_module>
- Shapes: <representative_shapes>
- Dtypes: <dtype_list>
- Accuracy reference: <reference_op_or_test>
- Artifact root: <run_root>

Workflow:
1. Prove with profiling that this operator path is a real bottleneck.
2. Inventory source semantics: shape, dtype, layout, stream, workspace, in-place
   behavior, graph-mode constraints, and error handling.
3. Define an interface contract before writing integration code.
4. Start with the lowest-risk implementation path. Escalate to kernel-pilot only
   when existing fused operators or library paths cannot meet the target.
5. Add a focused correctness test and at least one end-to-end smoke case.
6. Benchmark the operator path and the full model before/after.
7. Record migration risks, failed attempts, and reusable shape lessons.
```

## Kernel Pilot Gate

```text
Before writing a custom NPU kernel for <operator>, verify all gates:

- xLLM still misses the target after non-kernel changes.
- Profiling shows this kernel family explains at least 1% of relevant stage time
  or is the clear blocking latency tail.
- There is a stable correctness reference.
- Representative shapes cover the production path.
- Integration risk is lower than reusing an existing fused or library operator.

If any gate fails, return a no-go report and choose a safer optimization path.
```
