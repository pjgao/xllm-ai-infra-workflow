# xLLM NPU Operator Migration Runbook

This runbook summarizes migration lessons from two operator repositories:

- `https://gitcode.com/sinle4cat/torch_npu_ops`
- `https://gitcode.com/xLLM-AI/xllm_ops`

Do not copy source files directly. Use them to identify project structure,
registration patterns, build artifacts, validation scope, and integration risks.

## Repository Patterns

### torch_npu_ops

Observed roles:

- `ops_npu/` and `custom_functions_npu/`: ATB and torch_npu-facing C++ helpers.
- `ascendc_npu/`: small AscendC/aclnn examples exposed to Python.
- `npu_python_extension/`: `NpuExtension` + pybind11 registration for Python tests.
- `triton_npu/`: Triton-Ascend AOT flow; `setup.py` runs pytest, collects
  `.npubin` and json files into a binary directory.
- root `CMakeLists.txt`: builds `torch_npu_kernels` and `triton_adapter`, then
  runs Triton AOT generation as a post-build step.

Migration meaning:

- Good source for PyTorch/torch_npu wrappers and Triton-Ascend AOT packaging.
- Use it when xLLM needs a light wrapper or AOT binary loading pattern.
- Check for hidden sync, cache clearing, binary path stability, and test-only
  assumptions before using in serving.

### xllm_ops

Observed roles:

- `xllm_ops/<op>/op_host/`: proto, def, tiling header/source.
- `xllm_ops/<op>/op_kernel/`: AICore kernel implementation.
- `common/stub/op_api/`: op_api and Level0 stubs for Python/C++ tests.
- `atb_customize/`: ATB custom operation registration and configs.
- `test/cpp_test/`: C++ accuracy/performance tests.
- `test/python_test/`: `NpuExtension`, `RegisterOps.cpp`, Python wrappers and tests.
- root build: `bash build.sh`; CMake supports selecting `ASCEND_COMPUTE_UNIT`
  and operator name.

Migration meaning:

- Good source for full AscendC custom op layout.
- Use it when the operator needs tiling, workspace, AICore kernel, generated
  aclnn/op_api entry, or custom ATB packaging.
- Verify op name consistency across proto, def, tiling, kernel, CMake, wrapper,
  and Python/C++ tests.

## Migration Decision Matrix

| Candidate | Prefer When | Watch For |
|---|---|---|
| PyTorch / torch_npu | Existing aclnn/torch_npu op is close and overhead is small | implicit sync, extra allocations, graph capture compatibility |
| Triton-Ascend AOT | Medium-grain fused logic, Python Triton source exists, binary can be generated at build time | cache pollution, `.npubin` naming, json/kernel name mismatch, binary path |
| AscendC custom op | Need deterministic tiling, workspace, custom memory movement, or high AICore efficiency | op_host/kernel contract, dynamic shape, arch specialization, package install |
| ATB customize | Need ATB graph/operator integration already used by xLLM | parameter JSON, graph mode, padding/chunk prefill/MTP compatibility |

## End-to-End Migration Checklist

### 1. Evidence

- Baseline run has warmup.
- Profiling identifies the exact op family or host gap.
- Workload is fixed: model, TP/DP, batch, input/output length, sampling params.
- The candidate op is measured against a real xLLM bottleneck, not only a microbenchmark.

### 2. Interface Contract

Document:

- tensor shape, dtype, layout, stride and contiguous requirements;
- scalar and optional arguments;
- output allocation and inplace behavior;
- dynamic shape sources such as `decode_step`, `actual_seq_lengths`,
  `block_table`, `num_accepted_tokens`;
- workspace size and lifetime;
- stream semantics and whether the wrapper can run without host synchronization.

### 3. Implementation Mapping

Typical AscendC mapping:

```text
op_host/*_proto.cpp      -> public op schema
op_host/*_def.cpp        -> op definition and infer rules
op_host/*_tiling.*       -> tiling data and workspace decisions
op_kernel/*.cpp/.h       -> AICore implementation
CMakeLists.txt           -> register op subdirectory
RegisterOps.cpp          -> EXEC_NPU_CMD(aclnn*) wrapper for tests
xLLM callsite            -> replace old torch/aclnn/ATB path
```

Typical Triton-Ascend mapping:

```text
triton_src/*.py          -> Triton kernel and pytest reference
setup.py                 -> run pytest and copy .npubin/json
kernel_registry.*        -> load AOT binaries by kernel name
torch_api/*.cpp          -> C++ wrapper and args builder
xLLM CMake               -> build adapter and define TRITON_BINARY_PATH
```

Typical PyTorch/torch_npu mapping:

```text
NPU layer callsite       -> small wrapper around torch_npu/aclnn op
reference function       -> pure torch or existing xLLM implementation
unit test                -> deterministic random tensors on NPU
profiling check          -> verify no unexpected host sync/copy
```

## Validation Levels

| Level | Purpose | Required Evidence |
|---|---|---|
| L1 micro accuracy | prove numerical equivalence | max/mean error, failed shape dump |
| L2 micro performance | prove op-level benefit | warmup/repeat, p50/p90, baseline op |
| L3 xLLM smoke | prove serving path works | fixed prompts, deterministic sampling |
| L4 xLLM accuracy | prove no model regression | target dataset subset or full task |
| L5 profiling | prove bottleneck moved as expected | before/after kernel and timeline notes |

## Common Failure Modes

- Wrapper allocates tensors with wrong dtype or on CPU.
- Optional tensor semantics differ from original op.
- Tiling assumes static shape but decode has dynamic `seq_len` or accepted tokens.
- Triton AOT binary name differs from kernel registry key.
- Build uses stale binary cache after source changes.
- Graph mode captures one shape but replay receives another.
- MTP path changes `num_accepted_tokens`, conv/cache position, or padding behavior.
- Profiling run is used as formal performance data.

## PR Evidence Rule

An operator migration PR should include:

- why the old path is slow or unsafe;
- why the chosen operator path is correct;
- exact files changed and runtime phase affected;
- precision evidence and workload scope;
- warmed-up performance delta;
- profiling evidence that the expected hotspot changed.

