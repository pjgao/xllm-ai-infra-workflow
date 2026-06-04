# Qwen3.5 MTP Optimization Dossier

## Metadata

| Field | Value |
|---|---|
| framework | xLLM |
| model_family | Qwen3.5 |
| scenario | MTP / speculative decoding / graph mode / sampling postprocess |
| status | living dossier |

## Key Paths

| Path or Symbol | Why It Matters |
|---|---|
| `MTPWorkerImpl::step_decode` | draft generation and target validation entry |
| `MTPWorkerImpl::run_validate` | target-model verify path; graph and chunk-prefill interactions surface here |
| `MTPWorkerImpl::update_decode_step_input` | position, KV length, accepted-token bookkeeping |
| `SpeculativeWorkerImpl::step` | target/draft orchestration boundary |
| `GraphPersistentParam::update` | decode graph persistent parameter update; must not be used for non-decode batches |
| `qwen3_gated_delta_net_base` | Qwen3.5 DeltaNet convolution path and MTP transpose/cache issues |
| sampling postprocess path | top_p/top_k can introduce host sync and accuracy/perf risks |

## Case: nst=1 Was Better Than nst=2

Initial expectation: `num_speculative_tokens=2` should improve throughput by reducing
the number of decode iterations.

Observed result: nst=2 regressed TTFT and TPOT on long-input workloads. nst=1 was the
better balance on the validated A3 setup because draft prefill, verify cost, and
DeltaNet state reserve dominated the extra speculative-token benefit.

Validation lesson:

- Compare the same code version and same workload.
- Always use warmup for formal perf results.
- Report TTFT, TPOT, TPS and acceptance counters together.
- Treat nst>1 as a new configuration that needs fresh accuracy and profiling evidence.

## Case: MTP Transpose Elimination

Intent: remove repeated transpose work in Qwen3.5 MTP verify convolution.

Change pattern:

- Cache the transposed convolution weight once instead of transposing every step.
- Keep the MTP verify path in the expected layout so round-trip transpose kernels disappear.
- Put the logic in the small operator/model-path code rather than broad scheduler code.

Evidence pattern:

- Precision: validate a deterministic small set before performance claims.
- Profiling: compare transpose kernel call count and device time before/after.
- Performance: compare baseline/current with the same warmup and workload.

Risk:

- Layout fixes can silently change semantics. Do not accept a pure performance win without
  a small deterministic accuracy check and at least one dataset slice.

## Case: MTP Graph/Chunked Prefill Fix

Symptom examples:

- `PagedAttentionOperation setup failed`
- `decode context position/kv_len mismatch`
- `ACL graph persistent param only supports decode`
- MTP output becomes garbled while non-MTP output is normal.

Root-cause pattern:

- Qwen3.5 MTP validate can enter a different forward type from ordinary decode.
- Decode-only graph persistent parameter logic must not be applied to non-decode batches.
- Chunked-prefill decisions in speculative verify must match the path expected by the
  ATB/spec-kernel or Qwen3.5 verify implementation.
- Position/KV length checks are sensitive to accepted-token rollback and cache update order.

How to localize:

1. Reproduce with one deterministic chat request before using a full benchmark.
2. Run A/B: no MTP, MTP with graph off, MTP with graph on.
3. Compare against the known-good preview branch for MTP-specific code paths.
4. Inspect whether the failing stack enters `run_validate`, graph capture, or
   `update_decode_step_input`.
5. After each fix, run repeated single-request smoke, then dataset-slice accuracy, then perf.

Expected fix shape:

- Preserve the original ATB spec-kernel path while explicitly allowing the Qwen3.5 verify path.
- Guard decode-only persistent graph parameter updates by forward type.
- Keep accepted-token/KV bookkeeping aligned between target and draft contexts.

## Case: Acceptance Rate Measurement Pitfall

Do not conclude MTP is enabled or healthy only from a client-side acceptance-rate field.
Client tools can infer acceptance from output/request shapes, and that may not match the
server-side speculative path.

For speculative decoding, capture these xLLM counters before and after a fixed prompt set:

| Counter | Meaning |
|---|---|
| `speculative_num_accepted_tokens_total` | tokens accepted from draft |
| `speculative_num_draft_tokens_total` | draft tokens proposed |

Use:

```text
true_accept_rate = accepted_delta / draft_delta
```

If client-side and server-side rates disagree, trust the server counters for MTP health.

## Case: Sampling Postprocess Top-P/Top-K

Observed optimization space:

- top_p/top_k postprocess can add host synchronization and small tensor copy overhead.
- Removing sampling from the workload changes the problem definition, so compare both
  deterministic and sampling-enabled scenarios separately.
- When the business scenario uses `temperature=0`, sampling-postprocess optimization may
  be low priority even if profiling shows a local hotspot.

Validation lesson:

- Use deterministic requests to protect accuracy when touching sampling code.
- For performance, keep sampling parameters identical between baseline/current.
- Do not mix profiling run timing with formal non-profiling evalscope timing.

## Regression Checklist

- [ ] `git status --short` is clean before build and before push.
- [ ] Submodules are updated before local CI.
- [ ] `python setup.py build test --device npu` passes for xLLM PR changes.
- [ ] Single deterministic prompt is readable.
- [ ] Fixed small prompt set has no new malformed output.
- [ ] Dataset slice or target task is run when a previous bug was data-dependent.
- [ ] Perf run uses warmup.
- [ ] MTP run records server-side speculative counters.
- [ ] Profiling run has a separate artifact directory and is not used as formal perf timing.
