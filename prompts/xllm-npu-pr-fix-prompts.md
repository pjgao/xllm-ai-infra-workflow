# xLLM NPU PR Fix Prompts

## Fix a PR Regression

```text
Use xllm-npu-accuracy-debug, xllm-npu-incident-triage, and xLLM git-workflow to
analyze PR <pr_number_or_branch>.

Problem:
- Symptom: <garbled_output | crash | CEval_drop | perf_regression | review_finding>
- Target branch: <branch>
- Suspect commit or PR: <commit_or_pr>
- Known good commit/branch: <good_commit_or_branch>
- Artifact root: <run_root>

Workflow:
1. Confirm local worktree, branch, author, and git status.
2. Reproduce the issue with the smallest deterministic request.
3. Check logs and code logic before editing.
4. If the introducing point is unclear, bisect between known good and bad.
5. Implement the smallest fix that addresses the root cause.
6. Validate with build/UT when relevant, then run the accuracy ladder:
   one prompt -> 5-10 prompts -> target dataset subset -> full target task.
7. For performance-sensitive fixes, run warmed-up before/after perf and attach
   profiling only to explain the bottleneck.
8. Update PR description with: what fails without the fix, why it fails, what
   changed, and how it was validated.
9. Record reusable lessons in the relevant skill or model PR history.
```

## Reply to Review Feedback

```text
Review PR feedback for <pr_number> and produce a code-backed reply.

Steps:
1. Pull the latest PR branch and confirm no local dirty changes.
2. Read the exact commented code and surrounding logic.
3. Decide whether the feedback is correctness, performance, style, test, or
   documentation.
4. If code must change, update the code first and run the appropriate gate.
5. Reply with a concise explanation tied to the new diff and validation result.
6. Push only after confirming the fork branch and PR head point to the expected
   commit.
```
