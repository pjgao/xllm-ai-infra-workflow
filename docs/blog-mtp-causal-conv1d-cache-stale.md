# MTP、Causal Conv1d 和一场由旧参数引起的重复输出

2026-06-02

`xllm` · `Qwen3.5` · `MTP` · `NPU` · `ACL Graph`

如果说 attention 的 KV cache 像一本不断往后追加的日记，模型每次都可以翻回去看很早以前写过的内容，那么 causal conv 的 cache 更像一张很短的便签纸。它不关心从开头到现在的全部历史，只保存最近几个 token 的状态。新的 token 来了，纸上的内容往前滑一格，最旧的被丢掉，最新的被写进去。

这件事本来很简单。直到我们让模型一次往前试探好几步。

Qwen3.5 的 MTP 路径里，draft model 会先猜几个 token，target model 再验证这些 token 到底能接受几个。于是 causal conv 这张便签纸不再是每轮固定滑一格，而是每条请求各滑各的：有的滑三格，有的滑一格，有的可能只滑半天最后一个 draft 都没通过。这个时候，告诉算子“写哪一行 cache”和“往前滑几格”的参数就变得非常重要。

这次问题的表象很奇怪：同一道 UDP 选择题，模型最后能给出 `answer:D`，但中间会反复输出 `</think>`、`。` 和 `answer:D`。它不像普通的数值误差，也不像某个 matmul 低了一点精度。它更像模型的某部分状态卡在了一个局部循环里。

后面证明，确实是状态出了问题。

---

## Causal Conv 的那张便签纸

先看最简单的 causal conv。假设 kernel size 是 4，那么当前位置的输出只依赖当前 token 和前 3 个 token：

```text
y[t] = f(x[t], x[t-1], x[t-2], x[t-3])
```

decode 的时候，我们不需要保存完整历史。只要保存最近 3 个状态：

```text
旧 cache: [A7, A8, A9]
新 token: A10
新 cache: [A8, A9, A10]
```

这就是 causal conv cache。它比 KV cache 小很多，因为它的感受野是固定长度。

MTP 让这个逻辑稍微复杂了一点。假设这轮要验证 4 个连续位置：

```text
A10, A11, A12, A13
```

那么 causal conv 会用到：

```text
计算 A10: [A7,  A8,  A9,  A10]
计算 A11: [A8,  A9,  A10, A11]
计算 A12: [A9,  A10, A11, A12]
计算 A13: [A10, A11, A12, A13]
```

所以 MTP verify 时需要 expanded cache：

```text
expanded_state_len = kernel_size - 1 + seq_len - 1
```

例如 kernel size 为 4，verify seq_len 为 4，就需要 6 个位置。这也是代码里检查 `conv_cache.size(1)` 的原因。

---

## 两个容易混淆的参数

有两个参数在这次问题里最关键。

第一个是 `cache_indices`，或者在新路径里叫 `linear_state_indices`。它们表示同一个东西：batch 里的每条请求应该读写 conv cache 的哪一行。

例如：

```text
batch[0] = 请求 A -> conv_cache[10]
batch[1] = 请求 B -> conv_cache[25]

linear_state_indices = [10, 25]
```

它回答的是“写哪里”。

第二个是 `num_accepted_tokens`。这是 MTP 独有的。它表示每条请求这轮接受了几个 draft token。

例如 MTP=3：

```text
请求 A draft: A10, A11, A12
target 全接受
num_accepted_tokens[A] = 3

请求 B draft: B6, B7, B8
target 只接受 B6
num_accepted_tokens[B] = 1

num_accepted_tokens = [3, 1]
```

它回答的是“滑几格”。

把这两个参数合起来，就是一次 MTP causal conv state 更新的完整语义：

```text
请求 A:
  写 conv_cache[10]
  向前推进 3 格

请求 B:
  写 conv_cache[25]
  向前推进 1 格
```

如果 `linear_state_indices` 错了，就会写错行。如果 `num_accepted_tokens` 错了，就会滑错格。两者任何一个错，GatedDeltaNet 的 recurrent state 都会被污染。

---

## 为什么开 MTP 才稳定暴露

这里有个容易误解的点。`cache_indices` 在 host 上并不是只有 MTP 才有风险。非 MTP 的 `npu_causal_conv1d` 也有 host `IntArrayRef` 参数，包括：

```text
query_start_loc
cache_indices
initial_state_mode
num_accepted_tokens
```

从原则上说，只要一个 host 参数会在 ACL graph replay 之间变化，它就有 stale 风险。

那为什么不开 MTP 的时候没有看到同样问题？

原因不是它绝对安全，而是非 MTP decode 的实际场景简单得多。不开 MTP 时，每条请求每轮只前进一个 token。只要同一个 decode batch 连续跑，`cache_indices` 可能很多轮都不变：

```text
第 1 轮: batch=[A,B], cache_indices=[10,25]
第 2 轮: batch=[A,B], cache_indices=[10,25]
第 3 轮: batch=[A,B], cache_indices=[10,25]
```

即使 graph capture 固化了 host 参数，真实 replay 值也刚好一样。

MTP 不一样。即使 batch 完全不变，`num_accepted_tokens` 也会变：

```text
第 1 轮: batch=[A,B], num_accepted_tokens=[3,1]
第 2 轮: batch=[A,B], num_accepted_tokens=[1,3]
第 3 轮: batch=[A,B], num_accepted_tokens=[2,2]
```

shape 没变，batch 没变，cache row 也没变，但每条请求真实推进几格变了。如果这个值被 capture 成旧的 host `IntArrayRef`，conv cache 就会立刻错位。

这就是 MTP 的特殊性。它把“状态推进几步”从一个固定规则，变成了每轮 target verify 的结果。

---

## 旧参数怎么污染状态

假设当前有两个请求：

```text
请求 A: UDP 可靠传输题
请求 B: 二叉树遍历题

A -> conv_cache[10]
B -> conv_cache[25]
```

这轮 MTP 的真实结果是：

```text
linear_state_indices = [10, 25]
num_accepted_tokens  = [3, 1]
```

正确更新应该是：

```text
A: [A7, A8, A9] -> [A10, A11, A12]
B: [B3, B4, B5] -> [B4,  B5,  B6]
```

如果 replay 还用了旧的 accepted-token 值：

```text
stale num_accepted_tokens = [1, 3]
```

那么会变成：

```text
A: [A7, A8, A9] -> [A8, A9, A10]
B: [B3, B4, B5] -> [B6, B7, B8]
```

A 的文本已经推进到了 A12，但 conv cache 像是只推进到了 A10。B 更糟，B7 和 B8 明明没有被接受，却进了状态。下一轮模型看到的文本、KV cache 和 conv cache 就不再一致。

如果同时 `cache_indices` 也是旧的，问题会更直接：

```text
stale cache_indices = [25, 10]
```

那就是 A 的状态写进 B 的 cache 行，B 的状态写进 A 的 cache 行。一个请求开始背着另一个请求的局部历史继续生成，输出自然很难保持正常。

重复输出就是这样来的。模型不是完全崩掉，而是某个 recurrent state 被污染后，logits 反复落回相似的局部区域，于是不断吐出 `。`、`answer:D` 或 `</think>`。

---

## 这不是 transpose 的问题

PR #1536 的初始动机是合理的。profiling 看到 MTP 路径里 transpose 多，而非 MTP decode 使用的 `npu_causal_conv1d` 路径更干净，于是自然会想到让 MTP verify 复用这条路径。

问题在于，`npu_causal_conv1d` 的关键动态参数是 host `IntArrayRef`：

```cpp
const torch::IntArrayRef query_start_loc_opt,
const torch::IntArrayRef cache_indices_opt,
const torch::IntArrayRef initial_state_mode_opt,
const torch::IntArrayRef num_accepted_tokens_opt,
```

这对普通 decode 可能看起来没事，但对 MTP verify 不合适。MTP 的 `num_accepted_tokens` 是同 shape 下也会变化的语义参数，不能作为 host 属性被 capture 固化。

最终修复没有继续沿着减少 transpose 的方向走，而是回到 `causal_conv1d_update_v2` 的 tensor 参数路径：

```cpp
conv1d_params.conv_state_indices = linear_state_indices;
conv1d_params.query_start_loc = q_cu_seq_lens;
conv1d_params.num_accepted_tokens = num_accepted_tokens;
```

这三个值都走 persistent device tensor。每次 replay 前，graph executor 把当前请求的最新值 copy 到 persistent buffer，capture 图里只保留 tensor 地址，而不是固化旧 host 数组。

代价是 `causal_conv1d_update_v2` 的 layout contract 不如 `npu_causal_conv1d` 省事。Qwen3.5 的 conv cache 是：

```text
[slot, state_len, dim]
```

update kernel 期望：

```text
[slot, dim, state_len]
```

所以修复里要传：

```cpp
conv_cache.transpose(1, 2)
```

权重也要：

```cpp
conv_weight.transpose(0, 1).contiguous()
```

也就是说，这次修复牺牲了一部分 transpose 优化收益，换回了正确性。

---

## 一个失败但重要的尝试

最理想的方案当然是既保留 `npu_causal_conv1d` 的少 transpose，又让动态参数走 tensor。我们也尝试过 `aclnnCausalConv1dTensorGetWorkspaceSize` 方向，把 `query_start_loc`、`cache_indices`、`num_accepted_tokens` 都作为 tensor optional input。

这个方向理论上是对的，但当前 CANN/算子组合下不稳定。Qwen3.5 MTP spec decode 的合法 shape 在 graph on 和 graph off 下都会段错误：

```text
x=[1,4,2560] BF16
weight=[4,2560] BF16
conv_state=[89,6,2560] BF16
query_start_loc=[2] int64
cache_indices=[1] int64
num_accepted_tokens=[1] int64
```

所以它不能作为当前修复。

这类失败很有价值。它把两个问题分开了：我们不是不知道正确方向，而是当前可用实现还不能承载这个方向。工程里很多时候就是这样，正确性修复和性能修复不一定能在同一个补丁里完成。

---

## 事后应该留下的规则

这次问题可以总结成一条 review 规则：

> 任何进入 ACL graph capture 的 host `IntArrayRef`，只要它在同一 graph replay 下可能变化，就必须被视为 correctness 风险。

对 MTP verify 来说，至少这三个参数不应该 host 化：

```text
linear_state_indices
q_cu_seq_lens
num_accepted_tokens
```

如果未来还想继续消除 transpose，有两个方向：

1. 等 `aclnnCausalConv1d` 的 tensor 参数路径在 Qwen3.5 MTP shape 下稳定。
2. 新增一个 graph-safe 的 fused spec causal conv，把 MTP verify 需要的动态参数全部设计成 tensor。

在此之前，不能把 PR #1536 原来的性能收益继续当成修复后的收益。当前补丁证明的是正确性，不是性能收益。性能需要重新用同口径 evalscope 和 profiling 测。

最终验证里，TP=4、MTP=3、graph on、chunk prefill on，UDP CEval 单题在 `max_tokens=4096` 下稳定输出 `answer:D`，没有再复现重复 `</think>`、重复 `。` 或重复 `answer:D`。

这说明问题链路基本闭合：

```text
host 动态参数 stale
  -> causal conv cache 状态污染
  -> 重复输出
  -> tensor persistent 参数修复后恢复
```

这不是一个很大的 bug，但它提醒我们一件事：图模式里，参数的“值”不只是值，还是生命周期。对 stateful 模块来说，生命周期错了，状态就会慢慢把答案带偏。

