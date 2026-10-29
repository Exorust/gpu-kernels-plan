# Companion: Week 1 Wednesday, "Online normalizer calculation for softmax" (Milakov and Gimelshein, 2018)

The paper: `sources/online-softmax-1805.02867.pdf` (9 pages; read sections 1-3, pages 1-4). The plan's task: derive the rescale step on paper and explain why a running max and a running sum are enough for one pass. No code today.

The paper is short and dense: it gives algorithms and one proof, and leaves out the intuition. This guide puts the intuition back. Keep it open next to the PDF, one section per part of the paper.

## Why the plan puts this paper here

```
| Wed | *paper* · Online softmax ([paper](https://arxiv.org/abs/1805.02867), sections 1-3). Derive the rescale step on paper: why a running max and a running sum are enough for one pass. No code today. Week 4 FlashAttention uses this exact step. |
```

Three later tasks reuse today's result: Thursday's Triton build, Week 2's vLLM `topk_softmax` (softmax then pick the top experts, which is the paper's section 4), and Week 4's FlashAttention, whose whole trick is this rescale applied to tiles of attention scores.

Source: README.md:25-25

## Part 0: why anyone counts memory accesses for softmax

```
Softmax of a row x of length V:   y_i = exp(x_i) / sum_j exp(x_j)
Work per element: about 1 exp, 1 add, 1 divide (a few FLOPs).
Bytes per element: at least 4 to read x_i and 4 to write y_i (FP32).
Arithmetic intensity: a few FLOPs / 8 bytes, well under 1 FLOP per byte.
H100 ridge point: 295 FLOPs per byte (see w1-mon-h100-numbers.md).
```

Softmax is memory-bound by a factor of hundreds. Its run time is set by how many times it touches GPU memory, not by how many exps it does. So the paper measures every algorithm in "memory accesses per element". Fewer accesses means proportionally faster. Doing two extra FLOPs per element to save one memory access is a good trade.

Source: tutor

## Section 1 (page 1): the claim

```
classical Softmax with fewer memory accesses and hypothesize that this reduction
```

The paper does not make softmax approximate. It computes the exact same numbers as the standard "safe" softmax, with one fewer pass over the input. The benchmark result in the abstract (up to 1.3x for softmax alone) is what you would predict from the access count: 4 accesses down to 3 is 1.33x for a memory-bound kernel.

Source: artifacts/sources/online-softmax-1805.02867.pdf p. 1

## Section 2 (page 2): naive softmax, two loads and one store

```
doing three memory accesses per vector element: two loads and one store.
```

Algorithm 1, in words:
1. Pass 1: read every `x_j`, add `exp(x_j)` into a running sum `d`.
2. Pass 2: read every `x_i` again, write `y_i = exp(x_i) / d`.

Two reads plus one write per element = 3 accesses. The pass structure is forced: you cannot write any `y_i` until you know `d`, and you know `d` only after reading the whole row.

Source: artifacts/sources/online-softmax-1805.02867.pdf p. 2

## Section 2 (page 2): why naive softmax breaks

```
algorithm 1 can overflow or underflow due to the exponent. There is a safe form of (1), which is
```

Worked example. The largest FP32 number is about 3.4e38, and `exp(89)` ≈ 4.5e38 is already larger. So a row with one logit equal to 89 gives `exp(89) = inf`, then `inf / inf = NaN`. Logits of 89 are normal in a language model's output layer. In the other direction, if every logit is -110, every `exp` rounds to 0 and you divide 0 by 0.

The fix (equation 2): subtract the row's maximum `m` from every `x` first. Softmax does not change, because the factor `exp(-m)` appears on top and bottom and cancels. After the shift, every exponent is at most 0, so every `exp` is at most 1 (no overflow), and the largest one is exactly `exp(0) = 1` (so the sum is at least 1; no 0/0).

Source: artifacts/sources/online-softmax-1805.02867.pdf p. 2

## Section 2 (page 2): safe softmax costs a third pass

```
one - final values yi , see algorithm 2; This results in 4 memory access per vector element overall.
```

Algorithm 2, in words:
1. Pass 1: read every `x_k`, keep the running max `m`.
2. Pass 2: read every `x_j` again, sum `exp(x_j - m)` into `d`.
3. Pass 3: read every `x_i` again, write `y_i = exp(x_i - m) / d`.

Three reads plus one write = 4 accesses. Pass 2 needs the final `m`, so it cannot start until pass 1 is done. This dependency, "the sum needs the max", is exactly what the paper removes.

Source: artifacts/sources/online-softmax-1805.02867.pdf p. 2

## Section 3 (pages 2-3): the online idea in one sentence

```
Essentially, the algorithm keeps the maximum value m and the normalization term d as it iterates
```

Algorithm 3 merges passes 1 and 2. It keeps a running max `m` and a running sum `d`, both updated as each new element arrives:

- `m_j = max(m_{j-1}, x_j)`
- `d_j = d_{j-1} × exp(m_{j-1} - m_j) + exp(x_j - m_j)`

Then one more pass writes `y_i = exp(x_i - m_V) / d_V`. Two reads plus one write = 3 accesses. The page 2 line that states the saving: "It reduces memory accesses from 4 down to 3 per vector element".

The factor `exp(m_{j-1} - m_j)` is the **rescale step**. The next section is today's exercise: derive it yourself.

Source: artifacts/sources/online-softmax-1805.02867.pdf p. 3

## Exercise: derive the rescale step yourself

```
Goal: after j elements, keep   d_j = sum_{k=1..j} exp(x_k - m_j)
where m_j = max(x_1, ..., x_j).

Step 1. Write d_{j-1} the same way, with m_{j-1}.
Step 2. A new element x_j arrives and the max may change to m_j.
        Every old term exp(x_k - m_{j-1}) is now "measured against the wrong max".
Step 3. Use exp(a - c) = exp(a - b) × exp(b - c) with a = x_k, b = m_{j-1}, c = m_j.
Step 4. Pull the common factor out of the sum. What do you get?
```

Try it before you read on. Answer:

`exp(x_k - m_j) = exp(x_k - m_{j-1}) × exp(m_{j-1} - m_j)` for every old `k`. The second factor does not depend on `k`, so it comes out of the sum:

`sum_{k<j} exp(x_k - m_j) = d_{j-1} × exp(m_{j-1} - m_j)`. Add the new term `exp(x_j - m_j)` and you have line 5 of Algorithm 3.

**Why a running max and a running sum are enough:** the old terms are never needed one by one. They only ever appear as their sum, and changing the max multiplies every one of them by the same number. So one multiply fixes the whole sum. That is the answer to the plan's question.

Two facts to check in your derivation: when the max does not change, the factor is `exp(0) = 1` and nothing is rescaled; when it does change, `m_j > m_{j-1}`, so the factor is less than 1 and the old sum shrinks. It never grows, so it cannot overflow.

Source: tutor

## Worked example with numbers

```
x = [1, 3, 2]

j=1: m = 1.  d = exp(1-1)                         = 1.0000
j=2: m = 3.  d = 1.0000 × exp(1-3) + exp(3-3)     = 0.1353 + 1 = 1.1353
j=3: m = 3.  d = 1.1353 × exp(3-3) + exp(2-3)     = 1.1353 + 0.3679 = 1.5032

Check with safe softmax: exp(1-3) + exp(3-3) + exp(2-3) = 0.1353 + 1 + 0.3679 = 1.5032
y = [0.1353, 1, 0.3679] / 1.5032 = [0.0900, 0.6652, 0.2447]   (sums to 1.0000)
```

At j=2 the max jumped from 1 to 3, so the old sum 1.0 was multiplied by `exp(-2)`. At j=3 the max stayed at 3, so the rescale factor was 1. Do this on paper with your own 4-number row, including one where the max changes twice.

Source: tutor

## Theorem 1 and the safety bounds (pages 3-4)

```
• dj is also bounded: 1 ≤ dj ≤ j, ∀j ∈ 1, V . It can be easily proven by induction. The
```

The proof on page 3 is the derivation you just did, written as induction: assume `d_{S-1}` is right, show `d_S` is right. If your derivation matched, skim it.

The bounds say why it is safe: every term `exp(x_k - m_j)` is at most 1, so `d_j` is at most `j`; the term for the current max is exactly 1, so `d_j` is at least 1. With FP32 `d` this works for rows of up to about 1.7e37 elements (the next line on page 4). In practice, the concern is not overflow but rounding, which is why kernels keep `m` and `d` in FP32 even when `x` is BF16.

Source: artifacts/sources/online-softmax-1805.02867.pdf p. 4

## Section 3.1 (page 4): merging two partial results, the part GPUs need

```
operation ⊕ is associative, which enables parallel evaluation of (3). It is also commutative, which
```

Algorithm 3 is a loop, one element at a time. A GPU wants many threads to each handle part of the row. Equation 4 says how to merge two partial results `(m_a, d_a)` and `(m_b, d_b)`:

- `m = max(m_a, m_b)`
- `d = d_a × exp(m_a - m) + d_b × exp(m_b - m)`

Each partial sum is rescaled to the shared max, then added. One element `x` is the pair `(x, 1)`.

"Associative" is the key word. It means you can merge in any grouping: `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`. That is exactly the property a tree reduction needs. So the Harris tree from Tuesday works for `(m, d)` pairs with `⊕` in place of `+`. Each thread runs Algorithm 3 on its own slice, then the block merges the pairs with a tree.

Check with the worked example, split as `[1, 3] | [2]`: left = `(3, 1.1353)`, right = `(2, 1)`. Merge: `m = 3`, `d = 1.1353 × exp(0) + 1 × exp(2 - 3) = 1.5032`. Same answer.

Source: artifacts/sources/online-softmax-1805.02867.pdf p. 4

## Section 4 (page 4): skim only, it returns in Week 2

```
Online Softmax (algorithm 3) does three memory accesses per vector element: one load for the
```

Section 4 adds a top-K search to the same single pass, so softmax plus top-K needs one read of the input. You do not need it today. In Week 2 Friday, vLLM's `topk_softmax_kernels.cu` does the same job for mixture-of-experts routing (softmax over experts, then pick the top K). Come back to page 5, Algorithm 4, then.

Source: artifacts/sources/online-softmax-1805.02867.pdf p. 4

## One caveat to carry into Thursday

```
The paper counts accesses to the input vector in GPU memory.
If the whole row fits in registers or shared memory, a kernel reads it from HBM once,
and the 3 safe-softmax passes run on-chip. Then online softmax saves no HBM traffic.
It pays off when the row does NOT fit on-chip (very long rows, processed in chunks),
and in FlashAttention, where the "row" of attention scores is never stored at all.
```

Keep this in mind tomorrow: the Triton tutorial loads each full row into on-chip memory, so it is already one HBM read. The online version matters for rows too long to hold at once. See `w1-thu-triton-softmax-companion.md`.

Source: tutor
