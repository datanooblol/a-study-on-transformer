# CLS Token and PAD Masking

## The Problem

The encoder outputs one vector per token — if you feed in 5 purchase tokens you get 5 output vectors back.
But we want **one single vector** that summarizes the whole sequence to feed into the regression head.

Two problems to solve:
1. How do we collapse a variable-length sequence into one fixed-size vector?
2. How do we stop the model from attending to PAD tokens that carry no real information?

CLS token solves problem 1. PAD masking solves problem 2.

---

## Setup

Say we have a batch of 2 customers, `MAX_PURCHASES = 5`, `d_model = 4`:

```
Customer 1 purchases: Health, Health, OPD   → ids: [1, 1, 3, 0, 0]  (2 PADs at end)
Customer 2 purchases: Life                  → ids: [2, 0, 0, 0, 0]  (4 PADs at end)
```

0 is reserved for PAD in all vocabularies.

---

## `__init__` — what gets created

```python
self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
```
`vocab_size + 1` because 0 is reserved for PAD — if you have 4 purchase types you need 5 slots:
- 0 = PAD
- 1 = Health
- 2 = Life
- 3 = CI
- 4 = Motor

`padding_idx=0` tells PyTorch to keep the PAD embedding as all zeros and never update it during training.

```python
self.cls = nn.Parameter(torch.randn(1, 1, d_model))
```
One learnable CLS vector, shape `(1, 1, d_model)`:
- first `1` — one sample, will be expanded to batch size in forward
- second `1` — one token position
- `d_model` — the actual vector content

`nn.Parameter` means this tensor is trained by backprop just like any other weight.
It starts random and gradually learns to be a good "collector" of sequence information.

```python
self.encoder = EncoderBlock(d_model, n_heads, max_len + 1)
```
`max_len + 1` because after prepending CLS the sequence becomes one token longer.

---

## `forward` — step by step

### Step 1 — build the padding mask

```python
pad_mask = tokens == 0   # (batch, seq_len)
```

```
tokens:   [[1, 1, 3, 0, 0],
            [2, 0, 0, 0, 0]]

pad_mask: [[F, F, F, T, T],   ← positions 3, 4 are PAD
            [F, T, T, T, T]]  ← positions 1, 2, 3, 4 are PAD
```

`True` = this position is padding, ignore it in attention.

---

### Step 2 — prepend a False column for CLS

```python
cls_col  = torch.zeros(batch, 1, dtype=torch.bool)   # all False
pad_mask = torch.cat([cls_col, pad_mask], dim=1)
```

```
pad_mask: [[F, F, F, F, T, T],   ← CLS column added at front
            [F, F, T, T, T, T]]
           CLS↑
```

CLS is never padding so its column is always False.
We add this column so the mask shape matches the sequence length after CLS is prepended.

---

### Step 3 — embed tokens and prepend CLS

```python
x   = self.embed(tokens)              # (batch, seq_len, d_model)
cls = self.cls.expand(batch, -1, -1)  # (1, 1, d_model) → (batch, 1, d_model)
x   = torch.cat([cls, x], dim=1)      # (batch, 1+seq_len, d_model)
```

```
before: x has 5 token vectors per customer
after : x has 6 vectors — CLS at position 0, then the 5 tokens

Customer 1: [CLS, Health, Health, OPD, PAD, PAD]
Customer 2: [CLS, Life,   PAD,    PAD, PAD, PAD]
```

`expand` doesn't copy memory — it just makes the single CLS vector look like it has batch size copies.

---

### Step 4 — run encoder

```python
out = self.encoder(x, pad_mask)   # (batch, 1+seq_len, d_model)
```

Every token attends to every other token — but PAD positions are masked to `-inf` so softmax gives
them ~0 weight and they contribute nothing to any other token's output.

CLS attends to all real tokens and collects their information into its own output vector.

---

### Step 5 — read CLS output

```python
return out[:, 0]   # (batch, d_model)
```

```
out[:, 0] → CLS* — summary of the whole sequence   ← we return this
out[:, 1] → Health* embedding                       ← ignored
out[:, 2] → Health* embedding                       ← ignored
...
```

One vector per customer that summarizes their entire purchase sequence.

---

## The Full Picture

```
tokens:   [1,    1,     3,    0,    0  ]
           ↓     ↓      ↓     ↓     ↓
embed:    [H,    H,    OPD,  PAD,  PAD ]   PAD = zeros, masked out in attention
           ↓
prepend:  [CLS,  H,     H,   OPD,  PAD,  PAD]
           ↓     ↓      ↓     ↓     ↓    ↓
encoder:  every token attends to every other (PADs ignored via mask)
           ↓
out:      [CLS*, H*,   H*,  OPD*, PAD*, PAD*]
           ↑
           only this is returned — it summarizes the whole sequence
```

---

## Why CLS and not just average pooling?

You could average all token vectors instead:
```python
return out[:, 1:].mean(dim=1)   # average over all real token positions
```

CLS is preferred because:
- averaging treats all tokens equally — CLS learns to weight them differently
- the model can learn what kind of summary is most useful for the task
- it's the same trick BERT uses — well established in practice

---

## Connection to train_insurance.py

| | `train_insurance.py` (decoder) | `insurance_representation_demo.py` (encoder) |
|---|---|---|
| goal | predict the **next event** in a sequence | predict a **single value** from the full history |
| reads | past events only (causal mask) | all events freely (no causal mask) |
| output used | last position `[:, -1]` | CLS position `[:, 0]` |
| positional encoding | RoPE | RoPE |
