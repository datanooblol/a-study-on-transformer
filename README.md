# A Study on Transformer

A study of transformer architecture — building and understanding each variant from scratch.

## Goal

Understand the three main transformer architectures by implementing them from scratch, then apply the encoder variant with contrastive learning to cluster customer interaction sequences.

## Transformer Architecture Variants

### 1. Encoder Only (`package/encoder_block.py`)

Each token attends to every other token freely — no causal mask, full bidirectional context.

**Use cases:** understanding and representing input, no generation involved
- Text classification (e.g. sentiment analysis)
- Sentence embeddings and similarity
- Named entity recognition
- **This project** — encode interaction sequences, then cluster the embeddings

**Real-world example:** BERT reads a full sentence and produces a rich embedding for each token.

**Test:**
```python
from encoder_block import EncoderBlock
import torch

encoder = EncoderBlock(d_model=64, n_heads=4, max_len=50)
src = torch.randn(2, 10, 64)                              # (batch, src_len, d_model)
out = encoder(src)
print(out.shape)                                          # (2, 10, 64)
```

---

### 2. Encoder + Decoder (`package/encoder_block.py` + `package/decoder_block.py`)

Encoder reads and compresses the source sequence. Decoder generates the target sequence one token at a time, attending to its own past tokens (causal mask) and the full encoder output (cross-attention).

**Use cases:** transforming one sequence into another
- Machine translation (English → French)
- Summarization (long document → short summary)
- Speech recognition (audio features → text)

**Real-world example:** the original "Attention is All You Need" paper — translate a sentence by encoding the source and decoding the target.

**Test:**
```python
from encoder_block import EncoderBlock
from decoder_block import DecoderBlock
import torch

encoder = EncoderBlock(d_model=64, n_heads=4, max_len=50)
decoder = DecoderBlock(d_model=64, n_heads=4, max_len=50)

src = torch.randn(2, 10, 64)   # source sequence (e.g. English)
tgt = torch.randn(2, 7, 64)    # target sequence (e.g. French so far)

enc_out = encoder(src)         # encoder reads the full source
out = decoder(tgt, enc_out)    # decoder attends to its past + encoder output
print(out.shape)               # (2, 7, 64)
```

---

### 3. Decoder Only (`package/gpt_decoder_block.py`)

Same as encoder but with a causal mask — each token can only attend to itself and previous tokens. No encoder, no cross-attention.

**Use cases:** generating text autoregressively from a single sequence
- Text generation and completion
- Code generation
- Chat and instruction following

**Real-world example:** GPT predicts the next token given all previous tokens, stacking many decoder blocks.

**Test:**
```python
from gpt_decoder_block import GPTDecoderBlock
import torch

block = GPTDecoderBlock(d_model=64, n_heads=4, max_len=50)
x = torch.randn(2, 7, 64)     # (batch, seq_len, d_model)
out = block(x)
print(out.shape)               # (2, 7, 64)
```

---

## How They Relate

```
EncoderBlock
  + causal mask                  → GPTDecoderBlock
  + cross-attention to encoder   → DecoderBlock (full encoder-decoder)
```

- `EncoderBlock` vs `GPTDecoderBlock` — identical structure, only adds causal mask
- `GPTDecoderBlock` vs `DecoderBlock` — adds `CrossAttention` + `norm2`, takes `enc_out` as input

---

## Applied Project: Insurance Purchase Sequence Prediction (Decoder-based)

Applying the decoder-only architecture to predict a customer's next insurance purchase given their history.

**Each purchase event has 3 features:**
- `policy` — which policy was bought (categorical, e.g. A, B, C)
- `age` — customer's age at purchase (continuous)
- `price` — price paid (continuous)

**How it works:**
- Each event is fused into one vector: `policy_embedding + age_projection + price_projection`
- The sequence of event vectors is fed into `GPTDecoderBlock` — causal mask ensures each event only sees past events
- Three prediction heads on the output: next policy (classification), next age (regression), next price (regression)
- Position `i` in the output predicts event `i+1` — so the last position forecasts the next purchase

**Files:**
- `package/insurance_decoder_block.py` — `EventEmbedding` + `InsuranceDecoderBlock` with three prediction heads
- `package/train_insurance.py` — mock dataset with two customer archetypes, training loop, and inference

**Two customer archetypes in the mock dataset:**

| Type | Policy cycle | Age start | Price range |
|------|-------------|-----------|-------------|
| A-type | A → C → A → C ... | 25–35 | 100–200 |
| B-type | B → D → B → D ... | 40–50 | 300–500 |

**Inference example — given a history, predict the next purchase:**
```python
# A-type customer: bought A at 25, C at 28, A at 31
# model should predict: next policy C, age ~34, price ~100
predict(model,
    customer_policies = ["A", "C", "A"],
    customer_ages     = [25.0, 28.0, 31.0],
    customer_prices   = [105.0, 95.0, 110.0],
    name              = "Customer 1 (A-type)"
)

# to add a new event, just append to each list
predict(model,
    customer_policies = ["A", "C", "A", "A"],   # new event: policy A
    customer_ages     = [25.0, 28.0, 31.0, 35.0], # new event: age 35
    customer_prices   = [105.0, 95.0, 110.0, 36.0], # new event: price 36
    name              = "Customer 1 (A-type, 4 events)"
)
```

**Note:** age and price are normalized before feeding into the model (`mean=0, std=1`) and denormalized on the way out — so predictions are in real units (years, currency).

---

## Project Pipeline (Encoder-based)

1. **Data Loading** (`package/data_loader.py`) — PyTorch `Dataset` + `DataLoader` for interaction sequences
2. **Contrastive Cropping** (`package/collate.py`) — crop two random subsequences (positive pair) from each sample
3. **Padding & Masking** (`package/pad_and_mask.py`) — pad variable-length sequences and produce attention masks
4. **Transformer Encoder** (`package/encoder_block.py`) — encode padded sequences into fixed-size embeddings
5. **Clustering** — (TBD) cluster the learned embeddings
