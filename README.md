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

---

## Appendix: Bringing Your Own Data

If you have real customer data instead of the mock dataset, `InsuranceDataset` takes your sequences directly:

```python
purchase_sequences = [["A", "C", "A"], ["B", "D", "B"], ["A", "C", "A", "C"]]
age_sequences      = [[25.0, 28.0, 31.0], [40.0, 43.0, 46.0], [30.0, 33.0, 36.0, 39.0]]
price_sequences    = [[105.0, 95.0, 110.0], [410.0, 390.0, 420.0], [100.0, 120.0, 95.0, 110.0]]
```

The sequences above have different lengths (3, 3, 4). `DataLoader` requires all samples in a batch to be the same shape, so you need a padding strategy. Two options:

---

### Option 1 — Pre-pad in `__init__`

Pad all sequences to the global max length once when the dataset is created.

```python
def pad_sequence(seq, max_len, pad_value):
    return seq + [pad_value] * (max_len - len(seq))

class InsuranceDataset(Dataset):
    def __init__(self, purchases, ages, prices):
        max_len = max(len(s) for s in purchases)

        self.purchases = []
        self.ages      = []
        self.prices    = []
        self.masks     = []  # True = padding, False = real data

        for p, a, pr in zip(purchases, ages, prices):
            pad_len = max_len - len(p)
            self.purchases.append(pad_sequence([policy_to_id[x] for x in p], max_len, 0))
            self.ages.append(pad_sequence(a,   max_len, 0.0))
            self.prices.append(pad_sequence(pr, max_len, 0.0))
            self.masks.append([False] * len(p) + [True] * pad_len)

    def __len__(self):
        return len(self.purchases)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.purchases[idx], dtype=torch.long),
            torch.tensor(self.ages[idx],      dtype=torch.float).unsqueeze(-1),
            torch.tensor(self.prices[idx],    dtype=torch.float).unsqueeze(-1),
            torch.tensor(self.masks[idx],     dtype=torch.bool),
        )

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Pros:**
- Simple — no extra code at the `DataLoader` level
- Padding is done once, not repeated every epoch

**Cons:**
- If one sequence is much longer than the rest, every sample gets padded to that length — wastes memory and compute
- Global max length is fixed at dataset creation time — can't add longer sequences later without rebuilding

---

### Option 2 — `collate_fn` in `DataLoader`

Keep `__getitem__` simple, pad per batch to the longest sequence in that batch.

```python
class InsuranceDataset(Dataset):
    def __init__(self, purchases, ages, prices):
        self.purchases = [[policy_to_id[p] for p in seq] for seq in purchases]
        self.ages      = ages
        self.prices    = prices

    def __len__(self):
        return len(self.purchases)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.purchases[idx], dtype=torch.long),
            torch.tensor(self.ages[idx],      dtype=torch.float).unsqueeze(-1),
            torch.tensor(self.prices[idx],    dtype=torch.float).unsqueeze(-1),
        )

def collate_fn(batch):
    policies, ages, prices = zip(*batch)

    policies_padded = torch.nn.utils.rnn.pad_sequence(policies, batch_first=True, padding_value=0)
    ages_padded     = torch.nn.utils.rnn.pad_sequence(ages,     batch_first=True, padding_value=0.0)
    prices_padded   = torch.nn.utils.rnn.pad_sequence(prices,   batch_first=True, padding_value=0.0)

    # pad mask: True where position is beyond the real sequence length
    lengths  = torch.tensor([len(p) for p in policies])
    max_len  = policies_padded.shape[1]
    pad_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)  # (batch, max_len)

    return policies_padded, ages_padded, prices_padded, pad_mask

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
```

**Pros:**
- Each batch only pads to its own longest sequence — less wasted compute when sequence lengths vary a lot
- `__getitem__` stays clean and simple
- Works naturally with streaming or dynamically growing datasets

**Cons:**
- Slightly more code to write and maintain
- Batch shape varies between batches — minor overhead but rarely an issue in practice

---

**Which to use for the insurance case:**
Option 1 is fine — customer purchase histories are short and similar in length. Option 2 becomes worth it when sequences vary widely, e.g. one customer has 3 purchases and another has 50.
