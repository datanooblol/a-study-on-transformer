import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

# ── Vocabularies ───────────────────────────────────────────────────────────────
# each sequence type has its own set of event tokens
# 0 is reserved for PAD in all three — padding positions are ignored in attention

PURCHASES = ["Health", "Life", "CI", "Motor"]
CLAIMS    = ["None", "OPD", "Hospital", "Accident"]
PAYMENTS  = ["Paid", "Late"]

purchase_to_id = {p: i + 1 for i, p in enumerate(PURCHASES)}  # 1-indexed, 0 = PAD
claim_to_id    = {c: i + 1 for i, c in enumerate(CLAIMS)}
payment_to_id  = {p: i + 1 for i, p in enumerate(PAYMENTS)}


# ── Mock dataset patterns ──────────────────────────────────────────────────────
# two customer archetypes with clear learnable patterns:
#
# High-risk customer:
#   purchases  : Health, Hospital claims, frequent Late payments
#   claim amount: high (driven by hospital visits and late payments)
#
# Low-risk customer:
#   purchases  : Life or Motor, None/OPD claims, mostly Paid on time
#   claim amount: low

def generate_customer(risk_type: str):
    """Generate one customer's three sequences + profile + next-year claim amount."""
    if risk_type == "high":
        purchases = ["Health"] * torch.randint(2, 5, (1,)).item()
        claims    = ["Hospital"] * torch.randint(2, 4, (1,)).item() + ["OPD"] * torch.randint(0, 2, (1,)).item()
        payments  = ["Late"] * torch.randint(2, 4, (1,)).item() + ["Paid"] * torch.randint(1, 3, (1,)).item()
        age       = float(torch.randint(45, 70, (1,)).item())
        income    = float(torch.randint(30000, 60000, (1,)).item())
    else:
        purchases = ["Life", "Motor"][:torch.randint(1, 3, (1,)).item()]
        claims    = ["None"] * torch.randint(2, 4, (1,)).item() + ["OPD"] * torch.randint(0, 2, (1,)).item()
        payments  = ["Paid"] * torch.randint(3, 6, (1,)).item()
        age       = float(torch.randint(25, 45, (1,)).item())
        income    = float(torch.randint(60000, 120000, (1,)).item())

    # claim amount is a function of the customer's history — this is what the model learns to predict
    hospital_count = sum(c == "Hospital" for c in claims)
    late_count     = sum(p == "Late" for p in payments)
    claim_amount   = (200 * hospital_count + 5000 * late_count + 0.05 * income + age * 20
                      + torch.randn(1).item() * 1000)

    return purchases, claims, payments, age, income, claim_amount


def pad(seq, max_len):
    """Pad a sequence with 0s (PAD token) up to max_len."""
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


# fixed max lengths per sequence type — purchase histories are short, payment histories longer
MAX_PURCHASES = 5
MAX_CLAIMS    = 5
MAX_PAYMENTS  = 8


class InsuranceDataset(Dataset):
    def __init__(self, n_customers: int = 400):
        self.samples = []
        for _ in range(n_customers // 2):
            self.samples.append(generate_customer("high"))
            self.samples.append(generate_customer("low"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        purchases, claims, payments, age, income, claim_amount = self.samples[idx]

        return (
            # sequences: padded integer ids, shape (max_len,)
            torch.tensor(pad([purchase_to_id[p] for p in purchases], MAX_PURCHASES), dtype=torch.long),
            torch.tensor(pad([claim_to_id[c]    for c in claims],    MAX_CLAIMS),    dtype=torch.long),
            torch.tensor(pad([payment_to_id[p]  for p in payments],  MAX_PAYMENTS),  dtype=torch.long),
            # profile: age and income as a 2-element float vector
            torch.tensor([age, income], dtype=torch.float),
            # label: next-year claim amount to predict
            torch.tensor([claim_amount], dtype=torch.float),
        )


# ── Normalizer ─────────────────────────────────────────────────────────────────
# fit once on the full training set, reuse the same mean/std everywhere

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, values: torch.Tensor):
        self.mean = values.mean().item()
        self.std  = values.std().item()

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

age_normalizer    = Normalizer()
income_normalizer = Normalizer()
claim_normalizer  = Normalizer()


# ── RoPE ───────────────────────────────────────────────────────────────────────
# identical to encoder_block.py — encodes relative position within each sequence

class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        theta     = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_len).float()
        angles    = torch.outer(positions, theta)
        self.register_buffer("freqs", torch.stack([angles.cos(), angles.sin()], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        freqs   = self.freqs[:seq_len].unsqueeze(0).unsqueeze(2)
        x_pairs = x.unflatten(-1, (-1, 2))
        x0, x1  = x_pairs[..., 0], x_pairs[..., 1]
        cos, sin = freqs[..., 0], freqs[..., 1]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)


# ── Multi-Head Attention ───────────────────────────────────────────────────────
# identical to encoder_block.py — full bidirectional attention, no causal mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RoPE(self.head_dim, max_len)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (batch, n_heads, seq_len, seq_len) — every token attends to every other token
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if pad_mask is not None:
            # pad_mask is (batch, seq_len) — True means padding, set those to -inf
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        attn = scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.out_proj(out)


# ── FeedForward ────────────────────────────────────────────────────────────────
# identical to encoder_block.py

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_model * expansion)
        self.ff2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(F.relu(self.ff1(x)))


# ── Encoder Block ──────────────────────────────────────────────────────────────
# identical to encoder_block.py — self-attention + FFN, each wrapped with residual + LayerNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads, max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, pad_mask))
        x = self.norm2(x + self.ffn(x))
        return x


# ── Sequence Encoder ───────────────────────────────────────────────────────────
# wraps EncoderBlock for one sequence type (purchases / claims / payments)
# uses a CLS token to summarize the whole sequence into one vector
#
# CLS token: a learnable vector prepended to the sequence before encoding
# after the encoder runs, the CLS position has attended to every real token
# so we read it as a fixed-size summary of the entire sequence
# this is the same trick BERT uses to produce sentence embeddings

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        # +1 for PAD token at index 0 — padding_idx=0 keeps its embedding as zeros
        self.embed   = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        # learnable CLS token — one vector shared across all sequences in the batch
        self.cls     = nn.Parameter(torch.randn(1, 1, d_model))
        self.encoder = EncoderBlock(d_model, n_heads, max_len + 1)  # +1 for CLS position

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, seq_len) — integer ids, 0 = PAD
        batch = tokens.shape[0]

        # build padding mask before prepending CLS — True = padding position
        # CLS is never padding so we prepend a False column for it
        pad_mask = tokens == 0                                          # (batch, seq_len)
        cls_col  = torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device)
        pad_mask = torch.cat([cls_col, pad_mask], dim=1)               # (batch, 1 + seq_len)

        # embed tokens and prepend CLS
        x   = self.embed(tokens)                                        # (batch, seq_len, d_model)
        cls = self.cls.expand(batch, -1, -1)                           # (batch, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                               # (batch, 1 + seq_len, d_model)

        # run encoder — CLS attends to all real tokens, padding positions are masked out
        out = self.encoder(x, pad_mask)                                 # (batch, 1 + seq_len, d_model)

        # read only the CLS position — it now holds a summary of the whole sequence
        return out[:, 0]                                                # (batch, d_model)


# ── Insurance Encoder Model ────────────────────────────────────────────────────
# three separate SequenceEncoders — one per sequence type
# each produces a d_model summary vector for its sequence
# all three are concatenated with the profile vector, then fed into a regression head
#
# this is the encoder equivalent of InsuranceDecoderBlock in insurance_decoder_block.py:
#   decoder → predicts the NEXT event in a sequence (autoregressive)
#   encoder → compresses the FULL history into one embedding, predicts a single value

class InsuranceEncoderModel(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        # one encoder per sequence type — each has its own vocab and max length
        self.purchase_encoder = SequenceEncoder(len(PURCHASES), d_model, n_heads, MAX_PURCHASES)
        self.claim_encoder    = SequenceEncoder(len(CLAIMS),    d_model, n_heads, MAX_CLAIMS)
        self.payment_encoder  = SequenceEncoder(len(PAYMENTS),  d_model, n_heads, MAX_PAYMENTS)

        # profile (age + income) projected to d_model/2 — smaller since it's just 2 scalars
        self.profile_proj = nn.Linear(2, d_model // 2)

        # regression head: concat all four vectors → predict claim amount
        # input size: 3 sequences × d_model + d_model//2 profile
        self.head = nn.Sequential(
            nn.Linear(3 * d_model + d_model // 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 1),
        )

    def forward(
        self,
        purchases: torch.Tensor,   # (batch, MAX_PURCHASES)
        claims:    torch.Tensor,   # (batch, MAX_CLAIMS)
        payments:  torch.Tensor,   # (batch, MAX_PAYMENTS)
        profile:   torch.Tensor,   # (batch, 2) — normalized age and income
    ) -> torch.Tensor:
        # encode each sequence into one summary vector
        purchase_vec = self.purchase_encoder(purchases)   # (batch, d_model)
        claim_vec    = self.claim_encoder(claims)          # (batch, d_model)
        payment_vec  = self.payment_encoder(payments)      # (batch, d_model)
        profile_vec  = self.profile_proj(profile)          # (batch, d_model//2)

        # fuse all four into one customer vector
        customer = torch.cat([purchase_vec, claim_vec, payment_vec, profile_vec], dim=1)

        # predict next-year claim amount from the fused customer vector
        return self.head(customer)                         # (batch, 1)


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    dataset    = InsuranceDataset(n_customers=400)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # fit normalizers on the full training set before any training
    all_ages    = torch.stack([s[3] for s in dataset])
    all_incomes = torch.stack([s[3] for s in dataset])
    all_claims  = torch.cat([s[4] for s in dataset])

    # collect age and income separately from profile tensor
    profiles    = torch.stack([s[3] for s in dataset])   # (n, 2)
    age_normalizer.fit(profiles[:, 0])
    income_normalizer.fit(profiles[:, 1])
    claim_normalizer.fit(all_claims)

    print(f"age    — mean: {age_normalizer.mean:.1f},    std: {age_normalizer.std:.1f}")
    print(f"income — mean: {income_normalizer.mean:.1f}, std: {income_normalizer.std:.1f}")
    print(f"claim  — mean: {claim_normalizer.mean:.1f},  std: {claim_normalizer.std:.1f}\n")

    model     = InsuranceEncoderModel(d_model=32, n_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    for epoch in range(30):
        total_loss = 0.0
        model.train()

        for purchases, claims, payments, profile, claim_amount in dataloader:
            # normalize profile and label before feeding into model
            profile_norm = torch.stack([
                age_normalizer.transform(profile[:, 0]),
                income_normalizer.transform(profile[:, 1]),
            ], dim=1)
            claim_norm = claim_normalizer.transform(claim_amount)

            pred = model(purchases, claims, payments, profile_norm)
            loss = loss_fn(pred, claim_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1:3d} | loss {total_loss / len(dataloader):.4f}")

    return model


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(model, purchases, claims, payments, age, income, name):
    purchase_ids = torch.tensor([pad([purchase_to_id[p] for p in purchases], MAX_PURCHASES)], dtype=torch.long)
    claim_ids    = torch.tensor([pad([claim_to_id[c]    for c in claims],    MAX_CLAIMS)],    dtype=torch.long)
    payment_ids  = torch.tensor([pad([payment_to_id[p]  for p in payments],  MAX_PAYMENTS)],  dtype=torch.long)

    profile_norm = torch.tensor([[
        age_normalizer.transform(torch.tensor(age)).item(),
        income_normalizer.transform(torch.tensor(income)).item(),
    ]], dtype=torch.float)

    model.eval()
    with torch.no_grad():
        pred_norm = model(purchase_ids, claim_ids, payment_ids, profile_norm)

    pred_claim = claim_normalizer.inverse_transform(pred_norm[0]).item()

    print(f"{name}:")
    print(f"  purchases : {purchases}")
    print(f"  claims    : {claims}")
    print(f"  payments  : {payments}")
    print(f"  age       : {age},  income: {income}")
    print(f"  predicted next-year claim : {pred_claim:.0f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("training...\n")
    model = train()

    print("\ninference:\n")

    # high-risk customer: many hospital claims, late payments → should predict high claim amount
    predict(model,
        purchases = ["Health", "Health"],
        claims    = ["Hospital", "Hospital", "OPD"],
        payments  = ["Late", "Late", "Paid"],
        age       = 60,
        income    = 40000,
        name      = "Customer 1 (high-risk)"
    )

    # low-risk customer: life/motor, no claims, always paid on time → should predict low claim amount
    predict(model,
        purchases = ["Life", "Motor"],
        claims    = ["None", "None", "OPD"],
        payments  = ["Paid", "Paid", "Paid", "Paid"],
        age       = 30,
        income    = 90000,
        name      = "Customer 2 (low-risk)"
    )
