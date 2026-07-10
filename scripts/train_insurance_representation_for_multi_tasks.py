import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

# ── Vocabularies ───────────────────────────────────────────────────────────────
# 0 is reserved for PAD in all three — padding positions are ignored in attention

PURCHASES = ["Health", "Life", "CI", "Motor"]
CLAIMS    = ["None", "OPD", "Hospital", "Accident"]
PAYMENTS  = ["Paid", "Late"]

purchase_to_id = {p: i + 1 for i, p in enumerate(PURCHASES)}
claim_to_id    = {c: i + 1 for i, c in enumerate(CLAIMS)}
payment_to_id  = {p: i + 1 for i, p in enumerate(PAYMENTS)}


# ── Mock dataset patterns ──────────────────────────────────────────────────────
# four prediction targets per customer:
#
#   claim_amount  — next-year total claim cost (regression)
#   lapse_prob    — probability of policy lapse / non-renewal (regression, 0–1)
#   pay_amount    — expected premium payment next year (regression)
#   risk_score    — composite risk index combining all signals (regression)
#
# high-risk customer: Health, Hospital claims, Late payments → high on all four
# low-risk customer : Life/Motor, None/OPD claims, Paid      → low on all four

def generate_customer(risk_type: str):
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

    hospital_count = sum(c == "Hospital" for c in claims)
    late_count     = sum(p == "Late" for p in payments)

    # claim_amount: driven by hospital visits, late payments, age, income
    claim_amount = (200 * hospital_count + 5000 * late_count + 0.05 * income
                    + age * 20 + torch.randn(1).item() * 1000)

    # lapse_prob: late payers and low-income customers are more likely to lapse
    lapse_prob = torch.sigmoid(torch.tensor(
        0.8 * late_count - 0.3 * (income / 10000) + torch.randn(1).item() * 0.3
    )).item()

    # pay_amount: premium is higher for Health/CI policies and older customers
    health_count = sum(p in ("Health", "CI") for p in purchases)
    pay_amount   = (3000 * health_count + 1500 * (len(purchases) - health_count)
                    + age * 50 + torch.randn(1).item() * 500)

    # risk_score: composite index — higher means riskier customer overall
    risk_score = (hospital_count * 2 + late_count * 3 + age / 10
                  + (1 - income / 120000) * 5 + torch.randn(1).item() * 0.5)

    return purchases, claims, payments, age, income, claim_amount, lapse_prob, pay_amount, risk_score


def pad(seq, max_len):
    """Pad a sequence with 0s (PAD token) up to max_len."""
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


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
        purchases, claims, payments, age, income, claim_amount, lapse_prob, pay_amount, risk_score = self.samples[idx]

        return (
            torch.tensor(pad([purchase_to_id[p] for p in purchases], MAX_PURCHASES), dtype=torch.long),
            torch.tensor(pad([claim_to_id[c]    for c in claims],    MAX_CLAIMS),    dtype=torch.long),
            torch.tensor(pad([payment_to_id[p]  for p in payments],  MAX_PAYMENTS),  dtype=torch.long),
            torch.tensor([age, income], dtype=torch.float),
            # four targets — each is a (1,) tensor so they stack cleanly in the dataloader
            torch.tensor([claim_amount], dtype=torch.float),
            torch.tensor([lapse_prob],   dtype=torch.float),
            torch.tensor([pay_amount],   dtype=torch.float),
            torch.tensor([risk_score],   dtype=torch.float),
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
lapse_normalizer  = Normalizer()
pay_normalizer    = Normalizer()
risk_normalizer   = Normalizer()


# ── Uncertainty Weighted Loss ──────────────────────────────────────────────────
# instead of manually tuning loss weights for each task, we learn them
#
# idea: each task has a learnable log_var (log of variance / uncertainty)
#   precision = exp(-log_var)  →  noisy/hard tasks get high variance → low precision → graded leniently
#   the 0.5 * log_var term is a regularizer that prevents all precisions from collapsing to zero
#
# reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" (2018)

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks: int):
        super().__init__()
        # one learnable log_var per task — initialized to 0 (precision = 1)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])          # high log_var → low precision
            total     = total + precision * loss + 0.5 * self.log_vars[i]
        return total


# ── RoPE ───────────────────────────────────────────────────────────────────────

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

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        attn = scores.softmax(dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.out_proj(out)


# ── FeedForward ────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_model * expansion)
        self.ff2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(F.relu(self.ff1(x)))


# ── Encoder Block ──────────────────────────────────────────────────────────────

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
# wraps EncoderBlock for one sequence type — CLS token summarizes the whole sequence

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.cls     = nn.Parameter(torch.randn(1, 1, d_model))
        self.encoder = EncoderBlock(d_model, n_heads, max_len + 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]

        pad_mask = tokens == 0
        cls_col  = torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device)
        pad_mask = torch.cat([cls_col, pad_mask], dim=1)

        x   = self.embed(tokens)
        cls = self.cls.expand(batch, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        out = self.encoder(x, pad_mask)
        return out[:, 0]                                                # (batch, d_model)


# ── Insurance Multi-Task Encoder Model ────────────────────────────────────────
# same encoder backbone as the single-task version
# the only difference: instead of one regression head, we have four — one per target
#
# all four heads share the same fused customer vector as input
# this is called hard parameter sharing — the encoder learns one representation
# that is useful for all tasks simultaneously

class InsuranceEncoderModel(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        self.purchase_encoder = SequenceEncoder(len(PURCHASES), d_model, n_heads, MAX_PURCHASES)
        self.claim_encoder    = SequenceEncoder(len(CLAIMS),    d_model, n_heads, MAX_CLAIMS)
        self.payment_encoder  = SequenceEncoder(len(PAYMENTS),  d_model, n_heads, MAX_PAYMENTS)

        self.profile_proj = nn.Linear(2, d_model // 2)

        # fused input size: 3 sequences × d_model + d_model//2 profile
        fused_dim = 3 * d_model + d_model // 2

        # four separate heads — each predicts one target independently
        # they all read from the same fused customer vector
        self.claim_head = nn.Sequential(nn.Linear(fused_dim, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, 1))
        self.lapse_head = nn.Sequential(nn.Linear(fused_dim, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, 1))
        self.pay_head   = nn.Sequential(nn.Linear(fused_dim, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, 1))
        self.risk_head  = nn.Sequential(nn.Linear(fused_dim, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, 1))

    def forward(self, purchases, claims, payments, profile):
        purchase_vec = self.purchase_encoder(purchases)
        claim_vec    = self.claim_encoder(claims)
        payment_vec  = self.payment_encoder(payments)
        profile_vec  = self.profile_proj(profile)

        # one shared customer vector — all four heads read from this
        customer = torch.cat([purchase_vec, claim_vec, payment_vec, profile_vec], dim=1)

        return (
            self.claim_head(customer),   # (batch, 1)
            self.lapse_head(customer),   # (batch, 1)
            self.pay_head(customer),     # (batch, 1)
            self.risk_head(customer),    # (batch, 1)
        )


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    dataset    = InsuranceDataset(n_customers=400)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # fit all normalizers on the full dataset before training
    profiles     = torch.stack([s[3] for s in dataset])
    age_normalizer.fit(profiles[:, 0])
    income_normalizer.fit(profiles[:, 1])
    claim_normalizer.fit(torch.cat([s[4] for s in dataset]))
    lapse_normalizer.fit(torch.cat([s[5] for s in dataset]))
    pay_normalizer.fit(torch.cat([s[6] for s in dataset]))
    risk_normalizer.fit(torch.cat([s[7] for s in dataset]))

    print(f"age    — mean: {age_normalizer.mean:.1f},    std: {age_normalizer.std:.1f}")
    print(f"income — mean: {income_normalizer.mean:.1f}, std: {income_normalizer.std:.1f}")
    print(f"claim  — mean: {claim_normalizer.mean:.1f},  std: {claim_normalizer.std:.1f}")
    print(f"lapse  — mean: {lapse_normalizer.mean:.3f},  std: {lapse_normalizer.std:.3f}")
    print(f"pay    — mean: {pay_normalizer.mean:.1f},    std: {pay_normalizer.std:.1f}")
    print(f"risk   — mean: {risk_normalizer.mean:.2f},   std: {risk_normalizer.std:.2f}\n")

    model       = InsuranceEncoderModel(d_model=32, n_heads=4)
    loss_weigher = UncertaintyWeightedLoss(n_tasks=4)
    optimizer   = torch.optim.Adam(list(model.parameters()) + list(loss_weigher.parameters()), lr=1e-3)
    mse         = nn.MSELoss()

    for epoch in range(30):
        total_loss = 0.0
        model.train()

        for purchases, claims, payments, profile, claim_amount, lapse_prob, pay_amount, risk_score in dataloader:
            profile_norm = torch.stack([
                age_normalizer.transform(profile[:, 0]),
                income_normalizer.transform(profile[:, 1]),
            ], dim=1)

            # normalize all four targets
            claim_norm = claim_normalizer.transform(claim_amount)
            lapse_norm = lapse_normalizer.transform(lapse_prob)
            pay_norm   = pay_normalizer.transform(pay_amount)
            risk_norm  = risk_normalizer.transform(risk_score)

            pred_claim, pred_lapse, pred_pay, pred_risk = model(purchases, claims, payments, profile_norm)

            # one MSE loss per task — UncertaintyWeightedLoss balances them automatically
            loss = loss_weigher([
                mse(pred_claim, claim_norm),
                mse(pred_lapse, lapse_norm),
                mse(pred_pay,   pay_norm),
                mse(pred_risk,  risk_norm),
            ])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            # show learned task weights so we can see how the model balances the four tasks
            weights = torch.exp(-loss_weigher.log_vars).detach()
            print(f"epoch {epoch+1:3d} | loss {total_loss / len(dataloader):.4f} "
                  f"| weights claim={weights[0]:.2f} lapse={weights[1]:.2f} "
                  f"pay={weights[2]:.2f} risk={weights[3]:.2f}")

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
        pred_claim, pred_lapse, pred_pay, pred_risk = model(purchase_ids, claim_ids, payment_ids, profile_norm)

    print(f"{name}:")
    print(f"  purchases : {purchases}")
    print(f"  claims    : {claims}")
    print(f"  payments  : {payments}")
    print(f"  age: {age},  income: {income}")
    print(f"  predicted claim amount : {claim_normalizer.inverse_transform(pred_claim[0]).item():.0f}")
    print(f"  predicted lapse prob   : {lapse_normalizer.inverse_transform(pred_lapse[0]).item():.3f}")
    print(f"  predicted pay amount   : {pay_normalizer.inverse_transform(pred_pay[0]).item():.0f}")
    print(f"  predicted risk score   : {risk_normalizer.inverse_transform(pred_risk[0]).item():.2f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("training...\n")
    model = train()

    print("\ninference:\n")

    predict(model,
        purchases = ["Health", "Health"],
        claims    = ["Hospital", "Hospital", "OPD"],
        payments  = ["Late", "Late", "Paid"],
        age       = 60,
        income    = 40000,
        name      = "Customer 1 (high-risk)"
    )

    predict(model,
        purchases = ["Life", "Motor"],
        claims    = ["None", "None", "OPD"],
        payments  = ["Paid", "Paid", "Paid", "Paid"],
        age       = 30,
        income    = 90000,
        name      = "Customer 2 (low-risk)"
    )
