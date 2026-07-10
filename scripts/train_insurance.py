import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from package.insurance_decoder_block import InsuranceDecoderBlock

# ── Policy vocabulary ──────────────────────────────────────────────────────────
POLICIES     = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
policy_to_id = {p: i for i, p in enumerate(POLICIES)}
id_to_policy = {i: p for p, i in policy_to_id.items()}

# ── Mock dataset patterns ──────────────────────────────────────────────────────
# two customer archetypes with clear learnable patterns:
#
# Type 1 — "A-type customer": starts young, buys cheap policies (A, C), small price
#   policy sequence : A → C → A → C → A ...
#   age             : starts 25-35, increases by 3-5 each purchase
#   price           : 100-200 range
#
# Type 2 — "B-type customer": starts older, buys expensive policies (B, D), large price
#   policy sequence : B → D → B → D → B ...
#   age             : starts 40-50, increases by 2-4 each purchase
#   price           : 300-500 range

def generate_customer(customer_type: str, seq_len: int = 5):
    """Generate one customer's purchase history as (policies, ages, prices)."""
    policies, ages, prices = [], [], []

    if customer_type == "A":
        age   = float(torch.randint(25, 35, (1,)).item())
        cycle = ["A", "C"]
    else:
        age   = float(torch.randint(40, 50, (1,)).item())
        cycle = ["B", "D"]

    for i in range(seq_len):
        policy = cycle[i % 2]
        price  = (100 + torch.randn(1).item() * 20) if customer_type == "A" else (400 + torch.randn(1).item() * 30)

        policies.append(policy_to_id[policy])
        ages.append(age)
        prices.append(price)

        age += float(torch.randint(2, 6, (1,)).item())  # age increases each purchase

    return policies, ages, prices


class InsuranceDataset(Dataset):
    def __init__(self, n_customers: int = 500, seq_len: int = 5):
        self.samples = []
        for _ in range(n_customers // 2):
            self.samples.append(generate_customer("A", seq_len))
            self.samples.append(generate_customer("B", seq_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        policies, ages, prices = self.samples[idx]
        return (
            torch.tensor(policies, dtype=torch.long),           # (seq_len,)
            torch.tensor(ages,     dtype=torch.float).unsqueeze(-1),   # (seq_len, 1)
            torch.tensor(prices,   dtype=torch.float).unsqueeze(-1),   # (seq_len, 1)
        )


# ── Normalizer ────────────────────────────────────────────────────────────────
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

age_normalizer   = Normalizer()
price_normalizer = Normalizer()


# ── Uncertainty Weighted Loss ────────────────────────────────────────────────
# instead of manually tuning loss weights, let the model learn how strictly
# to grade each task based on how noisy/predictable each one is

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, task_names):
        super().__init__()
        self.task_names = task_names
        # start all tasks as equally strict (log_var=0 → sigma=1 → precision=1)
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, losses: dict):
        total = 0.0
        for i, name in enumerate(self.task_names):
            log_var   = self.log_vars[i]
            precision = torch.exp(-log_var)   # 1 / sigma^2 — how strictly this task is graded
            total    += precision * losses[name] + log_var
        return total

    def precisions(self):
        # for logging — higher precision = model is grading this task more strictly
        return {name: torch.exp(-self.log_vars[i]).item() for i, name in enumerate(self.task_names)}


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    dataset    = InsuranceDataset(n_customers=500, seq_len=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # fit normalizers on the full training set before any training
    all_ages   = torch.cat([s[1] for s in dataset])
    all_prices = torch.cat([s[2] for s in dataset])
    age_normalizer.fit(all_ages)
    price_normalizer.fit(all_prices)
    print(f"age   — mean: {age_normalizer.mean:.1f}, std: {age_normalizer.std:.1f}")
    print(f"price — mean: {price_normalizer.mean:.1f}, std: {price_normalizer.std:.1f}\n")

    model        = InsuranceDecoderBlock(num_policies=10, d_model=64, n_heads=4, max_len=50)
    loss_weigher = UncertaintyWeightedLoss(task_names=["policy", "age", "price"])
    # include loss_weigher parameters so log_vars are trained alongside the model
    optimizer    = torch.optim.Adam(list(model.parameters()) + list(loss_weigher.parameters()), lr=1e-3)

    policy_loss_fn = nn.CrossEntropyLoss()  # classification — next policy
    cont_loss_fn   = nn.MSELoss()           # regression — next age and price

    for epoch in range(30):
        total_loss = 0.0

        for policy, age, price in dataloader:
            age_norm   = age_normalizer.transform(age)
            price_norm = price_normalizer.transform(price)

            policy_logits, age_pred, price_pred = model(policy, age_norm, price_norm)

            # position i predicts event i+1 — shift targets by 1
            # input : events 0..n-2   (all but last)
            # target: events 1..n-1   (all but first)
            pred_policy = policy_logits[:, :-1]   # (batch, seq_len-1, num_policies)
            pred_age    = age_pred[:, :-1]         # (batch, seq_len-1, 1)
            pred_price  = price_pred[:, :-1]       # (batch, seq_len-1, 1)

            tgt_policy  = policy[:, 1:]            # (batch, seq_len-1)
            tgt_age     = age_norm[:, 1:]          # (batch, seq_len-1, 1)
            tgt_price   = price_norm[:, 1:]        # (batch, seq_len-1, 1)

            # CrossEntropyLoss expects (batch, num_classes, seq_len)
            loss_policy = policy_loss_fn(pred_policy.transpose(1, 2), tgt_policy)
            loss_age    = cont_loss_fn(pred_age, tgt_age)
            loss_price  = cont_loss_fn(pred_price, tgt_price)

            loss = loss_weigher({"policy": loss_policy, "age": loss_age, "price": loss_price})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1:3d} | loss {total_loss / len(dataloader):.4f} | precisions {loss_weigher.precisions()}")

    return model


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(model, customer_policies, customer_ages, customer_prices, name):
    policy = torch.tensor([[policy_to_id[p] for p in customer_policies]])  # (1, seq_len)
    age    = torch.tensor([customer_ages],  dtype=torch.float).unsqueeze(-1)  # (1, seq_len, 1)
    price  = torch.tensor([customer_prices], dtype=torch.float).unsqueeze(-1) # (1, seq_len, 1)

    age_norm   = age_normalizer.transform(age)
    price_norm = price_normalizer.transform(price)

    model.eval()
    with torch.no_grad():
        policy_logits, age_pred, price_pred = model(policy, age_norm, price_norm)

    # read from last position — forecast for the next event
    next_policy_id = policy_logits[0, -1].argmax().item()
    next_policy    = id_to_policy[next_policy_id]
    next_age   = age_normalizer.inverse_transform(age_pred[0, -1]).item()
    next_price = price_normalizer.inverse_transform(price_pred[0, -1]).item()

    # top 3 policy probabilities so we can see model confidence
    probs      = policy_logits[0, -1].softmax(dim=-1)
    top3_probs, top3_ids = probs.topk(3)
    top3       = [(id_to_policy[i.item()], f"{p.item():.0%}") for i, p in zip(top3_ids, top3_probs)]

    print(f"{name}:")
    print(f"  history          : {customer_policies}")
    print(f"  next policy      : {next_policy}  (top 3: {top3})")
    print(f"  next age         : {next_age:.1f}")
    print(f"  next price       : {next_price:.1f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("training...\n")
    model = train()

    print("\ninference:\n")

    # A-type customer: should predict C next, age ~33, price ~100
    predict(model,
        customer_policies = ["A", "C", "A"],
        customer_ages     = [25.0, 28.0, 31.0],
        customer_prices   = [105.0, 95.0, 110.0],
        name              = "Customer 1 (A-type)"
    )

    # B-type customer: should predict D next, age ~47, price ~400
    predict(model,
        customer_policies = ["B", "D", "B"],
        customer_ages     = [40.0, 43.0, 46.0],
        customer_prices   = [410.0, 390.0, 420.0],
        name              = "Customer 2 (B-type)"
    )

    predict(model,
        customer_policies = ["A"],
        customer_ages     = [25.0],
        customer_prices   = [105.0],
        name              = "Customer 3 (A-type)"
    )

    predict(model,
        customer_policies = ["B", "D",],
        customer_ages     = [40.0, 43.0],
        customer_prices   = [410.0, 390.0],
        name              = "Customer 4 (B-type)"
    )