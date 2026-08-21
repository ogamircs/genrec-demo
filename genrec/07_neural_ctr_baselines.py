"""Step 7: Neural CTR baselines — DeepFM and Wide & Deep.

Two deep architectures from the CTR-prediction literature, applied to the same
leave-one-out next-item task as every other baseline. They differ in exactly one
place — how low-order feature interactions get into the model — which is the
comparison this stage exists to make:

  DeepFM      (Guo et al. 2017,  arXiv 1703.04247)
      y = w0 + sum_j w_xj + sum_{j<k} <v_j, v_k> + MLP([v_1; ...; v_F])
      An FM component and a deep MLP *share one embedding table*; the FM term
      learns all pairwise interactions automatically, no feature engineering.

  Wide & Deep (Cheng et al. 2016, arXiv 1606.07792)
      y = w_wide . [x, phi(x)] + MLP([v_1; ...; v_F]) + b
      A linear "wide" model over the raw fields *plus hand-built cross-product
      features* phi(x) (memorization), jointly trained with a deep MLP that has
      its own embeddings (generalization). DeepFM's contribution was replacing
      phi(x) with the FM term and sharing the embeddings.

Fields (F = 6), all categorical, all derived from the training split only:

    user_id, user_activity_bucket, user_avg_star_bucket,
    item_id, item_popularity_bucket, item_avg_star_bucket

Wide & Deep cross-products phi(x) (the wide part's memorization capacity):

    user_id x item_popularity_bucket      user_activity x item_popularity
    user_id x item_avg_star_bucket        user_avg_star x item_avg_star

  (user_id x item_id is deliberately excluded: with one row per observed pair it
  would be a lookup table over the training positives and generalize to nothing.)

Both are trained as implicit feedback with negative sampling — the standard way
to drive a CTR model for top-N retrieval: every observed interaction is a
positive, plus `n_neg` uniformly sampled unvisited items per positive, binary
cross-entropy. Deviation from the Wide & Deep paper: it trains the wide half
with FTRL+L1 and the deep half with AdaGrad; PyTorch has no FTRL, so both halves
(and both models) use Adam, which also keeps the comparison optimizer-neutral.

Hyperparameters and epoch count are chosen per architecture on an internal
validation split (each user's last *training* interaction held out), so no test
information is used; each model is then refit on the full training data for the
winning epoch count. `--sweep` re-runs the 18-config grid behind DEFAULTS.

Outputs results/recs_deepfm.jsonl and results/recs_widedeep.jsonl in the same
format as the other baselines (top-10 normalized business names per user,
visited items masked).

Run with the same environment as steps 01-05 (torch + pandas + pyarrow):
  python genrec/07_neural_ctr_baselines.py                 # both models
  python genrec/07_neural_ctr_baselines.py --arch widedeep --sweep
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TOP_N = 10
BATCH = 8192
L2 = 1e-6
MAX_EPOCHS = 60
PATIENCE = 8            # validation epochs without improvement before stopping
SEED = 42
N_BUCKETS = 12          # slots reserved per bucketed side feature

ARCHS = ("deepfm", "widedeep")

# tuned with --sweep on the internal validation split (18-config grid per model,
# dim x n_neg x dropout). Both architectures picked the same winner; the one
# consistent effect across the grid was more negatives (16 > 8 > 4 at every dim,
# for both models), not more capacity.
DEFAULTS = {
    "deepfm":   {"dim": 16, "hidden": (128, 64), "dropout": 0.0, "n_neg": 16, "lr": 1e-3},
    "widedeep": {"dim": 16, "hidden": (128, 64), "dropout": 0.0, "n_neg": 16, "lr": 1e-3},
}

# cross-products for the Wide & Deep wide part, as (user field, item field)
# indices into the 3-column user/item feature rows built below.
CROSSES = ((0, 1), (0, 2), (1, 1), (2, 2))


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def count_bucket(counts: np.ndarray) -> np.ndarray:
    """log2 bucket of an interaction count: 0 -> 0, 1 -> 1, 2-3 -> 2, ... capped."""
    b = np.zeros(len(counts), dtype=np.int64)
    nz = counts > 0
    b[nz] = np.floor(np.log2(counts[nz])).astype(np.int64) + 1
    return np.clip(b, 0, N_BUCKETS - 1)


def star_bucket(means: np.ndarray, fallback: float) -> np.ndarray:
    """avg rating -> half-star bucket 1..9 (1.0 .. 5.0); unseen -> global mean."""
    m = np.where(np.isnan(means), fallback, means)
    return np.clip(np.rint(m * 2).astype(np.int64) - 1, 1, N_BUCKETS - 1)


class Features:
    """Per-user and per-item categorical field rows, computed from one fit set.

    `user_loc` / `item_loc` hold the raw category values per field (used for the
    cross-products); `user_glob` / `item_glob` hold the same values offset into a
    single shared embedding vocabulary so no two fields collide.
    """

    def __init__(self, n_u: int, n_i: int, uu: np.ndarray, ii: np.ndarray,
                 rr: np.ndarray):
        u_cnt = np.bincount(uu, minlength=n_u).astype(np.float64)
        i_cnt = np.bincount(ii, minlength=n_i).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            u_mean = np.bincount(uu, weights=rr, minlength=n_u) / u_cnt
            i_mean = np.bincount(ii, weights=rr, minlength=n_i) / i_cnt
        mu = float(rr.mean())

        self.user_loc = np.stack([np.arange(n_u), count_bucket(u_cnt),
                                  star_bucket(u_mean, mu)], axis=1)
        self.item_loc = np.stack([np.arange(n_i), count_bucket(i_cnt),
                                  star_bucket(i_mean, mu)], axis=1)
        self.user_card = [n_u, N_BUCKETS, N_BUCKETS]
        self.item_card = [n_i, N_BUCKETS, N_BUCKETS]

        offs, total = [], 0
        for c in self.user_card + self.item_card:
            offs.append(total)
            total += c
        self.vocab = total
        self.user_glob = self.user_loc + np.array(offs[:3])
        self.item_glob = self.item_loc + np.array(offs[3:])

        # cross-product vocabularies, also offset into one table
        self.cross_size, self.cross_off, tot = [], [], 0
        for a, b in CROSSES:
            sz = self.user_card[a] * self.item_card[b]
            self.cross_size.append(sz)
            self.cross_off.append(tot)
            tot += sz
        self.cross_vocab = tot


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CTRBase(nn.Module):
    """Holds the feature tables; subclasses turn (user, item) indices into logits."""

    def __init__(self, feats: Features):
        super().__init__()
        self.register_buffer("user_glob", torch.as_tensor(feats.user_glob))
        self.register_buffer("item_glob", torch.as_tensor(feats.item_glob))
        self.n_fields = feats.user_glob.shape[1] + feats.item_glob.shape[1]

    def fields(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.user_glob[u], self.item_glob[i]], dim=1)

    @staticmethod
    def mlp(n_in: int, hidden, dropout: float) -> nn.Sequential:
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)


class DeepFM(CTRBase):
    """FM component + deep MLP over one shared embedding table."""

    def __init__(self, feats: Features, cfg: dict):
        super().__init__(feats)
        self.emb = nn.Embedding(feats.vocab, cfg["dim"])   # shared by both parts
        self.lin = nn.Embedding(feats.vocab, 1)            # FM order-1 weights
        self.bias = nn.Parameter(torch.zeros(1))
        self.deep = self.mlp(self.n_fields * cfg["dim"], cfg["hidden"], cfg["dropout"])
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.lin.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        x = self.fields(u, i)
        v = self.emb(x)                                     # (B, F, d)
        first = self.lin(x).sum(dim=1).squeeze(-1) + self.bias
        s = v.sum(dim=1)
        second = 0.5 * ((s * s).sum(-1) - (v * v).sum(dim=(1, 2)))
        return first + second + self.deep(v.flatten(1)).squeeze(-1)


class WideDeep(CTRBase):
    """Linear wide part over raw fields + cross-products, plus a deep MLP.

    The two halves are jointly trained and, unlike DeepFM, do not share
    parameters: the wide part is a sparse linear model, the deep part has its
    own embedding table.
    """

    def __init__(self, feats: Features, cfg: dict):
        super().__init__(feats)
        self.register_buffer("user_loc", torch.as_tensor(feats.user_loc))
        self.register_buffer("item_loc", torch.as_tensor(feats.item_loc))
        self.register_buffer("cross_mul", torch.as_tensor(
            [feats.item_card[b] for _, b in CROSSES]))
        self.register_buffer("cross_off", torch.as_tensor(feats.cross_off))
        self.ua = [a for a, _ in CROSSES]
        self.ib = [b for _, b in CROSSES]

        self.wide = nn.Embedding(feats.vocab, 1)            # raw-field weights
        self.cross = nn.Embedding(feats.cross_vocab, 1)     # cross-product weights
        self.bias = nn.Parameter(torch.zeros(1))
        self.emb = nn.Embedding(feats.vocab, cfg["dim"])    # deep-only embeddings
        self.deep = self.mlp(self.n_fields * cfg["dim"], cfg["hidden"], cfg["dropout"])
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.cross.weight)

    def cross_index(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """phi(x): one index per cross-product, offset into the shared table."""
        ul = self.user_loc[u][:, self.ua]                   # (B, C)
        il = self.item_loc[i][:, self.ib]                   # (B, C)
        return ul * self.cross_mul + il + self.cross_off

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        x = self.fields(u, i)
        wide = (self.wide(x).sum(dim=1).squeeze(-1)
                + self.cross(self.cross_index(u, i)).sum(dim=1).squeeze(-1)
                + self.bias)
        deep = self.deep(self.emb(x).flatten(1)).squeeze(-1)
        return wide + deep


def build_model(arch: str, feats: Features, cfg: dict) -> CTRBase:
    return (DeepFM if arch == "deepfm" else WideDeep)(feats, cfg)


# ---------------------------------------------------------------------------
# Training / scoring
# ---------------------------------------------------------------------------

def sample_negatives(seen: np.ndarray, u_pos: np.ndarray, n_i: int,
                     rng: np.random.Generator, n_neg: int):
    """Uniform negatives, rejecting items the user interacted with in `seen`."""
    u_rep = np.repeat(u_pos, n_neg)
    neg = rng.integers(0, n_i, size=u_rep.shape[0])
    bad = seen[u_rep, neg]
    while bad.any():
        neg[bad] = rng.integers(0, n_i, size=int(bad.sum()))
        bad = seen[u_rep, neg]
    return u_rep, neg


@torch.no_grad()
def score_all(model: CTRBase, n_u: int, n_i: int, device: torch.device,
              chunk: int = 16) -> np.ndarray:
    """Full (n_users, n_items) score matrix, a few users at a time."""
    model.eval()
    out = np.empty((n_u, n_i), dtype=np.float32)
    items = torch.arange(n_i, device=device)
    for s in range(0, n_u, chunk):
        u = torch.arange(s, min(s + chunk, n_u), device=device)
        c = int(u.shape[0])
        logits = model(u.repeat_interleave(n_i), items.repeat(c))
        out[s:s + c] = logits.view(c, n_i).float().cpu().numpy()
    return out


def top_n_names(scores: np.ndarray, seen: np.ndarray, users: list,
                item_names: list, n: int = TOP_N) -> dict:
    """Rank by score, mask visited items, dedupe by normalized name (as in step 2)."""
    masked = scores.copy()
    masked[seen] = -np.inf
    order = np.argsort(-masked, axis=1, kind="stable")
    recs = {}
    for u_pos, uid in enumerate(users):
        names, dedup = [], set()
        for i in order[u_pos]:
            if not np.isfinite(masked[u_pos, i]):
                break
            nm = item_names[i]
            if nm not in dedup:
                dedup.add(nm)
                names.append(nm)
                if len(names) == n:
                    break
        recs[uid] = names
    return recs


def train(arch: str, uu: np.ndarray, ii: np.ndarray, rr: np.ndarray,
          seen: np.ndarray, n_u: int, n_i: int, device: torch.device,
          epochs: int, cfg: dict, val_target=None, item_names=None, users=None,
          verbose: bool = True, seed: int = SEED):
    """Fit one architecture on the given interactions.

    With val_target, tracks validation HR@10 each epoch, keeps the best weights
    and early-stops; otherwise trains for exactly `epochs` epochs.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    feats = Features(n_u, n_i, uu, ii, rr)
    model = build_model(arch, feats, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=L2)
    loss_fn = nn.BCEWithLogitsLoss()

    n_pos = len(uu)
    hr_history = []
    best_hr, best_state, best_epoch, since_best = -1.0, None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        u_neg, i_neg = sample_negatives(seen, uu, n_i, rng, cfg["n_neg"])
        u_all = np.concatenate([uu, u_neg])
        i_all = np.concatenate([ii, i_neg])
        y_all = np.concatenate([np.ones(n_pos, dtype=np.float32),
                                np.zeros(len(u_neg), dtype=np.float32)])
        perm = rng.permutation(len(u_all))
        u_t = torch.as_tensor(u_all[perm], device=device)
        i_t = torch.as_tensor(i_all[perm], device=device)
        y_t = torch.as_tensor(y_all[perm], device=device)

        total = 0.0
        for s in range(0, len(u_t), BATCH):
            ub, ib = u_t[s:s + BATCH], i_t[s:s + BATCH]
            loss = loss_fn(model(ub, ib), y_t[s:s + BATCH])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ub)
        msg = f"    epoch {epoch:2d}/{epochs} loss {total / len(u_t):.4f}"

        if val_target is not None:
            scores = score_all(model, n_u, n_i, device)
            recs = top_n_names(scores, seen, users, item_names)
            hr = float(np.mean([val_target[u] in recs[u] for u in users]))
            hr_history.append(hr)
            msg += f"  val HR@10 {hr:.4f}"
            if hr > best_hr:
                best_hr, best_epoch, since_best = hr, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
        if verbose:
            print(msg, flush=True)
        if val_target is not None and since_best >= PATIENCE:
            if verbose:
                print(f"    early stop (no val gain for {PATIENCE} epochs); "
                      f"best epoch {best_epoch} HR@10 {best_hr:.4f}")
            break

    if val_target is not None and best_state is not None:
        model.load_state_dict(best_state)
        model.best_epoch, model.best_hr = best_epoch, best_hr
    return model, hr_history


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="both", choices=list(ARCHS) + ["both"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="skip the validation phase and train this many epochs")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--neg", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search dim/n_neg/dropout on the validation split and exit")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="init/negative-sampling seed (see the stability note in REPORT.md)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename, e.g. --tag _s7")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    archs = list(ARCHS) if args.arch == "both" else [args.arch]
    device = torch.device(args.device)
    print(f"device: {device}")

    train_df = pd.read_parquet(DATA / "train_interactions.parquet")
    # implicit feedback: a handful of users rated the same business twice, keep
    # the most recent (train is date-sorted) -- same as the SAR/BPR baselines
    train_df = train_df.drop_duplicates(subset=["user_id", "business_id"], keep="last")

    users = sorted(train_df["user_id"].unique())
    items = sorted(train_df["business_id"].unique())
    u_idx = {u: n for n, u in enumerate(users)}
    i_idx = {b: n for n, b in enumerate(items)}
    n_u, n_i = len(users), len(items)
    print(f"train matrix: {n_u} users x {n_i} items, {len(train_df)} unique pairs")

    id2name = dict(zip(train_df["business_id"], train_df["business_name"].map(norm_name)))
    item_names = [id2name[b] for b in items]

    uu = train_df["user_id"].map(u_idx).to_numpy()
    ii = train_df["business_id"].map(i_idx).to_numpy()
    rr = train_df["stars"].to_numpy(dtype=np.float64)
    seen = np.zeros((n_u, n_i), dtype=bool)
    seen[uu, ii] = True

    # each user's last *training* interaction, held out for model selection;
    # the test items are never touched, so nothing here leaks
    val_idx = train_df.groupby("user_id", sort=False).tail(1).index
    val_df, fit_df = train_df.loc[val_idx], train_df.drop(index=val_idx)
    val_target = {r.user_id: norm_name(r.business_name) for r in val_df.itertuples()}
    f_uu = fit_df["user_id"].map(u_idx).to_numpy()
    f_ii = fit_df["business_id"].map(i_idx).to_numpy()
    f_rr = fit_df["stars"].to_numpy(dtype=np.float64)
    f_seen = np.zeros((n_u, n_i), dtype=bool)
    f_seen[f_uu, f_ii] = True

    def fit_validated(arch, cfg, verbose=True):
        return train(arch, f_uu, f_ii, f_rr, f_seen, n_u, n_i, device, MAX_EPOCHS,
                     cfg, val_target=val_target, item_names=item_names,
                     users=users, verbose=verbose, seed=args.seed)

    for arch in archs:
        d = DEFAULTS[arch]
        cfg = {"dim": args.dim or d["dim"], "hidden": d["hidden"],
               "dropout": d["dropout"] if args.dropout is None else args.dropout,
               "n_neg": args.neg or d["n_neg"], "lr": args.lr or d["lr"]}
        print(f"\n=== {arch} ===")

        if args.sweep:
            grid = [dict(d, dim=dim, n_neg=n_neg, dropout=dr)
                    for dim in (16, 32, 64)
                    for n_neg in (4, 8, 16)
                    for dr in (0.0, 0.2)]
            print(f"sweeping {len(grid)} configs on the validation split "
                  f"(HR@10 of the best epoch per config)")
            rows = []
            for c in grid:
                t0 = time.time()
                m, _ = fit_validated(arch, c, verbose=False)
                rows.append((c, m.best_hr, m.best_epoch))
                print(f"  dim {c['dim']:3d}  n_neg {c['n_neg']:2d}  dropout {c['dropout']:.1f}"
                      f"  -> HR@10 {m.best_hr:.4f} @ epoch {m.best_epoch:2d}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
                del m
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            best = max(rows, key=lambda r: r[1])
            print(f"  best: dim {best[0]['dim']}, n_neg {best[0]['n_neg']}, "
                  f"dropout {best[0]['dropout']}, epoch {best[2]} -> HR@10 {best[1]:.4f}")
            continue

        n_epochs = args.epochs
        if n_epochs is None:
            t0 = time.time()
            print(f"  config: {cfg}")
            val_model, hist = fit_validated(arch, cfg)
            n_epochs = val_model.best_epoch
            print(f"  best validation epoch: {n_epochs} "
                  f"(HR@10 {max(hist):.4f}) in {time.time()-t0:.0f}s")
            del val_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        t0 = time.time()
        print(f"  refitting on all {len(train_df)} pairs for {n_epochs} epochs")
        model, _ = train(arch, uu, ii, rr, seen, n_u, n_i, device, n_epochs, cfg,
                         seed=args.seed)
        recs = top_n_names(score_all(model, n_u, n_i, device), seen, users, item_names)
        out = f"recs_{arch}{args.tag}.jsonl"
        with open(RESULTS / out, "w", encoding="utf-8") as f:
            for uid, names in recs.items():
                f.write(json.dumps({"user_id": uid, "recs": names}) + "\n")
        print(f"  saved results/{out} ({len(recs)} users)")
        print(f"  {arch} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
