"""Step 8: PinSage — Graph Convolutional Neural Networks for Web-Scale
Recommender Systems (Ying et al., KDD 2018, arXiv 1806.01973).

PinSage is a GCN over a bipartite item/collection graph. Its four contributions,
all implemented here:

  1. Importance-based neighborhoods (Sec 3.2). N(u) is not the k-hop graph
     neighborhood; it is the top-T nodes by L1-normalized *visit count* of short
     random walks with restart started at u — in the limit of infinite walks,
     the Personalized PageRank scores w.r.t. u. Fixed |N(u)| = T bounds the
     memory footprint of a minibatch regardless of how popular a node is.

  2. Importance pooling (Sec 3.2). The symmetric aggregator gamma in Algorithm 1
     is a *weighted* mean, weighted by those same normalized visit counts.

         CONVOLVE(z_u, {z_v : v in N(u)}, alpha, gamma)          [Algorithm 1]
           n_u       <- gamma({ReLU(Q h_v + q) : v in N(u)}, alpha)
           z_u^new   <- ReLU(W . concat(z_u, n_u) + w)
           z_u^new   <- z_u^new / ||z_u^new||_2

     K such layers are stacked (Q, q, W, w shared across nodes, distinct per
     layer), then a final dense head z_u <- G_2 . ReLU(G_1 h_u^(K) + g).

  3. Max-margin ranking loss with curriculum hard negatives (Sec 3.3).

         J(z_q, z_i) = E_{n~P_n(q)} max{0, z_q . z_n - z_q . z_i + delta}

     500 negatives are sampled once per minibatch and shared by every example in
     it. Uniform negatives alone are too easy, so items ranked in a band of the
     PPR ordering w.r.t. q ("related, but less related than i") are added as hard
     negatives on a curriculum: none in epoch 1, n-1 at epoch n, capped at 6.

  4. Minibatch construction by re-indexing (Sec 3.3) and layer-wise inference
     (Sec 3.4). Algorithm 2 collects the K-hop neighborhood of a minibatch, maps
     it onto a compact local index space, and convolves on that subgraph only.
     Inference computes each layer once for the whole catalog rather than
     re-deriving shared neighbors per query — the single-machine equivalent of
     the paper's MapReduce pipeline.

Graph. Users are the paper's boards (no features), businesses are its pins (all
the features). Random walks are business -> user -> business, so N(u) is a set of
businesses and the paper's "even number of convolutional layers" footnote is
satisfied by construction: each item-level layer is two bipartite hops.

Item features x_u. The paper concatenates VGG-16 visual embeddings, Word2Vec
annotation embeddings and log(node degree). The Yelp analogue, from the same CSV
the rest of the pipeline reads:

    categories (490-way, mean-pooled bag)   name tokens (hashed 8k bag)
    city (57-way)                           lat / lon, is_open
    log1p(degree), mean star                <- training split only

`business_stars` and `business_review_count` are deliberately *not* used: they
are whole-Yelp aggregates that include the held-out review. Degree and mean star
are recomputed from whichever split is being fit, exactly as in step 7.

Two deviations, both forced and both load-bearing:

  * The paper's x_u are frozen pretrained vectors; there is no Yelp equivalent,
    so the content encoder above is trained end to end. There is still no
    per-item ID embedding anywhere in the model — PinSage is inductive, an item
    is only ever its content plus its graph position — which is the honest
    handicap against the ID-embedding baselines in step 7. `--item-id-emb`
    measures what dropping that constraint would buy.
  * The paper runs synchronous SGD across 16 GPUs with the linear scaling rule.
    The warmup-then-exponential-decay schedule is kept; the optimizer is Adam,
    since the large-batch multi-GPU regime that motivates SGD does not apply on
    one card. `--optimizer sgd` restores it.

Recommendation follows the paper's homefeed protocol (Sec 4.1): score every item
by its maximum embedding similarity to one of the user's most recently visited
items, mask visited items, take the top 10.

`--variant` reproduces the Table 1 ablation ladder (max-pooling, mean-pooling,
mean-pooling-xent, mean-pooling-hard, pinsage).

Reproducibility: `--seed` fixes initialization and every sample, but the backward
pass of the sparse neighbourhood aggregation accumulates with CUDA float atomics,
so two runs of the same seed still differ — about +/-0.002 HR@10, and 0.0201 to
0.0292 across 6 seeds under the full protocol. Read the spread, not a single run;
`--reps` averages sweep configs over seeds for that reason.

Run with the torch environment used by steps 03/04/07:
  python genrec/08_pinsage.py                       # validated fit + refit
  python genrec/08_pinsage.py --sweep main          # dim x T x margin grid
  python genrec/08_pinsage.py --variant mean-pooling --tag _meanpool
"""
import argparse
import json
import math
import re
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CSV = ROOT.parent / "yelp_reviews.csv"

TOP_N = 10
SEED = 42
MAX_EPOCHS = 40
PATIENCE = 8

# --- Sec 3.2 random walks -------------------------------------------------
N_WALKS = 1000        # walk chains started per node
WALK_LEN = 10         # 2-hop (item->user->item) steps per chain
RESTART_P = 0.5       # restart probability, i.e. walks *with restart*
MAX_RANK = 2000       # deepest PPR rank materialized (bounds the hard-neg band)

# --- Sec 3.3 loss ---------------------------------------------------------
N_SHARED_NEG = 500    # negatives sampled once per minibatch, shared by all pairs
MAX_HARD_NEG = 6      # curriculum ceiling: epoch n contributes min(n-1, 6)

# --- features -------------------------------------------------------------
NAME_HASH = 8192
D_CAT, D_NAME, D_CITY = 64, 64, 16
N_NUMERIC = 5         # log1p(degree), mean star, lat, lon, is_open

RECENT_CHOICES = (1, 2, 3, 5, 10, 20)   # homefeed query size, picked on val
RECENT_MAX = max(RECENT_CHOICES)

VARIANTS = ("pinsage", "mean-pooling", "max-pooling", "mean-pooling-xent",
            "mean-pooling-hard")
HARD_VARIANTS = {"pinsage", "mean-pooling-hard"}   # Table 1 column definitions

# Only `margin` is actually resolved by --sweep: 0.1 > 0.3 > 0.5 in all six
# dim x T cells, three seeds each. T, dim, the hard-negative band and the walk
# count all land inside the run-to-run noise (see REPORT.md), so those keep the
# paper's own values rather than a validation argmax fitted to noise.
DEFAULTS = {
    "dim": 128,           # d, output of every convolutional layer and the head
    "hidden": 256,        # m, column space of Q (paper: d=1024, m=2048)
    "layers": 2,          # K
    "neighbors": 50,      # T (paper's default, and its Table 4 best)
    "margin": 0.1,        # delta
    "batch": 2048,        # paper sweeps 512-4096 and reports 2048 as best
    "lr": 3e-3,
    "l2": 1e-6,
    "walks": N_WALKS,     # walk chains per node behind the PPR estimate
    "hard_band": (500, 2000),  # PPR rank band for hard negatives (paper: 2000-5000)
    "decay": 0.95,        # per-epoch exponential decay after the warmup epoch
}


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


# ---------------------------------------------------------------------------
# Bipartite graph
# ---------------------------------------------------------------------------

def build_csr(rows: np.ndarray, cols: np.ndarray, n_rows: int):
    """Sorted adjacency of one side of the bipartite graph as (ptr, idx)."""
    order = np.argsort(rows, kind="stable")
    idx = cols[order].astype(np.int64)
    ptr = np.zeros(n_rows + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(rows, minlength=n_rows))
    return ptr, idx


class Bipartite:
    """User<->item adjacency, resident on the GPU (Sec 3.3: the whole graph and
    feature matrix live next to the compute, so no CPU/GPU producer-consumer
    pipeline is needed at this scale)."""

    def __init__(self, uu: np.ndarray, ii: np.ndarray, n_u: int, n_i: int,
                 device: torch.device):
        iu_ptr, iu_idx = build_csr(ii, uu, n_i)   # item -> users
        ui_ptr, ui_idx = build_csr(uu, ii, n_u)   # user -> items
        self.iu_ptr = torch.as_tensor(iu_ptr, device=device)
        self.iu_idx = torch.as_tensor(iu_idx, device=device)
        self.ui_ptr = torch.as_tensor(ui_ptr, device=device)
        self.ui_idx = torch.as_tensor(ui_idx, device=device)
        self.n_u, self.n_i, self.device = n_u, n_i, device
        self.degree = np.bincount(ii, minlength=n_i).astype(np.float64)


def _step(cur: torch.Tensor, ptr: torch.Tensor, idx: torch.Tensor,
          gen: torch.Generator):
    """One hop: from each node in `cur`, jump to a uniformly random neighbor.

    Returns (next, ok). A degree-0 node has no neighbour to hop to — and in a
    bipartite graph it cannot simply stay put, since the following hop indexes
    the *other* side's adjacency — so it returns index 0 with ok=False and the
    caller drops the walker. Items with degree 0 do occur on the validation
    split, where a user's last training interaction is held out.
    """
    lo, deg = ptr[cur], ptr[cur + 1] - ptr[cur]
    r = torch.rand(cur.shape, device=cur.device, generator=gen)
    off = (r * deg.clamp(min=1)).long().clamp(max=(deg - 1).clamp(min=0))
    # lo of a degree-0 node runs one past the end of idx at the tail; clamp the
    # lookup into range and discard it via `ok`
    nxt = idx[(lo + off).clamp(max=idx.shape[0] - 1)]
    ok = deg > 0
    return torch.where(ok, nxt, torch.zeros_like(nxt)), ok


@torch.no_grad()
def random_walk_neighborhoods(g: Bipartite, n_walks: int, walk_len: int,
                              restart_p: float, top_t: int, max_rank: int,
                              seed: int, chunk: int = 256, verbose: bool = True):
    """Sec 3.2: L1-normalized visit counts of short random walks with restart.

    Returns
      nbr    (n_i, T)         importance-based neighborhood N(u), self excluded
      alpha  (n_i, T)         L1-normalized visit counts over N(u), rows sum to 1
      ranked (n_i, max_rank)  the same ordering carried deeper, for hard negatives
      n_hit  (n_i,)           how many distinct items each source actually reached
    """
    dev = g.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    n_i, max_rank = g.n_i, min(max_rank, g.n_i)

    nbr = torch.zeros(n_i, top_t, dtype=torch.long, device=dev)
    alpha = torch.zeros(n_i, top_t, device=dev)
    ranked = torch.zeros(n_i, max_rank, dtype=torch.int32, device=dev)
    n_hit = torch.zeros(n_i, dtype=torch.long, device=dev)

    t0 = time.time()
    for s in range(0, n_i, chunk):
        src = torch.arange(s, min(s + chunk, n_i), device=dev)
        c = int(src.shape[0])
        counts = torch.zeros(c, n_i, dtype=torch.int32, device=dev)
        row = torch.arange(c, device=dev).repeat_interleave(n_walks)
        start = src.repeat_interleave(n_walks)
        cur = start.clone()
        for _ in range(walk_len):
            cur, ok_u = _step(cur, g.iu_ptr, g.iu_idx, gen)   # item -> user
            cur, ok_i = _step(cur, g.ui_ptr, g.ui_idx, gen)   # user -> item
            alive = ok_u & ok_i
            counts.view(-1).scatter_add_(0, row * n_i + cur,
                                         alive.to(torch.int32))   # dead add 0
            back = torch.rand(cur.shape, device=dev, generator=gen) < restart_p
            cur = torch.where(back | ~alive, start, cur)

        counts[torch.arange(c, device=dev), src] = 0      # u is not its own neighbor
        n_hit[s:s + c] = (counts > 0).sum(1)
        top = counts.topk(max_rank, dim=1)
        ranked[s:s + c] = top.indices.to(torch.int32)
        w = top.values[:, :top_t].float()
        nbr[s:s + c] = top.indices[:, :top_t]
        alpha[s:s + c] = w / w.sum(1, keepdim=True).clamp(min=1e-12)
        del counts, top

    # An isolated item reaches nothing: fall back to a self-loop so CONVOLVE
    # degenerates to n_u = ReLU(Q h_u + q) instead of aggregating over garbage.
    dead = n_hit == 0
    if dead.any():
        ids = torch.arange(n_i, device=dev)[dead]
        nbr[dead] = ids.unsqueeze(1)
        alpha[dead] = 0.0
        alpha[dead, 0] = 1.0
    if verbose:
        print(f"  walks: {n_walks} chains x {walk_len} steps x {n_i} items "
              f"in {time.time() - t0:.0f}s; reached "
              f"{float(n_hit.float().mean()):.0f} distinct items on average, "
              f"{int(dead.sum())} isolated")
    return nbr, alpha, ranked, n_hit


# ---------------------------------------------------------------------------
# Item features  (x_u in Algorithm 2, line 8)
# ---------------------------------------------------------------------------

def load_item_frame(item_ids: list) -> pd.DataFrame:
    """Static business attributes, cached so the 231 MB CSV is read once."""
    cache = DATA / "item_features.parquet"
    cols = ["business_id", "business_name", "business_categories", "business_city",
            "business_latitude", "business_longitude", "business_open"]
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = (pd.read_csv(CSV, usecols=cols)
              .drop_duplicates("business_id").reset_index(drop=True))
        df.to_parquet(cache, index=False)
        print(f"  cached item attributes -> data/{cache.name}")
    return df.set_index("business_id").reindex(item_ids)


class ItemContent:
    """Categorical/numeric field arrays for every item, split-independent parts
    built once; `degree` and `mean_star` are supplied per fit."""

    def __init__(self, frame: pd.DataFrame):
        cats = frame["business_categories"].fillna("").str.split(";")
        vocab = sorted({c.strip() for row in cats for c in row if c.strip()})
        self.cat_vocab = {c: n for n, c in enumerate(vocab)}
        self.n_cat = len(vocab) + 1                     # +1 = "no categories"
        self.cat_bag, self.cat_off = self._bag(
            [[self.cat_vocab[c.strip()] for c in row if c.strip()] for row in cats],
            fallback=len(vocab))

        toks = frame["business_name"].fillna("").map(
            lambda s: [zlib.crc32(t.encode()) % NAME_HASH
                       for t in re.findall(r"[a-z0-9]+", s.lower())])
        self.name_bag, self.name_off = self._bag(list(toks), fallback=0)

        city = frame["business_city"].fillna("?").str.strip().str.lower()
        cvocab = {c: n for n, c in enumerate(sorted(city.unique()))}
        self.city = city.map(cvocab).to_numpy()
        self.n_city = len(cvocab)

        lat = frame["business_latitude"].to_numpy(dtype=np.float64)
        lon = frame["business_longitude"].to_numpy(dtype=np.float64)
        self.lat = (lat - lat.mean()) / (lat.std() + 1e-9)
        self.lon = (lon - lon.mean()) / (lon.std() + 1e-9)
        self.is_open = frame["business_open"].fillna(True).to_numpy(dtype=np.float64)

    @staticmethod
    def _bag(rows: list, fallback: int):
        """Ragged id lists -> (flat values, per-row offsets) for an EmbeddingBag."""
        flat, offs = [], []
        for r in rows:
            offs.append(len(flat))
            flat.extend(r if r else [fallback])
        return np.asarray(flat, dtype=np.int64), np.asarray(offs, dtype=np.int64)

    def numeric(self, degree: np.ndarray, mean_star: np.ndarray) -> np.ndarray:
        """log(node degree) is the paper's own third feature block."""
        d = np.log1p(degree)
        d = (d - d.mean()) / (d.std() + 1e-9)
        s = np.nan_to_num(mean_star, nan=float(np.nanmean(mean_star))) / 5.0
        return np.stack([d, s, self.lat, self.lon, self.is_open], axis=1)


class FeatureEncoder(nn.Module):
    """x_u: mean-pooled category bag || mean-pooled name bag || city || numerics."""

    def __init__(self, content: ItemContent, numeric: np.ndarray,
                 n_items: int, item_id_emb: int = 0):
        super().__init__()
        self.cat = nn.EmbeddingBag(content.n_cat, D_CAT, mode="mean")
        self.name = nn.EmbeddingBag(NAME_HASH, D_NAME, mode="mean")
        self.city = nn.Embedding(content.n_city, D_CITY)
        for e in (self.cat, self.name, self.city):
            nn.init.normal_(e.weight, std=0.05)

        self.register_buffer("cat_bag", torch.as_tensor(content.cat_bag))
        self.register_buffer("cat_off", torch.as_tensor(content.cat_off))
        self.register_buffer("name_bag", torch.as_tensor(content.name_bag))
        self.register_buffer("name_off", torch.as_tensor(content.name_off))
        self.register_buffer("city_idx", torch.as_tensor(content.city))
        self.register_buffer("num", torch.as_tensor(numeric, dtype=torch.float32))
        self.register_buffer("cat_len",
                             torch.as_tensor(np.diff(np.append(content.cat_off,
                                                               len(content.cat_bag)))))
        self.register_buffer("name_len",
                             torch.as_tensor(np.diff(np.append(content.name_off,
                                                               len(content.name_bag)))))
        self.ids = None
        if item_id_emb:                     # ablation only; breaks inductiveness
            self.ids = nn.Embedding(n_items, item_id_emb)
            nn.init.normal_(self.ids.weight, std=0.05)
        self.dim = D_CAT + D_NAME + D_CITY + N_NUMERIC + item_id_emb

    def _gather_bag(self, nodes, bag, off, length, table):
        """EmbeddingBag over an arbitrary subset of rows of a ragged table."""
        lens = length[nodes]
        starts = off[nodes]
        pos = torch.repeat_interleave(starts, lens) + (
            torch.arange(int(lens.sum()), device=nodes.device)
            - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens))
        new_off = torch.cumsum(lens, 0) - lens
        return table(bag[pos], new_off)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        parts = [self._gather_bag(nodes, self.cat_bag, self.cat_off,
                                  self.cat_len, self.cat),
                 self._gather_bag(nodes, self.name_bag, self.name_off,
                                  self.name_len, self.name),
                 self.city(self.city_idx[nodes]),
                 self.num[nodes]]
        if self.ids is not None:
            parts.append(self.ids(nodes))
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Algorithm 1 / Algorithm 2
# ---------------------------------------------------------------------------

class Convolve(nn.Module):
    """Algorithm 1. `pool` selects gamma: importance (weighted mean), mean, max."""

    def __init__(self, d_in: int, d_out: int, hidden: int, pool: str):
        super().__init__()
        self.Q = nn.Linear(d_in, hidden)            # Q, q
        self.W = nn.Linear(d_in + hidden, d_out)    # W, w
        self.pool = pool

    def forward(self, h: torch.Tensor, self_pos: torch.Tensor,
                nbr_pos: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """h: (n_prev, d_in) previous-layer states, indexed in the *previous*
        level's local space. self_pos/nbr_pos map this level's nodes into it."""
        msg = F.relu(self.Q(h))                     # ReLU(Q h_v + q), per node
        if self.pool == "max":
            gathered = msg[nbr_pos]                 # (n, T, hidden)
            gathered = gathered.masked_fill((alpha == 0).unsqueeze(-1), float("-inf"))
            n_u = gathered.max(dim=1).values
            n_u = torch.nan_to_num(n_u, neginf=0.0)
        else:
            w = alpha if self.pool == "importance" else (alpha > 0).float()
            w = w / w.sum(1, keepdim=True).clamp(min=1e-12)
            # sum_v w_uv * msg_v as one sparse matmul: never materializes (n, T, hidden)
            n, t = nbr_pos.shape
            rows = torch.arange(n, device=h.device).repeat_interleave(t)
            adj = torch.sparse_coo_tensor(
                torch.stack([rows, nbr_pos.reshape(-1)]),
                w.reshape(-1), (n, h.shape[0])).coalesce()
            n_u = torch.sparse.mm(adj, msg)
        z = F.relu(self.W(torch.cat([h[self_pos], n_u], dim=1)))
        return z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)


class PinSage(nn.Module):
    """Stacked convolutions plus the G_1 / G_2 / g head (Algorithm 2, lines 14-15)."""

    def __init__(self, encoder: FeatureEncoder, cfg: dict, pool: str):
        super().__init__()
        self.encoder = encoder
        d, m, K = cfg["dim"], cfg["hidden"], cfg["layers"]
        dims = [encoder.dim] + [d] * K
        self.convs = nn.ModuleList(
            [Convolve(dims[k], dims[k + 1], m, pool) for k in range(K)])
        self.G1 = nn.Linear(d, d)
        self.G2 = nn.Linear(d, d, bias=False)
        self.K = K

    def forward(self, levels: list, nbr_pos: list, alpha: list,
                self_pos: list) -> torch.Tensor:
        """`levels[0]` are the global ids needed at layer 0, `levels[K]` the
        targets; the *_pos lists index each level into the one below it."""
        h = self.encoder(levels[0])
        for k in range(self.K):
            h = self.convs[k](h, self_pos[k], nbr_pos[k], alpha[k])
        z = self.G2(F.relu(self.G1(h)))
        return z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)


class Sampler:
    """Algorithm 2 lines 1-7: expand a minibatch to its K-hop importance
    neighborhood and re-index it onto a compact local id space."""

    def __init__(self, nbr: torch.Tensor, alpha: torch.Tensor, K: int):
        self.nbr, self.alpha, self.K = nbr, alpha, K
        self.n_i = nbr.shape[0]
        self.scratch = torch.full((self.n_i,), -1, dtype=torch.long,
                                  device=nbr.device)

    def __call__(self, targets: torch.Tensor):
        levels = [targets]
        for _ in range(self.K):
            prev = levels[0]
            levels.insert(0, torch.unique(
                torch.cat([prev, self.nbr[prev].reshape(-1)])))
        nbr_pos, alpha, self_pos = [], [], []
        for k in range(self.K):
            below, here = levels[k], levels[k + 1]
            self.scratch[below] = torch.arange(below.shape[0], device=below.device)
            nbr_pos.append(self.scratch[self.nbr[here]])
            alpha.append(self.alpha[here])
            self_pos.append(self.scratch[here])
            self.scratch[below] = -1
        return levels, nbr_pos, alpha, self_pos


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def consecutive_pairs(df: pd.DataFrame, u_idx: dict, i_idx: dict):
    """L (Eq. 1): (q, i) where the user visited i immediately after q."""
    q, p = [], []
    for _, seq in df.groupby("user_id", sort=False)["business_id"]:
        s = [i_idx[b] for b in seq]
        for a, b in zip(s, s[1:]):
            if a != b:
                q.append(a)
                p.append(b)
    return np.asarray(q, dtype=np.int64), np.asarray(p, dtype=np.int64)


def sample_hard(ranked: torch.Tensor, n_hit: torch.Tensor, q: torch.Tensor,
                n_hard: int, band: tuple, gen: torch.Generator) -> torch.Tensor:
    """Sec 3.3 hard negatives: items in a PPR rank band w.r.t. the query.

    The paper's absolute band (ranks 2000-5000 of two billion items) has no
    meaning in a 10k catalog, so the *role* of the band is what is kept — related
    to q, but well below the items a top-10 list would contain — and (lo, hi) is
    tuned on the validation split. Queries whose walks never reached `lo`
    distinct items fall back to uniform negatives.
    """
    lo, hi = band
    hi = min(hi, ranked.shape[1])
    span = hi - lo
    dev = ranked.device
    r = torch.rand(q.shape[0], n_hard, device=dev, generator=gen)
    avail = (n_hit[q].clamp(max=hi) - lo).clamp(min=0).unsqueeze(1)
    rel = (r * avail.clamp(min=1)).long().clamp(max=span - 1)
    off = lo + rel
    hard = ranked[q.unsqueeze(1), off].long()
    fallback = torch.randint(0, ranked.shape[0], hard.shape, device=dev, generator=gen)
    return torch.where(avail > 0, hard, fallback)


def margin_loss(zq, zi, z_shared, z_hard, shared_valid, hard_valid, delta):
    """Eq. 1, averaged over the shared and hard negatives of each pair.

    `*_valid` drops any sampled negative that *is* this pair's positive item.
    The paper samples 500 uniform negatives out of two billion, where that has
    probability ~0; out of 10,165 it happens for ~5% of pairs, and leaving it in
    contributes a constant +delta hinge that fights the positive term.
    """
    pos = (zq * zi).sum(-1, keepdim=True)
    scores, valid = [zq @ z_shared.t()], [shared_valid.float()]
    if z_hard is not None:
        scores.append((zq.unsqueeze(1) * z_hard).sum(-1))
        valid.append(hard_valid.float())
    s, v = torch.cat(scores, 1), torch.cat(valid, 1)
    return (F.relu(s - pos + delta) * v).sum() / v.sum().clamp(min=1.0)


def xent_loss(zq, zi, z_shared, z_hard, shared_valid, hard_valid, _delta):
    """The GraphSAGE cross-entropy objective, for the mean-pooling-xent row."""
    v = shared_valid.float()
    pos = F.logsigmoid((zq * zi).sum(-1)).mean()
    neg = (F.logsigmoid(-(zq @ z_shared.t())) * v).sum() / v.sum().clamp(min=1.0)
    return -(pos + neg)


class Fit:
    """One (graph, features, walks) bundle — rebuilt per split so the validation
    fit never sees the edges it is scored on."""

    def __init__(self, df: pd.DataFrame, u_idx: dict, i_idx: dict, content: ItemContent,
                 cfg: dict, device: torch.device, seed: int, verbose: bool = True):
        uu = df["user_id"].map(u_idx).to_numpy()
        ii = df["business_id"].map(i_idx).to_numpy()
        n_u, n_i = len(u_idx), len(i_idx)
        self.graph = Bipartite(uu, ii, n_u, n_i, device)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_star = (np.bincount(ii, weights=df["stars"].to_numpy(float),
                                     minlength=n_i) / self.graph.degree)
        self.numeric = content.numeric(self.graph.degree, mean_star)

        self.nbr, self.alpha, self.ranked, self.n_hit = random_walk_neighborhoods(
            self.graph, cfg.get("walks", N_WALKS), WALK_LEN, RESTART_P,
            cfg["neighbors"], MAX_RANK, seed, verbose=verbose)

        self.seen = np.zeros((n_u, n_i), dtype=bool)
        self.seen[uu, ii] = True
        self.hist = np.full((n_u, RECENT_MAX), -1, dtype=np.int64)
        for uid, seq in df.groupby("user_id", sort=False)["business_id"]:
            s = [i_idx[b] for b in seq][::-1][:RECENT_MAX]      # most recent first
            self.hist[u_idx[uid], :len(s)] = s
        self.pairs = consecutive_pairs(df, u_idx, i_idx)


@torch.no_grad()
def embed_all(model: PinSage, sampler: Sampler, n_i: int, device: torch.device,
              chunk: int = 4096) -> torch.Tensor:
    """Sec 3.4: embeddings for the whole catalog, each layer evaluated once per
    node rather than once per query — the single-machine MapReduce equivalent."""
    model.eval()
    out = torch.empty(n_i, model.G2.out_features, device=device)
    for s in range(0, n_i, chunk):
        targets = torch.arange(s, min(s + chunk, n_i), device=device)
        out[s:s + targets.shape[0]] = model(*sampler(targets))
    return out


@torch.no_grad()
def score_and_rank(Z: torch.Tensor, hist: np.ndarray, seen: np.ndarray,
                   recent: int, item_names: list, users: list,
                   device: torch.device, chunk: int = 256):
    """Homefeed protocol (Sec 4.1): nearest items to one of the user's most
    recent visits, with visited items masked and names deduped as in step 2."""
    h = torch.as_tensor(hist[:, :recent], device=device)
    valid = h >= 0
    recs = {}
    for s in range(0, h.shape[0], chunk):
        hb, vb = h[s:s + chunk].clamp(min=0), valid[s:s + chunk]
        sims = torch.einsum("urd,id->uri", Z[hb], Z)
        sims = sims.masked_fill(~vb.unsqueeze(-1), float("-inf"))
        scores = sims.max(dim=1).values
        scores[torch.as_tensor(seen[s:s + chunk], device=device)] = float("-inf")
        order = scores.argsort(dim=1, descending=True)[:, :TOP_N * 5].cpu().numpy()
        for r in range(order.shape[0]):
            names, dedup = [], set()
            for i in order[r]:
                nm = item_names[i]
                if nm not in dedup:
                    dedup.add(nm)
                    names.append(nm)
                    if len(names) == TOP_N:
                        break
            recs[users[s + r]] = names
    return recs


def train(fit: Fit, content: ItemContent, cfg: dict, variant: str, epochs: int,
          device: torch.device, n_i: int, seed: int, item_names=None, users=None,
          val_target=None, verbose: bool = True):
    """Fit one model; with `val_target`, also picks the best (epoch, recent)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    pool = {"pinsage": "importance", "max-pooling": "max"}.get(variant, "mean")
    loss_fn = xent_loss if variant == "mean-pooling-xent" else margin_loss
    use_hard = variant in HARD_VARIANTS

    encoder = FeatureEncoder(content, fit.numeric, n_i, cfg.get("item_id_emb", 0))
    model = PinSage(encoder, cfg, pool).to(device)
    sampler = Sampler(fit.nbr, fit.alpha, cfg["layers"])
    opt = (torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                           weight_decay=cfg["l2"]) if cfg.get("optimizer") == "sgd"
           else torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["l2"]))

    q_all, i_all = fit.pairs
    q_all = torch.as_tensor(q_all, device=device)
    i_all = torch.as_tensor(i_all, device=device)
    n_pairs, batch = q_all.shape[0], cfg["batch"]
    steps = math.ceil(n_pairs / batch)

    best = {"hr": -1.0, "epoch": 0, "recent": RECENT_CHOICES[0], "state": None}
    since = 0
    for epoch in range(1, epochs + 1):
        model.train()
        n_hard = min(epoch - 1, MAX_HARD_NEG) if use_hard else 0
        perm = torch.randperm(n_pairs, device=device, generator=gen)
        total, expansion = 0.0, 0
        for step in range(steps):
            # Sec 3.3: linear warmup across the first epoch, then exp decay.
            frac = (step + 1) / steps if epoch == 1 else 1.0
            scale = frac * (cfg["decay"] ** max(0, epoch - 1))
            for gp in opt.param_groups:
                gp["lr"] = cfg["lr"] * scale

            sl = perm[step * batch:(step + 1) * batch]
            q, i = q_all[sl], i_all[sl]
            shared = torch.randint(0, n_i, (N_SHARED_NEG,), device=device, generator=gen)
            hard = (sample_hard(fit.ranked, fit.n_hit, q, n_hard,
                                cfg["hard_band"], gen) if n_hard else None)

            parts = [q, i, shared] + ([hard.reshape(-1)] if n_hard else [])
            targets = torch.unique(torch.cat(parts))
            levels, nbr_pos, alpha, self_pos = sampler(targets)
            expansion += levels[0].shape[0]

            z = model(levels, nbr_pos, alpha, self_pos)
            loc = torch.full((n_i,), -1, dtype=torch.long, device=device)
            loc[targets] = torch.arange(targets.shape[0], device=device)
            z_hard = z[loc[hard]] if n_hard else None
            hard_valid = (hard != i.unsqueeze(1)) if n_hard else None

            shared_valid = shared.unsqueeze(0) != i.unsqueeze(1)
            loss = loss_fn(z[loc[q]], z[loc[i]], z[loc[shared]], z_hard,
                           shared_valid, hard_valid, cfg["margin"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * q.shape[0]

        msg = (f"    epoch {epoch:2d}/{epochs} loss {total / n_pairs:.4f} "
               f"hard {n_hard}")
        if val_target is not None:
            Z = embed_all(model, sampler, n_i, device)
            hrs = {}
            for r in RECENT_CHOICES:
                recs = score_and_rank(Z, fit.hist, fit.seen, r, item_names,
                                      users, device)
                hrs[r] = float(np.mean([val_target[u] in recs[u] for u in users]))
            r_best = max(hrs, key=hrs.get)
            msg += f"  val HR@10 {hrs[r_best]:.4f} (recent={r_best})"
            if hrs[r_best] > best["hr"]:
                best.update(hr=hrs[r_best], epoch=epoch, recent=r_best,
                            state={k: v.detach().clone()
                                   for k, v in model.state_dict().items()})
                since = 0
            else:
                since += 1
        if verbose:
            print(msg + f"  |L0|~{expansion // steps}", flush=True)
        if val_target is not None and since >= PATIENCE:
            if verbose:
                print(f"    early stop; best epoch {best['epoch']} "
                      f"HR@10 {best['hr']:.4f} recent={best['recent']}")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.best = best
    return model, sampler


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pinsage", choices=VARIANTS)
    ap.add_argument("--epochs", type=int, default=None,
                    help="skip the validation phase and train this many epochs")
    ap.add_argument("--recent", type=int, default=None,
                    help="homefeed query size; default is picked on validation")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None, help="K (paper: 2)")
    ap.add_argument("--neighbors", type=int, default=None, help="T (paper: 50)")
    ap.add_argument("--margin", type=float, default=None, help="delta")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hard-band", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--walks", type=int, default=None,
                    help="walk chains per node; more chains = a closer "
                         "approximation to the Personalized PageRank limit")
    ap.add_argument("--item-id-emb", type=int, default=0,
                    help="ablation: add a transductive per-item embedding")
    ap.add_argument("--optimizer", default="adam", choices=("adam", "sgd"))
    ap.add_argument("--sweep", choices=("main", "band"), default=None)
    ap.add_argument("--reps", type=int, default=3,
                    help="seeds averaged per sweep config; the backward pass of "
                         "the sparse aggregation uses float atomics, so a single "
                         "run is worth about +/-0.002 HR@10 of noise")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"device: {device}  variant: {args.variant}")

    cfg = dict(DEFAULTS)
    for k in ("dim", "hidden", "layers", "neighbors", "margin", "batch", "lr",
              "walks"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.dim is not None and args.hidden is None:
        cfg["hidden"] = 2 * args.dim          # keep the paper's m = 2d ratio
    if args.hard_band is not None:
        cfg["hard_band"] = tuple(args.hard_band)
    cfg["optimizer"] = args.optimizer
    cfg["item_id_emb"] = args.item_id_emb

    train_df = pd.read_parquet(DATA / "train_interactions.parquet")
    train_df = train_df.drop_duplicates(subset=["user_id", "business_id"], keep="last")
    users = sorted(train_df["user_id"].unique())
    items = sorted(train_df["business_id"].unique())
    u_idx = {u: n for n, u in enumerate(users)}
    i_idx = {b: n for n, b in enumerate(items)}
    n_i = len(items)
    print(f"graph: {len(users)} users x {n_i} items, {len(train_df)} edges")

    id2name = dict(zip(train_df["business_id"], train_df["business_name"].map(norm_name)))
    item_names = [id2name[b] for b in items]
    content = ItemContent(load_item_frame(items))
    print(f"item features: {content.n_cat - 1} categories, {content.n_city} cities")

    # model selection on each user's last *training* interaction; the graph for
    # this phase excludes those edges, so nothing about the split leaks in
    val_idx = train_df.groupby("user_id", sort=False).tail(1).index
    val_df, fit_df = train_df.loc[val_idx], train_df.drop(index=val_idx)
    val_target = {r.user_id: norm_name(r.business_name) for r in val_df.itertuples()}

    def validated(c, verbose=True, seed=None):
        seed = args.seed if seed is None else seed
        f = Fit(fit_df, u_idx, i_idx, content, c, device, seed, verbose)
        m, _ = train(f, content, c, args.variant, MAX_EPOCHS, device, n_i,
                     seed, item_names, users, val_target, verbose)
        return m

    if args.sweep:
        if args.sweep == "main":
            grid = [dict(cfg, dim=d, hidden=2 * d, neighbors=t, margin=mg)
                    for d in (64, 128) for t in (10, 20, 50) for mg in (0.1, 0.3, 0.5)]
            keys = ("dim", "neighbors", "margin")
        else:
            grid = [dict(cfg, hard_band=b)
                    for b in ((50, 200), (100, 500), (200, 1000), (500, 2000))]
            keys = ("hard_band",)
        print(f"sweeping {len(grid)} configs x {args.reps} seeds "
              f"on the validation split")
        rows = []
        for c in grid:
            t0 = time.time()
            runs = []
            for r in range(args.reps):
                m = validated(c, verbose=False, seed=args.seed + r)
                runs.append(m.best)
                del m
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            hr = float(np.mean([b["hr"] for b in runs]))
            epoch = int(round(float(np.mean([b["epoch"] for b in runs]))))
            recent = max(set(b["recent"] for b in runs),
                         key=[b["recent"] for b in runs].count)
            rows.append((c, hr, epoch, recent, [b["hr"] for b in runs]))
            print("  " + "  ".join(f"{k} {c[k]}" for k in keys)
                  + f"  -> HR@10 {hr:.4f} (min {min(runs, key=lambda b: b['hr'])['hr']:.4f}"
                    f" max {max(runs, key=lambda b: b['hr'])['hr']:.4f})"
                    f" @ epoch ~{epoch} recent={recent}"
                    f"  ({time.time() - t0:.0f}s)", flush=True)
        b = max(rows, key=lambda r: r[1])
        print("  best: " + "  ".join(f"{k} {b[0][k]}" for k in keys)
              + f"  mean HR@10 {b[1]:.4f} @ epoch ~{b[2]} recent={b[3]}")
        return

    print(f"  config: { {k: v for k, v in cfg.items() if k != 'optimizer'} }")
    epochs, recent = args.epochs, args.recent
    if epochs is None or recent is None:
        t0 = time.time()
        m = validated(cfg)
        epochs = epochs or m.best["epoch"]
        recent = recent or m.best["recent"]
        print(f"  validation: epoch {m.best['epoch']} recent {m.best['recent']} "
              f"HR@10 {m.best['hr']:.4f} in {time.time() - t0:.0f}s")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    t0 = time.time()
    print(f"  refitting on all {len(train_df)} edges for {epochs} epochs")
    full = Fit(train_df, u_idx, i_idx, content, cfg, device, args.seed)
    model, sampler = train(full, content, cfg, args.variant, epochs, device, n_i,
                           args.seed)
    Z = embed_all(model, sampler, n_i, device)
    recs = score_and_rank(Z, full.hist, full.seen, recent, item_names, users, device)

    out = f"recs_pinsage{args.tag}.jsonl"
    with open(RESULTS / out, "w", encoding="utf-8") as f:
        for uid in users:
            f.write(json.dumps({"user_id": uid, "recs": recs[uid]}) + "\n")
    print(f"  saved results/{out} ({len(recs)} users, recent={recent}) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
