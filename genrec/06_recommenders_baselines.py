"""Step 6: Additional baselines from the recommenders-team library
(https://github.com/recommenders-team/recommenders):

  - SAR : Smart Adaptive Recommendations — item co-occurrence (Jaccard)
          similarity x time-decayed user affinity. `recommenders.models.sar.SAR`.
  - BPR : Bayesian Personalized Ranking — implicit-feedback matrix
          factorization with a pairwise ranking loss, via Cornac (the backend
          the recommenders library wraps for BPR).

Run with the .venv-rec Python (recommenders needs Python <= 3.11):
  .venv-rec/Scripts/python genrec/06_recommenders_baselines.py

Outputs results/recs_sar.jsonl and results/recs_bpr.jsonl in the same format
as the other baselines (top-10 normalized business names per test user).
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TOP_N = 10


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def save_name_recs(scores: np.ndarray, users: list, item_names: list,
                   seen: np.ndarray, name: str):
    """scores: (n_users, n_items); seen: bool mask of training interactions."""
    masked = scores.astype(np.float32, copy=True)
    masked[seen] = -np.inf
    order = np.argsort(-masked, axis=1, kind="stable")
    with open(RESULTS / f"recs_{name}.jsonl", "w", encoding="utf-8") as f:
        for u_idx, uid in enumerate(users):
            names, dedup = [], set()
            for i in order[u_idx]:
                if not np.isfinite(masked[u_idx, i]):
                    break
                nm = item_names[i]
                if nm not in dedup:
                    dedup.add(nm)
                    names.append(nm)
                    if len(names) == TOP_N:
                        break
            f.write(json.dumps({"user_id": uid, "recs": names}) + "\n")
    print(f"  saved results/recs_{name}.jsonl")


def main():
    train = pd.read_parquet(DATA / "train_interactions.parquet")
    train["timestamp"] = pd.to_datetime(train["date"]).astype("int64") // 10**9

    users = sorted(train["user_id"].unique())
    items = sorted(train["business_id"].unique())
    u_idx = {u: n for n, u in enumerate(users)}
    i_idx = {b: n for n, b in enumerate(items)}
    id2name = dict(zip(train["business_id"], train["business_name"].map(norm_name)))
    item_names = [id2name[b] for b in items]

    seen = np.zeros((len(users), len(items)), dtype=bool)
    seen[train["user_id"].map(u_idx), train["business_id"].map(i_idx)] = True

    # --- SAR (recommenders.models.sar) --------------------------------------
    from recommenders.models.sar import SAR

    t0 = time.time()
    # a few users rated the same business twice; keep the most recent rating
    # (train is date-sorted, and the matrix baselines behave the same way)
    sar_train = train.rename(columns={"user_id": "userID", "business_id": "itemID",
                                      "stars": "rating"})[
        ["userID", "itemID", "rating", "timestamp"]] \
        .drop_duplicates(subset=["userID", "itemID"], keep="last")
    print(f"  SAR train: {len(sar_train)} pairs "
          f"({len(train) - len(sar_train)} duplicate ratings collapsed)")
    sar = SAR(col_user="userID", col_item="itemID", col_rating="rating",
              col_timestamp="timestamp", similarity_type="jaccard",
              time_decay_coefficient=30, timedecay_formula=True, normalize=False)
    sar.fit(sar_train)

    # top-k frame (ask for depth 30 so name-dedupe can still fill 10 slots);
    # remove_seen=True is SAR's own already-visited masking
    topk = sar.recommend_k_items(sar_train[["userID"]].drop_duplicates(),
                                 top_k=30, remove_seen=True)
    scores = np.full((len(users), len(items)), -np.inf, dtype=np.float32)
    r = topk["userID"].map(u_idx).to_numpy()
    c = topk["itemID"].map(i_idx).to_numpy()
    scores[r, c] = topk["prediction"].to_numpy(dtype=np.float32)
    save_name_recs(scores, users, item_names, np.zeros_like(seen), "sar")
    print(f"  SAR done in {time.time()-t0:.0f}s")

    # --- BPR (Cornac, the backend recommenders wraps) ------------------------
    import cornac

    t0 = time.time()
    triplets = list(train[["user_id", "business_id", "stars"]].itertuples(index=False, name=None))
    train_set = cornac.data.Dataset.from_uir(triplets, seed=42)
    bpr = cornac.models.BPR(k=100, max_iter=200, learning_rate=0.01,
                            lambda_reg=0.001, seed=42, verbose=False)
    bpr.fit(train_set)

    # score all pairs from the learned factors, mapped back to our index order
    U = np.asarray(bpr.u_factors)          # (n_users_cornac, k)
    V = np.asarray(bpr.i_factors)          # (n_items_cornac, k)
    bi = np.asarray(bpr.i_biases)          # (n_items_cornac,)
    full = U @ V.T + bi[None, :]
    u_order = [train_set.uid_map[u] for u in users]
    i_order = [train_set.iid_map[b] for b in items]
    scores = full[np.ix_(u_order, i_order)]
    save_name_recs(scores, users, item_names, seen, "bpr")
    print(f"  BPR done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
