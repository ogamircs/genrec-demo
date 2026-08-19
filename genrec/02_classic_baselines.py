"""Step 2: Classic recommenders (numpy reimplementation of the notebook's
surprise models) + full-catalog top-10 recommendations for the leave-one-out
benchmark.

Models (notebook's tuned hyperparameters):
  - rank_based      : avg rating, min 50 interactions (notebook Model 1)
  - popularity      : most-rated items (extra reference baseline)
  - user_knn        : KNNBasic user-based, cosine, k=40, min_k=6
  - item_knn        : KNNBasic item-based, msd,    k=30, min_k=9
  - svd             : FunkSVD, n_epochs=20, lr_all=0.01, reg_all=0.2

Also runs the notebook's own protocol (>=100-review cohort, random 80/20
rating split, RMSE + precision/recall@10) as a sanity check that this
reimplementation matches the notebook's reported numbers.
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
RESULTS.mkdir(exist_ok=True)

TOP_N = 10


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


# ---------------------------------------------------------------------------
# Similarities (surprise semantics: computed over co-rated entries only)
# ---------------------------------------------------------------------------

def cosine_sim(R: np.ndarray, B: np.ndarray) -> np.ndarray:
    """surprise cosine: sim(a,b) = sum_common r_a*r_b / (||r_a||_common * ||r_b||_common).
    R: ratings (rows = entities), B: binary mask. min_support=1."""
    num = R @ R.T
    sq = R * R
    d1 = sq @ B.T          # d1[a,b] = sum over common of r_a^2
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = num / (np.sqrt(d1) * np.sqrt(d1.T))
    sim[~np.isfinite(sim)] = 0.0
    return sim.astype(np.float32)


def msd_sim(R: np.ndarray, B: np.ndarray) -> np.ndarray:
    """surprise msd: sim = 1 / (msd + 1), msd = mean squared diff over common."""
    sq = R * R
    q = sq @ B.T                     # q[a,b] = sum_common r_a^2
    s = R @ R.T
    cnt = B @ B.T
    d = q + q.T - 2.0 * s            # sum of squared diffs over common
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = cnt / (d + cnt)        # == 1 / (d/cnt + 1)
    sim[cnt < 1] = 0.0
    sim[~np.isfinite(sim)] = 0.0
    return sim.astype(np.float32)


# ---------------------------------------------------------------------------
# KNNBasic full prediction matrix
# ---------------------------------------------------------------------------

def knn_full_predict(sim: np.ndarray, R: np.ndarray, B: np.ndarray,
                     k: int, min_k: int, mu: float, user_based: bool) -> np.ndarray:
    """Predicted rating matrix (users x items), surprise KNNBasic formula.

    user_based: neighbors = users who rated the item, weighted by user sim.
    item_based: neighbors = items rated by the user, weighted by item sim.
    All sims are >= 0 here, so 'top-k by sim then keep sim>0' == surprise's
    nlargest(k) + positive-sim accumulation.
    """
    if user_based:
        # num[u,i] = sum_v sim(u,v) r(v,i) over ALL raters; fix popular items after
        num = sim @ R
        den = sim @ B
        cnt = (sim > 0).astype(np.float32) @ B
        raters = [np.nonzero(B[:, i])[0] for i in range(B.shape[1])]
        popular = [i for i, r in enumerate(raters) if len(r) > k]
        print(f"    user_knn: correcting {len(popular)} items with >{k} raters")
        for i in popular:
            idx = raters[i]
            r_vals = R[idx, i]
            s_sub = sim[:, idx]                              # (n_users, m)
            part = np.argpartition(s_sub, len(idx) - k, axis=1)[:, len(idx) - k:]
            top_s = np.take_along_axis(s_sub, part, axis=1)
            top_r = r_vals[part]
            num[:, i] = (top_s * top_r).sum(axis=1)
            den[:, i] = top_s.sum(axis=1)
            cnt[:, i] = (top_s > 0).sum(axis=1)
    else:
        num = R @ sim.T                                       # (n_users, n_items)
        den = B @ sim.T
        cnt = B.astype(np.float32) @ (sim > 0).astype(np.float32).T
        n_users = R.shape[0]
        heavy = [u for u in range(n_users) if int(B[u].sum()) > k]
        print(f"    item_knn: correcting {len(heavy)} users with >{k} rated items")
        for u in heavy:
            idx = np.nonzero(B[u])[0]
            r_vals = R[u, idx]
            s_sub = sim[:, idx]                              # (n_items, m)
            part = np.argpartition(s_sub, len(idx) - k, axis=1)[:, len(idx) - k:]
            top_s = np.take_along_axis(s_sub, part, axis=1)
            top_r = r_vals[part]
            num[u, :] = (top_s * top_r).sum(axis=1)
            den[u, :] = top_s.sum(axis=1)
            cnt[u, :] = (top_s > 0).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        est = num / den
    bad = (cnt < min_k) | ~np.isfinite(est)
    est[bad] = mu
    return est.astype(np.float32)


# ---------------------------------------------------------------------------
# FunkSVD (surprise.SVD semantics)
# ---------------------------------------------------------------------------

def funk_svd(uids: np.ndarray, iids: np.ndarray, ratings: np.ndarray,
             n_users: int, n_items: int, n_factors=100, n_epochs=20,
             lr=0.01, reg=0.2, seed=1):
    rng = np.random.RandomState(seed)
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    P = rng.normal(0, 0.1, (n_users, n_factors))
    Q = rng.normal(0, 0.1, (n_items, n_factors))
    mu = ratings.mean()
    for epoch in range(n_epochs):
        sse = 0.0
        for u, i, r in zip(uids, iids, ratings):
            pu, qi = P[u], Q[i]
            err = r - (mu + bu[u] + bi[i] + pu @ qi)
            sse += err * err
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            P[u] = pu + lr * (err * qi - reg * pu)
            Q[i] = qi + lr * (err * pu - reg * qi)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"    svd epoch {epoch + 1}/{n_epochs} train RMSE "
                  f"{np.sqrt(sse / len(ratings)):.4f}")
    return mu, bu, bi, P, Q


# ---------------------------------------------------------------------------
# Top-N name-level recommendation lists
# ---------------------------------------------------------------------------

def top_n_names(scores: np.ndarray, B: np.ndarray, users: list, item_names: list,
                n=TOP_N) -> dict:
    """Rank items per user by score, mask seen items, dedupe by normalized name."""
    masked = scores.copy()
    masked[B > 0] = -np.inf
    order = np.argsort(-masked, axis=1, kind="stable")
    recs = {}
    for u_idx, uid in enumerate(users):
        names, seen = [], set()
        for i in order[u_idx]:
            if not np.isfinite(masked[u_idx, i]):
                break
            nm = item_names[i]
            if nm not in seen:
                seen.add(nm)
                names.append(nm)
                if len(names) == n:
                    break
        recs[uid] = names
    return recs


def save_recs(recs: dict, name: str):
    with open(RESULTS / f"recs_{name}.jsonl", "w", encoding="utf-8") as f:
        for uid, names in recs.items():
            f.write(json.dumps({"user_id": uid, "recs": names}) + "\n")
    print(f"  saved results/recs_{name}.jsonl ({len(recs)} users)")


# ---------------------------------------------------------------------------
# Main benchmark on the leave-one-out cohort
# ---------------------------------------------------------------------------

def main_benchmark():
    train = pd.read_parquet(DATA / "train_interactions.parquet")
    test = pd.read_parquet(DATA / "test_items.parquet")

    users = sorted(train["user_id"].unique())
    items = sorted(train["business_id"].unique())
    u_idx = {u: n for n, u in enumerate(users)}
    i_idx = {b: n for n, b in enumerate(items)}
    n_u, n_i = len(users), len(items)
    print(f"train matrix: {n_u} users x {n_i} items, {len(train)} ratings")

    # cold-item ceiling: test targets never seen in training
    train_names = set(train["business_name"].map(norm_name))
    cold = (~test["business_name"].map(norm_name).isin(train_names)).mean()
    print(f"test targets whose name never appears in training: {cold:.1%} (HR ceiling {1-cold:.1%})")

    R = np.zeros((n_u, n_i), dtype=np.float32)
    uu = train["user_id"].map(u_idx).to_numpy()
    ii = train["business_id"].map(i_idx).to_numpy()
    rr = train["stars"].to_numpy(dtype=np.float32)
    R[uu, ii] = rr
    B = (R > 0).astype(np.float32)
    mu = float(rr.mean())

    id2name = dict(zip(train["business_id"], train["business_name"].map(norm_name)))
    item_names = [id2name[b] for b in items]

    # --- rank-based & popularity -------------------------------------------
    t0 = time.time()
    counts = B.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = R.sum(axis=0) / counts
    eligible = counts >= 50
    print(f"rank_based: {int(eligible.sum())} items with >=50 train interactions")
    rank_scores = np.where(eligible, avg, -np.inf)[None, :].repeat(n_u, axis=0)
    save_recs(top_n_names(rank_scores, B, users, item_names), "rank_based")
    pop_scores = counts[None, :].repeat(n_u, axis=0)
    save_recs(top_n_names(pop_scores, B, users, item_names), "popularity")
    print(f"  rank/popularity done in {time.time()-t0:.0f}s")

    # --- user-user KNN (cosine, k=40, min_k=6) ------------------------------
    t0 = time.time()
    sim_u = cosine_sim(R, B)
    est = knn_full_predict(sim_u, R, B, k=40, min_k=6, mu=mu, user_based=True)
    save_recs(top_n_names(est, B, users, item_names), "user_knn")
    del sim_u, est
    print(f"  user_knn done in {time.time()-t0:.0f}s")

    # --- item-item KNN (msd, k=30, min_k=9) ---------------------------------
    t0 = time.time()
    sim_i = msd_sim(R.T, B.T)
    est = knn_full_predict(sim_i, R, B, k=30, min_k=9, mu=mu, user_based=False)
    save_recs(top_n_names(est, B, users, item_names), "item_knn")
    del sim_i, est
    print(f"  item_knn done in {time.time()-t0:.0f}s")

    # --- FunkSVD (tuned: n_epochs=20, lr=0.01, reg=0.2) ---------------------
    t0 = time.time()
    mu_s, bu, bi, P, Q = funk_svd(uu, ii, rr, n_u, n_i)
    est = (mu_s + bu[:, None] + bi[None, :] + P @ Q.T).astype(np.float32)
    save_recs(top_n_names(est, B, users, item_names), "svd")
    print(f"  svd done in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# Notebook-protocol sanity check (>=100 cohort, random 80/20 rating split)
# ---------------------------------------------------------------------------

def knn_pair_predict(sim, R, B, k, min_k, mu, pairs, user_based):
    out = np.empty(len(pairs), dtype=np.float64)
    for n, (u, i) in enumerate(pairs):
        if user_based:
            neigh = np.nonzero(B[:, i])[0]
            sims = sim[u, neigh]
            r_vals = R[neigh, i]
        else:
            neigh = np.nonzero(B[u])[0]
            sims = sim[i, neigh]
            r_vals = R[u, neigh]
        if len(neigh) > k:
            part = np.argpartition(sims, len(neigh) - k)[len(neigh) - k:]
            sims, r_vals = sims[part], r_vals[part]
        pos = sims > 0
        if pos.sum() < min_k or sims[pos].sum() == 0:
            out[n] = mu
        else:
            out[n] = (sims[pos] * r_vals[pos]).sum() / sims[pos].sum()
    return np.clip(out, 0, 5)


def precision_recall_at_k(df_pred, k=10, threshold=3.5):
    precisions, recalls = [], []
    for _, grp in df_pred.groupby("user_id"):
        grp = grp.sort_values("est", ascending=False)
        n_rel = (grp["true"] >= threshold).sum()
        top = grp.head(k)
        n_rec_k = (top["est"] >= threshold).sum()
        n_rel_and_rec = ((top["true"] >= threshold) & (top["est"] >= threshold)).sum()
        precisions.append(n_rel_and_rec / n_rec_k if n_rec_k else 0)
        recalls.append(n_rel_and_rec / n_rel if n_rel else 0)
    p, r = float(np.mean(precisions)), float(np.mean(recalls))
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def sanity_check():
    print("\n=== Notebook-protocol sanity check (>=100 cohort, 80/20 split) ===")
    df = pd.read_csv(ROOT.parent / "yelp_reviews.csv",
                     usecols=["user_id", "business_id", "stars"])
    vc = df["user_id"].value_counts()
    df = df[df["user_id"].isin(vc[vc >= 100].index)].reset_index(drop=True)
    print(f"cohort: {df['user_id'].nunique()} users, {len(df)} ratings")

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(df))
    n_test = int(round(len(df) * 0.2))
    test_df = df.iloc[perm[:n_test]]
    train_df = df.iloc[perm[n_test:]]

    users = sorted(train_df["user_id"].unique())
    items = sorted(train_df["business_id"].unique())
    u_idx = {u: n for n, u in enumerate(users)}
    i_idx = {b: n for n, b in enumerate(items)}
    R = np.zeros((len(users), len(items)), dtype=np.float32)
    R[train_df["user_id"].map(u_idx), train_df["business_id"].map(i_idx)] = \
        train_df["stars"].to_numpy(dtype=np.float32)
    B = (R > 0).astype(np.float32)
    mu = float(train_df["stars"].mean())

    # only test pairs whose user AND item exist in train (surprise drops others)
    mask = test_df["user_id"].isin(u_idx) & test_df["business_id"].isin(i_idx)
    tp = test_df[mask]
    pairs = list(zip(tp["user_id"].map(u_idx), tp["business_id"].map(i_idx)))
    true = tp["stars"].to_numpy(dtype=np.float64)
    print(f"test pairs evaluated: {len(pairs)} of {len(test_df)}")

    results = {}

    sim_u = cosine_sim(R, B)
    est = knn_pair_predict(sim_u, R, B, 40, 6, mu, pairs, user_based=True)
    results["user_knn(cos,k40,mk6)"] = (true, est)
    del sim_u

    sim_i = msd_sim(R.T, B.T)
    est = knn_pair_predict(sim_i, R, B, 30, 9, mu, pairs, user_based=False)
    results["item_knn(msd,k30,mk9)"] = (true, est)
    del sim_i

    mu_s, bu, bi, P, Q = funk_svd(
        train_df["user_id"].map(u_idx).to_numpy(),
        train_df["business_id"].map(i_idx).to_numpy(),
        train_df["stars"].to_numpy(dtype=np.float64),
        len(users), len(items))
    est = np.clip(np.array([mu_s + bu[u] + bi[i] + P[u] @ Q[i] for u, i in pairs]), 0, 5)
    results["svd(tuned)"] = (true, est)

    sanity = {}
    for name, (t, e) in results.items():
        rmse = float(np.sqrt(np.mean((t - e) ** 2)))
        dfp = pd.DataFrame({"user_id": tp["user_id"].to_numpy(), "true": t, "est": e})
        p, r, f1 = precision_recall_at_k(dfp)
        sanity[name] = {"rmse": round(rmse, 4), "precision@10": round(p, 3),
                        "recall@10": round(r, 3), "f1@10": round(f1, 3)}
        print(f"{name:26s} RMSE {rmse:.4f}  P@10 {p:.3f}  R@10 {r:.3f}  F1@10 {f1:.3f}")

    with open(RESULTS / "sanity_check_notebook_protocol.json", "w") as f:
        json.dump(sanity, f, indent=2)


if __name__ == "__main__":
    main_benchmark()
    sanity_check()
