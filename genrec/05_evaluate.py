"""Step 5: Unified evaluation — HR@5/10 and NDCG@5/10 at normalized-name level
for GenRec and all classic baselines on the same leave-one-out test set."""
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODELS = ["popularity", "rank_based", "user_knn", "item_knn", "svd",
          "sar", "bpr", "deepfm", "widedeep",
          "pinsage_maxpool", "pinsage_meanpool", "pinsage_meanpool_xent",
          "pinsage_meanpool_hard", "pinsage",
          "genrec", "genrec_3b_free", "genrec_3b", "genrec_3b_xs"]
LABELS = {
    "popularity": "Popularity (most-rated)",
    "rank_based": "Rank-based (avg rating)",
    "user_knn": "User-user KNN (cosine, k=40)",
    "item_knn": "Item-item KNN (msd, k=30)",
    "svd": "SVD / matrix factorization (tuned)",
    "sar": "SAR (recommenders-team lib)",
    "bpr": "BPR (Cornac via recommenders)",
    "deepfm": "DeepFM (torch, FM + deep MLP)",
    "widedeep": "Wide & Deep (torch, crosses + MLP)",
    "pinsage_maxpool": "PinSage abl. max-pooling",
    "pinsage_meanpool": "PinSage abl. mean-pooling",
    "pinsage_meanpool_xent": "PinSage abl. mean-pooling-xent",
    "pinsage_meanpool_hard": "PinSage abl. mean-pooling-hard",
    "pinsage": "PinSage (GCN, importance pooling)",
    "genrec": "GenRec v1 (1B quick, free beams)",
    "genrec_3b_free": "GenRec (3B full, free 10 beams)",
    "genrec_3b": "GenRec v2 (3B full, constrained)",
    "genrec_3b_xs": "GenRec v3 (3B, seen-excluded trie)",
}


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def metrics_for(recs: dict, targets: dict, k: int = 10):
    """Leave-one-out metrics with a single relevant item per user.

    Precision@k uses the fixed denominator k; with one relevant item this
    makes Recall@k == HR@k and Precision@k == HR@k / k. F1 is computed
    per user (2PR/(P+R) on a hit, 0 on a miss) and averaged.
    """
    n = len(targets)
    hr5 = hr10 = ndcg5 = ndcg10 = p10 = r10 = f1_10 = 0.0
    for uid, target in targets.items():
        rec = recs.get(uid, [])
        rank = rec.index(target) + 1 if target in rec[:k] else None
        if rank is not None:
            gain = 1.0 / math.log2(rank + 1)
            hr10 += 1
            ndcg10 += gain
            if rank <= 5:
                hr5 += 1
                ndcg5 += gain
            p, r = 1.0 / k, 1.0
            p10 += p
            r10 += r
            f1_10 += 2 * p * r / (p + r)
    return {"HR@5": hr5 / n, "NDCG@5": ndcg5 / n,
            "HR@10": hr10 / n, "NDCG@10": ndcg10 / n,
            "Precision@10": p10 / n, "Recall@10": r10 / n, "F1@10": f1_10 / n}


def main():
    test = pd.read_parquet(DATA / "test_items.parquet")
    targets = {r.user_id: norm_name(r.business_name) for r in test.itertuples()}
    print(f"test users: {len(targets)}\n")

    all_metrics = {}
    for m in MODELS:
        path = RESULTS / f"recs_{m}.jsonl"
        if not path.exists():
            print(f"(skipping {m}: {path.name} not found)")
            continue
        recs = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs[r["user_id"]] = [norm_name(x) for x in r["recs"]]
        all_metrics[m] = metrics_for(recs, targets)

    header = (f"{'model':34s} {'HR@5':>7s} {'NDCG@5':>7s} {'HR@10':>7s} "
              f"{'NDCG@10':>8s} {'P@10':>7s} {'R@10':>7s} {'F1@10':>7s}")
    print(header)
    print("-" * len(header))
    for m, met in all_metrics.items():
        print(f"{LABELS[m]:34s} {met['HR@5']:7.4f} {met['NDCG@5']:7.4f} "
              f"{met['HR@10']:7.4f} {met['NDCG@10']:8.4f} {met['Precision@10']:7.4f} "
              f"{met['Recall@10']:7.4f} {met['F1@10']:7.4f}")

    with open(RESULTS / "final_metrics.json", "w") as f:
        json.dump({LABELS[m]: met for m, met in all_metrics.items()}, f, indent=2)
    print("\nsaved results/final_metrics.json")


if __name__ == "__main__":
    main()
