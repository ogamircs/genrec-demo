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

MODELS = ["popularity", "rank_based", "user_knn", "item_knn", "svd", "genrec"]
LABELS = {
    "popularity": "Popularity (most-rated)",
    "rank_based": "Rank-based (avg rating, notebook M1)",
    "user_knn": "User-user KNN (cosine, k=40)",
    "item_knn": "Item-item KNN (msd, k=30)",
    "svd": "SVD / matrix factorization (tuned)",
    "genrec": "GenRec (Llama-3.2-1B + LoRA)",
}


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def metrics_for(recs: dict, targets: dict):
    n = len(targets)
    hr5 = hr10 = ndcg5 = ndcg10 = 0.0
    for uid, target in targets.items():
        rec = recs.get(uid, [])
        rank = rec.index(target) + 1 if target in rec[:10] else None
        if rank is not None:
            gain = 1.0 / math.log2(rank + 1)
            hr10 += 1
            ndcg10 += gain
            if rank <= 5:
                hr5 += 1
                ndcg5 += gain
    return {"HR@5": hr5 / n, "NDCG@5": ndcg5 / n,
            "HR@10": hr10 / n, "NDCG@10": ndcg10 / n}


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

    header = f"{'model':38s} {'HR@5':>8s} {'NDCG@5':>8s} {'HR@10':>8s} {'NDCG@10':>8s}"
    print(header)
    print("-" * len(header))
    for m, met in all_metrics.items():
        print(f"{LABELS[m]:38s} {met['HR@5']:8.4f} {met['NDCG@5']:8.4f} "
              f"{met['HR@10']:8.4f} {met['NDCG@10']:8.4f}")

    with open(RESULTS / "final_metrics.json", "w") as f:
        json.dump({LABELS[m]: met for m, met in all_metrics.items()}, f, indent=2)
    print("\nsaved results/final_metrics.json")


if __name__ == "__main__":
    main()
