"""Step 1: Shared data prep for GenRec vs classic recommender benchmark.

Leave-one-out split (GenRec paper protocol): per user, the chronologically
last review is the test target; everything earlier is training data.
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

CSV = ROOT.parent / "yelp_reviews.csv"
MIN_REVIEWS = 20
MAX_HISTORY = 15          # max prior items in a prompt
MIN_CONTEXT = 3           # min prior items required for a training example
N_TRAIN_EXAMPLES = None   # None = keep all windows (the quick 1B run used 30_000)
SEED = 42

INSTRUCTION = (
    "Given a list of restaurants the user has visited in chronological order, "
    "predict the restaurant the user will visit next."
)


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def main() -> None:
    df = pd.read_csv(CSV, usecols=["user_id", "business_id", "business_name", "stars", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df["business_name"] = df["business_name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    vc = df["user_id"].value_counts()
    cohort = vc[vc >= MIN_REVIEWS].index
    df = df[df["user_id"].isin(cohort)].reset_index(drop=True)
    print(f"cohort users: {df['user_id'].nunique()}, interactions: {len(df)}")

    # Stable sort by date preserves original row order for same-day reviews.
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    # Leave-one-out: last row per user (chronologically) is the test item.
    last_idx = df.groupby("user_id", sort=False).tail(1).index
    test = df.loc[last_idx].copy()
    train = df.drop(index=last_idx).copy()
    assert len(train) + len(test) == len(df)
    print(f"train interactions: {len(train)}, test items: {len(test)}")

    # keep date: SAR-style models use timestamps for time-decayed affinity
    train.to_parquet(DATA / "train_interactions.parquet", index=False)
    test.to_parquet(DATA / "test_items.parquet", index=False)

    # Catalog of normalized names (from the full cohort data).
    catalog = df[["business_id", "business_name"]].drop_duplicates("business_id").copy()
    catalog["norm_name"] = catalog["business_name"].map(norm_name)
    catalog.to_csv(DATA / "catalog.csv", index=False)
    print(f"catalog: {catalog['business_id'].nunique()} businesses, "
          f"{catalog['norm_name'].nunique()} unique normalized names")

    # Per-user chronological name sequences (training part only).
    histories = train.groupby("user_id", sort=False)["business_name"].agg(list).to_dict()
    test_names = dict(zip(test["user_id"], test["business_name"]))
    with open(DATA / "user_histories.json", "w", encoding="utf-8") as f:
        json.dump({"histories": histories, "test_names": test_names}, f)

    # Sliding-window training examples over training sequences only.
    examples = []
    for user, seq in histories.items():
        for i in range(MIN_CONTEXT, len(seq)):
            hist = seq[max(0, i - MAX_HISTORY):i]
            examples.append({
                "user_id": user,
                "instruction": INSTRUCTION,
                "input": "Restaurants visited: " + ", ".join(hist),
                "output": seq[i],
            })
    print(f"total sliding-window examples: {len(examples)}")

    rng = random.Random(SEED)
    if N_TRAIN_EXAMPLES is not None and len(examples) > N_TRAIN_EXAMPLES:
        examples = rng.sample(examples, N_TRAIN_EXAMPLES)
    rng.shuffle(examples)

    # Leakage check: no training example may target the user's held-out item
    # (impossible by construction since windows only cover the train sequence,
    # but assert it anyway).
    leaks = sum(1 for e in examples if norm_name(e["output"]) == norm_name(test_names[e["user_id"]])
                and e["output"] not in histories[e["user_id"]])
    assert leaks == 0, f"{leaks} leaked examples"

    with open(DATA / "genrec_train.jsonl", "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {len(examples)} training examples -> genrec_train.jsonl")

    lens = [len(e["input"]) for e in examples]
    print(f"input char lengths: mean {sum(lens)/len(lens):.0f}, max {max(lens)}")


if __name__ == "__main__":
    main()
