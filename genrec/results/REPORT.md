# GenRec on Yelp Reviews — Initial Benchmark Report

**Date:** 2026-08-18 · **Hardware:** RTX 5070 Ti (16GB), Windows 11 · **Stack:** unsloth 2026.8.18, torch 2.10, transformers 5.5

## What was done

Implemented **GenRec** (arXiv 2307.00457, "GenRec: LLM for Generative Recommendation") on the Yelp reviews dataset and benchmarked it against classic algorithm baselines.

- **Data:** 229,907 Yelp reviews → cohort of **1,988 users with ≥20 reviews** (106,230 interactions, 10,230 businesses).
- **Split:** leave-one-out per the paper — each user's chronologically last review is the held-out test item; everything earlier is training data. 2.8% of test targets never appear in training (HR ceiling 97.2%).
- **GenRec:** Llama-3.2-1B-Instruct + LoRA (r=16, 11.3M trainable params, 4-bit QLoRA via unsloth). 30,000 instruction-formatted examples (history of ≤15 restaurant *names* → next name), loss on target name only, lr 3e-4, 2 epochs, 21 min training. Inference: beam search (10 beams) generates a top-10 name list per user; 3.8 min for all 1,988 users.
- **Classic baselines:** numpy implementations of the standard surprise-style recommenders (scikit-surprise is not installable here — source-only package, no C compiler). Validated two ways: (1) full-matrix predictions match an independent per-pair implementation to 1e-6; (2) under a classic rating-prediction protocol (≥100-review cohort, random 80/20 split) they land in the expected range — RMSE 0.93–0.99, F1@10 0.50–0.57.

## Main result — leave-one-out next-item recommendation (1,988 users, name-level match)

With one relevant item per user, Recall@10 = HR@10 and Precision@10 = HR@10 / 10; F1@10 is computed per user and averaged.

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | P@10 | R@10 | F1@10 |
|---|---|---|---|---|---|---|---|
| **BPR (Cornac via recommenders)** | **0.0241** | **0.0148** | **0.0428** | **0.0208** | **0.0043** | **0.0428** | **0.0078** |
| SAR (recommenders-team lib) | 0.0186 | 0.0119 | 0.0312 | 0.0159 | 0.0031 | 0.0312 | 0.0057 |
| GenRec (Llama-3.2-1B + LoRA) | 0.0141 | 0.0092 | 0.0171 | 0.0102 | 0.0017 | 0.0171 | 0.0031 |
| Popularity (most-rated) | 0.0111 | 0.0063 | 0.0277 | 0.0116 | 0.0028 | 0.0277 | 0.0050 |
| Rank-based (avg rating) | 0.0040 | 0.0028 | 0.0075 | 0.0040 | 0.0008 | 0.0075 | 0.0014 |
| SVD / matrix factorization (tuned) | 0.0010 | 0.0005 | 0.0020 | 0.0009 | 0.0002 | 0.0020 | 0.0004 |
| Item-item KNN (msd, k=30) | 0.0015 | 0.0008 | 0.0015 | 0.0008 | 0.0002 | 0.0015 | 0.0003 |
| User-user KNN (cosine, k=40) | 0.0000 | 0.0000 | 0.0015 | 0.0005 | 0.0002 | 0.0015 | 0.0003 |

*(bold = column winner)*

SAR and BPR come from the [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders) library (SAR via `recommenders.models.sar`; BPR via Cornac, the backend that library wraps): SAR = Jaccard item co-occurrence × time-decayed user affinity (jaccard, 30-day decay); BPR = implicit-feedback MF with a pairwise ranking loss (k=100, 200 iters, lr 0.01, reg 0.001, seed 42).

### Key findings

1. **Ranking-optimized CF wins overall.** BPR and SAR — the two additions from the recommenders-team library — top every metric. Models *trained to rank* (pairwise loss, co-occurrence affinity) beat models trained to predict ratings by an order of magnitude at top-N, and on this sparse, weakly-sequential data they also beat the 1B LLM.
2. **GenRec beats every rating-prediction baseline.** Best HR@5 among the original six (7–14× the personalized rating predictors), but it trails BPR (0.0241) and SAR (0.0186) at top-5 and falls behind popularity too at depth 10. Two reasons for the depth-10 gap: Yelp restaurant visits are weakly sequential (unlike movie-watching), and GenRec's deduped beam lists average only 8.9 names vs the classics' full 10 (more beams would close this gap).
3. **Rating-predictors are poor top-N retrievers.** KNN/SVD models predict *ratings* well (RMSE ≈ 0.93–0.99, F1@10 ≈ 0.5–0.57 under the rating-prediction protocol) yet their top-N lists almost never contain the next-visited restaurant — the classic mismatch between rating prediction and next-item retrieval. Those metrics only rank items the user already rated in the test split, which is a far easier task than retrieving 1 item out of 10,230.
4. **The LLM learned the catalog.** 97.7% of generated names are real businesses from the data (no constrained decoding used), and 10.9% of raw generations were restaurants the user had already visited (filtered out, matching the classics' masking of seen items).
5. **Numbers are in the paper's ballpark for sparse data.** The paper's GenRec got HR@10 = 0.0251 on Amazon Toys (sparse) and lost to its baseline there too, vs HR@10 = 0.1311 on dense MovieLens where it won. Our Yelp result (HR@10 = 0.0171, ahead of every rating-prediction baseline but behind ranking-optimized CF) is consistent with that pattern.

### Honest caveats

- One training run, no hyperparameter search, 2 epochs, 30k of 98k possible training windows — these are *initial* metrics.
- CoClustering was skipped (most complex to implement, adds little signal).
- Name-level evaluation merges chain locations (e.g. all "Subway" branches count as one item).

### Reproduce

```
# python = C:\Users\pc\.unsloth\studio\unsloth_studio\Scripts\python.exe
python genrec/01_prepare_data.py      # split + training examples
python genrec/02_classic_baselines.py # numpy baselines + sanity check
python genrec/03_genrec_train.py      # ~21 min on RTX 5070 Ti
python genrec/04_genrec_infer.py      # ~4 min
python genrec/05_evaluate.py          # final table
```
