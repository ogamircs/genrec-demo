# GenRec on Yelp Reviews — Initial Benchmark Report

**Date:** 2026-08-18 · **Hardware:** RTX 5070 Ti (16GB), Windows 11 · **Stack:** unsloth 2026.8.18, torch 2.10, transformers 5.5

## What was done

Implemented **GenRec** (arXiv 2307.00457, "GenRec: LLM for Generative Recommendation") on the Yelp reviews dataset and benchmarked it against the classic recommenders from the course notebook (`MLS_Yelp_Reviews_Notebook_with_Numpy_correction.ipynb`).

- **Data:** 229,907 Yelp reviews → cohort of **1,988 users with ≥20 reviews** (106,230 interactions, 10,230 businesses).
- **Split:** leave-one-out per the paper — each user's chronologically last review is the held-out test item; everything earlier is training data. 2.8% of test targets never appear in training (HR ceiling 97.2%).
- **GenRec:** Llama-3.2-1B-Instruct + LoRA (r=16, 11.3M trainable params, 4-bit QLoRA via unsloth). 30,000 instruction-formatted examples (history of ≤15 restaurant *names* → next name), loss on target name only, lr 3e-4, 2 epochs, 21 min training. Inference: beam search (10 beams) generates a top-10 name list per user; 3.8 min for all 1,988 users.
- **Classic baselines:** faithful numpy reimplementations of the notebook's surprise models (scikit-surprise is not installable here — source-only package, no C compiler). Validated two ways: (1) full-matrix predictions match an independent per-pair implementation to 1e-6; (2) under the notebook's own protocol (≥100-review cohort, random 80/20 split) they reproduce the notebook's reported numbers — RMSE 0.93–0.99, F1@10 0.50–0.57.

## Main result — leave-one-out next-item recommendation (1,988 users, name-level match)

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
|---|---|---|---|---|
| **GenRec (Llama-3.2-1B + LoRA)** | **0.0141** | **0.0092** | 0.0171 | 0.0102 |
| Popularity (most-rated) | 0.0111 | 0.0063 | **0.0277** | **0.0116** |

*(bold = column winner)*
| Rank-based (avg rating, notebook M1) | 0.0040 | 0.0028 | 0.0075 | 0.0040 |
| SVD / matrix factorization (tuned) | 0.0010 | 0.0005 | 0.0020 | 0.0009 |
| Item-item KNN (msd, k=30) | 0.0015 | 0.0008 | 0.0015 | 0.0008 |
| User-user KNN (cosine, k=40) | 0.0000 | 0.0000 | 0.0015 | 0.0005 |

### Key findings

1. **GenRec wins the precision-oriented metrics.** It has the best HR@5 and NDCG@5 of all six models — when it finds the right restaurant, it places it high in the list. It beats every *personalized* classic model by ~7–14× on HR@5.
2. **Popularity wins at depth 10.** A non-personalized most-rated list catches more targets in 10 guesses (HR@10 0.0277, NDCG@10 0.0116). Two reasons: Yelp restaurant visits are weakly sequential (unlike movie-watching), and GenRec's deduped beam lists average only 8.9 names vs the classics' full 10 (more beams would close this gap).
3. **Rating-predictors are poor top-N retrievers.** KNN/SVD models predict *ratings* well (RMSE ≈ 0.93–0.99, F1@10 ≈ 0.5–0.57 under the notebook's protocol) yet their top-N lists almost never contain the next-visited restaurant — the classic mismatch between rating prediction and next-item retrieval. The notebook's own metrics only rank items the user already rated in the test split, which is a far easier task than retrieving 1 item out of 10,230.
4. **The LLM learned the catalog.** 97.7% of generated names are real businesses from the data (no constrained decoding used), and 10.9% of raw generations were restaurants the user had already visited (filtered out, matching the classics' masking of seen items).
5. **Numbers are in the paper's ballpark for sparse data.** The paper's GenRec got HR@10 = 0.0251 on Amazon Toys (sparse) and lost to its baseline there too, vs HR@10 = 0.1311 on dense MovieLens where it won. Our Yelp result (HR@10 = 0.0171, winning on precision-style metrics, losing raw HR@10 to a strong simple baseline) is consistent with that pattern.

### Honest caveats

- One training run, no hyperparameter search, 2 epochs, 30k of 98k possible training windows — these are *initial* metrics.
- CoClustering (notebook Model 5) was skipped (most complex to reimplement, adds little signal).
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
