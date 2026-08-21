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
| DeepFM (torch, FM + deep MLP) | 0.0226 | 0.0136 | 0.0397 | 0.0190 | 0.0040 | 0.0397 | 0.0072 |
| Wide & Deep (torch, crosses + MLP) | 0.0206 | 0.0119 | 0.0387 | 0.0178 | 0.0039 | 0.0387 | 0.0070 |
| SAR (recommenders-team lib) | 0.0186 | 0.0119 | 0.0312 | 0.0159 | 0.0031 | 0.0312 | 0.0057 |
| PinSage (GCN, importance pooling) | 0.0171 | 0.0106 | 0.0282 | 0.0141 | 0.0028 | 0.0282 | 0.0051 |
| Popularity (most-rated) | 0.0111 | 0.0063 | 0.0277 | 0.0116 | 0.0028 | 0.0277 | 0.0050 |
| GenRec v2 (3B full, constrained 20 beams) | 0.0121 | 0.0079 | 0.0267 | 0.0124 | 0.0027 | 0.0267 | 0.0048 |
| GenRec v3 (v2 + seen-excluded trie) | 0.0121 | 0.0080 | 0.0262 | 0.0123 | 0.0026 | 0.0262 | 0.0048 |
| GenRec (3B full, free 10 beams) | 0.0126 | 0.0079 | 0.0216 | 0.0109 | 0.0022 | 0.0216 | 0.0039 |
| GenRec v1 (1B quick, free 10 beams) | 0.0141 | 0.0092 | 0.0171 | 0.0102 | 0.0017 | 0.0171 | 0.0031 |
| Rank-based (avg rating) | 0.0040 | 0.0028 | 0.0075 | 0.0040 | 0.0008 | 0.0075 | 0.0014 |
| SVD / matrix factorization (tuned) | 0.0010 | 0.0005 | 0.0020 | 0.0009 | 0.0002 | 0.0020 | 0.0004 |
| Item-item KNN (msd, k=30) | 0.0015 | 0.0008 | 0.0015 | 0.0008 | 0.0002 | 0.0015 | 0.0003 |
| User-user KNN (cosine, k=40) | 0.0000 | 0.0000 | 0.0015 | 0.0005 | 0.0002 | 0.0015 | 0.0003 |

*(bold = column winner)*

### Seed stability

BPR, DeepFM, Wide & Deep and PinSage are stochastic (random init + sampled negatives; PinSage's sparse
aggregation also accumulates gradients with non-deterministic float atomics, so even a fixed seed varies).
Re-running each under its full protocol — re-selecting the epoch count on the validation split every
time — over 6 seeds:

| Model | table row (seed 42) | mean ± sd over 6 seeds | range | mean hits @10 |
|---|---|---|---|---|
| BPR | 0.0428 | **0.0459 ± 0.0027** | 0.0423 – 0.0488 | 91 / 1,988 |
| DeepFM | 0.0397 | 0.0375 ± 0.0014 | 0.0352 – 0.0397 | 75 / 1,988 |
| Wide & Deep | 0.0387 | 0.0370 ± 0.0016 | 0.0352 – 0.0387 | 74 / 1,988 |
| PinSage | 0.0282 | 0.0253 ± 0.0035 | 0.0201 – 0.0292 | 50 / 1,988 |

The committed seed-42 run happens to be BPR's *worst* and close to the neural models' *best*, so the
single-run table understates BPR's lead: on means it is 91 hits vs 75, not 85 vs 79. DeepFM and
Wide & Deep are one hit apart with overlapping ranges — a tie. PinSage has the widest spread of the four
and its committed row also flatters it: on means it is level with popularity (50 hits vs 55), not above
it. Every other model in the table is deterministic given the data.

**DeepFM** ([arXiv:1703.04247](https://arxiv.org/abs/1703.04247)) and **Wide & Deep** ([arXiv:1606.07792](https://arxiv.org/abs/1606.07792)) are the deep-learning baselines, both in `genrec/07_neural_ctr_baselines.py` over identical data and tuning. DeepFM = an FM component and a deep MLP sharing one embedding table; Wide & Deep = a linear model over raw fields *plus four hand-built cross-products* (memorization), jointly trained with a separately-embedded MLP (generalization). Both use 6 categorical fields (user/item ids plus log-bucketed activity, popularity and average-rating fields derived from training data only), implicit feedback with 16 uniform negatives per positive, and binary cross-entropy. Dim/negatives/dropout came from an 18-config grid per model on an internal validation split (each user's last *training* interaction — the test items are never touched); both picked dim 16 / 16 negatives / no dropout, then refit on the full training set. ~1 min each on the RTX 5070 Ti.

**PinSage** ([arXiv:1806.01973](https://arxiv.org/abs/1806.01973), `genrec/08_pinsage.py`) is the graph model: a GCN over the bipartite user/business graph, with importance-based neighbourhoods (top-50 businesses by L1-normalized visit count of random walks with restart — Personalized PageRank in the limit), importance pooling (weighted mean over those counts), two convolutional layers, the paper's `G₂·ReLU(G₁h+g)` head, a max-margin ranking loss with 500 minibatch-shared negatives, and curriculum hard negatives from a PPR rank band (none in epoch 1, *n−1* at epoch *n*, capped at 6). It is the only model in the benchmark with **no item-ID parameters** — a business is its content features (categories, name tokens, city, lat/lon, log-degree, mean star) plus its graph position — which is also why adding a per-item embedding (`--item-id-emb 64`) changes nothing (0.0245 vs 0.0253 over 3 seeds). Recommendation uses the paper's homefeed protocol: nearest items to one of the user's most recently visited businesses. ~25 s end to end.

Two implementation checks worth recording. The Algorithm 2 subgraph re-indexing reproduces a full-catalog forward pass to `1.2e-07` (float32 round-off) for minibatches of 7/64/1,024 nodes. And the random-walk estimator really does approximate Personalized PageRank — against exact power iteration, 1,000 chains per node recover 55.4% of the true top-10 neighbourhood and 16,000 recover 82.5%, at 0.4 s and 1.5 s respectively.

SAR and BPR come from the [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders) library (SAR via `recommenders.models.sar`; BPR via Cornac, the backend that library wraps): SAR = Jaccard item co-occurrence × time-decayed user affinity (jaccard, 30-day decay); BPR = implicit-feedback MF with a pairwise ranking loss (k=100, 200 iters, lr 0.01, reg 0.001, seed 42).

GenRec variants (see README ablation): v1 = Llama-3.2-1B, 30k windows, 2 epochs, free 10-beam; v2 = Llama-3.2-3B, all 98k windows, 3 epochs (4.3 h), catalog-trie-constrained 20-beam (100% in-catalog, 1,958/1,988 full lists); v3 = v2 with per-user tries excluding visited names (no measurable effect). Training scale bought +26% HR@10, constrained decoding another +24%; top-5 slipped ~15% from popularity-skewed full-data training.

### Key findings

1. **Ranking-optimized CF wins overall.** BPR tops every metric. Models *trained to rank* (pairwise loss, co-occurrence affinity) beat models trained to predict ratings by an order of magnitude at top-N, and on this sparse, weakly-sequential data they also beat the LLM.
2. **Deep learning does not rescue the objective.** DeepFM and Wide & Deep finish second and third — comfortably ahead of SAR (75 and 74 mean hits vs 62) and popularity, and ~18% behind plain BPR (91 mean hits). BPR has no neural network at all; what it has is a pairwise ranking loss, while both deep models optimize pointwise classification over sampled negatives. Architecture lost to objective.
3. **The two deep architectures are indistinguishable.** DeepFM's selling point over Wide & Deep is replacing hand-built cross features with an FM term that learns all pairwise interactions. Here that is worth one hit out of 1,988. With only 6 fields — two of them carrying the real signal — four hand-picked crosses cover the same ground; DeepFM's edge needs a wide, messy feature space this dataset does not have. The one hyperparameter that mattered for both was the negative count (16 > 8 > 4 at every embedding dim); capacity did nothing consistent.
4. **PinSage reproduces four of its five Table 1 rungs — but not the headline.** Running the paper's own ablation ladder (`--variant`, 3 seeds each, test HR@10): mean-pooling-xent 0.0148 (worst, as in the paper), max-pooling 0.0221, mean-pooling 0.0258, mean-pooling-hard 0.0262 — the published order. Importance pooling, the paper's 46% → 67% jump, gives 0.0260: nothing. The reason is visible in the same data: T (10/20/50), the hard-negative band and even walk fidelity (55% → 83% agreement with exact PPR) all land inside run-to-run noise. Weighting a neighbourhood pays when it is a 50-item sample of three billion; a 2-hop expansion of a 10,165-item catalog already reaches ~9,700 of them, so the weighting has nothing to bite on. Only the margin δ resolved across an 18-config grid (0.1 > 0.3 > 0.5 in all six dim × T cells).
5. **PinSage trades accuracy for coverage.** It recommends 1,934 distinct businesses across the 1,988 lists, against BPR's 784, DeepFM's 678 and popularity's 31; its most-recommended item appears in 13.8% of lists against BPR's 38.1%. With no ID parameters it can only rank by content and graph position, and its neighbour structure is sensible on inspection (Pizzeria Bianco → La Grande Orange Pizzeria, Cibo, True Food Kitchen). That is exactly the profile that loses an HR@10 contest whose hits concentrate in popular restaurants.
4. **GenRec beats every rating-prediction baseline.** Best HR@5 among the original six (7–14× the personalized rating predictors), but it trails BPR (0.0241) and SAR (0.0186) at top-5 and falls behind popularity too at depth 10. Two reasons for the depth-10 gap: Yelp restaurant visits are weakly sequential (unlike movie-watching), and GenRec's deduped beam lists average only 8.9 names vs the classics' full 10 (more beams would close this gap).
6. **Rating-predictors are poor top-N retrievers.** KNN/SVD models predict *ratings* well (RMSE ≈ 0.93–0.99, F1@10 ≈ 0.5–0.57 under the rating-prediction protocol) yet their top-N lists almost never contain the next-visited restaurant — the classic mismatch between rating prediction and next-item retrieval. Those metrics only rank items the user already rated in the test split, which is a far easier task than retrieving 1 item out of 10,230.
7. **The LLM learned the catalog.** 97.7% of generated names are real businesses from the data (no constrained decoding used), and 10.9% of raw generations were restaurants the user had already visited (filtered out, matching the classics' masking of seen items).
8. **Numbers are in the paper's ballpark for sparse data.** The paper's GenRec got HR@10 = 0.0251 on Amazon Toys (sparse) and lost to its baseline there too, vs HR@10 = 0.1311 on dense MovieLens where it won. Our Yelp result (HR@10 = 0.0171, ahead of every rating-prediction baseline but behind ranking-optimized CF) is consistent with that pattern.

### Honest caveats

- One training run, no hyperparameter search, 2 epochs, 30k of 98k possible training windows — these are *initial* metrics. The neural CTR models are the exception: grid-searched on an internal validation split, which if anything flatters them relative to the untuned models.
- Absolute hit counts are small (55–97 hits out of 1,988 users), so differences under ~10 hits are within noise — see the seed-stability table above rather than reading the single-run ordering too closely.
- PinSage's validation score does not transfer. The *same* trained model scores 0.0407 on the internal
  validation split and 0.0211 on test — a far bigger drop than a model-free popularity control shows for
  those two splits (0.0257 → 0.0221, ratio 0.86 against PinSage's 0.52), and the median gap from query to
  target grows from 8 days at the validation step to 12 at the test step. Its `recent` hyperparameter,
  chosen on validation, is flat on test (0.022–0.029 across all six values).
- CoClustering was skipped (most complex to implement, adds little signal).
- Name-level evaluation merges chain locations (e.g. all "Subway" branches count as one item).

### Reproduce

```
# python = C:\Users\pc\.unsloth\studio\unsloth_studio\Scripts\python.exe
python genrec/01_prepare_data.py      # split + training examples
python genrec/02_classic_baselines.py # numpy baselines + sanity check
python genrec/03_genrec_train.py      # ~21 min on RTX 5070 Ti
python genrec/04_genrec_infer.py      # ~4 min
python genrec/07_neural_ctr_baselines.py  # DeepFM + Wide & Deep (--sweep re-runs the grid)
python genrec/08_pinsage.py               # PinSage (--sweep main|band, --variant for the ablations)
python genrec/05_evaluate.py          # final table
```
