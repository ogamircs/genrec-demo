# GenRec on Yelp — generative LLM recommendation vs. classic recommenders

An implementation of **GenRec** — *"GenRec: Large Language Model for Generative Recommendation"* (Ji et al., [arXiv:2307.00457](https://arxiv.org/abs/2307.00457)) — on Yelp restaurant reviews, benchmarked head-to-head against classic recommender systems on the same leave-one-out next-item task.

The same *generative recommendation* direction is being pursued at production scale in industry — see Netflix's post on their LLM-native GenRec foundation model: [GenRec: Towards LLM-native recommendation at Netflix](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3).

**TL;DR:** an LLM fine-tuned to literally *generate the name of the next restaurant a user will visit* beats every rating-prediction baseline — and scaling it (3B, all 98k training windows, catalog-constrained 20-beam decoding) lifts HR@10 a further 56%, past popularity on NDCG@10. But ranking-optimized collaborative filtering still tops every metric on this sparse, weakly-sequential dataset — BPR and SAR from the [recommenders-team library](https://github.com/recommenders-team/recommenders), plus **DeepFM** and **Wide & Deep**, the two standard deep-CTR architectures, which land second and third — models trained to *rank* beat models trained to predict ratings, and here they beat the LLM too (the paper saw the same sparse-data pattern on Amazon Toys). **PinSage**, the Pinterest graph convolutional network, lands mid-table: it is the only model here with no item-ID parameters at all, and it recommends 2.5× more distinct businesses than BPR — accuracy is not the only axis it moves.

## Results

Leave-one-out: each user's chronologically last review is held out; every model emits a top-10 list of business names; hit = held-out name in the list. 1,988 test users, catalog of 10,230 businesses. With one relevant item per user, Recall@10 = HR@10 and Precision@10 = HR@10 / 10; F1@10 is computed per user and averaged.

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | P@10 | R@10 | F1@10 |
|---|---|---|---|---|---|---|---|
| **BPR (Cornac via recommenders)** | **0.0241** | **0.0148** | **0.0428** | **0.0208** | **0.0043** | **0.0428** | **0.0078** |
| DeepFM (torch, FM + deep MLP) | 0.0226 | 0.0136 | 0.0397 | 0.0190 | 0.0040 | 0.0397 | 0.0072 |
| Wide & Deep (torch, crosses + MLP) | 0.0206 | 0.0119 | 0.0387 | 0.0178 | 0.0039 | 0.0387 | 0.0070 |
| SAR (recommenders-team lib) | 0.0186 | 0.0119 | 0.0312 | 0.0159 | 0.0031 | 0.0312 | 0.0057 |
| PinSage (GCN, importance pooling) | 0.0171 | 0.0106 | 0.0282 | 0.0141 | 0.0028 | 0.0282 | 0.0051 |
| Popularity (most-rated) | 0.0111 | 0.0063 | 0.0277 | 0.0116 | 0.0028 | 0.0277 | 0.0050 |
| GenRec v2 (3B full, constrained beams) | 0.0121 | 0.0079 | 0.0267 | 0.0124 | 0.0027 | 0.0267 | 0.0048 |
| GenRec v1 (1B quick, free beams) | 0.0141 | 0.0092 | 0.0171 | 0.0102 | 0.0017 | 0.0171 | 0.0031 |
| Rank-based (avg rating) | 0.0040 | 0.0028 | 0.0075 | 0.0040 | 0.0008 | 0.0075 | 0.0014 |
| SVD / FunkSVD (tuned) | 0.0010 | 0.0005 | 0.0020 | 0.0009 | 0.0002 | 0.0020 | 0.0004 |
| Item-item KNN (msd, k=30) | 0.0015 | 0.0008 | 0.0015 | 0.0008 | 0.0002 | 0.0015 | 0.0003 |
| User-user KNN (cosine, k=40) | 0.0000 | 0.0000 | 0.0015 | 0.0005 | 0.0002 | 0.0015 | 0.0003 |

*(bold = column winner)*

### Seed stability — read the table with error bars

BPR, DeepFM, Wide & Deep and PinSage are all stochastic (random init + sampled negatives; PinSage's
sparse aggregation also accumulates gradients with non-deterministic float atomics). Re-running each
under its **full protocol** — re-selecting the epoch count on the validation split every time — over
6 seeds shows the single-run row above is worth about ±2σ of noise:

| Model | table row (seed 42) | mean ± sd over 6 seeds | range | mean hits @10 |
|---|---|---|---|---|
| BPR | 0.0428 | **0.0459 ± 0.0027** | 0.0423 – 0.0488 | 91 / 1,988 |
| DeepFM | 0.0397 | 0.0375 ± 0.0014 | 0.0352 – 0.0397 | 75 / 1,988 |
| Wide & Deep | 0.0387 | 0.0370 ± 0.0016 | 0.0352 – 0.0387 | 74 / 1,988 |
| PinSage | 0.0282 | 0.0253 ± 0.0035 | 0.0201 – 0.0292 | 50 / 1,988 |

Three things follow, and all of them matter more than the table ordering:

1. **The seed-42 row understates BPR.** Its committed run is at the *bottom* of its own spread while
   the neural models' committed runs sit at the *top* of theirs. On means the gap is 91 vs 75 hits,
   not 85 vs 79 — BPR's lead is bigger than the headline table suggests, not smaller.
2. **DeepFM and Wide & Deep are indistinguishable** — 1 hit apart on means, with almost totally
   overlapping ranges. Whatever separates them on any single run is noise.
3. **PinSage's committed row also overstates it** — 0.0282 against a 0.0253 mean, and its ±0.0035 spread is
   the widest in the table. It sits level with popularity (0.0277) on means, not above it.

All other models in the table are deterministic given the data, so they have no such spread.

### Making GenRec compete — the ablation

Three upgrades were tested on top of the v1 quick recipe (Llama-3.2-1B, 30k training windows, 2 epochs, free 10-beam search):

| Variant | Model & data | Decoding | HR@5 | HR@10 | NDCG@10 |
|---|---|---|---|---|---|
| v1 (quick) | 1B · 30k ex. · 2 ep. | free, 10 beams | **0.0141** | 0.0171 | 0.0102 |
| + training scale | 3B · 98k ex. · 3 ep. | free, 10 beams | 0.0126 | 0.0216 | 0.0109 |
| **v2 (+ constrained)** | 3B · 98k ex. · 3 ep. | catalog trie, 20 beams | 0.0121 | **0.0267** | **0.0124** |
| v3 (+ seen-excluded) | 3B · 98k ex. · 3 ep. | per-user trie, 20 beams | 0.0121 | 0.0262 | 0.0123 |

Training scale bought +26% HR@10 and constrained decoding another +24% (**+56% total**, with 100% in-catalog generations and NDCG@10 above popularity). Excluding already-visited names from the trie changed nothing — the model wasn't wasting beams on revisits. The cost: top-5 slipped ~15%, traceable to the full-data training itself (popularity-skewed windows), not the decoding. BPR still leads by a wide margin.

**📊 Live interactive report:** <https://claude.ai/code/artifact/27a02b7c-cb62-4bd8-9aa6-4d1086ca18d5> — training curve, example generations, and diagnostics. Same content as [`genrec/results/report.html`](genrec/results/report.html); prose version in [`genrec/results/REPORT.md`](genrec/results/REPORT.md).

**How it all works** — pipeline data flow, the vectorized algorithms behind each baseline, and the full fine-tuning recipe (LoRA config, loss masking, training dynamics, beam-search inference): [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Methodologies tested

### 1. GenRec (LLM, generative) — the paper's method

Per the paper: user interaction histories are formatted as instruction-tuning examples using item **names** (not IDs), so the LLM can exploit name semantics.

- **Template** — System: *"You are a restaurant recommendation system."* · User: *"Given a list of restaurants the user has visited in chronological order, predict the restaurant the user will visit next. Restaurants visited: A, B, C…"* · Assistant: *the next restaurant's name* (the only tokens the loss sees).
- **LoRA:** 4-bit QLoRA via [unsloth](https://github.com/unslothai/unsloth) — r=16, α=32, all attention + MLP projections; lr 3e-4 (paper's value), AdamW-8bit, linear decay, effective batch 16.
- **v1 (quick):** `Llama-3.2-1B-Instruct`, 30k of the 98,278 sliding windows (≤15 prior visits each), 2 epochs — 21 min on an RTX 5070 Ti. Free 10-beam search (97.7% of generations in-catalog; lists average 8.9 names).
- **v2 (scaled):** `Llama-3.2-3B-Instruct`, all 98,278 windows, 3 epochs — 4.3 h. **Catalog-constrained 20-beam search**: a token trie over the 7,218 business names (`prefix_allowed_tokens_fn`) forces every beam to spell a real restaurant — 100% in-catalog, 1,958/1,988 full lists. `--exclude-seen` additionally builds per-user tries without visited names (measured: no effect).
- Inference runs through plain `transformers` + `peft` (unsloth's fast-generate KV cache is incompatible with beam search in transformers 5.x).

### 2–6. Classic algorithm baselines

Numpy implementations of the standard surprise-style recommenders, with tuned hyperparameters. `scikit-surprise` ships source-only and needs a C compiler, so the algorithms were written in vectorized numpy and **validated two ways**: full-matrix predictions match an independent per-pair implementation to 1e-6, and under a classic rating-prediction protocol (≥100-review cohort, random 80/20 rating split) they land in the expected range for these algorithms:

| Model (rating-prediction protocol) | RMSE | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|---|
| User-user KNN (cosine, k=40, min_k=6) | 0.9945 | 0.756 | 0.443 | 0.559 |
| Item-item KNN (msd, k=30, min_k=9) | 0.9807 | 0.693 | 0.394 | 0.502 |
| SVD (n_epochs=20, lr=0.01, reg=0.2) | 0.9322 | 0.788 | 0.447 | 0.570 |

- **Popularity** — items ranked by training rating count (non-personalized reference baseline).
- **Rank-based** — average rating among items with ≥50 interactions.
- **User-user KNN** — surprise `KNNBasic` semantics: cosine similarity over co-rated items only, top-k=40 neighbors, min_k=6, global-mean fallback.
- **Item-item KNN** — same structure, msd similarity, k=30, min_k=9.
- **FunkSVD** — biased MF (μ + bᵤ + bᵢ + qᵢ·pᵤ), 100 factors, SGD, tuned n_epochs=20 / lr=0.01 / reg=0.2.
- **CoClustering** — *skipped*: the most complex to implement, and its performance tracks SVD's.

### 7–8. Ranking-optimized baselines — from [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders)

Two algorithms added from the recommenders-team library (the former Microsoft Recommenders project), run via the actual library in a separate Python ≤3.11 environment (`genrec/06_recommenders_baselines.py`):

- **SAR** (Smart Adaptive Recommendations, `recommenders.models.sar.SAR`) — Jaccard item co-occurrence similarity × time-decayed user affinity (30-day decay on review timestamps), `recommend_k_items` with seen-item removal.
- **BPR** (Bayesian Personalized Ranking, via [Cornac](https://github.com/PreferredAI/cornac) — the backend the recommenders library wraps for BPR) — implicit-feedback matrix factorization with a pairwise ranking loss; k=100 factors, 200 iterations, lr 0.01, reg 0.001, seed 42. Scored over all items from the learned factors, seen items masked.

These two are the interesting contrast: they optimize *ranking* rather than rating error, and they top the results table — beating the rating predictors by an order of magnitude and, on this dataset, GenRec as well.

### 9–10. DeepFM and Wide & Deep — the deep-learning baselines (`genrec/07_neural_ctr_baselines.py`)

Two deep CTR architectures, one script, identical data and tuning. They differ in exactly one place — how low-order feature interactions enter the model — which is the whole point of running both.

**DeepFM** (Guo et al., [arXiv:1703.04247](https://arxiv.org/abs/1703.04247)): a factorization-machine component and a deep MLP that **share one embedding table**, so low-order interactions (FM) and high-order ones (MLP) are learned jointly from raw features with no manual feature engineering.

```
y = sigmoid( w0 + Σ w_xj                     ← FM order-1
                + Σ_{j<k} ⟨v_j, v_k⟩         ← FM order-2 (pairwise interactions)
                + MLP([v_1; …; v_F]) )       ← deep component, same embeddings
```

**Wide & Deep** (Cheng et al., [arXiv:1606.07792](https://arxiv.org/abs/1606.07792)) — the architecture DeepFM was written to improve on: a linear **wide** model over the raw fields plus **hand-built cross-product features** φ(x) (memorization), jointly trained with a **deep** MLP that has its own separate embeddings (generalization). DeepFM's contribution was replacing φ(x) with the FM term and sharing the embeddings; running both measures whether that replacement is worth anything here.

```
DeepFM      y = w0 + Σ w_xj + Σ_{j<k} ⟨v_j, v_k⟩ + MLP([v_1; …; v_F])
Wide & Deep y = w_wide · [x, φ(x)] + MLP([v_1; …; v_F]) + b
```

- **Cross-products φ(x)** for the wide part: `user_id × item_popularity`, `user_id × item_avg_star`, `user_activity × item_popularity`, `user_avg_star × item_avg_star`. `user_id × item_id` is deliberately excluded — with one row per observed pair it would be a lookup table over the training positives and generalize to nothing.
- **Fields (F = 6), all categorical, all derived from the training split only:** `user_id`, `user_activity_bucket`, `user_avg_star_bucket`, `item_id`, `item_popularity_bucket`, `item_avg_star_bucket` (log₂ buckets for counts, half-star buckets for averages). The bucketed fields let the model share statistical strength across users/items with similar profiles — the reason to use a *field-aware* model rather than plain MF.
- **Objective (both):** implicit feedback with negative sampling, the standard way to drive a CTR model for top-N retrieval — every observed interaction is a positive, plus 16 uniformly sampled unvisited items per positive as negatives (resampled every epoch), binary cross-entropy. Deviation from the Wide & Deep paper: it trains the wide half with FTRL+L1 and the deep half with AdaGrad; PyTorch has no FTRL, so both halves use Adam — which also keeps the DeepFM comparison optimizer-neutral.
- **Tuning (both, identically):** an 18-config grid (embedding dim × negatives × dropout) on an **internal validation split** — each user's last *training* interaction, held out; the test items are never touched. Both architectures picked the same winner: dim 16, 16 negatives, no dropout, MLP 128→64, Adam lr 1e-3. Each is then refit on the full training data for its validation-selected epoch count (`--sweep` re-runs the grid).
- Written directly in PyTorch (~1 min per model end-to-end on the RTX 5070 Ti, including the validation phase) and scored over all 10,165 items per user with visited items masked, exactly like the other baselines.

**Result: second and third, and tied with each other.** Both clear SAR (75 and 74 mean hits vs 62) and popularity comfortably, and both trail BPR (91 mean hits). The gap between *them* is one hit out of 1,988 — nothing. Two conclusions:

- **The FM component buys nothing over hand-built crosses here** — DeepFM's headline improvement on Wide & Deep doesn't reproduce on this data, which has only 6 fields and 2 of them carrying real information (user id, item id). DeepFM's advantage is supposed to show up with many fields, where enumerating useful crosses by hand becomes hopeless; with a handful of fields, four hand-picked crosses cover the same ground.
- **Neither deep model catches a shallow one.** BPR is plain matrix factorization with no MLP at all. What it has is a *pairwise ranking loss*; both neural models optimize pointwise classification over sampled negatives. On this data the objective is worth more than the architecture.

The one hyperparameter that mattered was the number of negatives: 16 > 8 > 4 at every embedding dim, for both models. Capacity (dim 16 → 64) did nothing consistent.

### 11. PinSage — the graph neural network (`genrec/08_pinsage.py`)

*Ying et al., KDD 2018, [arXiv:1806.01973](https://arxiv.org/abs/1806.01973) — "Graph Convolutional Neural Networks for Web-Scale Recommender Systems," the GCN deployed at Pinterest.*

The only **graph** model here, and the only one with no item-ID parameters anywhere: PinSage is inductive, so a business is nothing but its content features plus its position in the user–business graph. Users play the paper's boards (no features), businesses play its pins (all the features), and walks run business → user → business, which makes each item-level convolution two bipartite hops.

- **Importance-based neighbourhoods.** `N(u)` is not the k-hop neighbourhood but the top-T businesses by L1-normalized visit count of short random walks with restart from `u` — Personalized PageRank in the limit. Measured against exact PPR here, 1,000 chains per node recover 55% of the true top-10 and 16,000 recover 83%; neither changes HR@10.
- **Importance pooling.** The aggregator is a *weighted* mean over those same visit counts, then `z_u ← ReLU(W·concat(h_u, n_u) + w)`, L2-normalized. Two such layers, then the paper's `G₂·ReLU(G₁h + g)` head. Implemented as a sparse mat-mul so the `(n, T, m)` message tensor (520 MB at T=50) never exists.
- **Item features.** The paper's VGG-16 + Word2Vec + log-degree becomes categories (490-way bag), hashed name tokens, city, lat/lon, is_open, log-degree and mean star. Global `business_stars` / `business_review_count` are excluded — they are whole-Yelp aggregates that include the held-out review.
- **Objective.** Max-margin ranking loss, 500 negatives shared per minibatch, plus curriculum hard negatives from a band of the PPR ranking w.r.t. the query — none in epoch 1, *n−1* at epoch *n*, capped at 6.
- **Recommendation** follows the paper's homefeed protocol: nearest items to one of the user's most recently visited businesses, visited items masked.

**The Table 1 ablation ladder, reproduced** (`--variant`, 3 seeds each, test HR@10):

| Variant | Paper (hit-rate) | Here (HR@10, 3-seed mean) |
|---|---|---|
| mean-pooling-xent | 29% (worst) | 0.0148 (worst) ✅ |
| max-pooling | 39% | 0.0221 ✅ |
| mean-pooling | 41% | 0.0258 ✅ |
| mean-pooling-hard | 46% | 0.0262 ✅ |
| **PinSage** (importance pooling) | **67%** | **0.0260** ❌ |

Four of the five rungs land in the paper's order. The one that does not is the paper's headline: **importance pooling buys nothing here**. That is consistent with everything else this dataset says about the neighbourhood — T doesn't resolve, walk fidelity doesn't resolve, and a 2-hop expansion already covers 93% of a 10,165-item catalog. Weighting a neighbourhood matters when it is a 50-item sample of three billion; it does not when it is most of the catalog.

**It buys diversity instead.** Across the 1,988 top-10 lists:

| Model | distinct businesses recommended | most-recommended item's reach | distinct top-1 |
|---|---|---|---|
| **PinSage** | **1,934** | **13.8%** of lists | **612** |
| BPR | 784 | 38.1% | 210 |
| DeepFM | 678 | 33.5% | 167 |
| Popularity | 31 | 88.6% | 9 |

PinSage is by far the least popularity-biased model in the benchmark — which is exactly why it loses on an HR@10 metric whose hits are concentrated in popular restaurants, and exactly what a content-plus-graph inductive model should produce. Its nearest-neighbour structure is sensible on inspection (Pizzeria Bianco → La Grande Orange Pizzeria, Cibo, True Food Kitchen — all independent downtown Phoenix restaurants).

### Evaluation protocols

- **Main benchmark:** leave-one-out next-item, HR@5/10 and NDCG@5/10 (NDCG = 1/log₂(rank+1)) at **normalized-name level** (chains like "Subway" merge to one item; applied equally to all models). Classic models rank the full catalog with seen items masked; GenRec generates freely.
- **Sanity check:** a classic rating-prediction protocol (RMSE + precision/recall@10 at threshold 3.5) — used only to validate the numpy implementations. The gap between these two tables is itself a finding: rating-prediction metrics only rank items the user already rated, which is a far easier task than retrieving 1 item out of 10,230.

## Key findings

1. **Ranking-optimized CF wins overall** — BPR tops every metric even after GenRec's upgrades: models trained to rank beat models trained to predict ratings by an order of magnitude at top-N, and on this sparse data they beat the LLM too.
2. **Deep ≠ better: the objective is what matters.** DeepFM and Wide & Deep — both with MLPs, 6 feature fields and identical grid searches — land second and third, ~18% behind plain BPR on 6-seed means (75 and 74 hits vs 91). BPR is a shallow matrix factorization; what it has is a pairwise ranking loss, while both neural models optimize pointwise classification over sampled negatives.
3. **The two deep architectures are indistinguishable.** DeepFM's selling point over Wide & Deep is replacing hand-built cross features with an FM term. With 6 fields, that replacement is worth one hit out of 1,988 — it needs a wide, messy feature space to pay off, which this dataset does not have.
4. **PinSage reproduces its own ablation ladder, except the headline.** Four of the paper's five Table 1 rungs land in the published order (mean-pooling-xent worst, then max-pooling, then mean-pooling, then +hard negatives). The one that does not is importance pooling itself — the paper's 46% → 67% jump is 0.0262 → 0.0260 here. Weighting a neighbourhood pays when it is a 50-item sample of three billion; a 2-hop expansion of a 10,165-item catalog already reaches 95% of it, so there is nothing left to weight.
5. **Accuracy is not the only axis.** PinSage recommends 1,934 distinct businesses against BPR's 784 and popularity's 31, and its most-recommended item reaches only 13.8% of lists against BPR's 38.1%. It has no item-ID parameters at all, so it can only rank by content and graph position — which costs it HR@10 on a metric whose hits concentrate in popular restaurants, and buys the least popularity-biased catalogue coverage in the benchmark.
6. **Almost nothing about PinSage resolves at this scale.** Across an 18-config grid at 3 seeds each, only the margin δ clears the noise floor (0.1 > 0.3 > 0.5 in all six cells); T, embedding size, the hard-negative band and even the random-walk fidelity (55% → 83% top-10 agreement with exact Personalized PageRank) all land inside run-to-run variance. The defaults therefore keep the paper's values rather than a validation argmax fitted to noise.
7. **The GenRec upgrades worked, up to a ceiling** — 3B + full data + constrained beams lifted HR@10 56% and pushed NDCG@10 past popularity, at a ~15% top-5 cost from popularity-skewed full-data training. GenRec v2 lands within 4% of popularity at HR@10 and clearly above every rating-prediction baseline — still well short of BPR.
8. **Rating predictors are poor top-N retrievers** despite good RMSE — optimizing squared error rewards obscure items with a few perfect ratings.
9. **The LLM exploits name semantics** — e.g., a user whose history is full of frozen-yogurt shops gets four more yogurt shops in the top-10, including the exact held-out one. This is the paper's core motivation for names over IDs.
10. **Consistent with the paper:** GenRec won on dense MovieLens (HR@10 0.131) but lost to its baseline on sparse Amazon Toys (HR@10 0.025); this Yelp result sits in the sparse-data pattern.

## Repository layout

```
genrec/
  01_prepare_data.py       # leave-one-out split + 30k GenRec training examples
  02_classic_baselines.py  # classic algorithm baselines + sanity check
  03_genrec_train.py       # unsloth QLoRA fine-tune (~21 min)
  04_genrec_infer.py       # 10-beam generation of top-10 lists (~4 min)
  05_evaluate.py           # unified HR / NDCG / Precision / Recall / F1 table
  06_recommenders_baselines.py  # SAR + BPR from the recommenders-team library
  07_neural_ctr_baselines.py    # DeepFM + Wide & Deep (torch, ~1 min each)
  08_pinsage.py                 # PinSage GCN + Table 1 ablations (torch, ~25 s)
  results/                 # metrics, rec lists, REPORT.md, report.html
  data/                    # (generated by 01, gitignored)
  models/                  # (generated by 03, gitignored)
```

## Setup & reproduction

1. **Environment:** Python 3.13 with `unsloth`, `torch` (CUDA), `transformers`, `trl`, `peft`, `datasets`, `pandas`, `pyarrow`, `numpy`, plus an NVIDIA GPU (~6 GB VRAM is enough for the 1B model). The easiest path is an [Unsloth](https://docs.unsloth.ai/) install, which pulls everything.
2. **Data:** the big CSV is not committed. Yelp review data is available at <https://huggingface.co/datasets/Yelp/yelp_review_full>. This project was run on the classic Yelp academic export `yelp_reviews.csv` (229,907 reviews, Phoenix metro, 2005–2013) with columns `user_id, business_id, business_name, stars, date, text, …` — the pipeline needs those five named columns, so any Yelp review export containing them will work. Place the file as `yelp_reviews.csv` in the repo root.
3. **Run** (each step is independent and resumable):

```bash
python genrec/01_prepare_data.py
python genrec/02_classic_baselines.py
python genrec/03_genrec_train.py                  # v2 recipe (3B, 3 epochs, ~4.3h)
python genrec/04_genrec_infer.py                  # constrained 20-beam (~22 min)
python genrec/06_recommenders_baselines.py        # see env note below
python genrec/07_neural_ctr_baselines.py          # DeepFM + Wide & Deep (needs torch)
python genrec/08_pinsage.py                       # PinSage (needs torch)
python genrec/05_evaluate.py

# v1 quick recipe (set N_TRAIN_EXAMPLES = 30_000 in 01 first):
python genrec/03_genrec_train.py --base unsloth/Llama-3.2-1B-Instruct --epochs 2 --out genrec_lora
python genrec/04_genrec_infer.py --adapter genrec_lora --beams 10 --out recs_genrec.jsonl --unconstrained
```

The SAR/BPR step needs its own environment because `recommenders` supports Python ≤3.11 and (as of 1.2.1) NumPy <2:

```bash
py -3.11 -m venv .venv-rec
.venv-rec/Scripts/pip install recommenders pyarrow "numpy==1.26.4"
.venv-rec/Scripts/python genrec/06_recommenders_baselines.py
```

## Caveats & next steps

- Initial metrics: one training run, no hyperparameter search, 2 epochs, 30k of 98k possible training windows. The neural CTR baselines are the exception — they get a validation grid search, and their seed spread is measured above.
- PinSage's validation score does not transfer: the *same* model scores 0.0407 on the internal validation split and 0.0211 on test, a bigger drop than a model-free popularity control shows for the same two splits (0.86 ratio vs 0.52). Its `recent` hyperparameter, picked on validation, is flat on test. Read its 6-seed spread, not its committed row.
- First improvements to try: 15–20 beams (fills the top-10 lists), the fuller cohort (users with ≥10 reviews: 4,393 users / 138k interactions), 3+ epochs, and constrained decoding over the catalog.

## References

- Ji, Li, Xu, Hua, Ge, Tan, Zhang — *GenRec: Large Language Model for Generative Recommendation*, [arXiv:2307.00457](https://arxiv.org/abs/2307.00457) (the implemented paper).
- Netflix Tech Blog — [*GenRec: Towards LLM-native recommendation at Netflix*](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3) (related industry work on the same idea).
- Guo, Tang, Ye, Li, He — *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*, [arXiv:1703.04247](https://arxiv.org/abs/1703.04247) (IJCAI 2017).
- Cheng et al. — *Wide & Deep Learning for Recommender Systems*, [arXiv:1606.07792](https://arxiv.org/abs/1606.07792) (DLRS 2016). Both are implemented in `07_neural_ctr_baselines.py`.
- [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders) — source of the SAR and BPR baselines (SAR: `recommenders.models.sar`; BPR: [Cornac](https://github.com/PreferredAI/cornac), the backend the library wraps). Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009.
