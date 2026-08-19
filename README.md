# GenRec on Yelp — generative LLM recommendation vs. classic recommenders

An implementation of **GenRec** — *"GenRec: Large Language Model for Generative Recommendation"* (Ji et al., [arXiv:2307.00457](https://arxiv.org/abs/2307.00457)) — on Yelp restaurant reviews, benchmarked head-to-head against classic recommender systems on the same leave-one-out next-item task.

The same *generative recommendation* direction is being pursued at production scale in industry — see Netflix's post on their LLM-native GenRec foundation model: [GenRec: Towards LLM-native recommendation at Netflix](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3).

**TL;DR:** an LLM fine-tuned to literally *generate the name of the next restaurant a user will visit* beats every rating-prediction baseline — and scaling it (3B, all 98k training windows, catalog-constrained 20-beam decoding) lifts HR@10 a further 56%, past popularity on NDCG@10. But ranking-optimized collaborative filtering (BPR and SAR, added from the [recommenders-team library](https://github.com/recommenders-team/recommenders)) still tops every metric on this sparse, weakly-sequential dataset — models trained to *rank* beat models trained to predict ratings, and here they beat the LLM too (the paper saw the same sparse-data pattern on Amazon Toys).

## Results

Leave-one-out: each user's chronologically last review is held out; every model emits a top-10 list of business names; hit = held-out name in the list. 1,988 test users, catalog of 10,230 businesses. With one relevant item per user, Recall@10 = HR@10 and Precision@10 = HR@10 / 10; F1@10 is computed per user and averaged.

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | P@10 | R@10 | F1@10 |
|---|---|---|---|---|---|---|---|
| **BPR (Cornac via recommenders)** | **0.0241** | **0.0148** | **0.0428** | **0.0208** | **0.0043** | **0.0428** | **0.0078** |
| SAR (recommenders-team lib) | 0.0186 | 0.0119 | 0.0312 | 0.0159 | 0.0031 | 0.0312 | 0.0057 |
| Popularity (most-rated) | 0.0111 | 0.0063 | 0.0277 | 0.0116 | 0.0028 | 0.0277 | 0.0050 |
| GenRec v2 (3B full, constrained beams) | 0.0121 | 0.0079 | 0.0267 | 0.0124 | 0.0027 | 0.0267 | 0.0048 |
| GenRec v1 (1B quick, free beams) | 0.0141 | 0.0092 | 0.0171 | 0.0102 | 0.0017 | 0.0171 | 0.0031 |
| Rank-based (avg rating) | 0.0040 | 0.0028 | 0.0075 | 0.0040 | 0.0008 | 0.0075 | 0.0014 |
| SVD / FunkSVD (tuned) | 0.0010 | 0.0005 | 0.0020 | 0.0009 | 0.0002 | 0.0020 | 0.0004 |
| Item-item KNN (msd, k=30) | 0.0015 | 0.0008 | 0.0015 | 0.0008 | 0.0002 | 0.0015 | 0.0003 |
| User-user KNN (cosine, k=40) | 0.0000 | 0.0000 | 0.0015 | 0.0005 | 0.0002 | 0.0015 | 0.0003 |

*(bold = column winner)*

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

### Evaluation protocols

- **Main benchmark:** leave-one-out next-item, HR@5/10 and NDCG@5/10 (NDCG = 1/log₂(rank+1)) at **normalized-name level** (chains like "Subway" merge to one item; applied equally to all models). Classic models rank the full catalog with seen items masked; GenRec generates freely.
- **Sanity check:** a classic rating-prediction protocol (RMSE + precision/recall@10 at threshold 3.5) — used only to validate the numpy implementations. The gap between these two tables is itself a finding: rating-prediction metrics only rank items the user already rated, which is a far easier task than retrieving 1 item out of 10,230.

## Key findings

1. **Ranking-optimized CF wins overall** — BPR and SAR (from the recommenders-team library) top every metric even after GenRec's upgrades: models trained to rank beat models trained to predict ratings by an order of magnitude at top-N, and on this sparse data they beat the LLM too.
2. **The GenRec upgrades worked, up to a ceiling** — 3B + full data + constrained beams lifted HR@10 56% and pushed NDCG@10 past popularity, at a ~15% top-5 cost from popularity-skewed full-data training. GenRec v2 lands within 4% of popularity at HR@10 and clearly above every rating-prediction baseline — still well short of BPR.
3. **Rating predictors are poor top-N retrievers** despite good RMSE — optimizing squared error rewards obscure items with a few perfect ratings.
4. **The LLM exploits name semantics** — e.g., a user whose history is full of frozen-yogurt shops gets four more yogurt shops in the top-10, including the exact held-out one. This is the paper's core motivation for names over IDs.
5. **Consistent with the paper:** GenRec won on dense MovieLens (HR@10 0.131) but lost to its baseline on sparse Amazon Toys (HR@10 0.025); this Yelp result sits in the sparse-data pattern.

## Repository layout

```
genrec/
  01_prepare_data.py       # leave-one-out split + 30k GenRec training examples
  02_classic_baselines.py  # classic algorithm baselines + sanity check
  03_genrec_train.py       # unsloth QLoRA fine-tune (~21 min)
  04_genrec_infer.py       # 10-beam generation of top-10 lists (~4 min)
  05_evaluate.py           # unified HR / NDCG / Precision / Recall / F1 table
  06_recommenders_baselines.py  # SAR + BPR from the recommenders-team library
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

- Initial metrics: one training run, no hyperparameter search, 2 epochs, 30k of 98k possible training windows.
- First improvements to try: 15–20 beams (fills the top-10 lists), the fuller cohort (users with ≥10 reviews: 4,393 users / 138k interactions), 3+ epochs, and constrained decoding over the catalog.

## References

- Ji, Li, Xu, Hua, Ge, Tan, Zhang — *GenRec: Large Language Model for Generative Recommendation*, [arXiv:2307.00457](https://arxiv.org/abs/2307.00457) (the implemented paper).
- Netflix Tech Blog — [*GenRec: Towards LLM-native recommendation at Netflix*](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3) (related industry work on the same idea).
- [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders) — source of the SAR and BPR baselines (SAR: `recommenders.models.sar`; BPR: [Cornac](https://github.com/PreferredAI/cornac), the backend the library wraps). Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009.
