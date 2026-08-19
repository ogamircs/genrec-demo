"""Step 4: GenRec inference — beam-search top-10 next-restaurant generation
for every test user (leave-one-out target).

Two decoding modes:
  - constrained (default): every beam is forced to spell a real catalog
    business name via a token trie + prefix_allowed_tokens_fn, so all beam
    mass lands on valid, distinct candidates (standard generative-retrieval
    trick);
  - --unconstrained: free-form generation (the v1 quick-run behavior).

Uses plain transformers + PEFT (not unsloth's fast-generate patch, whose
tuple KV cache is incompatible with beam search in transformers 5.x).

  v1 quick run: --adapter genrec_lora --beams 10 --out recs_genrec.jsonl --unconstrained
  v2 big run:   defaults below
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MAX_HISTORY = 15
SYSTEM_PROMPT = "You are a restaurant recommendation system."
INSTRUCTION = (
    "Given a list of restaurants the user has visited in chronological order, "
    "predict the restaurant the user will visit next."
)


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def build_trie(token_seqs):
    """Token trie over catalog names. Node = {token_id: child}; terminal nodes
    carry the key -1. Allowed continuations = children + eos when terminal."""
    root = {}
    for ids in token_seqs:
        node = root
        for t in ids:
            node = node.setdefault(t, {})
        node[-1] = True
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="genrec_lora_3b")
    ap.add_argument("--beams", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="recs_genrec_3b.jsonl")
    ap.add_argument("--unconstrained", action="store_true")
    ap.add_argument("--exclude-seen", action="store_true",
                    help="per-user tries that exclude already-visited names, so "
                         "all beam mass lands on unseen candidates")
    args = ap.parse_args()

    adapter_dir = str(ROOT / "models" / args.adapter)
    base_name = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
    print(f"adapter: {args.adapter}  base: {base_name}  beams: {args.beams}  "
          f"constrained: {not args.unconstrained}")
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    blob = json.load(open(DATA / "user_histories.json", encoding="utf-8"))
    histories = blob["histories"]
    users = list(histories.keys())

    trie = None
    name_tokens = None
    if not args.unconstrained:
        catalog_names = pd.read_csv(DATA / "catalog.csv")["business_name"].unique()
        name_tokens = {n: tokenizer(n, add_special_tokens=False)["input_ids"]
                       for n in catalog_names}
        if not args.exclude_seen:
            trie = build_trie(name_tokens.values())
        print(f"catalog trie over {len(catalog_names)} display names "
              f"(per-user, seen excluded)" if args.exclude_seen else
              f"catalog trie over {len(catalog_names)} display names")

    print(f"generating top-10 recs for {len(users)} users")
    out_path = RESULTS / args.out
    f_out = open(out_path, "w", encoding="utf-8")
    t0 = time.time()

    for start in range(0, len(users), args.batch):
        batch = users[start:start + args.batch]
        prompts = []
        for uid in batch:
            hist = histories[uid][-MAX_HISTORY:]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INSTRUCTION + "\n\nRestaurants visited: "
                                            + ", ".join(hist)},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=24,
            num_beams=args.beams,
            num_return_sequences=args.beams,
            do_sample=False,
            early_stopping=True,
            pad_token_id=eos_id,
        )
        if not args.unconstrained:
            if args.exclude_seen:
                row_tries = []
                for uid in batch:
                    seen_hist = {norm_name(n) for n in histories[uid]}
                    row_tries.append(build_trie(
                        ids for n, ids in name_tokens.items()
                        if norm_name(n) not in seen_hist))
            else:
                row_tries = [trie] * len(batch)

            def allowed_fn(batch_id, sent):
                node = row_tries[batch_id]
                for t in sent[prompt_len:].tolist():
                    nxt = node.get(t)
                    if nxt is None:
                        return [eos_id]
                    node = nxt
                allowed = [t for t in node if t != -1]
                if -1 in node:
                    allowed.append(eos_id)
                return allowed or [eos_id]
            gen_kwargs["prefix_allowed_tokens_fn"] = allowed_fn

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

        for b, uid in enumerate(batch):
            beams = texts[b * args.beams:(b + 1) * args.beams]
            seen_hist = {norm_name(n) for n in histories[uid]}
            raw, filtered, dedup = [], [], set()
            for t in beams:
                nm = norm_name(t)
                if not nm or nm in dedup:
                    continue
                dedup.add(nm)
                raw.append(nm)
                if nm not in seen_hist and len(filtered) < 10:
                    filtered.append(nm)
            f_out.write(json.dumps({"user_id": uid, "recs": filtered,
                                    "recs_raw": raw}) + "\n")

        if (start // args.batch) % 25 == 0:
            done = start + len(batch)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(users)} users  ({rate:.1f} users/s, "
                  f"eta {(len(users) - done) / max(rate, 1e-9) / 60:.1f} min)", flush=True)
    f_out.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_path}")

    # diagnostics
    catalog = set(pd.read_csv(DATA / "catalog.csv")["norm_name"])
    n_recs = n_in_cat = n_repeat_dropped = 0
    n_lists = full_lists = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_recs += len(r["recs_raw"])
            n_in_cat += sum(1 for n in r["recs_raw"] if n in catalog)
            n_repeat_dropped += len(r["recs_raw"]) - len(r["recs"])
            n_lists += 1
            full_lists += len(r["recs"]) >= 10
    print(f"in-catalog rate: {n_in_cat / n_recs:.1%}  "
          f"already-visited/overflow dropped: {n_repeat_dropped / n_recs:.1%}  "
          f"full 10-name lists: {full_lists}/{n_lists}")


if __name__ == "__main__":
    main()
