"""Step 4: GenRec inference — beam-search top-10 next-restaurant generation
for every test user (leave-one-out target).

Uses plain transformers + PEFT (not unsloth's fast-generate patch, whose
tuple KV cache is incompatible with beam search in transformers 5.x).
"""
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MAX_HISTORY = 15
NUM_BEAMS = 10
BATCH_USERS = 8
SYSTEM_PROMPT = "You are a restaurant recommendation system."
INSTRUCTION = (
    "Given a list of restaurants the user has visited in chronological order, "
    "predict the restaurant the user will visit next."
)


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def main():
    adapter_dir = str(ROOT / "models" / "genrec_lora")
    base = AutoModelForCausalLM.from_pretrained(
        "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit",
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blob = json.load(open(DATA / "user_histories.json", encoding="utf-8"))
    histories = blob["histories"]
    users = list(histories.keys())
    print(f"generating top-{NUM_BEAMS} recs for {len(users)} users")

    out_path = RESULTS / "recs_genrec.jsonl"
    f_out = open(out_path, "w", encoding="utf-8")
    t0 = time.time()

    for start in range(0, len(users), BATCH_USERS):
        batch = users[start:start + BATCH_USERS]
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
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=24,
                num_beams=NUM_BEAMS,
                num_return_sequences=NUM_BEAMS,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for b, uid in enumerate(batch):
            beams = texts[b * NUM_BEAMS:(b + 1) * NUM_BEAMS]
            seen_hist = {norm_name(n) for n in histories[uid]}
            raw, filtered, dedup = [], [], set()
            for t in beams:
                nm = norm_name(t)
                if not nm or nm in dedup:
                    continue
                dedup.add(nm)
                raw.append(nm)
                if nm not in seen_hist:
                    filtered.append(nm)
            f_out.write(json.dumps({"user_id": uid, "recs": filtered,
                                    "recs_raw": raw}) + "\n")

        if (start // BATCH_USERS) % 25 == 0:
            done = start + len(batch)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(users)} users  ({rate:.1f} users/s, "
                  f"eta {(len(users) - done) / max(rate, 1e-9) / 60:.1f} min)", flush=True)
    f_out.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_path}")

    # diagnostics
    import pandas as pd
    catalog = set(pd.read_csv(DATA / "catalog.csv")["norm_name"])
    n_recs = n_in_cat = n_repeat_dropped = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_recs += len(r["recs_raw"])
            n_in_cat += sum(1 for n in r["recs_raw"] if n in catalog)
            n_repeat_dropped += len(r["recs_raw"]) - len(r["recs"])
    print(f"in-catalog rate: {n_in_cat / n_recs:.1%}  "
          f"already-visited recs dropped: {n_repeat_dropped / n_recs:.1%}")


if __name__ == "__main__":
    main()
