"""Step 3: GenRec fine-tuning (unsloth QLoRA on Llama-3.2-1B-Instruct).

Paper setup adapted to a 1B model: instruction-formatted next-item examples,
LoRA, lr 3e-4, AdamW, linear decay, loss on the target item name only.
"""
from unsloth import FastLanguageModel  # noqa: F401  (must be first import)

import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LEN = 320
SYSTEM_PROMPT = "You are a restaurant recommendation system."


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    rows = [json.loads(l) for l in open(DATA / "genrec_train.jsonl", encoding="utf-8")]
    print(f"training examples: {len(rows)}")

    def to_text(row):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"] + "\n\n" + row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list(rows).map(to_text, remove_columns=[c for c in rows[0]])
    print("sample formatted example:\n", ds[0]["text"][:600])

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(MODELS / "checkpoints"),
            dataset_text_field="text",
            max_length=MAX_SEQ_LEN,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=2,
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            warmup_steps=100,
            optim="adamw_8bit",
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="no",
            seed=42,
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # verify masking: only the item name (+eot) should remain unmasked
    sample = trainer.train_dataset[0]
    labels = sample["labels"]
    kept = [t for t, l in zip(sample["input_ids"], labels) if l != -100]
    print("unmasked target tokens decode to:", repr(tokenizer.decode(kept)))

    stats = trainer.train()
    print("train runtime (s):", stats.metrics.get("train_runtime"))
    print("final train loss:", stats.metrics.get("train_loss"))

    model.save_pretrained(str(MODELS / "genrec_lora"))
    tokenizer.save_pretrained(str(MODELS / "genrec_lora"))
    print("adapter saved to models/genrec_lora")

    # smoke test: greedy generation on 5 training prompts
    FastLanguageModel.for_inference(model)
    for row in rows[:5]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"] + "\n\n" + row["input"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=24,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        print(f"  target: {row['output']!r:45s} generated: {gen!r}")


if __name__ == "__main__":
    main()
