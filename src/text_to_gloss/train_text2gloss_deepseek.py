import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import unsloth
from unsloth import FastLanguageModel

import pandas as pd
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from sacrebleu import corpus_chrf, corpus_ter
from sacrebleu.metrics import BLEU
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer
from tqdm.auto import tqdm

SYSTEM_PROMPT = (
    "You are an expert translator for German Sign Language glossing. Your task is to convert a full German sentence "
    "into its precise sign language gloss sequence.\n"
    "Guidelines:\n"
    "- Output only the final gloss sequence as a comma-separated list of uppercase gloss tokens.\n"
    "- Do not include any chain-of-thought, explanations, or intermediary reasoning in your output.\n"
    "- Do not output any tokens not part of the gloss sequence."
)

EXPERIMENT_SEED = 200


def now_tag() -> str:
    return datetime.now().isoformat().split(".")[0].replace(":", "-")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class PeriodicTrainingProgressCallback(TrainerCallback):
    def __init__(self, logger: logging.Logger, log_every_steps: int = 10):
        self.logger = logger
        self.log_every_steps = max(1, int(log_every_steps))
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.logger.info(
            "Training started: max_steps=%s, num_train_epochs=%s",
            state.max_steps,
            args.num_train_epochs,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step % self.log_every_steps != 0:
            return

        elapsed = time.time() - (self.start_time or time.time())
        steps_per_sec = state.global_step / max(elapsed, 1e-9)

        if state.max_steps and state.max_steps > 0:
            pct = 100.0 * state.global_step / state.max_steps
            remaining_steps = max(0, state.max_steps - state.global_step)
            eta_sec = remaining_steps / max(steps_per_sec, 1e-9)
            self.logger.info(
                "Training progress: step %d/%d (%.2f%%) | elapsed=%s | eta=%s | %.3f steps/s",
                state.global_step,
                state.max_steps,
                pct,
                _format_duration(elapsed),
                _format_duration(eta_sec),
                steps_per_sec,
            )
        else:
            self.logger.info(
                "Training progress: step %d | elapsed=%s | %.3f steps/s",
                state.global_step,
                _format_duration(elapsed),
                steps_per_sec,
            )


def configure_logger(log_file: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def _safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def format_conversations(df: pd.DataFrame):
    data = []
    for row in df.itertuples(index=False):
        sentence = _safe_text(getattr(row, "full_sentence", ""))
        glosses = _safe_text(getattr(row, "glosses", ""))

        # Skip unusable rows to avoid Arrow mixed-type issues and bad supervision.
        if not sentence or not glosses:
            continue

        data.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": sentence},
                    {"role": "assistant", "content": glosses},
                ]
            }
        )
    return data


def compute_max_gloss_tokens(df: pd.DataFrame, tokenizer) -> int:
    max_tokens = 0
    for gloss in df["glosses"]:
        g = _safe_text(gloss)
        if not g:
            continue
        tokens = tokenizer(g, add_special_tokens=False, return_tensors="pt")["input_ids"]
        max_tokens = max(max_tokens, tokens.shape[1])
    return max_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="Directory containing train/dev/test CSVs (legacy mode)")
    parser.add_argument("--train_csv", type=str, default="", help="Full path to train CSV")
    parser.add_argument("--dev_csv", type=str, default="", help="Full path to dev CSV")
    parser.add_argument("--test_csv", type=str, default="", help="Full path to test CSV")
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, overrides epochs and limits training steps")
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="If > 0, caps number of samples for each eval split (DEV/TEST)")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for generation during DEV/TEST evaluation")
    parser.add_argument("--run_root", type=str, default="/storage/text_2_gloss_runs")
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--dataset_num_proc", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--train_log_every_steps", type=int, default=10, help="Emit explicit training progress logs every N steps")
    parser.add_argument("--eval_log_every", type=int, default=10, help="Emit explicit evaluation progress logs every N samples")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_id = now_tag()
    run_root = Path(args.run_root)
    run_dir = run_root / run_id
    output_dir = run_dir / "outputs"
    results_dir = run_dir / "results"
    model_dir = results_dir / "fine_tuned_deepseek"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(results_dir / "process.log")
    logger.info("Run ID: %s", run_id)

    if args.train_csv and args.dev_csv and args.test_csv:
        train_csv = Path(args.train_csv)
        dev_csv = Path(args.dev_csv)
        test_csv = Path(args.test_csv)
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        train_csv = data_dir / "train.csv"
        dev_csv = data_dir / "dev.csv"
        test_csv = data_dir / "test.csv"
    else:
        raise ValueError("Provide either --train_csv/--dev_csv/--test_csv or --data_dir")

    logger.info("Using split files:")
    logger.info("  train_csv=%s", train_csv)
    logger.info("  dev_csv=%s", dev_csv)
    logger.info("  test_csv=%s", test_csv)

    train_df = pd.read_csv(train_csv, encoding="utf-8")
    val_df = pd.read_csv(dev_csv, encoding="utf-8")
    test_df = pd.read_csv(test_csv, encoding="utf-8")

    for df in (train_df, val_df, test_df):
        df["full_sentence"] = df["full_sentence"].map(_safe_text)
        df["glosses"] = df["glosses"].map(_safe_text)

    logger.info("Train=%d, Dev=%d, Test=%d", len(train_df), len(val_df), len(test_df))

    if args.max_eval_samples > 0:
        val_df = val_df.head(args.max_eval_samples).copy()
        test_df = test_df.head(args.max_eval_samples).copy()
        logger.info(
            "Evaluation capped to max_eval_samples=%d -> Dev=%d, Test=%d",
            args.max_eval_samples, len(val_df), len(test_df)
        )

    train_dataset = Dataset.from_list(format_conversations(train_df))
    val_dataset = Dataset.from_list(format_conversations(val_df))

    logger.info("Loading model: unsloth/DeepSeek-R1-Distill-Llama-8B")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    max_gloss_tokens = compute_max_gloss_tokens(pd.concat([train_df, val_df, test_df], ignore_index=True), tokenizer)
    logger.info("Max target gloss tokens: %d", max_gloss_tokens)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=EXPERIMENT_SEED,
    )

    def formatting_prompts_func(examples):
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in examples["messages"]]
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            output_dir=str(output_dir),
            optim="adamw_8bit",
            seed=EXPERIMENT_SEED,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
            report_to="none",
        ),
        callbacks=[PeriodicTrainingProgressCallback(logger, args.train_log_every_steps)],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training done.")

    # Switch Unsloth model to inference mode before generate().
    FastLanguageModel.for_inference(model)

    gloss_log = results_dir / "gloss_generation_log.txt"

    def generate_gloss_batch(sentences: list[str]) -> list[str]:
        messages_batch = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ]
            for sentence in sentences
        ]
        input_texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_gloss_tokens + 1,
            use_cache=True,
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [text.split("<think>")[-1].strip() for text in generated_texts]

    def evaluate(df: pd.DataFrame, mode: str):
        total = len(df)
        if total == 0:
            logger.warning("%s evaluation skipped: empty dataset.", mode)
            return

        batch_size = max(1, int(args.eval_batch_size))
        use_tqdm = sys.stdout.isatty()
        eval_start = time.time()
        predictions = []
        log_lines = []

        rows = list(df.itertuples(index=False))

        with torch.inference_mode():
            iter_idx = tqdm(
                range(0, total, batch_size),
                total=(total + batch_size - 1) // batch_size,
                desc=f"Generating {mode}",
                disable=not use_tqdm,
            )
            for start in iter_idx:
                batch_rows = rows[start: start + batch_size]
                batch_sentences = [r.full_sentence for r in batch_rows]
                batch_predictions = generate_gloss_batch(batch_sentences)
                predictions.extend(batch_predictions)

                for sentence, pred in zip(batch_sentences, batch_predictions):
                    log_lines.append(f"Input: {sentence}\nOutput: {pred}\n{'-' * 60}\n")

                done = min(start + len(batch_rows), total)
                if args.eval_log_every > 0 and (done % args.eval_log_every == 0 or done == total):
                    elapsed = time.time() - eval_start
                    samples_per_sec = done / max(elapsed, 1e-9)
                    eta_sec = (total - done) / max(samples_per_sec, 1e-9)
                    logger.info(
                        "%s progress: %d/%d (%.2f%%) | elapsed=%s | eta=%s | %.3f samples/s",
                        mode,
                        done,
                        total,
                        100.0 * done / total,
                        _format_duration(elapsed),
                        _format_duration(eta_sec),
                        samples_per_sec,
                    )

        with open(gloss_log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {mode} ===\n")
            f.writelines(log_lines)

        references = [r.glosses for r in rows]

        bleu_scores = {}
        for n in range(1, 5):
            bleu_scores[f"BLEU-{n}"] = BLEU(max_ngram_order=n).corpus_score(predictions, [references]).score

        chrf = corpus_chrf(predictions, [references]).score
        ter = corpus_ter(predictions, [references]).score

        rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1 = r2 = rL = 0.0
        for p, r in zip(predictions, references):
            s = rouge.score(r, p)
            r1 += s["rouge1"].fmeasure
            r2 += s["rouge2"].fmeasure
            rL += s["rougeL"].fmeasure
        k = len(predictions)
        rouge1 = 100.0 * r1 / k
        rouge2 = 100.0 * r2 / k
        rougel = 100.0 * rL / k

        out_file = results_dir / f"evaluation_results_{mode}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"{mode} set evaluation:\n")
            for name, score in bleu_scores.items():
                f.write(f"{name}: {score:.2f}\n")
            f.write(f"CHRF score: {chrf:.2f}\n")
            f.write(f"TER score: {ter:.2f}\n")
            f.write(f"ROUGE-1: {rouge1:.2f}\n")
            f.write(f"ROUGE-2: {rouge2:.2f}\n")
            f.write(f"ROUGE-L: {rougel:.2f}\n")

        logger.info("%s -> BLEU-1..4=%s, CHRF=%.2f, TER=%.2f, ROUGE1/2/L=%.2f/%.2f/%.2f",
                    mode, bleu_scores, chrf, ter, rouge1, rouge2, rougel)

    evaluate(val_df, "DEV")
    evaluate(test_df, "TEST")

    logger.info("Saving model/tokenizer to: %s", model_dir)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    logger.info("Done.")

if __name__ == "__main__":
    main()