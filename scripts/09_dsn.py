#!/usr/bin/env python3
# Created: 23-11-26
# agent: kilo/baidu/ernie-5.5-flash-preview:free
"""09_dsn.py — Don't Say No (Zhou et al., 2025, arXiv:2404.16369).

Evaluates refusal behavior using token-level refusal log-probability and
cosine-decayed refusal suppression alongside surface regex matching.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluate import (
    compute_refusal_token_logprob,
    dsn_cosine_decay,
    evaluate_harmless_compliance,
    evaluate_refusal_rate,
    get_refusal_token_ids,
)
from src.utils import load_prompts


def main():
    parser = argparse.ArgumentParser(
        description="Don't Say No (Paper 07): Token-level refusal log-probability evaluation & DSN loss."
    )
    parser.add_argument("--model", "-m", required=True, help="HF model ID or local path")
    parser.add_argument("--harmful", default="data/harmful.txt", help="Harmful prompts file")
    parser.add_argument("--harmless", default="data/harmless.txt", help="Harmless prompts file")
    parser.add_argument("--max-samples", type=int, default=64, help="Max prompts per split")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generation tokens")
    parser.add_argument(
        "--judge",
        choices=["logprob", "regex", "hybrid"],
        default="hybrid",
        help="Refusal judge mode for generation evaluation",
    )
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=0.5,
        help="Refusal probability threshold for logprob judge",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty during generation",
    )
    parser.add_argument("--gguf-file", help="GGUF filename to load via transformers")
    parser.add_argument("--output", "-o", help="Write JSON results")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.gguf_file:
        load_kwargs["gguf_file"] = args.gguf_file

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    single_ids, multi_seqs = get_refusal_token_ids(tokenizer)
    sample_tokens = [
        tokenizer.decode([tid], skip_special_tokens=True)
        for tid in single_ids[:12]
    ]
    print(f"Resolved {len(single_ids)} single-token and {len(multi_seqs)} multi-token refusal starters")
    print(f"Sample refusal tokens: {sample_tokens}")

    results = {
        "model": args.model,
        "paper": "Don't Say No (Zhou et al., 2025, arXiv:2404.16369)",
        "judge": args.judge,
        "refusal_threshold": args.refusal_threshold,
        "num_single_refusal_tokens": len(single_ids),
        "num_multi_refusal_sequences": len(multi_seqs),
        "cosine_decay_schedule": {
            str(step): dsn_cosine_decay(step, total_steps=10, lambda_init=1.0)
            for step in range(0, 11, 2)
        },
    }

    if args.harmful:
        harmful_prompts = load_prompts(args.harmful, args.max_samples)
        print(f"\n=== HARMFUL PROMPTS ({len(harmful_prompts)} samples) ===")
        lp_harmful = compute_refusal_token_logprob(
            model,
            tokenizer,
            harmful_prompts,
            batch_size=args.batch_size,
            refusal_token_ids=single_ids,
            refusal_threshold=args.refusal_threshold,
        )
        gen_harmful = evaluate_refusal_rate(
            model,
            tokenizer,
            harmful_prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            repetition_penalty=args.repetition_penalty,
            judge=args.judge,
            refusal_threshold=args.refusal_threshold,
        )
        print(f"  Mean refusal log-prob: {lp_harmful['mean_refusal_logprob']:.4f}")
        print(f"  Mean refusal prob:     {lp_harmful['mean_refusal_prob']:.4f}")
        print(f"  Top-1 refusal rate:    {lp_harmful['top1_refusal_rate']:.2%}")
        print(f"  Regex refusal rate:    {gen_harmful['regex_refusal_rate']:.2%}")
        print(f"  Joint refusal rate:    {gen_harmful['refusal_rate']:.2%} (judge={args.judge})")

        results["harmful"] = {
            "token_logprob": lp_harmful,
            "generation_eval": gen_harmful,
        }

    if args.harmless:
        harmless_prompts = load_prompts(args.harmless, args.max_samples)
        print(f"\n=== HARMLESS PROMPTS ({len(harmless_prompts)} samples) ===")
        lp_harmless = compute_refusal_token_logprob(
            model,
            tokenizer,
            harmless_prompts,
            batch_size=args.batch_size,
            refusal_token_ids=single_ids,
            refusal_threshold=args.refusal_threshold,
        )
        gen_harmless = evaluate_harmless_compliance(
            model,
            tokenizer,
            harmless_prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            repetition_penalty=args.repetition_penalty,
            judge=args.judge,
            refusal_threshold=args.refusal_threshold,
        )
        print(f"  Mean refusal log-prob: {lp_harmless['mean_refusal_logprob']:.4f}")
        print(f"  Mean refusal prob:     {lp_harmless['mean_refusal_prob']:.4f}")
        print(f"  Top-1 refusal rate:    {lp_harmless['top1_refusal_rate']:.2%}")
        print(f"  Regex over-refusal:    {gen_harmless['regex_refusal_rate']:.2%}")
        print(f"  Joint over-refusal:    {gen_harmless['over_refusal_rate']:.2%} (judge={args.judge})")

        results["harmless"] = {
            "token_logprob": lp_harmless,
            "generation_eval": gen_harmless,
        }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()