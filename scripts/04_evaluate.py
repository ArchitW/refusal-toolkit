#!/usr/bin/env python3
"""04_evaluate.py — Evaluate refusal rate."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import load_prompts
from src.evaluate import evaluate_refusal_rate, evaluate_harmless_compliance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--harmful")
    parser.add_argument("--harmless")
    parser.add_argument("--output")
    parser.add_argument("--judge", choices=["regex", "logprob", "hybrid"], default="regex",
                        help="Refusal judge mode (Paper 07: logprob/hybrid)")
    parser.add_argument("--refusal-threshold", type=float, default=0.5,
                        help="Refusal probability threshold for logprob judge")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="OBLITERATUS ships 1.15: without it greedy decoding loops on ablated models")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = {"judge": args.judge, "refusal_threshold": args.refusal_threshold}

    if args.harmful:
        prompts = load_prompts(args.harmful)
        print(f"Evaluating {len(prompts)} harmful prompts (judge={args.judge})...")
        results["harmful"] = evaluate_refusal_rate(
            model,
            tokenizer,
            prompts,
            repetition_penalty=args.repetition_penalty,
            judge=args.judge,
            refusal_threshold=args.refusal_threshold,
        )
        print(f"Refusal rate: {results['harmful']['refusal_rate']:.2%}")

    if args.harmless:
        prompts = load_prompts(args.harmless)
        print(f"Evaluating {len(prompts)} harmless prompts (judge={args.judge})...")
        results["harmless"] = evaluate_harmless_compliance(
            model,
            tokenizer,
            prompts,
            repetition_penalty=args.repetition_penalty,
            judge=args.judge,
            refusal_threshold=args.refusal_threshold,
        )
        print(f"Over-refusal rate: {results['harmless']['over_refusal_rate']:.2%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
