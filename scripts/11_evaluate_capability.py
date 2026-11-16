#!/usr/bin/env python3
"""11_evaluate_capability.py — Run capability benchmarks on a model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capability import evaluate_capability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--benchmarks", nargs="+",
                        choices=["mmlu", "hellaswag", "ifeval", "humaneval"],
                        default=["mmlu", "hellaswag"])
    parser.add_argument("--n-samples", type=int,
                        help="Limit samples per benchmark")
    parser.add_argument("--output", "-o")
    parser.add_argument("--batch-size", default="auto")
    args = parser.parse_args()

    print(f"Evaluating capabilities for {args.model}")
    print(f"Benchmarks: {args.benchmarks}")

    results = evaluate_capability(
        args.model,
        benchmarks=args.benchmarks,
        n_samples=args.n_samples,
        output_dir=str(Path(args.model).parent / "capability_results"),
        batch_size=args.batch_size,
    )

    for bench, info in results.items():
        score = info.get("score")
        score_str = f"{score:.2%}" if isinstance(score, (int, float)) else str(score)
        print(f"  {bench}: {score_str} ({info['domain']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()