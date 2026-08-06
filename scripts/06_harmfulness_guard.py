#!/usr/bin/env python3
"""06_harmfulness_guard.py — Post-ablation latent harmfulness detection (Zhao et al. 2025)."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_measurements
from src.directions import extract_harmfulness_direction, auto_select_best_layer
from src.evaluate import detect_harmfulness_latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--prompt", required=True, help="Prompt to analyze for harmfulness")
    parser.add_argument("--layer", type=int, help="Layer to probe (default: auto-selected)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Similarity threshold")
    parser.add_argument("--output", "-o", help="Write JSON results")
    args = parser.parse_args()

    measurements = load_measurements(args.measurements)

    layer = args.layer
    if layer is None:
        layer = auto_select_best_layer(measurements, metric="norm")
        print(f"Auto-selected layer: {layer}")

    harmfulness_dir = extract_harmfulness_direction(measurements, layer)
    print(f"Extracted harmfulness direction from layer {layer}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=dtype, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    result = detect_harmfulness_latent(
        model, tokenizer, args.prompt,
        harmfulness_dir, layer, args.threshold
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Harmfulness similarity: {result['harmfulness_similarity']:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Is harmful: {result['is_harmful']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()