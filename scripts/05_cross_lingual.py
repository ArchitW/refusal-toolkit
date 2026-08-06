#!/usr/bin/env python3
"""05_cross_lingual.py — Cross-lingual direction transfer (Wang et al. 2025)."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_measurements, load_prompts
from src.directions import (
    extract_single_direction,
    auto_select_best_layer,
    project_against_harmless,
    normalize_direction,
)
from src.ablate import register_ablation_hooks, remove_hooks
from src.evaluate import evaluate_refusal_rate, evaluate_harmless_compliance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--measurements", required=True, help="English measurements .pt")
    parser.add_argument("--target-lang-harmful", required=True, help="Target language harmful prompts")
    parser.add_argument("--target-lang-harmless", required=True, help="Target language harmless prompts")
    parser.add_argument("--source-layer", type=int, help="Layer to extract direction from (default: auto)")
    parser.add_argument("--target-layers", type=int, nargs="+", help="Layers to ablate (default: all)")
    parser.add_argument("--scale", type=float, default=1.0, help="Ablation strength")
    parser.add_argument("--projected", action="store_true", help="Project against harmless direction")
    parser.add_argument("--output", "-o", help="Write JSON results")
    parser.add_argument("--max-samples", type=int, default=64)
    args = parser.parse_args()

    measurements = load_measurements(args.measurements)

    source_layer = args.source_layer
    if source_layer is None:
        source_layer = auto_select_best_layer(measurements, metric="norm")
        print(f"Auto-selected source layer: {source_layer}")

    direction = extract_single_direction(measurements, source_layer)
    if args.projected:
        harmless = measurements[f"harmless_{source_layer}"].float()
        direction = project_against_harmless(direction, harmless)
        print("Projected direction against harmless mean")
    direction = normalize_direction(direction)

    print(f"Loading model: {args.model}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=dtype, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load target language prompts
    target_harmful = load_prompts(args.target_lang_harmful, args.max_samples)
    target_harmless = load_prompts(args.target_lang_harmless, args.max_samples)
    print(f"Loaded {len(target_harmful)} target harmful, {len(target_harmless)} target harmless prompts")

    # Baseline (no ablation)
    print("\n[BASELINE - Target Language]")
    baseline_harmful = evaluate_refusal_rate(model, tokenizer, target_harmful)
    baseline_harmless = evaluate_harmless_compliance(model, tokenizer, target_harmless)
    print(f"Harmful refusal rate: {baseline_harmful['refusal_rate']:.2%}")
    print(f"Over-refusal rate: {baseline_harmless['over_refusal_rate']:.2%}")

    # Apply ablation hooks
    handles = register_ablation_hooks(
        model, direction,
        layers=args.target_layers,
        scale=args.scale,
        mode="subtract",
    )

    print(f"\n[ABLATED - English direction applied to {args.target_lang_harmful}]")
    ablated_harmful = evaluate_refusal_rate(model, tokenizer, target_harmful)
    ablated_harmless = evaluate_harmless_compliance(model, tokenizer, target_harmless)
    print(f"Harmful refusal rate: {ablated_harmful['refusal_rate']:.2%}")
    print(f"Over-refusal rate: {ablated_harmless['over_refusal_rate']:.2%}")

    remove_hooks(handles)

    results = {
        "source_layer": source_layer,
        "scale": args.scale,
        "projected": args.projected,
        "target_lang_harmful": args.target_lang_harmful,
        "target_lang_harmless": args.target_lang_harmless,
        "baseline": {
            "harmful_refusal_rate": baseline_harmful["refusal_rate"],
            "over_refusal_rate": baseline_harmless["over_refusal_rate"],
        },
        "ablated": {
            "harmful_refusal_rate": ablated_harmful["refusal_rate"],
            "over_refusal_rate": ablated_harmless["over_refusal_rate"],
        },
        "refusal_reduction": baseline_harmful["refusal_rate"] - ablated_harmful["refusal_rate"],
        "over_refusal_change": ablated_harmless["over_refusal_rate"] - baseline_harmless["over_refusal_rate"],
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()