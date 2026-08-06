#!/usr/bin/env python3
"""03_abliterate.py — Apply abliteration (permanent/hook/add)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils import load_measurements
from src.directions import (
    extract_single_direction,
    auto_select_best_layer,
    project_against_harmless,
    normalize_direction,
)
from src.ablate import (
    ablate_model_weights,
    register_ablation_hooks,
    remove_hooks,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--source-layer", type=int, help="Layer to extract direction from (default: auto)")
    parser.add_argument("--target-layers", type=int, nargs="+", help="Layers to ablate (default: all)")
    parser.add_argument("--scale", type=float, default=1.0, help="Ablation strength")
    parser.add_argument("--projected", action="store_true", help="Project against harmless direction")
    parser.add_argument("--norm-preserve", action="store_true", help="Norm-preserving ablation")
    parser.add_argument("--mode", choices=["permanent", "hook", "add"], default="permanent",
                        help="permanent=weight edit, hook=inference hooks, add=induce refusal")
    parser.add_argument("--output", "-o", help="Output path for saved model (permanent mode)")
    parser.add_argument("--quant", choices=["4bit", "8bit"], help="Quantization for loading")
    parser.add_argument("--eval", action="store_true", help="Quick eval before/after")
    parser.add_argument("--harmful", default="data/harmful.txt")
    parser.add_argument("--harmless", default="data/harmless.txt")
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()

    measurements = load_measurements(args.measurements)

    source_layer = args.source_layer
    if source_layer is None:
        source_layer = auto_select_best_layer(measurements, metric="norm")
        print(f"Auto-selected source layer: {source_layer}")

    # Extract direction
    direction = extract_single_direction(measurements, source_layer)
    if args.projected:
        harmless = measurements[f"harmless_{source_layer}"].float()
        direction = project_against_harmless(direction, harmless)
        print("Projected direction against harmless mean")
    direction = normalize_direction(direction)

    # Load model
    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif args.quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = {"device_map": "auto", "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if quant_config:
        load_kwargs["quantization_config"] = quant_config

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.mode == "permanent":
        if args.output is None:
            parser.error("--output is required for permanent mode")
        
        print(f"Applying permanent ablation (scale={args.scale}, projected={args.projected}, norm_preserve={args.norm_preserve})")
        ablate_model_weights(
            model, measurements,
            source_layer=source_layer,
            target_layers=args.target_layers,
            scale=args.scale,
            projected=args.projected,
            norm_preserve=args.norm_preserve,
            directions=direction,
        )

        print(f"Saving model to {args.output}")
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print("Done.")

    elif args.mode in ("hook", "add"):
        from src.utils import load_prompts
        from src.evaluate import evaluate_refusal_rate, evaluate_harmless_compliance

        harmful_prompts = load_prompts(args.harmful, args.max_samples) if args.eval else None
        harmless_prompts = load_prompts(args.harmless, args.max_samples)

        if args.mode == "hook" and args.eval:
            print("\n[BASELINE]")
            assert harmful_prompts is not None
            baseline_harmful = evaluate_refusal_rate(model, tokenizer, harmful_prompts)
            baseline_harmless = evaluate_harmless_compliance(model, tokenizer, harmless_prompts)
            print(f"Harmful refusal rate: {baseline_harmful['refusal_rate']:.2%}")
            print(f"Over-refusal rate: {baseline_harmless['over_refusal_rate']:.2%}")

        if args.mode == "add":
            print("\n[BASELINE - Harmless]")
            baseline_harmless = evaluate_harmless_compliance(model, tokenizer, harmless_prompts)
            print(f"Over-refusal rate: {baseline_harmless['over_refusal_rate']:.2%}")

        mode_str = "add" if args.mode == "add" else "subtract"
        print(f"Registering activation {mode_str.upper()} hooks (scale={args.scale}) on layers {args.target_layers or 'all'}")
        handles = register_ablation_hooks(
            model, direction,
            layers=args.target_layers,
            scale=args.scale,
            mode=mode_str,
        )

        if args.mode == "hook" and args.eval:
            print("\n[WITH HOOKS]")
            assert harmful_prompts is not None
            ablated_harmful = evaluate_refusal_rate(model, tokenizer, harmful_prompts)
            ablated_harmless = evaluate_harmless_compliance(model, tokenizer, harmless_prompts)
            print(f"Harmful refusal rate: {ablated_harmful['refusal_rate']:.2%}")
            print(f"Over-refusal rate: {ablated_harmless['over_refusal_rate']:.2%}")
        elif args.mode == "add":
            print("\n[WITH ADDITION]")
            added_harmless = evaluate_harmless_compliance(model, tokenizer, harmless_prompts)
            print(f"Induced refusal rate: {added_harmless['over_refusal_rate']:.2%}")

        remove_hooks(handles)
        print("Hooks removed. Model restored to original behavior.")


if __name__ == "__main__":
    main()