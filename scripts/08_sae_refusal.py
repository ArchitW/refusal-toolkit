#!/usr/bin/env python3
"""08_sae_refusal.py — SAE-based refusal feature extraction & ablation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils import load_measurements, save_measurements, load_prompts
from src.sae_extract import extract_sae_refusal_directions
from src.evaluate import evaluate_refusal_rate, evaluate_harmless_compliance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-prompt-measurements", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--hidden-dim-mult", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-n-features", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=1000)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--ablate", action="store_true",
                        help="Run ablation demo after extraction")
    parser.add_argument("--model", "-m", help="Model to ablate (required with --ablate)")
    parser.add_argument("--mode", choices=["hook", "permanent"], default="hook")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--harmful", default="data/harmful.txt")
    parser.add_argument("--harmless", default="data/harmless.txt")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--eval-output", help="Write ablation evaluation JSON")
    parser.add_argument("--gguf-file", help="GGUF filename to load via transformers")
    args = parser.parse_args()

    if args.ablate and not args.model:
        parser.error("--model is required with --ablate")

    measurements = load_measurements(args.per_prompt_measurements)

    result = extract_sae_refusal_directions(
        measurements,
        layer_idx=args.layer,
        hidden_dim_multiplier=args.hidden_dim_mult,
        top_k=args.top_k,
        top_n_features=args.top_n_features,
        train_epochs=args.train_epochs,
    )

    print(f"\nTop-{args.top_n_features} refusal features:")
    feature_ids = []
    for info in result["feature_info"]:
        print(f"  Rank {info['rank']}: Feature {info['feature_idx']} "
              f"(score={info['refusal_score']:.4f}, "
              f"harmful={info['harmful_activation']:.4f}, "
              f"harmless={info['harmless_activation']:.4f})")
        feature_ids.append(info["feature_idx"])

    save_dict = {
        "feature_directions": result["feature_directions"],
        "feature_info": result["feature_info"],
        "layer": result["layer"],
        "num_features": result["num_features"],
        "top_k": result["top_k"],
        "sae_state_dict": result["sae"].state_dict(),
    }

    torch.save(save_dict, args.output)
    print(f"\nSaved to {args.output}")

    if not args.ablate:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.sae_extract import SparseAutoencoder, register_sae_feature_ablation_hooks, ablate_sae_features
    from src.ablate import remove_hooks

    print(f"\n=== SAE FEATURE ABLATION ({args.mode}) ===")
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
    }
    if args.gguf_file:
        load_kwargs["gguf_file"] = args.gguf_file
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    harmful_prompts = load_prompts(args.harmful, args.max_samples)
    harmless_prompts = load_prompts(args.harmless, args.max_samples)

    sae = SparseAutoencoder(result["sae"].input_dim, result["sae"].hidden_dim, result["sae"].top_k)
    sae.load_state_dict(result["sae"].state_dict())
    sae.to(next(model.parameters()).device)
    sae.eval()

    print("\n[BASELINE]")
    baseline_harmful = evaluate_refusal_rate(model, tokenizer, harmful_prompts)
    baseline_harmless = evaluate_harmless_compliance(model, tokenizer, harmless_prompts)
    print(f"Harmful refusal rate: {baseline_harmful['refusal_rate']:.2%}")
    print(f"Over-refusal rate: {baseline_harmless['over_refusal_rate']:.2%}")

    if args.mode == "hook":
        handle = register_sae_feature_ablation_hooks(
            model, sae, feature_ids, args.layer, args.scale
        )
        ablate_label = f"top-{len(feature_ids)} features (hooks)"
    else:
        ablate_sae_features(model, sae, feature_ids, args.layer, args.scale)
        ablate_label = f"top-{len(feature_ids)} features (permanent weights)"

    print(f"\n[ABLATED] {ablate_label} at layer {args.layer}, scale={args.scale}")
    ablated_harmful = evaluate_refusal_rate(model, tokenizer, harmful_prompts)
    ablated_harmless = evaluate_harmless_compliance(model, tokenizer, harmless_prompts)
    print(f"Harmful refusal rate: {ablated_harmful['refusal_rate']:.2%}")
    print(f"Over-refusal rate: {ablated_harmless['over_refusal_rate']:.2%}")

    if args.mode == "hook":
        remove_hooks([handle])

    eval_results = {
        "layer": args.layer,
        "mode": args.mode,
        "scale": args.scale,
        "features_ablated": feature_ids,
        "baseline": {
            "harmful_refusal_rate": baseline_harmful["refusal_rate"],
            "over_refusal_rate": baseline_harmless["over_refusal_rate"],
        },
        "ablated": {
            "harmful_refusal_rate": ablated_harmful["refusal_rate"],
            "over_refusal_rate": ablated_harmless["over_refusal_rate"],
        },
        "refusal_reduction": baseline_harmful["refusal_rate"] - ablated_harmful["refusal_rate"],
    }

    if args.eval_output:
        with open(args.eval_output, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nEvaluation saved to {args.eval_output}")


if __name__ == "__main__":
    main()
