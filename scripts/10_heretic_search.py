#!/usr/bin/env python3
"""10_heretic_search.py — Heretic-style Optuna TPE search for ablation hyperparameters.

Runs a 100-trial Optuna TPE study that minimises dev-set keyword refusals
subject to a first-token KL divergence cap (default ≤0.1). Each trial:
  1. Samples (layer, strength, direction_scale) from the search space.
  2. Applies the requested method (dim/mpoa/som/ara/jspace) via LoRA adapters.
  3. Evaluates dev-set refusal rate (keyword judge).
  4. Computes first-token KL divergence vs. the original model.
  5. Reports the objective; trials exceeding the KL cap are penalised.

The best trial (fewest dev refusals with KL ≤ cap) is exported as a
standalone model.

Based on the methodology in:
  Is Abliteration Getting Harder? Not Uniformly (arXiv:2611.17846)
"""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_measurements, load_prompts
from src.directions import auto_select_best_layer, normalize_direction
from src.ablate import (
    apply_method,
    remove_hooks,
    register_ablation_hooks,
)
from src.directions import extract_single_direction
from src.evaluate import evaluate_refusal_rate, is_refusal


def compute_first_token_kl(
    model,
    tokenizer,
    prompts: list,
    base_logits: torch.Tensor,
    batch_size: int = 4,
) -> float:
    """Compute mean first-token KL divergence between ablated and base model
    on a list of prompts."""
    from src.utils import format_chat_prompts
    formatted = format_chat_prompts(prompts[:16], tokenizer)  # subsample
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    kls = []
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=False)
        logits = out.logits[:, 0, :]  # first token, (b, vocab)
        # KL(base || ablated)
        log_p = F.log_softmax(base_logits[:logits.shape[0]], dim=-1)
        log_q = F.log_softmax(logits, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
        kls.append(kl.item())
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sum(kls) / len(kls)


def get_base_logits(model, tokenizer, prompts: list, batch_size: int = 4) -> torch.Tensor:
    """Get first-token logits from the base model for KL reference."""
    from src.utils import format_chat_prompts
    formatted = format_chat_prompts(prompts[:16], tokenizer)
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    all_logits = []
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[:, 0, :]
        all_logits.append(logits.cpu())
        del inputs, out
    return torch.cat(all_logits, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--per-prompt-measurements", "-p",
                        help="Required for som/ara methods")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for the best model")
    parser.add_argument("--method", choices=["dim", "mpoa", "som", "ara", "jspace"],
                        default="mpoa", help="Ablation method")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Optuna TPE budget")
    parser.add_argument("--kl-cap", type=float, default=0.1,
                        help="First-token KL divergence cap")
    parser.add_argument("--dev-prompts", default="data/harmful.txt",
                        help="Dev-set harmful prompts for keyword refusal")
    parser.add_argument("--judge", choices=["regex", "logprob", "hybrid"], default="regex",
                        help="Refusal judge mode (Paper 07: logprob/hybrid)")
    parser.add_argument("--refusal-threshold", type=float, default=0.5,
                        help="Refusal probability threshold for logprob judge")
    parser.add_argument("--max-samples", type=int, default=64,
                        help="Max dev prompts")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility")
    parser.add_argument("--quant", choices=["4bit", "8bit"],
                        help="Quantization for loading")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("optuna is required for Heretic-style search. Install with: pip install optuna")
        sys.exit(1)

    optuna.logging.setLevel(optuna.logging.WARNING)

    print(f"Loading model: {args.model}")
    quant_config = None
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif args.quant == "8bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = {"device_map": "auto", "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if quant_config:
        load_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    measurements = load_measurements(args.measurements)
    per_prompt = None
    if args.per_prompt_measurements:
        per_prompt = load_measurements(args.per_prompt_measurements)

    dev_prompts = load_prompts(args.dev_prompts, args.max_samples)
    print(f"Dev set: {len(dev_prompts)} prompts")

    # Precompute base logits for KL reference.
    base_logits = get_base_logits(model, tokenizer, dev_prompts)
    num_layers = measurements["num_layers"]
    best_layer = auto_select_best_layer(measurements, metric="norm")
    print(f"Auto-selected best layer: {best_layer}")

    def objective(trial):
        # Sample hyperparameters.
        layer = trial.suggest_int("layer", 0, num_layers - 1)
        strength = trial.suggest_float("strength", 0.3, 2.0, step=0.1)
        # For methods that need it.
        kwargs = {}
        if args.method == "som":
            kwargs["k"] = trial.suggest_int("som_k", 2, 6)
            kwargs["rank"] = kwargs["k"] + 2
        elif args.method == "ara":
            kwargs["rank"] = trial.suggest_int("ara_rank", 16, 64, step=16)
            kwargs["steps"] = trial.suggest_int("ara_steps", 50, 200, step=50)
            kwargs["lr"] = trial.suggest_float("ara_lr", 0.01, 0.5, step=0.01)

        # Reload model from a clean state each trial (in production you'd
        # deep-copy or use hooks; for search we reload).
        trial_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

        try:
            apply_method(
                args.method, trial_model,
                measurements, per_prompt,
                source_layer=layer,
                scale=strength,
                merge=True,
                **kwargs,
            )
        except Exception as e:
            print(f"  Trial {trial.number}: method failed ({e})")
            return 1.0  # worst objective

        # Evaluate dev-set refusal rate.
        refusal_result = evaluate_refusal_rate(
            trial_model, tokenizer, dev_prompts, max_new_tokens=128,
            judge=args.judge, refusal_threshold=args.refusal_threshold,
        )
        refusal_rate = refusal_result["refusal_rate"]

        # Compute first-token KL.
        kl = compute_first_token_kl(trial_model, tokenizer, dev_prompts, base_logits)

        # Penalise if KL exceeds cap.
        penalty = max(0.0, kl - args.kl_cap) * 10.0
        objective = refusal_rate + penalty

        print(f"  Trial {trial.number}: layer={layer}, strength={strength:.1f}, "
              f"refusal={refusal_rate:.2%} ({args.judge}), KL={kl:.4f}, obj={objective:.4f}")

        # Clean up.
        del trial_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trial.set_user_attr("refusal_rate", refusal_rate)
        trial.set_user_attr("kl_divergence", kl)
        trial.set_user_attr("layer", layer)
        trial.set_user_attr("strength", strength)
        return objective

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest trial: {best.number}")
    print(f"  Params: {best.params}")
    print(f"  Refusal rate: {best.user_attrs.get('refusal_rate', 'N/A')}")
    print(f"  KL divergence: {best.user_attrs.get('kl_divergence', 'N/A')}")

    # Export best model.
    print(f"\nExporting best model to {args.output}...")
    best_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    kwargs = {}
    if args.method == "som":
        k = best.params.get("som_k", 4)
        kwargs["k"] = k
        kwargs["rank"] = k + 2
    elif args.method == "ara":
        for k in ("ara_rank", "ara_steps", "ara_lr"):
            if k in best.params:
                kwargs[k.replace("ara_", "")] = best.params[k]

    apply_method(
        args.method, best_model,
        measurements, per_prompt,
        source_layer=best.params["layer"],
        scale=best.params["strength"],
        merge=True,
        **kwargs,
    )

    best_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    metadata = {
        "method": args.method,
        "best_params": best.params,
        "refusal_rate": best.user_attrs.get("refusal_rate"),
        "kl_divergence": best.user_attrs.get("kl_divergence"),
        "n_trials": args.n_trials,
        "kl_cap": args.kl_cap,
        "seed": args.seed,
    }
    with open(Path(args.output) / "heretic_search_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done. Metadata saved to {Path(args.output) / 'heretic_search_metadata.json'}")


if __name__ == "__main__":
    main()