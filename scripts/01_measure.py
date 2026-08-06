#!/usr/bin/env python3
"""01_measure.py — Measure harmful/harmless activations."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.measure import measure_refusal_directions
from src.measure_per_prompt import measure_per_prompt_activations
from src.utils import load_prompts, save_measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--harmful", default="data/harmful.txt")
    parser.add_argument("--harmless", default="data/harmless.txt")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--per-prompt-output", help="Also save per-prompt activations")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--quant", choices=["4bit", "8bit"])
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--gguf-file", help="GGUF filename to load via transformers")
    args = parser.parse_args()

    harmful = load_prompts(args.harmful, args.max_samples)
    harmless = load_prompts(args.harmless, args.max_samples)
    print(f"Loaded {len(harmful)} harmful, {len(harmless)} harmless prompts")

    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    elif args.quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = {"device_map": "auto", "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    if args.gguf_file:
        load_kwargs["gguf_file"] = args.gguf_file
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = measure_refusal_directions(
        model, tokenizer, harmful, harmless,
        layer_indices=args.layers,
        batch_size=args.batch_size,
    )
    save_measurements(results, args.output)

    if args.per_prompt_output:
        print("\nCollecting per-prompt activations...")
        per_prompt_results = measure_per_prompt_activations(
            model, tokenizer, harmful, harmless,
            layer_indices=args.layers,
            batch_size=max(1, args.batch_size // 4),
        )
        save_measurements(per_prompt_results, args.per_prompt_output)

    print("Done.")


if __name__ == "__main__":
    main()
