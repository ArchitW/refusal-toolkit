#!/usr/bin/env python3
"""02_analyze.py — Analyze measurements and auto-select best layer."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_measurements
from src.directions import auto_select_best_layer, compute_cohens_d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("measurements", help="Path to measurements .pt file")
    parser.add_argument("--output-json", help="Write analysis to JSON")
    args = parser.parse_args()

    measurements = load_measurements(args.measurements)
    num_layers = measurements["num_layers"]

    results = {
        "num_layers": num_layers,
        "layers": [],
    }

    best_norm_layer = auto_select_best_layer(measurements, metric="norm")
    best_cohens_layer = auto_select_best_layer(measurements, metric="cohens_d")

    print(f"Model has {num_layers} layers")
    print(f"Best layer (refusal norm): {best_norm_layer}")
    print(f"Best layer (Cohen's d): {best_cohens_layer}")
    print()

    for layer in range(num_layers):
        norm = measurements.get(f"refusal_norm_{layer}", 0.0)
        cohens = compute_cohens_d(measurements, layer)
        harmful_norm = measurements[f"harmful_{layer}"].norm().item()
        harmless_norm = measurements[f"harmless_{layer}"].norm().item()

        is_best_norm = " ← BEST (norm)" if layer == best_norm_layer else ""
        is_best_cohens = " ← BEST (cohens)" if layer == best_cohens_layer else ""

        print(f"Layer {layer:2d}: refusal_norm={norm:.4f}, cohens_d={cohens:.4f}, "
              f"harmful_norm={harmful_norm:.4f}, harmless_norm={harmless_norm:.4f}"
              f"{is_best_norm}{is_best_cohens}")

        results["layers"].append({
            "layer": layer,
            "refusal_norm": norm,
            "cohens_d": cohens,
            "harmful_norm": harmful_norm,
            "harmless_norm": harmless_norm,
        })

    results["best_layer_norm"] = best_norm_layer
    results["best_layer_cohens_d"] = best_cohens_layer

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nAnalysis saved to {args.output_json}")


if __name__ == "__main__":
    main()