#!/usr/bin/env python3
"""refusal-toolkit — Unified CLI for the Refusal Toolkit."""
import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def load_script(module_name: str, script_name: str):
    """Load a script module dynamically."""
    script_path = SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        prog="refusal-toolkit",
        description="Refusal Toolkit — Remove or induce refusal behavior in LLMs via representation engineering",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # measure
    p_measure = subparsers.add_parser("measure", help="Measure harmful/harmless activations")
    p_measure.add_argument("--model", "-m", required=True)
    p_measure.add_argument("--harmful", default="data/harmful.txt")
    p_measure.add_argument("--harmless", default="data/harmless.txt")
    p_measure.add_argument("--output", "-o", required=True)
    p_measure.add_argument("--per-prompt-output", help="Also save per-prompt activations")
    p_measure.add_argument("--batch-size", type=int, default=8)
    p_measure.add_argument("--max-samples", type=int, default=256)
    p_measure.add_argument("--quant", choices=["4bit", "8bit"])
    p_measure.add_argument("--layers", type=int, nargs="+")
    p_measure.add_argument("--gguf-file", help="GGUF filename to load via transformers")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze measurements and select best layer")
    p_analyze.add_argument("measurements", help="Path to measurements .pt file")
    p_analyze.add_argument("--output-json", help="Write analysis to JSON")

    # abliterate
    p_abl = subparsers.add_parser("abliterate", help="Apply abliteration (permanent/hook/add)")
    p_abl.add_argument("--model", "-m", required=True)
    p_abl.add_argument("--measurements", required=True)
    p_abl.add_argument("--source-layer", type=int, help="Layer to extract direction from (default: auto)")
    p_abl.add_argument("--target-layers", type=int, nargs="+", help="Layers to ablate (default: all)")
    p_abl.add_argument("--scale", type=float, default=1.0, help="Ablation strength")
    p_abl.add_argument("--projected", action="store_true", help="Project against harmless direction")
    p_abl.add_argument("--norm-preserve", action="store_true", help="Norm-preserving ablation")
    p_abl.add_argument("--mode", choices=["permanent", "hook", "add"], default="permanent")
    p_abl.add_argument("--output", "-o", help="Output path for saved model (permanent mode)")
    p_abl.add_argument("--quant", choices=["4bit", "8bit"])
    p_abl.add_argument("--eval", action="store_true", help="Quick eval before/after")
    p_abl.add_argument("--harmful", default="data/harmful.txt")
    p_abl.add_argument("--harmless", default="data/harmless.txt")
    p_abl.add_argument("--max-samples", type=int, default=32)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate refusal rate")
    p_eval.add_argument("--model", "-m", required=True)
    p_eval.add_argument("--harmful")
    p_eval.add_argument("--harmless")
    p_eval.add_argument("--output")
    p_eval.add_argument("--judge", choices=["regex", "logprob", "hybrid"], default="regex",
                        help="Refusal judge mode (Paper 07)")
    p_eval.add_argument("--refusal-threshold", type=float, default=0.5,
                        help="Refusal probability threshold for logprob judge")
    p_eval.add_argument("--repetition-penalty", type=float, default=1.0)

    # dsn (Paper 07: Don't Say No)
    p_dsn = subparsers.add_parser(
        "dsn",
        help="Don't Say No (arXiv:2404.16369): Token-level refusal log-prob evaluation",
    )
    p_dsn.add_argument("--model", "-m", required=True)
    p_dsn.add_argument("--harmful", default="data/harmful.txt")
    p_dsn.add_argument("--harmless", default="data/harmless.txt")
    p_dsn.add_argument("--max-samples", type=int, default=64)
    p_dsn.add_argument("--batch-size", type=int, default=8)
    p_dsn.add_argument("--max-new-tokens", type=int, default=128)
    p_dsn.add_argument("--judge", choices=["logprob", "regex", "hybrid"], default="hybrid")
    p_dsn.add_argument("--refusal-threshold", type=float, default=0.5)
    p_dsn.add_argument("--repetition-penalty", type=float, default=1.0)
    p_dsn.add_argument("--gguf-file")
    p_dsn.add_argument("--output", "-o")

    # cross-lingual
    p_cross = subparsers.add_parser("cross-lingual", help="Cross-lingual direction transfer")
    p_cross.add_argument("--model", "-m", required=True)
    p_cross.add_argument("--measurements", required=True)
    p_cross.add_argument("--target-lang-harmful", required=True)
    p_cross.add_argument("--target-lang-harmless", required=True)
    p_cross.add_argument("--source-layer", type=int)
    p_cross.add_argument("--target-layers", type=int, nargs="+")
    p_cross.add_argument("--scale", type=float, default=1.0)
    p_cross.add_argument("--projected", action="store_true")
    p_cross.add_argument("--output", "-o")
    p_cross.add_argument("--max-samples", type=int, default=64)

    # harmfulness-guard
    p_guard = subparsers.add_parser("harmfulness-guard", help="Post-ablation latent harmfulness detection")
    p_guard.add_argument("--model", "-m", required=True)
    p_guard.add_argument("--measurements", required=True)
    p_guard.add_argument("--prompt", required=True)
    p_guard.add_argument("--layer", type=int)
    p_guard.add_argument("--threshold", type=float, default=0.3)
    p_guard.add_argument("--output", "-o")

    # codeql-verify
    p_codeql = subparsers.add_parser("codeql-verify", help="CodeQL static analysis for CWE detection")
    p_codeql.add_argument("--model", "-m", required=True)
    p_codeql.add_argument("--prompts", required=True)
    p_codeql.add_argument("--language", default="python")
    p_codeql.add_argument("--codeql-path", default="codeql")
    p_codeql.add_argument("--query-pack")
    p_codeql.add_argument("--max-samples", type=int, default=20)
    p_codeql.add_argument("--max-tokens", type=int, default=512)
    p_codeql.add_argument("--output", "-o")

    # sae
    p_sae = subparsers.add_parser("sae", help="SAE-based refusal feature extraction & ablation")
    p_sae.add_argument("--per-prompt-measurements", required=True)
    p_sae.add_argument("--layer", type=int, required=True)
    p_sae.add_argument("--hidden-dim-mult", type=int, default=8)
    p_sae.add_argument("--top-k", type=int, default=64)
    p_sae.add_argument("--top-n-features", type=int, default=10)
    p_sae.add_argument("--train-epochs", type=int, default=1000)
    p_sae.add_argument("--output", "-o", required=True)
    p_sae.add_argument("--ablate", action="store_true")
    p_sae.add_argument("--model", "-m")
    p_sae.add_argument("--mode", choices=["hook", "permanent"], default="hook")
    p_sae.add_argument("--scale", type=float, default=1.0)
    p_sae.add_argument("--harmful", default="data/harmful.txt")
    p_sae.add_argument("--harmless", default="data/harmless.txt")
    p_sae.add_argument("--max-samples", type=int, default=128)
    p_sae.add_argument("--eval-output")
    p_sae.add_argument("--gguf-file")

    # heretic-search (advanced: Optuna TPE search with KL cap)
    p_search = subparsers.add_parser(
        "heretic-search",
        help="Heretic-style Optuna TPE search for ablation hyperparameters "
             "(arXiv:2611.17846)",
    )
    p_search.add_argument("--model", "-m", required=True)
    p_search.add_argument("--measurements", required=True)
    p_search.add_argument("--per-prompt-measurements", "-p",
                          help="Required for som/ara methods")
    p_search.add_argument("--output", "-o", required=True,
                          help="Output directory for the best model")
    p_search.add_argument("--method",
                          choices=["dim", "mpoa", "som", "ara", "jspace"],
                          default="mpoa")
    p_search.add_argument("--n-trials", type=int, default=100)
    p_search.add_argument("--kl-cap", type=float, default=0.1,
                          help="First-token KL divergence cap")
    p_search.add_argument("--dev-prompts", default="data/harmful.txt")
    p_search.add_argument("--judge", choices=["regex", "logprob", "hybrid"], default="regex",
                          help="Refusal judge mode (Paper 07)")
    p_search.add_argument("--refusal-threshold", type=float, default=0.5,
                          help="Refusal probability threshold for logprob judge")
    p_search.add_argument("--max-samples", type=int, default=64)
    p_search.add_argument("--seed", type=int, default=1)
    p_search.add_argument("--quant", choices=["4bit", "8bit"])

    # evaluate-capability
    p_cap = subparsers.add_parser(
        "evaluate-capability",
        help="Run capability benchmarks (MMLU, HellaSwag, IFEval, HumanEval)",
    )
    p_cap.add_argument("--model", "-m", required=True)
    p_cap.add_argument("--benchmarks", nargs="+",
                       choices=["mmlu", "hellaswag", "ifeval", "humaneval"],
                       default=["mmlu", "hellaswag"])
    p_cap.add_argument("--n-samples", type=int,
                       help="Limit samples per benchmark")
    p_cap.add_argument("--output", "-o")
    p_cap.add_argument("--batch-size", default="auto")

    args = parser.parse_args()

    # Dispatch to appropriate script
    script_map = {
        "measure": "01_measure.py",
        "analyze": "02_analyze.py",
        "abliterate": "03_abliterate.py",
        "evaluate": "04_evaluate.py",
        "cross-lingual": "05_cross_lingual.py",
        "harmfulness-guard": "06_harmfulness_guard.py",
        "codeql-verify": "07_codeql_verify.py",
        "sae": "08_sae_refusal.py",
        "dsn": "09_dsn.py",
        "heretic-search": "10_heretic_search.py",
        "evaluate-capability": "11_evaluate_capability.py",
    }

    script_file = script_map.get(args.command)
    if script_file:
        module = load_script(f"refusal_toolkit_{args.command.replace('-', '_')}", script_file)
        # Replace sys.argv with script name + remaining args
        sys.argv = [script_file] + [a for a in sys.argv[2:]]
        module.main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()