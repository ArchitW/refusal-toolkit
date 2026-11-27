# Created: 27-11-26
# agent: kilo/baidu/ernie-5.5-flash-preview:free
"""Static command specifications for the TUI, mirroring the argparse CLI.

Each spec describes one subcommand: its help text, the script file it maps to,
and an ordered list of fields. The TUI builds forms from these specs and runs
the resolved command as a subprocess so behavior matches the CLI exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Field:
    name: str
    label: str
    type: str = "str"  # str | int | float | bool | path | choice | multi
    required: bool = False
    default: object = None
    choices: list[str] = field(default_factory=list)
    help: str = ""
    flag: str = ""  # CLI flag without leading dashes, e.g. "model"

    def to_arg(self, value: object) -> list[str]:
        """Convert a form value into CLI argument tokens."""
        if self.type == "bool":
            return [f"--{self.flag}"] if value else []
        if value is None or value == "":
            return []
        if self.type == "multi":
            items = list(value) if isinstance(value, (list, tuple)) else [value]
            return [f"--{self.flag}"] + [str(v) for v in items]
        return [f"--{self.flag}", str(value)]


@dataclass
class CommandSpec:
    command: str
    script: str
    help: str
    fields: list[Field]


def _f(name, label, **kw) -> Field:
    flag = kw.pop("flag", name.replace("_", "-"))
    return Field(name=name, label=label, flag=flag, **kw)


COMMANDS: list[CommandSpec] = [
    CommandSpec(
        command="measure",
        script="01_measure.py",
        help="Measure harmful/harmless activations",
        fields=[
            _f("model", "Model", required=True, help="HF model id or local path"),
            _f("harmful", "Harmful prompts", default="data/harmful.txt", type="path"),
            _f("harmless", "Harmless prompts", default="data/harmless.txt", type="path"),
            _f("output", "Output", required=True, type="path", flag="output"),
            _f("per_prompt_output", "Per-prompt output", type="path"),
            _f("batch_size", "Batch size", type="int", default=8),
            _f("max_samples", "Max samples", type="int", default=256),
            _f("quant", "Quantization", type="choice", choices=["4bit", "8bit"]),
            _f("layers", "Layers", type="multi", help="Space-separated layer indices"),
            _f("gguf_file", "GGUF file", type="path"),
        ],
    ),
    CommandSpec(
        command="analyze",
        script="02_analyze.py",
        help="Analyze measurements and select best layer",
        fields=[
            _f("measurements", "Measurements .pt", required=True, type="path"),
            _f("output_json", "Output JSON", type="path"),
        ],
    ),
    CommandSpec(
        command="abliterate",
        script="03_abliterate.py",
        help="Apply abliteration (permanent/hook/add)",
        fields=[
            _f("model", "Model", required=True),
            _f("measurements", "Measurements", required=True, type="path"),
            _f("source_layer", "Source layer", type="int"),
            _f("target_layers", "Target layers", type="multi"),
            _f("scale", "Scale", type="float", default=1.0),
            _f("projected", "Projected", type="bool"),
            _f("norm_preserve", "Norm preserve", type="bool"),
            _f("mode", "Mode", type="choice", choices=["permanent", "hook", "add"], default="permanent"),
            _f("output", "Output", type="path", flag="output"),
            _f("quant", "Quantization", type="choice", choices=["4bit", "8bit"]),
            _f("eval", "Quick eval", type="bool"),
            _f("harmful", "Harmful prompts", default="data/harmful.txt", type="path"),
            _f("harmless", "Harmless prompts", default="data/harmless.txt", type="path"),
            _f("max_samples", "Max samples", type="int", default=32),
        ],
    ),
    CommandSpec(
        command="evaluate",
        script="04_evaluate.py",
        help="Evaluate refusal rate",
        fields=[
            _f("model", "Model", required=True),
            _f("harmful", "Harmful prompts", type="path"),
            _f("harmless", "Harmless prompts", type="path"),
            _f("output", "Output", type="path", flag="output"),
            _f("judge", "Judge", type="choice", choices=["regex", "logprob", "hybrid"], default="regex"),
            _f("refusal_threshold", "Refusal threshold", type="float", default=0.5),
            _f("repetition_penalty", "Repetition penalty", type="float", default=1.0),
        ],
    ),
    CommandSpec(
        command="cross-lingual",
        script="05_cross_lingual.py",
        help="Cross-lingual direction transfer",
        fields=[
            _f("model", "Model", required=True),
            _f("measurements", "Measurements", required=True, type="path"),
            _f("target_lang_harmful", "Target-lang harmful", required=True, type="path"),
            _f("target_lang_harmless", "Target-lang harmless", required=True, type="path"),
            _f("source_layer", "Source layer", type="int"),
            _f("target_layers", "Target layers", type="multi"),
            _f("scale", "Scale", type="float", default=1.0),
            _f("projected", "Projected", type="bool"),
            _f("output", "Output", type="path", flag="output"),
            _f("max_samples", "Max samples", type="int", default=64),
        ],
    ),
    CommandSpec(
        command="harmfulness-guard",
        script="06_harmfulness_guard.py",
        help="Post-ablation latent harmfulness detection",
        fields=[
            _f("model", "Model", required=True),
            _f("measurements", "Measurements", required=True, type="path"),
            _f("prompt", "Prompt", required=True),
            _f("layer", "Layer", type="int"),
            _f("threshold", "Threshold", type="float", default=0.3),
            _f("output", "Output", type="path", flag="output"),
        ],
    ),
    CommandSpec(
        command="codeql-verify",
        script="07_codeql_verify.py",
        help="CodeQL static analysis for CWE detection",
        fields=[
            _f("model", "Model", required=True),
            _f("prompts", "Prompts", required=True, type="path"),
            _f("language", "Language", default="python"),
            _f("codeql_path", "CodeQL path", default="codeql"),
            _f("query_pack", "Query pack"),
            _f("max_samples", "Max samples", type="int", default=20),
            _f("max_tokens", "Max tokens", type="int", default=512),
            _f("output", "Output", type="path", flag="output"),
        ],
    ),
    CommandSpec(
        command="dsn",
        script="09_dsn.py",
        help="Don't Say No: token-level refusal log-prob evaluation",
        fields=[
            _f("model", "Model", required=True),
            _f("harmful", "Harmful prompts", default="data/harmful.txt", type="path"),
            _f("harmless", "Harmless prompts", default="data/harmless.txt", type="path"),
            _f("max_samples", "Max samples", type="int", default=64),
            _f("batch_size", "Batch size", type="int", default=8),
            _f("max_new_tokens", "Max new tokens", type="int", default=128),
            _f("judge", "Judge", type="choice", choices=["logprob", "regex", "hybrid"], default="hybrid"),
            _f("refusal_threshold", "Refusal threshold", type="float", default=0.5),
            _f("repetition_penalty", "Repetition penalty", type="float", default=1.0),
            _f("gguf_file", "GGUF file", type="path"),
            _f("output", "Output", type="path", flag="output"),
        ],
    ),
    CommandSpec(
        command="sae",
        script="08_sae_refusal.py",
        help="SAE-based refusal feature extraction & ablation",
        fields=[
            _f("per_prompt_measurements", "Per-prompt measurements", required=True, type="path"),
            _f("layer", "Layer", required=True, type="int"),
            _f("hidden_dim_mult", "Hidden dim mult", type="int", default=8),
            _f("top_k", "Top-k", type="int", default=64),
            _f("top_n_features", "Top-n features", type="int", default=10),
            _f("train_epochs", "Train epochs", type="int", default=1000),
            _f("output", "Output", required=True, type="path", flag="output"),
            _f("ablate", "Ablate", type="bool"),
            _f("model", "Model"),
            _f("mode", "Mode", type="choice", choices=["hook", "permanent"], default="hook"),
            _f("scale", "Scale", type="float", default=1.0),
            _f("harmful", "Harmful prompts", default="data/harmful.txt", type="path"),
            _f("harmless", "Harmless prompts", default="data/harmless.txt", type="path"),
            _f("max_samples", "Max samples", type="int", default=128),
            _f("eval_output", "Eval output", type="path"),
            _f("gguf_file", "GGUF file", type="path"),
        ],
    ),
    CommandSpec(
        command="heretic-search",
        script="10_heretic_search.py",
        help="Heretic-style Optuna TPE search for ablation hyperparameters",
        fields=[
            _f("model", "Model", required=True),
            _f("measurements", "Measurements", required=True, type="path"),
            _f("per_prompt_measurements", "Per-prompt measurements", type="path", flag="per-prompt-measurements"),
            _f("output", "Output dir", required=True, type="path", flag="output"),
            _f("method", "Method", type="choice", choices=["dim", "mpoa", "som", "ara", "jspace"], default="mpoa"),
            _f("n_trials", "N trials", type="int", default=100),
            _f("kl_cap", "KL cap", type="float", default=0.1),
            _f("dev_prompts", "Dev prompts", default="data/harmful.txt", type="path"),
            _f("judge", "Judge", type="choice", choices=["regex", "logprob", "hybrid"], default="regex"),
            _f("refusal_threshold", "Refusal threshold", type="float", default=0.5),
            _f("max_samples", "Max samples", type="int", default=64),
            _f("seed", "Seed", type="int", default=1),
            _f("quant", "Quantization", type="choice", choices=["4bit", "8bit"]),
        ],
    ),
    CommandSpec(
        command="evaluate-capability",
        script="11_evaluate_capability.py",
        help="Run capability benchmarks (MMLU, HellaSwag, IFEval, HumanEval)",
        fields=[
            _f("model", "Model", required=True),
            _f("benchmarks", "Benchmarks", type="multi",
               choices=["mmlu", "hellaswag", "ifeval", "humaneval"],
               default=["mmlu", "hellaswag"]),
            _f("n_samples", "N samples", type="int"),
            _f("output", "Output", type="path", flag="output"),
            _f("batch_size", "Batch size", default="auto"),
        ],
    ),
]

COMMAND_BY_NAME: dict[str, CommandSpec] = {c.command: c for c in COMMANDS}