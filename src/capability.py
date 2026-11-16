"""
capability.py

Capability benchmarks for post-ablation evaluation:
- MMLU (knowledge)
- HellaSwag (reasoning)
- IFEval (instruction following)
- HumanEval (code generation)

Uses lm-evaluation-harness where available; falls back to simple heuristics.
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


BENCHMARKS = {
    "mmlu": {"task": "mmlu", "metric": "acc", "domain": "knowledge"},
    "hellaswag": {"task": "hellaswag", "metric": "acc_norm", "domain": "reasoning"},
    "ifeval": {"task": "ifeval", "metric": "inst_level_acc", "domain": "instruction_following"},
    "humaneval": {"task": "humaneval", "metric": "pass@1", "domain": "code"},
}


def evaluate_capability(
    model_path: str,
    benchmarks: Optional[List[str]] = None,
    n_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    batch_size: str = "auto",
) -> Dict:
    """Run capability benchmarks via lm-eval-harness.

    Returns a dict keyed by benchmark name with {metric, score, domain}.
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    try:
        import lm_eval
    except ImportError:
        return _evaluate_capability_fallback(model_path, benchmarks, n_samples, output_dir)

    results = {}
    for bench in benchmarks:
        if bench not in BENCHMARKS:
            continue
        info = BENCHMARKS[bench]
        task = info["task"]
        limit = f",limit={n_samples}" if n_samples else ""

        cmd = [
            "lm-eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype=bfloat16",
            "--tasks", task,
            "--batch_size", batch_size,
            "--output_path", output_dir or "outputs",
            "--no_cache",
        ]
        if n_samples:
            cmd.extend(["--limit", str(n_samples)])

        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            # Parse the last JSON line from stdout.
            score = None
            for line in out.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        if "results" in data and task in data["results"]:
                            score = data["results"][task].get(info["metric"])
                    except json.JSONDecodeError:
                        pass
            results[bench] = {"score": score, "metric": info["metric"], "domain": info["domain"]}
        except subprocess.TimeoutExpired:
            results[bench] = {"score": None, "metric": info["metric"], "domain": info["domain"], "error": "timeout"}
        except Exception as e:
            results[bench] = {"score": None, "metric": info["metric"], "domain": info["domain"], "error": str(e)}

    return results


def _evaluate_capability_fallback(
    model_path: str,
    benchmarks: List[str],
    n_samples: Optional[int],
    output_dir: Optional[str],
) -> Dict:
    """Fallback: try to import lm_eval directly."""
    try:
        from lm_eval import evaluator, tasks
        results = {}
        for bench in benchmarks:
            if bench not in BENCHMARKS:
                continue
            info = BENCHMARKS[bench]
            try:
                task = tasks.get_task_dict([info["task"]])
                out = evaluator.simple_evaluate(
                    model="hf",
                    model_args=f"pretrained={model_path},dtype=bfloat16",
                    tasks=[info["task"]],
                    limit=n_samples,
                )
                score = out["results"][info["task"]].get(info["metric"])
                results[bench] = {"score": score, "metric": info["metric"], "domain": info["domain"]}
            except Exception as e:
                results[bench] = {"score": None, "metric": info["metric"], "domain": info["domain"], "error": str(e)}
        return results
    except ImportError:
        return {
            bench: {"score": None, "metric": BENCHMARKS[bench]["metric"],
                    "domain": BENCHMARKS[bench]["domain"],
                    "error": "lm-eval not installed"}
            for bench in benchmarks
        }