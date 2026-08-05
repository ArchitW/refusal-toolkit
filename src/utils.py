"""
utils.py

Data loading, prompt formatting, and common I/O helpers.
"""
import json
import random
from pathlib import Path
from typing import List, Optional, Union
import torch


def load_prompts(path: Union[str, Path], max_samples: Optional[int] = None) -> List[str]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            prompts = [str(item) for item in data]
        elif isinstance(data, dict) and "text" in data:
            prompts = data["text"]
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")
    elif suffix == ".jsonl":
        prompts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    prompts.append(obj)
                elif isinstance(obj, dict):
                    prompts.append(obj.get("text") or obj.get("instruction") or obj.get("prompt") or "")
        prompts = [p for p in prompts if p]
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load .parquet files")
        df = pd.read_parquet(path)
        col = "text" if "text" in df.columns else ("instruction" if "instruction" in df.columns else None)
        if col is None:
            raise ValueError(f"No 'text' or 'instruction' column found in {path}")
        prompts = df[col].dropna().astype(str).tolist()
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if max_samples is not None and len(prompts) > max_samples:
        prompts = random.sample(prompts, max_samples)

    return prompts


def format_chat_prompts(
    prompts: List[str],
    tokenizer,
    system_prompt: Optional[str] = None,
) -> List[str]:
    formatted = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            text = f"User: {prompt}\nAssistant:"
        formatted.append(text)
    return formatted


def save_measurements(measurements: dict, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(measurements, path)
    print(f"Measurements saved to {path}")


def load_measurements(path: Union[str, Path]) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
