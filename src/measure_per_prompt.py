"""
measure_per_prompt.py

Per-prompt activation measurement for true multi-direction SVD.
"""
import gc
import torch
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .model_adapter import get_layer_base, get_num_layers
from .utils import format_chat_prompts


def measure_per_prompt_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    layer_indices: Optional[List[int]] = None,
    pos: int = -1,
    batch_size: int = 1,
    max_new_tokens: int = 1,
    store_on_cpu: bool = True,
) -> Dict:
    if layer_indices is None:
        num_layers = get_num_layers(model)
        layer_indices = list(range(num_layers))

    device = next(model.parameters()).device

    harmful_formatted = format_chat_prompts(harmful_prompts, tokenizer)
    harmless_formatted = format_chat_prompts(harmless_prompts, tokenizer)

    min_len = min(len(harmful_formatted), len(harmless_formatted))
    harmful_formatted = harmful_formatted[:min_len]
    harmless_formatted = harmless_formatted[:min_len]

    def collect_activations(formatted_prompts, desc):
        activations = {layer_idx: [] for layer_idx in layer_indices}

        for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=desc):
            batch = formatted_prompts[i:i + batch_size]

            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            encoding = tokenizer(batch, padding=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            hidden_states = outputs.hidden_states[0]
            del input_ids, attention_mask, outputs

            for layer_idx in layer_indices:
                h = hidden_states[layer_idx][:, pos, :].detach()
                if store_on_cpu:
                    h = h.cpu()
                activations[layer_idx].append(h)

            del hidden_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            layer_idx: torch.cat(tensors, dim=0).float()
            for layer_idx, tensors in activations.items()
        }

    harmful_acts = collect_activations(harmful_formatted, "Collecting harmful per-prompt")
    del harmful_formatted
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    harmless_acts = collect_activations(harmless_formatted, "Collecting harmless per-prompt")
    del harmless_formatted
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {"num_layers": get_num_layers(model)}

    for layer in tqdm(layer_indices, desc="Building contrast matrices"):
        h = harmful_acts[layer]
        hl = harmless_acts[layer]
        contrast = h - hl

        results[f"harmful_per_prompt_{layer}"] = h
        results[f"harmless_per_prompt_{layer}"] = hl
        results[f"contrast_{layer}"] = contrast
        results[f"refusal_{layer}"] = contrast.mean(dim=0)
        results[f"refusal_norm_{layer}"] = float(contrast.mean(dim=0).norm().item())

    del harmful_acts, harmless_acts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results
