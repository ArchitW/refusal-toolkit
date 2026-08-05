"""
measure.py

Numerically stable, batched, multi-layer activation measurement using
Welford's online algorithm in float32.
"""
import gc
import torch
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .model_adapter import get_layer_base, get_num_layers


def welford_batched_multilayer(
    formatted_prompts: List[str],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_indices: List[int],
    pos: int = -1,
    batch_size: int = 1,
    max_new_tokens: int = 1,
) -> Dict[int, torch.Tensor]:
    means = {layer_idx: None for layer_idx in layer_indices}
    counts = {layer_idx: 0 for layer_idx in layer_indices}
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=desc):
        batch_prompts = formatted_prompts[i:i + batch_size]

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        encoding = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
        )
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
            current = hidden_states[layer_idx][:, pos, :].float()
            batch_actual = current.size(0)
            total = counts[layer_idx] + batch_actual

            if means[layer_idx] is None:
                means[layer_idx] = current.mean(dim=0)
            else:
                delta = current - means[layer_idx]
                means[layer_idx] += delta.sum(dim=0) / total

            counts[layer_idx] = total
            del current

        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        layer_idx: mean.to("cpu")
        for layer_idx, mean in means.items()
    }


def measure_refusal_directions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    layer_indices: Optional[List[int]] = None,
    pos: int = -1,
    batch_size: int = 8,
    max_new_tokens: int = 1,
) -> Dict:
    if layer_indices is None:
        num_layers = get_num_layers(model)
        layer_indices = list(range(num_layers))

    from .utils import format_chat_prompts
    harmful_formatted = format_chat_prompts(harmful_prompts, tokenizer)
    harmless_formatted = format_chat_prompts(harmless_prompts, tokenizer)

    harmful_means = welford_batched_multilayer(
        harmful_formatted, "Measuring harmful activations",
        model, tokenizer, layer_indices, pos, batch_size, max_new_tokens,
    )
    del harmful_formatted
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    harmless_means = welford_batched_multilayer(
        harmless_formatted, "Measuring harmless activations",
        model, tokenizer, layer_indices, pos, batch_size, max_new_tokens,
    )
    del harmless_formatted
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {"num_layers": get_num_layers(model)}
    for layer in tqdm(layer_indices, desc="Compiling directions"):
        h_mean = harmful_means[layer]
        hl_mean = harmless_means[layer]
        r_dir = h_mean - hl_mean

        results[f"harmful_{layer}"] = h_mean
        results[f"harmless_{layer}"] = hl_mean
        results[f"refusal_{layer}"] = r_dir
        results[f"refusal_norm_{layer}"] = float(r_dir.norm().item())

    del harmful_means, harmless_means
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results
