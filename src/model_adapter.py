"""
model_adapter.py

Auto-detects model internal structure to support Llama, Qwen, Gemma, Mistral,
StableLM, and other HF architectures without hardcoding attribute paths.
"""
import torch
from typing import Tuple, List
from transformers import PreTrainedModel


def get_layer_base(model: PreTrainedModel):
    """
    Navigate to the module that contains the transformer layers.
    Handles nested structures (e.g. multimodal models with language_model).
    """
    candidates = [
        "model",
        "transformer",
        "language_model",
        "model.language_model",
        "transformer.language_model",
    ]
    for path in candidates:
        mod = model
        for part in path.split("."):
            if hasattr(mod, part):
                mod = getattr(mod, part)
            else:
                mod = None
                break
        if mod is not None:
            layer_candidates = ["layers", "h", "blocks", "encoder"]
            for lc in layer_candidates:
                if hasattr(mod, lc):
                    return getattr(mod, lc), path + "." + lc
    raise ValueError(f"Could not auto-detect layer structure for {type(model).__name__}")


def get_num_layers(model: PreTrainedModel) -> int:
    layers, _ = get_layer_base(model)
    return len(layers)


def get_hidden_size(model: PreTrainedModel) -> int:
    config = model.config
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "d_model"):
        return config.d_model
    if hasattr(config, "n_embd"):
        return config.n_embd
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise ValueError("Could not determine hidden_size from config")


def get_attention_output_proj(layer) -> torch.nn.Linear:
    candidates = [
        "self_attn.o_proj", "attn.o_proj", "attention.o_proj",
        "self_attention.o_proj", "linear_attn.out_proj",
        "attn.out_proj", "attention.out_proj",
        "self_attn.out_proj", "conv.out_proj",
    ]
    for cand in candidates:
        mod = layer
        for part in cand.split("."):
            if hasattr(mod, part):
                mod = getattr(mod, part)
            else:
                mod = None
                break
        if mod is not None and isinstance(mod, torch.nn.Linear):
            return mod
    raise ValueError(f"Could not find attention output projection in layer {type(layer).__name__}")


def get_mlp_down_proj(layer) -> torch.nn.Linear:
    candidates = [
        "mlp.down_proj", "feed_forward.down_proj", "ffn.down_proj",
        "mlp.dense_4h_to_h", "feed_forward.dense_4h_to_h",
        "feed_forward.w2", "mlp.w2",
    ]
    for cand in candidates:
        mod = layer
        for part in cand.split("."):
            if hasattr(mod, part):
                mod = getattr(mod, part)
            else:
                mod = None
                break
        if mod is not None and isinstance(mod, torch.nn.Linear):
            return mod
    raise ValueError(f"Could not find MLP down projection in layer {type(layer).__name__}")


def has_tied_weights(model: PreTrainedModel) -> bool:
    config = model.config
    return getattr(config, "tie_word_embeddings", False)


def get_model_family(model: PreTrainedModel) -> str:
    name = type(model).__name__.lower()
    if "llama" in name:
        return "llama"
    if "qwen" in name:
        return "qwen"
    if "gemma" in name:
        return "gemma"
    if "mistral" in name:
        return "mistral"
    if "stablelm" in name:
        return "stablelm"
    if "phi" in name:
        return "phi"
    return "unknown"
