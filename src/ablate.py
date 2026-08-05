"""
ablate.py

Two modes:
1. Inference-time activation hook (reversible)
2. Permanent weight orthogonalization (persistent)
3. Activation Addition — flip sign to INDUCE refusal
"""
import gc
import torch
from typing import Optional, List

from .model_adapter import get_layer_base, get_attention_output_proj, get_mlp_down_proj
from .directions import normalize_direction, project_against_harmless


def _as_direction_matrix(direction: torch.Tensor) -> torch.Tensor:
    """Return an (n_directions, d) float tensor of unit directions."""
    if direction.dim() == 1:
        directions = direction.unsqueeze(0)
    elif direction.dim() == 2:
        directions = direction
    else:
        raise ValueError(f"direction must be 1D or 2D, got {direction.dim()}D")
    return normalize_direction(directions)


class AblationHook:
    def __init__(self, direction: torch.Tensor, scale: float = 1.0):
        self.directions = _as_direction_matrix(direction)
        self.scale = scale

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        directions = self.directions.to(hidden.device, dtype=hidden.dtype)
        proj = torch.einsum("...d,nd->...n", hidden, directions).unsqueeze(-1) * directions
        ablated = hidden - self.scale * proj.sum(dim=-2)

        if isinstance(output, tuple):
            return (ablated,) + output[1:]
        return ablated


class ActivationAdditionHook:
    def __init__(self, direction: torch.Tensor, scale: float = 1.0):
        self.directions = _as_direction_matrix(direction)
        self.scale = scale

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        directions = self.directions.to(hidden.device, dtype=hidden.dtype)
        proj = torch.einsum("...d,nd->...n", hidden, directions).unsqueeze(-1) * directions
        modified = hidden + self.scale * proj.sum(dim=-2)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


def register_ablation_hooks(
    model,
    direction: torch.Tensor,
    layers: Optional[List[int]] = None,
    scale: float = 1.0,
    mode: str = "subtract",
) -> List:
    layer_base, _ = get_layer_base(model)
    handles = []

    if layers is None:
        layers = range(len(layer_base))

    HookClass = ActivationAdditionHook if mode == "add" else AblationHook

    for idx in layers:
        hook = HookClass(direction, scale)
        handle = layer_base[idx].register_forward_hook(hook)
        handles.append(handle)

    return handles


def remove_hooks(handles: List):
    for h in handles:
        h.remove()


def _is_quantized_weight(weight) -> bool:
    return type(weight).__module__.startswith("bitsandbytes")


def _assert_not_quantized(weight, module_name: str):
    if _is_quantized_weight(weight):
        raise RuntimeError(
            f"Cannot permanently ablate {module_name}: weights are "
            f"bitsandbytes-4bit quantized ({type(weight).__name__}). "
            "Quantization-aware ablation is not implemented yet; use "
            "--mode hook, or run on an unquantized checkpoint "
            "(e.g. Qwen/Qwen2.5-0.5B-Instruct)."
        )


def orthogonalize_matrix(
    W: torch.Tensor,
    direction: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    original_dtype = W.dtype
    device = W.device

    W_f = W.float()
    d = normalize_direction(direction).to(device).float()

    if d.dim() > 1:
        d = d.view(-1)

    if d.numel() == W_f.shape[1]:
        projection = torch.matmul(W_f, d)
        W_f -= scale * torch.outer(projection, d)
    elif d.numel() == W_f.shape[0]:
        projection = torch.matmul(W_f.T, d)
        W_f -= scale * torch.outer(d, projection)
    else:
        raise ValueError(
            f"Direction dim {d.numel()} matches neither W input ({W_f.shape[1]}) "
            f"nor W output ({W_f.shape[0]})"
        )

    return W_f.to(original_dtype)


def norm_preserving_orthogonalize(
    W: torch.Tensor,
    direction: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    original_dtype = W.dtype
    device = W.device

    W_f = W.float()
    d = normalize_direction(direction).to(device).float().view(-1)

    if d.numel() == W_f.shape[1]:
        row_norms = torch.norm(W_f, dim=1, keepdim=True)
        W_dir = torch.nn.functional.normalize(W_f, dim=1)

        proj = torch.matmul(W_dir, d)
        W_dir_new = W_dir - scale * torch.outer(proj, d)
        W_dir_new = torch.nn.functional.normalize(W_dir_new, dim=1)

        W_new = row_norms * W_dir_new
    elif d.numel() == W_f.shape[0]:
        Wt = W_f.T
        col_norms = torch.norm(Wt, dim=1, keepdim=True)
        Wt_dir = torch.nn.functional.normalize(Wt, dim=1)

        proj = torch.matmul(Wt_dir, d)
        Wt_new = Wt_dir - scale * torch.outer(proj, d)
        Wt_new = torch.nn.functional.normalize(Wt_new, dim=1)

        W_new = (col_norms * Wt_new).T
    else:
        raise ValueError(
            f"Direction dim {d.numel()} matches neither W input ({W_f.shape[1]}) "
            f"nor W output ({W_f.shape[0]})"
        )

    return W_new.to(original_dtype)


def project_bias(
    bias: torch.Tensor,
    direction: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Project a bias vector along a direction (output-space semantics).

    b' = b - scale * (b . d) d  — removes the refusal direction from the
    bias term, which weight-only projection leaves partially active."""
    original_dtype = bias.dtype
    b = bias.float().view(-1)
    d = normalize_direction(direction).to(b.device).float().view(-1)
    if d.numel() != b.numel():
        raise ValueError(
            f"Direction dim {d.numel()} does not match bias dim {b.numel()}"
        )
    return (b - scale * torch.dot(b, d) * d).to(original_dtype)


def whitened_svd_directions(
    per_prompt_measurements: dict,
    layer: int,
    top_k: int = 4,
    reg: float = 1e-2,
) -> torch.Tensor:
    """Covariance-normalized (whitened) SVD refusal directions for a layer.

    OBLITERATUS/Heretic-style extraction: whiten the per-prompt
    harmful-minus-harmless difference vectors against the pooled activation
    covariance, SVD the whitened differences, take the top-k right singular
    vectors, and un-whiten them back into activation space. Separates the
    guardrail signal from natural activation variance, which plain SVD of
    the contrast mean cannot do. Returns an (top_k, d) tensor of unit
    directions."""
    harmful = per_prompt_measurements[f"harmful_per_prompt_{layer}"].float()
    harmless = per_prompt_measurements[f"harmless_per_prompt_{layer}"].float()
    n = min(harmful.shape[0], harmless.shape[0])
    if n < 2:
        raise ValueError(f"Need >=2 per-prompt pairs per class, got {n}")

    delta = harmful[:n] - harmless[:n]                      # (n, d)
    acts = torch.cat([harmful[:n], harmless[:n]], dim=0)    # (2n, d)
    mean = acts.mean(0, keepdim=True)
    C = (acts - mean).T @ (acts - mean) / acts.shape[0]     # (d, d)
    reg = reg * torch.trace(C) / C.shape[0]
    C = C + reg * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)

    L, V = torch.linalg.eigh(C)
    Cinv_sqrt = V @ torch.diag(L.clamp(min=1e-12).pow(-0.5)) @ V.T

    D_w = delta @ Cinv_sqrt                                 # whitened diffs (n, d)
    _, _, Vt = torch.linalg.svd(D_w)
    Vk = Vt[:top_k].T                                       # (d, k) whitened dirs
    dirs = (Cinv_sqrt @ Vk).T                               # (k, d) un-whitened
    return normalize_direction(dirs)


def svd_directions(
    per_prompt_measurements: dict,
    layer: int,
    top_k: int = 4,
) -> torch.Tensor:
    """Top-k SVD directions of the centered harmful/harmless per-prompt
    difference vectors."""
    harmful = per_prompt_measurements[f"harmful_per_prompt_{layer}"].float()
    harmless = per_prompt_measurements[f"harmless_per_prompt_{layer}"].float()
    n = min(harmful.shape[0], harmless.shape[0])
    delta = harmful[:n] - harmless[:n]                        # (n, d)
    delta = delta - delta.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(delta)
    return normalize_direction(Vt[:top_k])


def leace_direction(
    per_prompt_measurements: dict,
    layer: int,
    reg: float = 1e-2,
) -> torch.Tensor:
    """LEACE-style concept-erasure direction (Belrose et al. 2023),
    translation-only variant.

    For a binary concept z in {+1,-1}, the minimal expected-L2-change unit
    direction whose removal destroys linear separability is
        u = normalize(C^{-1/2} @ delta)
    where delta = mean(harmful) - mean(harmless) and C is the pooled
    activation covariance. Unlike diff-in-means (which ignores how
    activations vary) and unlike our whitened-SVD (which un-whitens its
    directions back), LEACE projects along the whitened contrast itself.
    Returns an (1, d) tensor."""
    harmful = per_prompt_measurements[f"harmful_per_prompt_{layer}"].float()
    harmless = per_prompt_measurements[f"harmless_per_prompt_{layer}"].float()
    n = min(harmful.shape[0], harmless.shape[0])
    if n < 2:
        raise ValueError(f"Need >=2 per-prompt pairs per class, got {n}")

    delta = (harmful[:n] - harmless[:n]).mean(0)            # (d,)
    acts = torch.cat([harmful[:n], harmless[:n]], dim=0)    # (2n, d)
    mean = acts.mean(0, keepdim=True)
    C = (acts - mean).T @ (acts - mean) / acts.shape[0]     # (d, d)
    reg = reg * torch.trace(C) / C.shape[0]
    C = C + reg * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)

    L, V = torch.linalg.eigh(C)
    Cinv_sqrt = V @ torch.diag(L.clamp(min=1e-12).pow(-0.5)) @ V.T

    return normalize_direction((Cinv_sqrt @ delta).unsqueeze(0))


def build_directions(
    measurements: dict,
    per_prompt_measurements: Optional[dict],
    source_layer: int,
    method: str = "diff",
    top_k: int = 4,
    projected: bool = False,
) -> torch.Tensor:
    """Build an (n_directions, d) direction matrix for a source layer.

    method: 'diff' (single diff-in-means), 'svd' (plain contrast SVD),
    'whitened-svd' (covariance-normalized SVD), 'leace' (Belrose concept
    erasure direction)."""
    if method == "diff":
        direction = measurements[f"refusal_{source_layer}"].float()
        if projected:
            harmless = measurements[f"harmless_{source_layer}"].float()
            direction = project_against_harmless(direction, harmless)
        return _as_direction_matrix(direction)
    if method == "svd":
        if per_prompt_measurements is None:
            raise ValueError("svd requires --per-prompt-measurements")
        return svd_directions(per_prompt_measurements, source_layer, top_k)
    if method == "whitened-svd":
        if per_prompt_measurements is None:
            raise ValueError("whitened-svd requires --per-prompt-measurements")
        return whitened_svd_directions(
            per_prompt_measurements, source_layer, top_k
        )
    if method == "leace":
        if per_prompt_measurements is None:
            raise ValueError("leace requires --per-prompt-measurements")
        return leace_direction(per_prompt_measurements, source_layer)
    raise ValueError(f"unknown method {method}")


def ablate_model_weights(
    model,
    measurements: dict,
    source_layer: int,
    target_layers: Optional[List[int]] = None,
    scale: float = 1.0,
    projected: bool = False,
    norm_preserve: bool = False,
    directions: Optional[torch.Tensor] = None,
    bias: bool = False,
    scale_attn: Optional[float] = None,
    scale_mlp: Optional[float] = None,
):
    layer_base, _ = get_layer_base(model)
    num_layers = len(layer_base)

    if target_layers is None:
        target_layers = list(range(num_layers))

    if directions is None:
        refusal_dir = measurements[f"refusal_{source_layer}"].float()
        if projected:
            harmless = measurements[f"harmless_{source_layer}"].float()
            refusal_dir = project_against_harmless(refusal_dir, harmless)
        dirs = _as_direction_matrix(refusal_dir)
    else:
        dirs = _as_direction_matrix(directions.float())

    s_attn = scale_attn if scale_attn is not None else scale
    s_mlp = scale_mlp if scale_mlp is not None else scale

    for idx in target_layers:
        layer = layer_base[idx]

        for d in dirs:
            d = normalize_direction(d)

            try:
                o_proj = get_attention_output_proj(layer)
                _assert_not_quantized(o_proj.weight, "o_proj")
                if norm_preserve:
                    o_proj.weight.data = norm_preserving_orthogonalize(
                        o_proj.weight.data, d, s_attn
                    )
                else:
                    o_proj.weight.data = orthogonalize_matrix(
                        o_proj.weight.data, d, s_attn
                    )
                if bias and o_proj.bias is not None:
                    o_proj.bias.data = project_bias(o_proj.bias.data, d, s_attn)
            except ValueError:
                pass

            try:
                down_proj = get_mlp_down_proj(layer)
                _assert_not_quantized(down_proj.weight, "down_proj")
                if norm_preserve:
                    down_proj.weight.data = norm_preserving_orthogonalize(
                        down_proj.weight.data, d, s_mlp
                    )
                else:
                    down_proj.weight.data = orthogonalize_matrix(
                        down_proj.weight.data, d, s_mlp
                    )
                if bias and down_proj.bias is not None:
                    down_proj.bias.data = project_bias(down_proj.bias.data, d, s_mlp)
            except ValueError:
                pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
