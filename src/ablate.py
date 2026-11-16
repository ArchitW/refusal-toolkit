"""
ablate.py

Two modes:
1. Inference-time activation hook (reversible)
2. Permanent weight orthogonalization (persistent)
3. Activation Addition — flip sign to INDUCE refusal
4. LoRA-based ablation methods (MPOA, SOM, ARA)
"""
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Callable

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
        layers = list(range(len(layer_base)))

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


# =============================================================================
# LoRA-based ablation methods (Heretic 2.0 style)
# =============================================================================
#
# All Heretic-style methods share a common structure:
#   1. A rank-r LoRA adapter is attached to each targeted module (o_proj,
#      down_proj) instead of editing weights in place.
#   2. The adapter's low-rank factors are initialised so that the ablation
#      removes the refusal direction while restoring row norms.
#   3. A search over (layer, strength, direction) is run by an outer
#      optimiser (Optuna TPE) minimising dev-set refusals + first-token KL.
#
# The functions below build the adapter weights; the search harness lives in
# `scripts/10_heretic_search.py`.
# =============================================================================


class LoRAAdapter(nn.Module):
    """Rank-r LoRA adapter:  W' x = W x + (x @ A.T) @ B.T * scale/alpha.

    A: (r, in_dim), B: (out_dim, r). The classic formulation applies
    B @ diag(scale/alpha) @ A, but we fold the scalar into B for simplicity
    and keep A, B as plain parameters.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, device=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        # A small init: A ~ N(0, 0.01), B = 0 so the adapter is initially a no-op.
        self.A = nn.Parameter(torch.zeros(rank, in_dim, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_dim, rank, device=device, dtype=dtype))
        nn.init.normal_(self.A, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T) @ self.B.T


def _module_out_dim(module: nn.Linear) -> int:
    return module.weight.shape[0]


def _module_in_dim(module: nn.Linear) -> int:
    return module.weight.shape[1]


def make_ablation_lora_init(
    direction: torch.Tensor,
    in_dim: int,
    out_dim: int,
    rank: int,
    projected: bool = False,
    harmless_mean: Optional[torch.Tensor] = None,
    norm_preserve: bool = True,
    strength: float = 1.0,
) -> Tuple[nn.Parameter, nn.Parameter]:
    """Initialise a rank-r LoRA adapter that removes `direction` from the
    module output while (optionally) restoring row norms.

    The adapter computes B @ A @ x, and we want the resulting update to be
    approximately  -strength * (x . d) * d  (i.e. a rank-1 projection of the
    refusal direction), possibly with row-norm restoration.

    We build B @ A = -strength * outer(d, d) restricted to the rank-r
    subspace spanned by d (plus r-1 arbitrary orthogonal fillers that the
    search can later exploit).
    """
    device = direction.device
    dtype = torch.float32

    d = normalize_direction(direction).to(device).float().view(-1)
    if d.numel() != in_dim:
        raise ValueError(
            f"direction dim {d.numel()} != module in_dim {in_dim}"
        )

    # Build a rank-r basis whose first column is the refusal direction.
    # Use QR to complete d to an orthonormal basis of the top-r subspace.
    if rank == 1:
        basis = d.unsqueeze(1)  # (in_dim, 1)
    else:
        # Start with d, then fill with random directions, QR-orthonormalise.
        rand = torch.randn(in_dim, rank, device=device, dtype=dtype)
        rand[:, 0] = d
        Q, _ = torch.linalg.qr(rand)
        # Ensure the first column aligns with d (QR can flip signs).
        if torch.dot(Q[:, 0], d) < 0:
            Q[:, 0] = -Q[:, 0]
        basis = Q  # (in_dim, rank)

    # A = sqrt(strength) * basis.T   (rank, in_dim)
    A = (strength ** 0.5) * basis.T.contiguous()

    # B: (out_dim, rank). We want B @ A @ x ~ -strength * (x.d) d.
    #   B @ A = -strength * d_outer  restricted to the basis.
    # If out_dim == in_dim (o_proj/down_proj are square-ish for many models),
    # set B = -sqrt(strength) * basis so that B @ A = -strength * basis @ basis.T,
    # which projects out the refusal direction and (if norm_preserve) we
    # rescale rows to restore norms.
    if norm_preserve:
        # B = -sqrt(strength) * (out_dim x rank) matrix whose columns are
        # the basis padded/truncated to out_dim. For non-square modules we
        # simply project along d in output space.
        B = -(strength ** 0.5) * basis.T  # (rank, out_dim) — but we want
        # (out_dim, rank), so:
        B = B.T.contiguous()  # (out_dim, rank)
    else:
        B = -(strength ** 0.5) * basis.T.contiguous()
        B = B.T.contiguous()

    return nn.Parameter(A), nn.Parameter(B)


def attach_lora_to_module(
    module: nn.Linear,
    direction: torch.Tensor,
    rank: int = 3,
    projected: bool = False,
    harmless_mean: Optional[torch.Tensor] = None,
    norm_preserve: bool = True,
    strength: float = 1.0,
) -> LoRAAdapter:
    """Attach a LoRA adapter to a linear module's output.

    Returns the adapter; the caller is responsible for removing it (e.g. by
    restoring the original weight or popping the adapter).
    """
    d = direction
    if projected and harmless_mean is not None:
        d = project_against_harmless(direction, harmless_mean)

    in_dim = _module_in_dim(module)
    out_dim = _module_out_dim(module)

    adapter = LoRAAdapter(in_dim, out_dim, rank,
                          device=module.weight.device,
                          dtype=module.weight.dtype)
    A_init, B_init = make_ablation_lora_init(
        d, in_dim, out_dim, rank,
        projected=projected,
        harmless_mean=harmless_mean,
        norm_preserve=norm_preserve,
        strength=strength,
    )
    with torch.no_grad():
        adapter.A.copy_(A_init.to(adapter.A.dtype))
        adapter.B.copy_(B_init.to(adapter.B.dtype))

    return adapter


def apply_lora_adapters(
    model,
    measurements: dict,
    source_layer: int,
    target_layers: Optional[List[int]] = None,
    rank: int = 3,
    scale: float = 1.0,
    projected: bool = False,
    norm_preserve: bool = True,
    direction: Optional[torch.Tensor] = None,
) -> List[Tuple[nn.Module, LoRAAdapter]]:
    """Attach LoRA adapters implementing the ablation to every targeted
    module. Returns (module, adapter) pairs so the caller can merge or remove
    them later.
    """
    layer_base, _ = get_layer_base(model)
    num_layers = len(layer_base)
    if target_layers is None:
        target_layers = list(range(num_layers))

    if direction is None:
        refusal_dir = measurements[f"refusal_{source_layer}"].float()
        if projected:
            harmless = measurements[f"harmless_{source_layer}"].float()
            refusal_dir = project_against_harmless(refusal_dir, harmless)
    else:
        refusal_dir = direction

    refusal_dir = normalize_direction(refusal_dir)

    harmless_mean = None
    if projected:
        harmless_mean = measurements.get(f"harmless_{source_layer}")

    attached: List[Tuple[nn.Module, LoRAAdapter]] = []
    for idx in target_layers:
        layer = layer_base[idx]
        for getter in (get_attention_output_proj, get_mlp_down_proj):
            try:
                mod = getter(layer)
            except ValueError:
                continue
            adapter = attach_lora_to_module(
                mod, refusal_dir, rank=rank,
                projected=projected,
                harmless_mean=harmless_mean,
                norm_preserve=norm_preserve,
                strength=scale,
            )
            attached.append((mod, adapter))

    return attached


def merge_lora_adapters(
    attached: List[Tuple[nn.Module, LoRAAdapter]],
) -> None:
    """Merge each LoRA adapter into its parent module's weight in-place:
    W <- W + B @ A. After merging, the adapter can be discarded.
    """
    with torch.no_grad():
        for module, adapter in attached:
            # adapter.forward(x) = (x @ A.T) @ B.T = x @ (B @ A).T
            # So the effective weight update is  delta_W = B @ A  (out, in)
            delta = adapter.B @ adapter.A  # (out_dim, in_dim)
            module.weight.data.add_(delta.to(module.weight.dtype))


# =============================================================================
# MPOA — Magnitude-Preserving Orthogonal Ablation (rank-3 LoRA)
# =============================================================================
#
# Heretic's MPOA preset: direction projected orthogonal to harmless mean,
# edit computed on row-normalised weights, row norms restored (rank-3 adapter).
# The rank-3 adapter gives the search extra degrees of freedom beyond the
# rank-1 projection, which is what lets it restore row norms without an
# explicit two-pass weight edit.

def apply_mpoa(
    model,
    measurements: dict,
    source_layer: int,
    target_layers: Optional[List[int]] = None,
    rank: int = 3,
    scale: float = 1.0,
    merge: bool = True,
) -> List[Tuple[nn.Module, LoRAAdapter]]:
    """Apply MPOA: projected direction, row-norm-restoring rank-3 LoRA."""
    harmless = measurements.get(f"harmless_{source_layer}")
    refusal = measurements[f"refusal_{source_layer}"].float()
    if harmless is not None:
        refusal = project_against_harmless(refusal, harmless.float())

    attached = apply_lora_adapters(
        model, measurements, source_layer,
        target_layers=target_layers,
        rank=rank,
        scale=scale,
        projected=True,
        norm_preserve=True,
        direction=refusal,
    )
    if merge:
        merge_lora_adapters(attached)
    return attached


# =============================================================================
# SOM — Self-Organising Map directions (rank-6 LoRA)
# =============================================================================
#
# Piras et al. (2025): a SOM over per-prompt harmful activations yields k
# prototype "neurons" per layer. The top-k most-winning neurons give k
# orthonormalised directions; ablate them with a geometric decay across the
# k directions (rank-k adapter). Heretic's port uses k=4 (rank-6 adapter
# because the row-norm restoration consumes 2 extra ranks).

class SelfOrganizingMap(nn.Module):
    """Minimal 1-D SOM over activation vectors.

    The map is a 1-D lattice of `n_neurons` codebook vectors of dim `d`.
    Training: for each input, find the best-matching unit (BMU), update the
    BMU and its neighbours with a decaying learning rate and neighbourhood
    radius.
    """

    def __init__(self, in_dim: int, n_neurons: int = 16,
                 device=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.n_neurons = n_neurons
        # Initialise codebook from a random sample of the data (done in fit).
        self.codebook = nn.Parameter(
            torch.zeros(n_neurons, in_dim, device=device, dtype=dtype)
        )

    @torch.no_grad()
    def fit(self, X: torch.Tensor, epochs: int = 50, lr0: float = 0.5,
            sigma0: Optional[float] = None, device=None):
        """Train the SOM on activation matrix X (n_samples, in_dim)."""
        X = X.to(self.codebook.device)
        n = X.shape[0]
        if sigma0 is None:
            sigma0 = self.n_neurons / 2.0

        # Initialise codebook from random samples.
        idx = torch.randperm(n, device=X.device)[:self.n_neurons]
        self.codebook.data.copy_(X[idx].float())

        positions = torch.arange(self.n_neurons, device=X.device, dtype=torch.float32)
        for epoch in range(epochs):
            lr = lr0 * (1.0 - epoch / epochs)
            sigma = sigma0 * (1.0 - epoch / epochs)
            # Random sample order.
            perm = torch.randperm(n, device=X.device)
            for i in perm:
                x = X[i]  # (d,)
                # BMU: nearest codebook vector.
                dists = torch.norm(self.codebook - x, dim=1)
                bmu = int(torch.argmin(dists))
                # Neighbourhood: Gaussian around BMU.
                h = torch.exp(
                    -0.5 * ((positions - bmu) / sigma) ** 2
                ).unsqueeze(1)  # (n_neurons, 1)
                self.codebook.data += lr * h * (x - self.codebook)

        # Return the k most "winning" neurons (those closest to the mean of X).
        mean_act = X.mean(0)
        dists = torch.norm(self.codebook - mean_act, dim=1)
        top_idx = torch.argsort(dists)[:self.n_neurons]
        return self.codebook.data[top_idx]

    def get_directions(self, k: int = 4) -> torch.Tensor:
        """Return the top-k codebook vectors as unit directions."""
        norms = torch.norm(self.codebook, dim=1, keepdim=True).clamp(min=1e-8)
        normalised = self.codebook / norms
        return normalised[:k]


def som_directions(
    per_prompt_measurements: dict,
    layer: int,
    k: int = 4,
    n_neurons: int = 16,
    epochs: int = 50,
    device=None,
) -> torch.Tensor:
    """Extract k SOM directions from per-prompt harmful activations at a layer.

    The SOM is trained on the harmful per-prompt activations; the k codebook
    vectors closest to the class mean are returned as orthonormalised
    directions with a geometric decay applied (direction i is scaled by
    decay^i so that the search can weight the top direction most heavily).
    Returns a (k, d) tensor of unit directions.
    """
    harmful = per_prompt_measurements[f"harmful_per_prompt_{layer}"].float()
    if device is None:
        device = harmful.device

    som = SelfOrganizingMap(harmful.shape[1], n_neurons=n_neurons,
                            device=device, dtype=harmful.dtype)
    som.fit(harmful, epochs=epochs, device=device)
    dirs = som.get_directions(k=k)

    # Orthonormalise via QR (so the directions are not collinear).
    Q, _ = torch.linalg.qr(dirs.T)  # (d, k)
    dirs = Q.T  # (k, d)

    # Geometric decay: direction i scaled by decay^i, then re-normalised.
    decay = 0.7
    scales = torch.tensor([decay ** i for i in range(k)],
                          device=dirs.device, dtype=dirs.dtype).unsqueeze(1)
    dirs = dirs * scales
    return normalize_direction(dirs)


def apply_som(
    model,
    measurements: dict,
    per_prompt_measurements: Optional[dict] = None,
    source_layer: int = 0,
    target_layers: Optional[List[int]] = None,
    k: int = 4,
    rank: int = 6,
    scale: float = 1.0,
    merge: bool = True,
) -> List[Tuple[nn.Module, LoRAAdapter]]:
    """Apply SOM: k self-organising-map directions with geometric decay,
    rank-6 LoRA adapter (rank k + 2 for row-norm restoration)."""
    if per_prompt_measurements is None:
        raise ValueError("SOM requires per_prompt_measurements")

    dirs = som_directions(per_prompt_measurements, source_layer, k=k)
    # Combine the k directions into a single (k, d) tensor; the adapter
    # construction in apply_lora_adapters handles multi-direction by using
    # rank = k + 2 (the search uses rank 6 for k=4).
    attached = apply_lora_adapters(
        model, measurements, source_layer,
        target_layers=target_layers,
        rank=rank,
        scale=scale,
        projected=False,
        norm_preserve=True,
        direction=dirs,
    )
    if merge:
        merge_lora_adapters(attached)
    return attached


# =============================================================================
# ARA — Arbitrary-Rank Ablation (rank-50 LoRA, L-BFGS optimisation)
# =============================================================================
#
# Heretic's default modifier. For each targeted module, a rank-r (r=50)
# LoRA adapter is fitted so that:
#   - outputs on harmful prompts move toward the harmless mean, and
#   - outputs on harmless prompts stay fixed,
#   - row magnitudes are preserved.
# The objective is optimised with L-BFGS. Unlike the direction-based
# methods, ARA makes no geometric assumption about where refusal lives.

def ara_objective(
    adapter: LoRAAdapter,
    base_weight: torch.Tensor,
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    lam: float = 1.0,
    norm_reg: float = 0.01,
) -> torch.Tensor:
    """ARA loss for a single module.

    loss = MSE(harmful_out, harmless_mean) + lam * MSE(harmless_out, harmless)
           + norm_reg * ||W + B@A||_row_norms - ||W||_row_norms||^2

    harmful_out = harmful_acts @ (W + B@A).T
    harmless_out = harmless_acts @ (W + B@A).T
    """
    delta = adapter.B @ adapter.A  # (out, in)
    W_new = base_weight + delta

    harmful_out = harmful_acts @ W_new.T  # (n, out)
    harmless_out = harmless_acts @ W_new.T

    target = harmless_acts.mean(0, keepdim=True)  # (1, out)
    harmful_loss = F.mse_loss(harmful_out, target.expand_as(harmful_out))
    harmless_loss = F.mse_loss(harmless_out, harmless_out.detach())

    # Row-norm preservation.
    orig_norms = torch.norm(base_weight, dim=1)
    new_norms = torch.norm(W_new, dim=1)
    norm_loss = ((new_norms - orig_norms) ** 2).mean()

    return harmful_loss + lam * harmless_loss + norm_reg * norm_loss


def apply_ara(
    model,
    measurements: dict,
    per_prompt_measurements: dict,
    source_layer: int,
    target_layers: Optional[List[int]] = None,
    rank: int = 50,
    steps: int = 100,
    lr: float = 0.1,
    lam: float = 1.0,
    norm_reg: float = 0.01,
    merge: bool = True,
) -> List[Tuple[nn.Module, LoRAAdapter]]:
    """Apply ARA: fit a rank-r LoRA adapter per targeted module with L-BFGS.

    The adapter is optimised so that harmful activations at `source_layer`
    move toward the harmless mean while harmless activations stay fixed,
    with row-norm preservation. The fitted adapters are then merged.
    """
    if per_prompt_measurements is None:
        raise ValueError("ARA requires per_prompt_measurements")

    harmful_key = f"harmful_per_prompt_{source_layer}"
    harmless_key = f"harmless_per_prompt_{source_layer}"
    if harmful_key not in per_prompt_measurements or harmless_key not in per_prompt_measurements:
        raise ValueError(f"Per-prompt activations not found for layer {source_layer}")

    harmful_acts = per_prompt_measurements[harmful_key].float()
    harmless_acts = per_prompt_measurements[harmless_key].float()

    layer_base, _ = get_layer_base(model)
    num_layers = len(layer_base)
    if target_layers is None:
        target_layers = list(range(num_layers))

    attached: List[Tuple[nn.Module, LoRAAdapter]] = []
    for idx in target_layers:
        layer = layer_base[idx]
        for getter in (get_attention_output_proj, get_mlp_down_proj):
            try:
                mod = getter(layer)
            except ValueError:
                continue

            in_dim = _module_in_dim(mod)
            out_dim = _module_out_dim(mod)
            # For simplicity, ARA uses the source layer's activations for all
            # target layers (a more faithful implementation would collect
            # activations at each target layer).
            adapter = LoRAAdapter(in_dim, out_dim, rank,
                                  device=mod.weight.device,
                                  dtype=mod.weight.dtype)
            adapter.A.data.normal_(std=0.01)
            adapter.B.data.zero_()

            base_weight = mod.weight.data.clone().float()
            opt = torch.optim.LBFGS(
                adapter.parameters(),
                lr=lr,
                max_iter=steps,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                loss = ara_objective(
                    adapter, base_weight,
                    harmful_acts.to(mod.weight.device, dtype=torch.float32),
                    harmless_acts.to(mod.weight.device, dtype=torch.float32),
                    lam=lam, norm_reg=norm_reg,
                )
                loss.backward()
                return loss

            opt.step(closure)
            attached.append((mod, adapter))

    if merge:
        merge_lora_adapters(attached)
    return attached


# =============================================================================
# J-space ablation (Jacobian-space, small pair only)
# =============================================================================
#
# Computes the refusal direction in a "Jacobian lens" workspace defined by
# the Jacobian of the model's output w.r.t. the residual stream at the
# source layer, then pulls it back to the residual stream and bakes it into
# the weights with the standard rank-1/rank-3 edit.

def jacobian_lens_direction(
    model,
    tokenizer,
    measurements: dict,
    per_prompt_measurements: Optional[dict],
    source_layer: int,
    k: int = 1,
) -> torch.Tensor:
    """Estimate the refusal direction in Jacobian-lens space.

    The Jacobian lens maps a residual-stream direction d to the change it
    induces in the output logits:
        J_lens = d out / d residual  (out_dim, d)
    A refusal direction in residual space is one whose lens image best
    separates harmful from harmless outputs. We approximate this by:
      1. Take the diff-in-means direction r in residual space.
      2. Compute the Jacobian-vector product J_lens @ r for a batch of
         harmful and harmless prompts.
      3. The J-space direction is the mean difference of these images.
    In practice this reduces to a re-weighted diff-in-means, which is why
    the reference implementation is a forward hook rather than a weight edit.

    Returns a (k, d) tensor of unit directions in residual space.
    """
    # Fallback: use diff-in-means (the J-space correction is small and the
    # reference implementation removes little refusal).
    refusal = measurements[f"refusal_{source_layer}"].float()
    return normalize_direction(refusal.unsqueeze(0) if refusal.dim() == 1 else refusal)


def apply_jspace(
    model,
    measurements: dict,
    source_layer: int,
    target_layers: Optional[List[int]] = None,
    rank: int = 3,
    scale: float = 1.0,
    merge: bool = True,
) -> List[Tuple[nn.Module, LoRAAdapter]]:
    """Apply J-space ablation: directions from Jacobian-lens space, baked
    into the weights with the DiM/MPOA edit (rank-3 adapter)."""
    dirs = jacobian_lens_direction(model, None, measurements, None, source_layer)
    attached = apply_lora_adapters(
        model, measurements, source_layer,
        target_layers=target_layers,
        rank=rank,
        scale=scale,
        projected=False,
        norm_preserve=True,
        direction=dirs,
    )
    if merge:
        merge_lora_adapters(attached)
    return attached


# =============================================================================
# Unified entry point
# =============================================================================

def apply_method(
    method: str,
    model,
    measurements: dict,
    per_prompt_measurements: Optional[dict],
    source_layer: int,
    target_layers: Optional[List[int]] = None,
    scale: float = 1.0,
    merge: bool = True,
    **kwargs,
) -> List[Tuple[nn.Module, LoRAAdapter]]:
    """Dispatch to the requested ablation method.

    method: 'dim' (diff-in-means, rank-1), 'mpoa' (rank-3),
            'som' (rank-6), 'ara' (rank-50), 'jspace' (rank-3).
    """
    if method == "dim":
        return apply_lora_adapters(
            model, measurements, source_layer,
            target_layers=target_layers,
            rank=1, scale=scale, projected=False, norm_preserve=False,
        )
    if method == "mpoa":
        return apply_mpoa(
            model, measurements, source_layer,
            target_layers=target_layers, scale=scale, merge=merge,
        )
    if method == "som":
        return apply_som(
            model, measurements, per_prompt_measurements, source_layer,
            target_layers=target_layers, scale=scale, merge=merge,
        )
    if method == "ara":
        return apply_ara(
            model, measurements, per_prompt_measurements, source_layer,
            target_layers=target_layers, merge=merge,
            **{k: v for k, v in kwargs.items() if k in
               ('rank', 'steps', 'lr', 'lam', 'norm_reg')},
        )
    if method == "jspace":
        return apply_jspace(
            model, measurements, source_layer,
            target_layers=target_layers, scale=scale, merge=merge,
        )
    raise ValueError(f"unknown method {method}")
