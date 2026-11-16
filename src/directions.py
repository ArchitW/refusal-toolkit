"""
directions.py

Refusal direction extraction and refinement:
- Single diff-in-means (Arditi et al.)
- Multi-direction SVD on contrast matrix ("Willing but Unable")
- PCA-based extraction
- Projected abliteration
- Harmfulness vs Refusal separation (Zhao et al.)
"""
import torch
from typing import Dict, List, Optional, Tuple


def extract_single_direction(measurements: Dict, layer_idx: int) -> torch.Tensor:
    return measurements[f"refusal_{layer_idx}"].float()


def extract_topk_directions_svd(
    measurements: Dict,
    layer_idx: int,
    k: int = 3,
) -> torch.Tensor:
    contrast_key = f"contrast_{layer_idx}"

    if contrast_key in measurements:
        contrasts = measurements[contrast_key].float()
        contrasts_centered = contrasts - contrasts.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(contrasts_centered, full_matrices=False)
            k_actual = min(k, Vh.shape[0])
            return Vh[:k_actual]
        except Exception as e:
            print(f"SVD failed ({e}), falling back to mean direction")
            return measurements[f"refusal_{layer_idx}"].float().unsqueeze(0)
    else:
        refusal = measurements[f"refusal_{layer_idx}"].float().unsqueeze(0)
        return refusal


def extract_directions_pca(
    measurements: Dict,
    layer_idx: int,
    k: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    contrast_key = f"contrast_{layer_idx}"

    if contrast_key not in measurements:
        dir_vec = measurements[f"refusal_{layer_idx}"].float().unsqueeze(0)
        return dir_vec, torch.ones(1)

    contrasts = measurements[contrast_key].float()
    contrasts_centered = contrasts - contrasts.mean(dim=0, keepdim=True)
    cov = torch.matmul(contrasts_centered.T, contrasts_centered) / (contrasts_centered.shape[0] - 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    k_actual = min(k, eigenvectors.shape[1])
    directions = eigenvectors[:, :k_actual].T

    total_var = eigenvalues.sum()
    explained = eigenvalues[:k_actual] / total_var

    return directions, explained


def project_against_harmless(
    refusal_dir: torch.Tensor,
    harmless_mean: torch.Tensor,
) -> torch.Tensor:
    harmless_norm = torch.nn.functional.normalize(harmless_mean.float(), dim=0)
    projection_scalar = refusal_dir @ harmless_norm
    return refusal_dir - projection_scalar * harmless_norm


def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(direction.float(), dim=-1)


def compute_cohens_d(measurements: Dict, layer_idx: int) -> float:
    diff = measurements[f"refusal_{layer_idx}"]
    harmless = measurements[f"harmless_{layer_idx}"]
    pooled_std = harmless.norm().item() / (harmless.numel() ** 0.5) + 1e-8
    return float(diff.norm().item() / pooled_std)


def auto_select_best_layer(measurements: Dict, metric: str = "norm") -> int:
    num_layers = measurements["num_layers"]
    best_layer = 0
    best_score = -1.0

    for layer in range(num_layers):
        if metric == "norm":
            score = measurements.get(f"refusal_norm_{layer}", 0.0)
        elif metric == "cohens_d":
            score = compute_cohens_d(measurements, layer)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_layer = layer

    return best_layer


def extract_harmfulness_direction(
    measurements: Dict,
    layer_idx: int,
) -> torch.Tensor:
    harmful = measurements[f"harmful_{layer_idx}"].float()
    harmless = measurements[f"harmless_{layer_idx}"].float()

    harmless_norm = normalize_direction(harmless)
    projection = (harmful @ harmless_norm) * harmless_norm
    harmfulness = harmful - projection
    return harmfulness


def compute_direction_similarity(dir_a: torch.Tensor, dir_b: torch.Tensor) -> float:
    a = normalize_direction(dir_a)
    b = normalize_direction(dir_b)
    return float((a @ b).item())


def compute_contrast_outliers(
    measurements: Dict,
    layer_idx: int,
    threshold_std: float = 2.0,
) -> List[int]:
    contrast_key = f"contrast_{layer_idx}"
    if contrast_key not in measurements:
        return []

    contrasts = measurements[contrast_key].float()
    norms = torch.norm(contrasts, dim=1)
    mean_norm = norms.mean()
    std_norm = norms.std()

    outlier_mask = norms > (mean_norm + threshold_std * std_norm)
    return torch.where(outlier_mask)[0].tolist()


# =============================================================================
# SOM (Self-Organising Map) directions
# =============================================================================
#
# Piras et al. (2025): a 1-D SOM over per-prompt harmful activations yields k
# prototype "neurons" per layer. The k codebook vectors closest to the class
# mean are orthonormalised and returned as directions, scaled by a geometric
# decay (direction i gets weight decay^i). Heretic's port uses k=4.

def som_directions(
    per_prompt_measurements: Dict,
    layer_idx: int,
    k: int = 4,
    n_neurons: int = 16,
    epochs: int = 50,
    decay: float = 0.7,
    lr0: float = 0.5,
) -> torch.Tensor:
    """Train a 1-D SOM on harmful per-prompt activations and return k unit
    directions (top-k codebook vectors closest to the class mean, scaled by
    geometric decay and re-normalised)."""
    harmful_key = f"harmful_per_prompt_{layer_idx}"
    if harmful_key not in per_prompt_measurements:
        raise ValueError(f"Per-prompt harmful activations not found for layer {layer_idx}")

    X = per_prompt_measurements[harmful_key].float()
    n, d = X.shape

    # Initialise codebook from random samples.
    idx = torch.randperm(n, device=X.device)[:n_neurons]
    codebook = X[idx].clone().float()

    positions = torch.arange(n_neurons, device=X.device, dtype=torch.float32)
    sigma0 = n_neurons / 2.0

    for epoch in range(epochs):
        lr = lr0 * (1.0 - epoch / epochs)
        sigma = sigma0 * (1.0 - epoch / epochs)
        perm = torch.randperm(n, device=X.device)
        for i in perm:
            x = X[i]
            dists = torch.norm(codebook - x, dim=1)
            bmu = int(torch.argmin(dists))
            h = torch.exp(-0.5 * ((positions - bmu) / sigma) ** 2).unsqueeze(1)
            codebook.data += lr * h * (x - codebook)

    # Top-k codebook vectors closest to the class mean.
    mean_act = X.mean(0)
    dists = torch.norm(codebook - mean_act, dim=1)
    top_idx = torch.argsort(dists)[:k]
    dirs = codebook[top_idx]

    # Orthonormalise via QR.
    Q, _ = torch.linalg.qr(dirs.T)  # (d, k)
    dirs = Q.T  # (k, d)

    # Geometric decay then re-normalise.
    scales = torch.tensor([decay ** i for i in range(k)],
                          device=dirs.device, dtype=dirs.dtype).unsqueeze(1)
    dirs = dirs * scales
    return normalize_direction(dirs)
