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
