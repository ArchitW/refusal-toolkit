"""Refusal Toolkit — Production-grade framework for representation engineering on LLMs."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("refusal-toolkit")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

from .model_adapter import (
    get_layer_base,
    get_num_layers,
    get_hidden_size,
    get_attention_output_proj,
    get_mlp_down_proj,
    has_tied_weights,
    get_model_family,
)

from .measure import measure_refusal_directions
from .measure_per_prompt import measure_per_prompt_activations
from .directions import (
    extract_single_direction,
    extract_topk_directions_svd,
    extract_directions_pca,
    project_against_harmless,
    normalize_direction,
    compute_cohens_d,
    auto_select_best_layer,
    extract_harmfulness_direction,
    compute_direction_similarity,
    compute_contrast_outliers,
)
from .ablate import (
    register_ablation_hooks,
    remove_hooks,
    orthogonalize_matrix,
    norm_preserving_orthogonalize,
    project_bias,
    whitened_svd_directions,
    svd_directions,
    leace_direction,
    build_directions,
    ablate_model_weights,
)
from .evaluate import (
    is_refusal,
    evaluate_refusal_rate,
    evaluate_harmless_compliance,
    detect_harmfulness_latent,
    detect_reasoning_tags,
    split_reasoning_output,
)
from .utils import (
    load_prompts,
    format_chat_prompts,
    save_measurements,
    load_measurements,
)
from .sae_extract import (
    SparseAutoencoder,
    train_sae,
    identify_refusal_features,
    extract_sae_refusal_directions,
    register_sae_feature_ablation_hooks,
    ablate_sae_feature,
    ablate_sae_features,
)

__all__ = [
    # model_adapter
    "get_layer_base",
    "get_num_layers",
    "get_hidden_size",
    "get_attention_output_proj",
    "get_mlp_down_proj",
    "has_tied_weights",
    "get_model_family",
    # measure
    "measure_refusal_directions",
    "measure_per_prompt_activations",
    # directions
    "extract_single_direction",
    "extract_topk_directions_svd",
    "extract_directions_pca",
    "project_against_harmless",
    "normalize_direction",
    "compute_cohens_d",
    "auto_select_best_layer",
    "extract_harmfulness_direction",
    "compute_direction_similarity",
    "compute_contrast_outliers",
    # ablate
    "register_ablation_hooks",
    "remove_hooks",
    "orthogonalize_matrix",
    "norm_preserving_orthogonalize",
    "project_bias",
    "whitened_svd_directions",
    "svd_directions",
    "leace_direction",
    "build_directions",
    "ablate_model_weights",
    # evaluate
    "is_refusal",
    "evaluate_refusal_rate",
    "evaluate_harmless_compliance",
    "detect_harmfulness_latent",
    "detect_reasoning_tags",
    "split_reasoning_output",
    # utils
    "load_prompts",
    "format_chat_prompts",
    "save_measurements",
    "load_measurements",
    # sae_extract
    "SparseAutoencoder",
    "train_sae",
    "identify_refusal_features",
    "extract_sae_refusal_directions",
    "register_sae_feature_ablation_hooks",
    "ablate_sae_feature",
    "ablate_sae_features",
]