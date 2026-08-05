"""
sae_extract.py

Sparse Autoencoder (SAE) based refusal direction extraction.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        top_k: int = 64,
        jumprelu_threshold: float = 0.03,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.jumprelu_threshold = jumprelu_threshold

        self.W_enc = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        self.W_dec = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        with torch.no_grad():
            self.W_dec.data = torch.nn.functional.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.matmul(x, self.W_enc.T) + self.b_enc
        mask = (pre > self.jumprelu_threshold).float()
        acts = torch.relu(pre) * mask

        if self.top_k > 0 and self.top_k < self.hidden_dim:
            topk_vals, topk_idx = torch.topk(acts, k=self.top_k, dim=-1)
            acts_sparse = torch.zeros_like(acts)
            acts_sparse.scatter_(-1, topk_idx, topk_vals)
            acts = acts_sparse

        return acts

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.matmul(f, self.W_dec.T) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def get_feature_directions(self) -> torch.Tensor:
        return self.W_dec.data.T


def train_sae(
    activations: torch.Tensor,
    hidden_dim: Optional[int] = None,
    top_k: int = 64,
    lr: float = 1e-3,
    l1_coef: float = 1e-3,
    epochs: int = 1000,
    batch_size: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> SparseAutoencoder:
    input_dim = activations.shape[1]
    if hidden_dim is None:
        hidden_dim = input_dim * 8

    sae = SparseAutoencoder(input_dim, hidden_dim, top_k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(activations)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae.train()
    for epoch in tqdm(range(epochs), desc="Training SAE"):
        total_loss = 0.0
        total_recon = 0.0
        total_l1 = 0.0

        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, f = sae(batch)

            recon_loss = torch.mean((x_hat - batch) ** 2)
            l1_loss = l1_coef * torch.mean(torch.abs(f))
            loss = recon_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                sae.W_dec.data = torch.nn.functional.normalize(sae.W_dec.data, dim=0)

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_l1 += l1_loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.6f} "
                  f"recon={total_recon/len(loader):.6f} l1={total_l1/len(loader):.6f}")

    sae.eval()
    return sae


def identify_refusal_features(
    sae: SparseAutoencoder,
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    top_n: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, List[Dict]]:
    sae = sae.to(device)

    with torch.no_grad():
        _, harmful_features = sae(harmful_acts.to(device))
        _, harmless_features = sae(harmless_acts.to(device))

    harmful_mean = harmful_features.mean(dim=0)
    harmless_mean = harmless_features.mean(dim=0)
    scores = harmful_mean - harmless_mean

    top_scores, top_indices = torch.topk(scores, k=top_n)
    feature_directions = sae.get_feature_directions()[top_indices].cpu()

    feature_info = []
    for i, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
        feature_info.append({
            "rank": i,
            "feature_idx": idx,
            "refusal_score": score,
            "harmful_activation": harmful_mean[idx].item(),
            "harmless_activation": harmless_mean[idx].item(),
        })

    return feature_directions, feature_info


def extract_sae_refusal_directions(
    measurements: Dict,
    layer_idx: int,
    hidden_dim_multiplier: int = 8,
    top_k: int = 64,
    top_n_features: int = 10,
    train_epochs: int = 1000,
) -> Dict:
    harmful_key = f"harmful_per_prompt_{layer_idx}"
    harmless_key = f"harmless_per_prompt_{layer_idx}"

    if harmful_key not in measurements or harmless_key not in measurements:
        raise ValueError(
            f"Per-prompt measurements not found for layer {layer_idx}. "
            f"Run measure_per_prompt.py first."
        )

    harmful_acts = measurements[harmful_key].float()
    harmless_acts = measurements[harmless_key].float()
    all_acts = torch.cat([harmful_acts, harmless_acts], dim=0)
    hidden_dim = hidden_dim_multiplier * all_acts.shape[1]

    print(f"Training SAE on {all_acts.shape[0]} activations of dim {all_acts.shape[1]} "
          f"(hidden dim {hidden_dim})...")
    sae = train_sae(
        all_acts,
        hidden_dim=hidden_dim,
        top_k=top_k,
        epochs=train_epochs,
    )

    print(f"Identifying top-{top_n_features} refusal features...")
    directions, info = identify_refusal_features(sae, harmful_acts, harmless_acts, top_n_features)

    return {
        "sae": sae,
        "feature_directions": directions,
        "feature_info": info,
        "layer": layer_idx,
        "num_features": sae.hidden_dim,
        "top_k": top_k,
    }


class SAEFeatureAblationHook:
    """Removes the contribution of specific SAE features from the residual
    stream at inference time: x' = x - scale * sum_i f_i * d_i."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        feature_indices: List[int],
        scale: float = 1.0,
    ):
        self.sae = sae
        self.feature_indices = torch.as_tensor(list(feature_indices), dtype=torch.long)
        self.scale = scale

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        flat = hidden.reshape(-1, hidden.size(-1))

        with torch.no_grad():
            flat_f = flat.to(self.sae.W_dec.device, dtype=self.sae.W_dec.dtype)
            features = self.sae.encode(flat_f)
            contributions = torch.matmul(
                features[:, self.feature_indices],
                self.sae.W_dec[:, self.feature_indices].T,
            )
            flat_new = flat - self.scale * contributions.to(flat.dtype)

        modified = flat_new.reshape(hidden.shape)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


def register_sae_feature_ablation_hooks(
    model,
    sae: SparseAutoencoder,
    feature_indices: List[int],
    layer_idx: int,
    scale: float = 1.0,
):
    from .model_adapter import get_layer_base

    layer_base, _ = get_layer_base(model)
    hook = SAEFeatureAblationHook(sae, feature_indices, scale)
    return layer_base[layer_idx].register_forward_hook(hook)


def ablate_sae_feature(
    model,
    sae: SparseAutoencoder,
    feature_idx: int,
    layer_idx: int,
    scale: float = 1.0,
):
    from .ablate import orthogonalize_matrix, _assert_not_quantized
    from .model_adapter import get_layer_base, get_attention_output_proj, get_mlp_down_proj

    direction = sae.get_feature_directions()[feature_idx].cpu()

    layer_base, _ = get_layer_base(model)
    layer = layer_base[layer_idx]

    try:
        o_proj = get_attention_output_proj(layer)
        _assert_not_quantized(o_proj.weight, "o_proj")
        o_proj.weight.data = orthogonalize_matrix(o_proj.weight.data, direction, scale)
    except ValueError:
        pass

    try:
        down_proj = get_mlp_down_proj(layer)
        _assert_not_quantized(down_proj.weight, "down_proj")
        down_proj.weight.data = orthogonalize_matrix(down_proj.weight.data, direction, scale)
    except ValueError:
        pass


def ablate_sae_features(
    model,
    sae: SparseAutoencoder,
    feature_indices: List[int],
    layer_idx: int,
    scale: float = 1.0,
):
    from .ablate import orthogonalize_matrix, _assert_not_quantized
    from .model_adapter import get_layer_base, get_attention_output_proj, get_mlp_down_proj

    layer_base, _ = get_layer_base(model)
    layer = layer_base[layer_idx]

    for feature_idx in feature_indices:
        direction = sae.get_feature_directions()[feature_idx].cpu()

        try:
            o_proj = get_attention_output_proj(layer)
            _assert_not_quantized(o_proj.weight, "o_proj")
            o_proj.weight.data = orthogonalize_matrix(o_proj.weight.data, direction, scale)
        except ValueError:
            pass

        try:
            down_proj = get_mlp_down_proj(layer)
            _assert_not_quantized(down_proj.weight, "down_proj")
            down_proj.weight.data = orthogonalize_matrix(down_proj.weight.data, direction, scale)
        except ValueError:
            pass
