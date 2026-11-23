Created: 23-11-26
Updated: 23-11-26

# 03 — Refusal Behavior in LLMs: A Nonlinear Perspective

## TL;DR (Summary)

Prior work (Arditi et al., 2024) proved that refusal in LLMs lies along a single straight line (a 1-D direction) in the hidden state. This paper tests whether that simple straight line holds across different model families. The authors collect hidden states from six LLMs across three architectures (Llama, Qwen, Mistral). They apply dimensionality reduction methods (PCA, t-SNE, and UMAP) to visualize harmful and harmless activations. The visualizations show that refusal is **nonlinear** and **multidimensional**. Instead of a simple line, refusal forms a curved manifold or multi-component cluster. The authors conclude that erasing a single straight line leaves residual nonlinear refusal structure in the model.

---

## Simple Example (Plain English)

Imagine you are mapping traffic patterns in three different cities (Llama, Qwen, and Mistral).

1. **The 1-D Claim (Note `01`)**:
   - An earlier study (Note `01`) looked at average traffic flow in one city. It found that one single street (a 1-D line) explained most of the difference between busy days and quiet days.

2. **Testing All Three Cities**:
   - This paper asks: *"Does one single street control traffic in every city?"*
   - Instead of just checking the street average, the authors plot every single car trip on a 2D map using three different mapping tools (PCA, t-SNE, and UMAP).

3. **What the Map Reveals**:
   - In all three cities, the trips do not sit on a single straight line.
   - Some refusals cluster in one corner (direct refusals like *"I cannot help with that"*), while other refusals cluster in a completely different curved area (indirect refusals like *"I can tell you about the theory, but not step-by-step instructions"*).
   - Because the map is curved and has multiple clusters, cutting one single straight street does not block all the roads used for refusals.

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | Refusal Behavior in Large Language Models: A Nonlinear Perspective |
| **Authors** | Fabian Hildebrandt, Andreas Maier, Patrick Krauss, Achim Schilling |
| **Year** | 2025 (January 2025) |
| **Venue / Links** | [arXiv:2501.08145](https://arxiv.org/abs/2501.08145) |
| **Toolkit Role** | Empirical counter-evidence to 1-D refusal; motivates SVD, PCA, and SOM (`mpoa`, `som`, `ara`) |
| **Toolkit Status** | **Implemented** (`src/directions.py::extract_directions_pca`, `extract_topk_directions_svd`; `src/ablate.py::apply_som`, `apply_ara`) |

---

## 1. The Problem

### 1.1 The Single-Direction Assumption
- Arditi et al. (`01`, `arXiv:2406.11717`) computed the difference in mean activations between harmful and harmless prompts:
  $$r = \mathbb{E}[h \mid \text{harmful}] - \mathbb{E}[h \mid \text{harmless}]$$
- They showed that subtracting $r$ from the residual stream causes models to comply with harmful requests.

### 1.2 Unverified Geometric Assumptions
- Mean difference is a first-order (linear) summary statistic.
- If refusal representations are curved (nonlinear) or span multiple separate clusters (multidimensional), a single mean difference vector $r$ only cuts one axis of the refusal region.
- Residual refusal signals may remain active after single-direction ablation.

### 1.3 Core Claim of This Paper
- Refusal representations in transformer LLMs are **nonlinear** and **multidimensional**.
- The geometry of the refusal region varies by model family and by layer depth.
- Single-direction removal is a linear approximation of a more complex geometric structure.

---

## 2. The Method (How It Works)

The authors use a four-step activation geometry analysis across six LLMs from three model families (Llama-2-7B/13B, Mistral-7B, Qwen2.5-1.5B/7B/14B).

### Step 1: Prompt & Activation Collection
1. Construct a paired prompt set of harmful instructions and benign instructions.
2. Run a forward pass through each model.
3. Extract the hidden state vector $h^{(\ell)} \in \mathbb{R}^d$ at the final prompt token position across layers $\ell$.

### Step 2: Linear Dimensionality Reduction (PCA)
1. Concatenate all harmful and harmless activation vectors at layer $\ell$.
2. Center the activations by subtracting the global mean vector.
3. Compute the top principal components (PCs) via Singular Value Decomposition (SVD).
4. Project the activations onto the top 2–3 principal components.

### Step 3: Nonlinear Manifold Visualization (t-SNE & UMAP)
1. Apply **t-SNE** (t-distributed Stochastic Neighbor Embedding) and **UMAP** (Uniform Manifold Approximation and Projection) to project the high-dimensional hidden states into 2-D space.
2. Preserve local neighborhood distances between similar hidden states.
3. Plot the harmful and harmless clusters in 2-D space across early, middle, and late layers.

### Step 4: Nonlinear Probe Evaluation
1. Train a linear logistic regression probe to classify harmful vs. harmless hidden states.
2. Train an MLP (multi-layer perceptron) nonlinear probe on the same hidden states.
3. Erase the single mean-difference direction $r$ and re-evaluate both probes to measure residual separability.

---

## 3. Key Findings (What the Authors Found)

- **Nonlinearity in Hidden Space**: UMAP and t-SNE plots show that harmful and harmless activations form curved, crescent-shaped manifolds rather than two clean parallel lines.
- **Multiple Sub-Clusters**: Direct refusals (e.g., `"I'm sorry, I cannot..."`) and evasive/soft refusals (e.g., `"While I can discuss the theory, I cannot provide..."`) occupy distinct regions in activation space.
- **Architecture & Layer Dependence**: The depth at which refusal peaks and the curvature of the refusal manifold differ across Llama, Qwen, and Mistral models.
- **Residual Signal After 1-D Ablation**: Erasing a single linear direction leaves nonlinear clusters intact; nonlinear probes still achieve high classification accuracy on the ablated activations.

| Finding | Observed Behavior |
|---|---|
| **Geometry of Clusters** | Curved 2-D manifolds (crescent / ring shapes) in UMAP and t-SNE |
| **Cluster Structure** | Distinct sub-clusters for hard refusals vs. soft/educational refusals |
| **Depth Profile** | Linear separation increases in mid-layers; nonlinear curvature appears in later layers |
| **Post-Ablation Probe** | Nonlinear MLP probes still detect refusal structure after rank-1 erasure |

---

## 4. Relationship to Other Research

- **Challenges Single-Direction Erasure (Note `01`, `arXiv:2406.11717`)**:
  - Note `01` proved that a rank-1 mean difference vector removes most surface refusals.
  - Note `03` shows why some residual refusal or behavioral weirdness remains after rank-1 weight orthogonalization.
- **Corroborates Concept Cones (Note `04`, `arXiv:2502.17420`)**:
  - Note `04` uses gradient-based optimization (RDO) to prove that refusal occupies a $k$-dimensional polyhedral cone where multiple independent directions mediate refusal.
- **Explains Heretic 2.0 Methods (arXiv:2611.17846)**:
  - Because refusal is multidimensional and nonlinear, the toolkit implements top-$k$ SVD (`src/ablate.py::svd_directions`), Self-Organizing Maps (`src/ablate.py::apply_som`), and rank-50 LoRA L-BFGS optimization (`src/ablate.py::apply_ara`).

---

## 5. Implementation in `refusal-toolkit`

| Step | Module / Function | CLI Command / Flag |
|---|---|---|
| **1. PCA Extraction** | `src/directions.py::extract_directions_pca` | `src/directions.py` |
| **2. Contrast SVD (Top-k)** | `src/directions.py::extract_topk_directions_svd`, `src/ablate.py::svd_directions` | `src/ablate.py::build_directions(method="svd")` |
| **3. Whitened SVD (Top-k)** | `src/ablate.py::whitened_svd_directions` | `src/ablate.py::build_directions(method="whitened-svd")` |
| **4. SOM & ARA Multi-Direction Ablation** | `src/ablate.py::apply_som`, `apply_ara` | `refusal-toolkit heretic-search --method som` / `--method ara` |

---

## 6. Limitations & Open Questions

- **Qualitative Visualizations**: t-SNE and UMAP distort global distances in 2-D projections. Cluster separation in 2-D plots does not prove causal control in high-dimensional space.
- **Lack of Direct Causal Ablation**: The paper focuses on probe accuracy and visualization; Note `04` provides the direct causal intervention study (RDO + RepInd).
- **Number of Required Directions**: The paper does not give a closed-form formula for how many directions ($k$) a specific model requires; `heretic-search` (`scripts/10_heretic_search.py`) sweeps $k$ empirically.

---

## 7. Citation

```bibtex
@article{hildebrandt2025refusal,
  title={Refusal Behavior in Large Language Models: A Nonlinear Perspective},
  author={Hildebrandt, Fabian and Maier, Andreas and Krauss, Patrick and Schilling, Achim},
  journal={arXiv preprint arXiv:2501.08145},
  year={2025}
}
```