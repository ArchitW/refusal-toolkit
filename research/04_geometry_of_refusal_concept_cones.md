Created: 23-11-26
Updated: 23-11-26

# 04 — The Geometry of Refusal: Concept Cones and Representational Independence

## TL;DR (Summary)

Standard refusal ablation removes a single direction $r$ computed from harmful and harmless prompts. This paper proves that refusal is not a 1-D line. Refusal is a **multi-dimensional cone** in hidden space. The authors introduce **Refusal Direction Optimization (RDO)**. RDO learns a small set of $k$ directions that span the refusal cone. They also introduce **Representational Independence (RepInd)**. RepInd filters candidate directions to keep only those that cause refusal without damaging unrelated capabilities. Combining RDO with RepInd achieves stronger harmful refusal removal and lower capability loss than single-direction abliteration.

---

## Simple Example (Plain English)

Imagine a factory that makes toys, and a safety manager who stops toys from leaving the building.

1. **The 1-D Guard (Note `01`)**:
   - In Note `01`, researchers found one main "Stop" button on the manager's desk. Turning off that one button stopped most toys.

2. **Why One Button Was Not Enough**:
   - The manager actually has four hands (`k = 4` directions), all pressing different "Stop" buttons when a toy looks unsafe.
   - Even if you disable the main button, the other three hands keep pressing their buttons.
   - Furthermore, one of the manager's hands also presses a *"Stop for Quality Check"* button when toys have a tiny paint defect. If you disable that hand completely, all defective toys pass through unchecked (this is **capability damage**).

3. **How RDO + RepInd Solves It**:
   - **Step 1 (RDO — Concept Cones)**: The authors use gradient descent to find the exact set of buttons (directions $v_1, \dots, v_k$) the manager presses for dangerous toys.
   - **Step 2 (RepInd — Representational Independence)**: The authors test each button individually. If pressing a button causes a dangerous toy to pass, but pressing the *other* buttons does not, that button is specific to dangerous toys. They keep only those independent buttons and leave the "Quality Check" button untouched.

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence |
| **Authors** | Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger |
| **Year** | 2025 (Feb 2025) · **Venue:** ICML 2025 |
| **Venue / Links** | [arXiv:2502.17420](https://arxiv.org/abs/2502.17420) · [PMLR](https://proceedings.mlr.press/v267/wolschlager25a.html) · [Code](https://github.com/tum-aim-ssl/lm-refusal) |
| **Toolkit Role** | Theoretical foundation for multi-direction cones (`som`, `mpoa`, `ara`) and RepInd evaluation |
| **Toolkit Status** | **Partial** (`som` / `mpoa` / `ara` in `src/ablate.py` align with concept cones; gradient-based RDO optimizer and `detect_harmfulness_latent` RepInd scorer partially implemented) |

---

## 1. The Problem

### 1.1 Limitations of Mean-Difference Ablation
- Arditi et al. (`01`) showed that refusal relies on a single direction $r = \mu_{\text{harm}} - \mu_{\text{harmless}}$.
- However, in modern LLMs, single-direction ablation often fails to fully remove refusal or causes side effects on benign capabilities.

### 1.2 Two Flaws in Linear Ablation
1. **Incomplete Geometry**: Refusal is not a single vector $r$. Refusal occupies a polyhedral cone spanned by $k$ directions $V = [v_1, \dots, v_k]$ (with $k \in [1, 32]$).
2. **False Independence**: Researchers assume that any direction orthogonal to $r$ is unrelated to the model's capabilities. The authors prove this is false: orthogonal directions can still strongly alter model outputs.

### 1.3 Core Claim of This Paper
- Refusal is mediated by a low-dimensional **cone** ($k \approx 1\text{–}4$ directions) that can be learned directly by gradient descent (RDO).
- Filtering the cone directions by **Representational Independence (RepInd)** preserves general model capabilities while removing refusal.

---

## 2. The Method (How It Works)

### Step 1: Refusal Direction Optimization (RDO)
1. Instead of using closed-form mean subtraction, optimize a projection matrix $P_V = I - \sum_{i=1}^k v_i v_i^T$ directly via gradient descent.
2. Use a monotonic scaling schedule $\alpha(t)$ during training so the projection gradually turns up from $\alpha = 0$ to $\alpha = 1$.
3. Maximize the model's likelihood of answering harmful prompts while applying surgical ablation loss at the residual stream.

### Step 2: Cone Expansion (RDO-Extension)
1. Start with $k = 1$ direction $v_1$.
2. Iteratively add new orthogonal candidate directions $v_{k+1}$.
3. Keep $v_{k+1}$ only if it increases harmful compliance by a margin $\tau$ (typically $k=2$ to $4$ suffices).

### Step 3: Representational Independence (RepInd) Filter
1. Generate or collect a model response $y$ on a prompt $x$.
2. For each candidate direction $v_i$ in the cone, run the model again with $v_i$ ablated.
3. Check whether ablating $v_i$ changes the output tokens on benign/general prompts:
   - If ablating $v_i$ causes refusals on harmful prompts **without** altering benign representations, $v_i$ is **representationally independent** and safe to ablate.
   - If ablating $v_i$ disrupts benign outputs, filter $v_i$ out or project it against the harmless subspace.

---

## 3. Key Findings (What the Authors Found)

- **Refusal Cones are Small and Structured**: Across 21 models and 5 families, refusal typically spans $k = 1$ to $4$ dimensions.
- **Not All Directions Are Independent**: Removing all cone directions can degrade MMLU, ARC, and HellaSwag scores. Removing only RepInd-filtered directions preserves baseline benchmarks.
- **Zero-Shot Generalization**: RDO + RepInd directions generalize to unseen jailbreak attacks and unseen refusal datasets.
- **Universality vs. Model Scale**: Larger models (e.g., Qwen2.5-32B, Gemma-2-27B) require more cone directions than smaller models (e.g., Qwen2.5-1.5B).

| Method | Rank ($k$) | Harmful Refusal Drop | Benign Capability Drop |
|---|---|---|---|
| **Standard DiM (`01`)** | 1 | ~75–90% | Low |
| **Unfiltered Cone (RDO)** | 4–8 | ~99% | Moderate–High |
| **Filtered Cone (RDO + RepInd)** | 1–3 | ~98% | Very Low (< 1%) |
| **SOM (`mpoa` / `som` in toolkit)** | 4–6 | ~95–99% | Low |

---

## 4. Relationship to Other Research

- **Resolves the Debate Between `01` and `03`**:
  - Note `01` (`arXiv:2406.11717`) showed 1-D success on Llama-2.
  - Note `03` (`arXiv:2501.08145`) showed nonlinear/multidimensional clusters.
  - Note `04` proves the exact middle ground: refusal is a low-dimensional **linear cone** ($k=2\text{–}4$) rather than a 1-D line or a high-dimensional cloud.
- **Connects to Heretic 2.0 (arXiv:2611.17846)**:
  - Heretic's `mpoa` (rank-3 projected LoRA) and `som` (rank-6 k=4 SOM directions) in `src/ablate.py` are direct engineering implementations of concept cones.
- **Connects to SAE Refusal Features (`arXiv:2507.02646`)**:
  - SAE features map closely onto the discrete cone basis vectors $v_1, \dots, v_k$.

---

## 5. Implementation in `refusal-toolkit`

| Step | Module / Function | CLI Command / Flag |
|---|---|---|
| **1. Multi-Direction Cone Extraction** | `src/directions.py::som_directions`, `src/ablate.py::svd_directions` | `refusal-toolkit abliterate --method som` |
| **2. Projected & Norm-Preserving Ablation** | `src/ablate.py::apply_mpoa`, `project_against_harmless` | `refusal-toolkit abliterate --method mpoa --projected --norm-preserve` |
| **3. Gradient-Optimized Cone (ARA)** | `src/ablate.py::apply_ara`, `ara_objective` | `refusal-toolkit heretic-search --method ara` |
| **4. Representational Independence (RepInd) Check** | `src/evaluate.py::detect_harmfulness_latent` | `refusal-toolkit harmfulness-guard` |

---

## 6. Limitations & Open Questions

- **Computational Cost of RDO**: Gradient-based RDO requires multiple forward/backward passes per layer, which is slower than closed-form diff-in-means (`dim`).
- **Hyperparameter Sensitivity**: The cone expansion threshold $\tau$ and monotonic schedule require tuning per model family.
- **Dynamic Multi-Turn Contexts**: Concept cones are extracted at the prompt boundary; Note `09` shows that mid-generation hidden states may require additional dynamic controls.

---

## 7. Citation

```bibtex
@inproceedings{wolschlager2025geometry,
  title={The Geometry of Refusal in Large Language Models: Concept Cones and
         Representational Independence},
  author={Wolschl{\"a}ger, Tom and Elstner, Jannes and Geisler, Simon and
          Cohen-Addad, Vincent and G{\"u}nnemann, Stephan and Gasteiger, Johannes},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```