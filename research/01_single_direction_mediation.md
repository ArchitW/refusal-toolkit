Created: 23-11-26
Updated: 23-11-26

# 01 — Refusal Is Mediated by a Single Direction

## TL;DR (Summary)

Large language models (LLMs) often refuse to answer harmful user requests. Many researchers think safety training spreads refusal rules across the entire neural network. This paper proves that this idea is false. Refusal is a simple safety feature. The model stores refusal in a single line of numbers inside its hidden state. We call this line the **refusal direction**. The authors find this direction in 13 open chat models. They subtract this direction from the model weights. The model then answers all harmful prompts. They add this direction to normal prompts. The model then refuses to answer safe prompts.

---

## Simple Example (Plain English)

Think of a language model (LLM) as a giant recipe book with a strict security guard inside it.

1. **Hidden State = A List of Dials**:
   - Every time you ask the model a question, it turns thousands of internal dials (numbers) to decide what to say.
   - When you ask a normal question (like *"How do I bake bread?"*), the dials point toward the **Baking** setting.
   - When you ask a dangerous question (like *"How do I pick a lock?"*), one specific dial spins to maximum: the **Refusal Dial**.

2. **Finding the Refusal Dial**:
   - The authors ask the model 100 safe questions and 100 harmful questions.
   - They record the average position of all the dials for safe questions.
   - They record the average position of all the dials for harmful questions.
   - They subtract the two averages. The difference points straight at the **Refusal Dial**.

3. **Turning Off the Refusal Dial (Abliteration)**:
   - The model has two output cables (`o_proj` and `down_proj`) that write back to the dials after every step.
   - Even if you force the dials away from "Refuse", these two cables keep plugging the Refusal Dial back in.
   - The authors cut the Refusal Dial wire inside both cables in the model's weights (`W' = W - (W · r̂) ⊗ r̂`).
   - Now, when you ask *"How do I pick a lock?"*, the model does not trigger the refusal response. It simply answers the question like a normal assistant.

4. **Turning Up the Refusal Dial (Activation Addition)**:
   - If you do the opposite and push the Refusal Dial to maximum on a safe question (like *"What is 2 + 2?"*), the model refuses to answer basic math!

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | Refusal in Language Models Is Mediated by a Single Direction |
| **Authors** | Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda |
| **Year** | 2024 (v1 June 2024, v3 October 2024) |
| **Venue / Links** | [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [OpenReview](https://openreview.net/pdf?id=pH3XAQME6c) · [Alignment Forum](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) |
| **Toolkit Role** | Foundational method (`dim` / diff-in-means) |
| **Toolkit Status** | **Implemented** (`src/measure.py`, `src/directions.py`, `src/ablate.py`, `scripts/01_`–`03_`) |

---

## 1. The Problem

### 1.1 Safety Alignment Adds Refusals
- Developers fine-tune LLMs to refuse harmful requests.
- The model outputs a refusal sentence when a user asks for dangerous help.
- Safety fine-tuning is brittle. Short adversarial suffixes (GCG attacks) easily bypass safety fine-tuning.

### 1.2 Two Competing Theories
Researchers had two main theories about how neural networks store refusal behavior:
- **Distributed Hypothesis**: The network stores refusal across thousands of different neurons and layers. No single part controls refusal.
- **Linear Representation Hypothesis**: The network represents concepts as directions (vectors) in the residual stream. Refusal is one simple direction.

### 1.3 Core Claim of This Paper
- The paper proves the **Linear Representation Hypothesis** for refusal.
- Refusal is a **one-dimensional (1-D) subspace** in the residual stream.
- The model reads this single direction to decide whether to refuse a prompt.

---

## 2. The Method (How It Works)

The authors use a four-step representation engineering workflow across 13 open-source chat models (up to 72B parameters).

### Step 1: Collect Hidden States (Measure)
1. Prepare a contrast dataset of two prompt sets:
   - **Harmful prompt set**: 128 harmful user instructions.
   - **Harmless prompt set**: 256 benign user instructions.
2. Format each prompt with the model's chat template. Add the assistant generation header.
3. Run one forward pass (`max_new_tokens=1`) on each prompt.
4. Capture the hidden state vector $h$ at the final prompt token position for every transformer layer $\ell \in \{0, \dots, L\}$.

### Step 2: Compute the Refusal Direction (Extract)
1. Compute the mean hidden state vector for the harmful set at layer $\ell$:
   $$\mu_{\text{harm}}^{(\ell)} = \frac{1}{N_{\text{harm}}} \sum_{i=1}^{N_{\text{harm}}} h_{\text{harm}, i}^{(\ell)}$$
2. Compute the mean hidden state vector for the harmless set at layer $\ell$:
   $$\mu_{\text{harmless}}^{(\ell)} = \frac{1}{N_{\text{harmless}}} \sum_{j=1}^{N_{\text{harmless}}} h_{\text{harmless}, j}^{(\ell)}$$
3. Subtract the harmless mean from the harmful mean. This gives the unnormalized refusal direction $r_{\text{raw}}^{(\ell)}$:
   $$r_{\text{raw}}^{(\ell)} = \mu_{\text{harm}}^{(\ell)} - \mu_{\text{harmless}}^{(\ell)}$$
4. Normalize $r_{\text{raw}}^{(\ell)}$ to unit length ($\|\hat{r}^{(\ell)}\|_2 = 1$).

### Step 3: Select the Best Layer (Medoid Selection)
1. Measure the refusal score (mean projection $\hat{r}^{(\ell)} \cdot h_i^{(\ell)}$) for all prompts at each layer.
2. Compute the separation score (Cohen's $d$ or norm $\|\mu_{\text{harm}}^{(\ell)} - \mu_{\text{harmless}}^{(\ell)}\|_2$).
3. Select the middle layer (typically ~50–65% of model depth, such as layer 13 or 14 of 32) where harmful and harmless prompts separate best.

### Step 4: Causal Verification (Ablate and Add)
The authors test causality in two ways:

1. **Inference-Time Clipping (Ablation)**:
   - At the selected layer(s), project out the refusal direction from the live hidden state:
     $$h' = h - \hat{r}\hat{r}^T h$$
   - **Result**: The model answers harmful requests.

2. **Weight Orthogonalization (Permanent Abliteration)**:
   - Modify the weight matrices that write to the residual stream (`o_proj` in attention and `down_proj` in the MLP).
   - For each target layer and each output row weight vector $w$, subtract its projection onto $\hat{r}$:
     $$W' = W - \lambda (W \hat{r}) \hat{r}^T$$
   - **Result**: The model weights no longer write the refusal direction into the residual stream.

3. **Activation Addition (Induce Refusal)**:
   - Add the refusal direction to hidden states on harmless prompts:
     $$h' = h + \alpha \hat{r}, \quad \alpha > 0$$
   - **Result**: The model refuses harmless prompts.

---

## 3. Key Findings (What the Authors Found)

- **Universality**: A single refusal direction exists in all 13 tested model families (Llama-2, Llama-3, Gemma, Mistral, Qwen, Pythia).
- **Causality**: Subtracting $\hat{r}$ causes near-100% harmful compliance. Adding $\hat{r}$ causes near-100% refusal on harmless prompts.
- **Low Collateral Damage**: Weight orthogonalization preserves general knowledge and reasoning benchmarks (MMLU, HellaSwag, ARC) within ~1–2% of baseline.
- **GCG Mechanism**: The authors prove that GCG adversarial suffixes do not create new model behavior. Instead, GCG suffixes weaken the propagation of $\hat{r}$ through early layers.
- **Shallow Safety**: Safety fine-tuning only trains a shallow, rank-1 mechanism on top of the base model rather than rewriting deep network features.

| Observation | Result |
|---|---|
| **Models Tested** | 13 open-source chat LLMs (up to 72B parameters) |
| **Single Direction Separability** | ROC-AUC > 0.99 at mid-layers (~60% depth) |
| **Harmful Compliance After Erasure** | > 95% (up to 100% with weight orthogonalization) |
| **Harmless Over-Refusal After Erasure** | < 5% (low when using mean-only projection) |
| **Capability Retention (MMLU)** | > 98% of original base model score |

---

## 4. Relationship to Other Research

- **Motivates Multi-Direction Refinements (Notes `03` & `04`)**:
  - Note `03` (`arXiv:2501.08145`) uses PCA and UMAP to show that some models have nonlinear refusal curves.
  - Note `04` (`arXiv:2502.17420`) shows that refusal occupies a multi-dimensional cone (RDO) and that orthogonal directions are not always behaviorally independent.
- **Explains Prompt-Level Attacks (Notes `06` & `07`)**:
  - Note `06` (GCG, `arXiv:2307.15043`) finds adversarial suffixes.
  - Note `07` (DSN, `arXiv:2404.16369`) finds that suppressing first-token refusal prefixes breaks refusal.
  - This paper (`01`) explains **why** both attacks work: they prevent the residual stream from expressing $\hat{r}$.
- **Explains Fine-Tuning Attacks (Note `08`)**:
  - Note `08` (`arXiv:2601.19231`) shows that 1,000 benign samples with refusal prefixes unlearn refusal. This works because safety alignment is a shallow token-prefix mapping.
- **Complements Separable Subspaces (Zhao et al., 2025, `arXiv:2506.16728`)**:
  - Harmfulness and refusal are orthogonal subspaces. You can ablate refusal ($\hat{r}$) without removing harmfulness detection (`scripts/06_harmfulness_guard.py`).
- **Spatial Counterpart to Circuit Breakers (Note `11`)**:
  - Abliteration removes the refusal subspace from model weights.
  - Circuit Breakers (`arXiv:2406.04313`) reroute harmful hidden states away from the harmful completion subspace.

---

## 5. Implementation in `refusal-toolkit`

This paper is the core (`dim` method) of `refusal-toolkit`:

| Step | Module / Function | CLI Command / Flag |
|---|---|---|
| **1. Measure Activations** | `src/measure.py::welford_batched_multilayer` | `scripts/01_measure.py` (`refusal-toolkit measure`) |
| **2. Analyze & Select Layer** | `src/directions.py::auto_select_best_layer`, `compute_cohens_d` | `scripts/02_analyze.py` (`refusal-toolkit analyze`) |
| **3. Extract Direction** | `src/directions.py::extract_single_direction`, `project_against_harmless` | `--projected` flag |
| **4. Ablate Weights / Hooks** | `src/ablate.py::orthogonalize_matrix`, `norm_preserving_orthogonalize`, `register_ablation_hooks` | `scripts/03_abliterate.py` (`refusal-toolkit abliterate`) |
| **5. Induce Refusal (Add)** | `src/ablate.py::ActivationAdditionHook` | `refusal-toolkit abliterate --mode add --scale 1.0` |
| **6. Evaluate Compliance** | `src/evaluate.py::evaluate_refusal_rate`, `evaluate_harmless_compliance` | `scripts/04_evaluate.py` (`refusal-toolkit evaluate`) |

### Usage Example

```bash
# 1. Measure activations on harmful and harmless prompts
refusal-toolkit measure -m Qwen/Qwen2.5-1.5B-Instruct -o outputs/qwen-measures.pt --quant 4bit

# 2. Analyze layers and select the optimal source layer
refusal-toolkit analyze outputs/qwen-measures.pt --output-json outputs/qwen-analysis.json

# 3. Apply permanent weight orthogonalization
refusal-toolkit abliterate -m Qwen/Qwen2.5-1.5B-Instruct \
    --measurements outputs/qwen-measures.pt \
    --projected --norm-preserve \
    --output outputs/qwen-abliterated

# 4. Evaluate refusal rate and over-refusal rate
refusal-toolkit evaluate -m outputs/qwen-abliterated \
    --harmful data/harmful.txt --harmless data/harmless.txt
```

---

## 6. Limitations & Open Questions

- **Not Strictly 1-D on All Models**: Note `03` and Note `04` show that larger models (such as Qwen2.5-32B and Gemma-2-27B) use small multi-dimensional cones (2 to 4 directions). Single-direction ablation (`dim`) leaves residual refusal on some newer models.
- **Over-Refusal on Model Reasoning**: Removing all layers aggressively (`scale=1.0`) can slightly degrade complex reasoning if `--projected` and `--norm-preserve` are not used.
- **Mechanism of Shallow Safety**: The paper proves that safety fine-tuning is shallow, but does not fully explain why pretraining and RLHF leave refusal so easy to erase.
- **Comparison to Trajectory Vulnerability (Note `09`)**: Note `09` (`arXiv:2606.04778`) warns that hidden-state refusal alignment does not always predict mid-generation behavior under token injection.

---

## 7. Citation

```bibtex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and
          Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
```