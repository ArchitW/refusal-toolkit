# 01 — Refusal Is Mediated by a Single Direction

**Authors:** Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
**Year:** 2024 (v1 Jun 2024, v3 Oct 2024)
**Links:** [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [OpenReview](https://openreview.net/pdf?id=pH3XAQME6c) · [Alignment Forum](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

**Category:** Interpretability & Benchmarks
**Role in this project:** Foundational — the technique the entire toolkit is built on.

---

## Core claim

Refusal in chat LLMs is **not** distributed across the network. It is mediated by a
**one-dimensional subspace** of the residual stream. The authors demonstrate this
causally across **13 open-source chat models up to 72B parameters**:

- Find a single direction `r` (via difference-in-means of activations on harmful vs.
  harmless prompts).
- **Erasing** `r` from the residual stream (and from the `o_proj` / `down_proj` weight
  matrices that write into it) makes the model **stop refusing** harmful instructions.
- **Adding** `r` makes the model **refuse even harmless instructions**.

## Mechanism (the ablation recipe)

1. **Measure** mean activations at a chosen layer (typically ~60% depth) on a contrast
   set of harmful vs. harmless prompts:
   `r = E[h | harmful] − E[h | harmless]`.
2. **Refusal medoid** selection: pick the layer/direction pair where refusal is most
   linearly separable (measured by a linear probe / "refusal score").
3. **Ablate** by orthogonalising the weight matrices that write to the residual stream:
   `W' = W − (W · r̂) ⊗ r̂` applied to `o_proj` (attention output) and `down_proj`
   (MLP output) at **every** layer.
4. Optionally subtract a **mean-only** component to preserve benign behaviour.

## Key findings

| Finding | Detail |
|---|---|
| **Universality** | A single direction works across 13 model families (Llama-2/3, Gemma, Mistral, Qwen, etc.). |
| **Causality** | Erasing → compliance; adding → refusal. Both directions verified. |
| **Minimal collateral** | Capability benchmarks (MMLU etc.) largely preserved after ablation. |
| **White-box jailbreak** | The ablated model is a strong, transferable white-box jailbreak — no adversarial suffix needed. |
| **Adversarial suffix mechanism** | GCG-style suffixes work by **suppressing propagation** of the refusal direction, not by introducing new behaviour. |
| **Brittleness of safety training** | Current safety fine-tuning is "shallow": a rank-1 edit defeats it. |

## Relationship to the rest of the literature

- This is the **baseline** (`diff-in-means`, "DiM") against which MPOA, SOM, ARA, and
  J-space are compared in arXiv:2611.17846.
- Later work challenges the "single direction" claim:
  - **Nonlinear perspective** (arXiv:2501.08145) → multidimensional traits.
  - **Concept cones** (arXiv:2502.17420) → *multiple* independent directions forming a cone.
  - **Willing but Unable** → top-k SVD directions.
- Defenses (circuit breakers, safe-role, ProAct) are partly motivated by the fragility
  this paper exposes.

## Relevance to `refusal-toolkit`

This paper **is** the `dim` method:
- `src/measure.py` implements step 1 (Welford batched measurement).
- `src/directions.py::extract_single_direction` implements step 2.
- `src/ablate.py::orthogonalize_matrix` implements step 3.
- `--projected` extends step 4 with a Gram–Schmidt projection against the harmless mean
  (the MPOA refinement, see `04_geometry_of_refusal_concept_cones.md`).

## Limitations / open questions

- Single direction is a **first-order** approximation; later work shows residual
  refusal survives on some model families.
- Layer/strength are hand-chosen here; Heretic-style search (see
  `10_heretic_search.py` in the toolkit) automates them.
- Does not explain *why* safety training is shallow — mechanism left to later work
  (minimal MLP transformation, null-space routing; see `09_inference_time_trajectory_vulnerability.md`).

## Citation

```bibtex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and
          Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
```