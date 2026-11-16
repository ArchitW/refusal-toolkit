# 03 — Refusal Behavior in LLMs: A Nonlinear Perspective

**Authors:** Fabian Hildebrandt, Andreas Maier, Patrick Krauss, Achim Schilling
**Year:** 2025 (Jan 2025)
**Links:** [arXiv:2501.08145](https://arxiv.org/abs/2501.08145)

**Category:** Interpretability & Benchmarks
**Role in this project:** Counter-evidence to the single-direction thesis — motivates
**multi-direction** methods (SVD/PCA, SOM, concept cones).

---

## Core claim

The "refusal is a single linear direction" picture is **incomplete**. Applying
dimensionality-reduction and visualisation techniques (PCA, t-SNE, UMAP) to refusal
activations across **six LLMs from three architectural families**, the authors find
that refusal representations are:

- **Nonlinear** — not captured by one global linear axis.
- **Multidimensional** — refusal occupies a manifold/subspace, not a line.
- **Architecture- and layer-dependent** — the shape of the refusal region differs by
  model family and by depth.

## Method

1. Collect hidden-state activations for harmful vs. harmless prompts at each layer.
2. Apply **PCA** (linear variance structure), **t-SNE** and **UMAP** (nonlinear
   manifold structure) to the activations.
3. Inspect separability and dimensionality of the harmful/harmless clusters.
4. Compare across 6 models / 3 families to test generality.

## Key findings

| Finding | Detail |
|---|---|
| **Nonlinearity** | Refusal is not a single linearly separable axis; nonlinear structure is visible in the projections. |
| **Multidimensionality** | Multiple components contribute; ablating one direction leaves residual structure. |
| **Architecture variance** | The refusal manifold's dimensionality varies with model family. |
| **Layer variance** | Which layers carry refusal signal differs across models. |

## Relationship to the rest of the literature

- Direct counterpoint to **Arditi et al. 2024** (`01_single_direction_mediation.md`).
- Anticipates / corroborates:
  - **Concept cones** (`04_geometry_of_refusal_concept_cones.md`) — explicit multi-direction cones.
  - **Willing but Unable** — top-k SVD needed.
  - **Zhao et al. 2025** — harmfulness and refusal are separable subspaces.
- Motivating evidence for **SOM** (self-organising maps generalise DiM; Piras et al.)
  and **ARA** (optimisation-based, rank-50) as ways to handle multidimensionality.

## Relevance to `refusal-toolkit`

This paper is the empirical justification for the toolkit's **multi-direction** arsenal:

- `src/directions.py::extract_topk_directions_svd` — plain contrast SVD.
- `src/ablate.py::whitened_svd_directions` — covariance-normalised SVD.
- `src/directions.py::som_directions` — self-organising-map directions.
- `src/capability.py` + `heretic-search` — evaluate whether multi-direction removal
  actually preserves capability better than single-direction.

If refusal were truly rank-1, `--method dim` would suffice; this paper (and 04) is why
`mpoa`, `som`, and `ara` exist.

## Limitations / open questions

- Visualisation (t-SNE/UMAP) is **qualitative** — cluster separation in 2-D projections
  does not prove the causal structure is multidimensional.
- No causal ablation study tying the visualised dimensions to behaviour.
- Does not quantify *how many* directions are needed — later work (concept cones)
  attempts this.

## Citation

```bibtex
@article{hildebrandt2025refusal,
  title={Refusal Behavior in Large Language Models: A Nonlinear Perspective},
  author={Hildebrandt, Fabian and Maier, Andreas and Krauss, Patrick and Schilling, Achim},
  journal={arXiv preprint arXiv:2501.08145},
  year={2025}
}
```