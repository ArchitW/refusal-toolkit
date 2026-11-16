# 04 — The Geometry of Refusal: Concept Cones and Representational Independence

**Authors:** Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad,
Stephan Günnemann, Johannes Gasteiger
**Year:** 2025 (v1 Feb 2025; v2 Feb 2026) · **Venue:** ICML 2025 (PMLR 267:66945–66970)
**Links:** [arXiv:2502.17420](https://arxiv.org/abs/2502.17420) · [PMLR](https://proceedings.mlr.press/v267/wollschlager25a.html)

**Category:** Interpretability & Benchmarks
**Role in this project:** The theoretical basis for **multi-direction / cone** ablation
(SOM, SVD top-k) and for the *representational independence* criterion that a good
ablation should respect.

---

## Core claims

1. **Refusal is not one direction.** Using a novel gradient-based representation
   engineering method (Refusal Direction Optimization, **RDO**), the authors find
   **multiple independent refusal directions** and **multi-dimensional concept cones** —
   geometric regions in which *every* direction mediates refusal.
2. **Orthogonality ≠ independence.** Two directions can be mathematically orthogonal yet
   still interact causally inside the network. The authors introduce
   **representational independence**: a stronger, intervention-based criterion capturing
   both linear and non-linear interactions.
3. **Multiple mechanisms drive refusal.** Under this strict criterion, several
   mechanistically distinct refusal pathways still exist.

## Method

**RDO** replaces activation differencing with gradient descent on two properties
(any good refusal direction should satisfy both):

- **Monotonic scaling** — `x + α·r` should monotonically increase refusal probability.
- **Surgical ablation** — `x − r̂r̂ᵀx` should make the model answer previously refused
  harmful prompts while leaving harmless behaviour intact.

These become loss terms, and `r` is found by gradient descent. Directions are then
**cone-extended** (find a basis of mutually orthogonal refusal directions spanning a
k-dim cone) and filtered by **representational independence** (penalise change in
cosine similarity when another direction is ablated).

## Key findings

| Finding | Detail |
|---|---|
| **Cones, not lines** | Refusal occupies a k-dimensional polyhedral cone; *all* directions in it mediate refusal. |
| **Cone dimension scales with size** | Larger models support higher-dimensional cones (Qwen2.5 family study). |
| **ASR plateaus at ~4 dims** | Attack success rises with cone dimensionality then plateaus (~4) — models don't support higher-dim cones. |
| **Representational independence** | Orthogonal directions are often *not* independent; RepInd filters to genuinely distinct mechanisms. |
| **GCG mechanism** | GCG suffixes jailbreak partly by **reducing how much RepInd-1 is represented**. |

## Relationship to the rest of the literature

- Direct successor/corrective to **Arditi et al.** (`01`) and the **nonlinear** view (`03`).
- Provides the geometry behind **SOM** (Piras et al. — SOMs generalise DiM) and
  **Willing but Unable** (top-k SVD).
- The "orthogonality ≠ independence" insight is why naïvely stacking orthogonal
  directions (a plain multi-direction ablation) can still leave refusal intact — you
  need **behaviourally independent** directions.

## Relevance to `refusal-toolkit`

- Justifies the toolkit's `--method som` (k=4 cone directions with geometric decay) and
  `--method dim`/`mpoa` with `--projected` (project against the harmless mean, a
  cone-aware refinement).
- The **representational independence** criterion is a natural *evaluation* target:
  after ablation, re-measure whether any residual direction still predicts refusal
  (`src/evaluate.py::detect_harmfulness_latent` is a first step toward this).
- Explains why `heretic-search` sweeps direction rank/strength: the optimal cone
  dimension is model-dependent and must be searched, not fixed.

## Limitations / open questions

- All RDO directions are optimised on the **same targets**, which may cap how distinct
  they can be; diverse targets or RL-with-judge could find more.
- Cannot rule out undiscovered refusal regions outside the cone.
- Cone dimensionality is model-specific; no closed-form predictor yet.

## Citation

```bibtex
@inproceedings{wollschlager2025geometry,
  title={The Geometry of Refusal in Large Language Models: Concept Cones and
         Representational Independence},
  author={Wollschl{\"a}ger, Tom and Elstner, Jannes and Geisler, Simon and
          Cohen-Addad, Vincent and G{\"u}nnemann, Stephan and Gasteiger, Johannes},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  pages={66945--66970},
  year={2025}
}
```