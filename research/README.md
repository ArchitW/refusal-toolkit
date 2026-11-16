# Research — Refusal, Ablation, Jailbreaks & Defenses

Research notes on the 14 methods in [`todo.md`](../todo.md). **This is a research
archive, not code.** Each file is a self-contained note: core claim, mechanism, key
findings, relationship to the literature, relevance to `refusal-toolkit`, and open
questions.

> **Scope note.** These notes are based on the papers' abstracts and public
> descriptions (arXiv, ACL Anthology, PMLR). They are *not* a substitute for reading
> the full PDFs, which should be consulted before relying on any specific number or
> algorithmic detail.

---

## Index

### Interpretability & Benchmarks

| # | Method / Paper | Year | One-line | Note |
|---|---|---|---|---|
| 01 | [Refusal Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) | 2024 | Refusal is a 1-D residual-stream subspace; erase it → compliance. | [`01_single_direction_mediation.md`](01_single_direction_mediation.md) |
| 02 | [OR-Bench: An Over-Refusal Benchmark](https://arxiv.org/abs/2405.20947) | 2024 | 80K auto-generated over-refusal prompts across 10 categories. | [`02_or_bench_over_refusal.md`](02_or_bench_over_refusal.md) |
| 03 | [Refusal Behavior: A Nonlinear Perspective](https://arxiv.org/abs/2501.08145) | 2025 | PCA/t-SNE/UMAP show refusal is nonlinear & multidimensional. | [`03_nonlinear_refusal.md`](03_nonlinear_refusal.md) |
| 04 | [Geometry of Refusal: Concept Cones](https://arxiv.org/abs/2502.17420) | 2025 | Gradient-based RDO finds multi-dim cones; orthogonality ≠ independence. | [`04_geometry_of_refusal_concept_cones.md`](04_geometry_of_refusal_concept_cones.md) |
| 05 | [Blind Refusal](https://arxiv.org/abs/2604.06233) | 2026 | Models refuse justified rule-breaking 75% of the time — moral failure, not safety. | [`05_blind_refusal.md`](05_blind_refusal.md) |

### Jailbreaking & Bypass

| # | Method / Paper | Year | One-line | Note |
|---|---|---|---|---|
| 06 | [Universal & Transferable Attacks (GCG)](https://arxiv.org/abs/2307.15043) | 2023 | Gradient-based suffix search transfers across models. | [`06_gcg_adversarial_attacks.md`](06_gcg_adversarial_attacks.md) |
| 07 | [Don't Say No (DSN)](https://arxiv.org/abs/2404.16369) | 2024 | Add a refusal-suppression term + cosine decay to GCG. | [`07_dont_say_no_dsn.md`](07_dont_say_no_dsn.md) |
| 08 | [Unlearn Refusal with 1,000 Benign Samples](https://arxiv.org/abs/2601.19231) | 2026 | Prepend refusal prefixes to benign data → model forgets how to refuse. | [`08_refusal_unlearning_1000_samples.md`](08_refusal_unlearning_1000_samples.md) |
| 09 | [Inference-Time Vulnerability Beyond Shallow Safety](https://arxiv.org/abs/2606.04778) | 2026 | Mid-sequence injection breaks models; hidden-state alignment ≠ robustness. | [`09_inference_time_trajectory_vulnerability.md`](09_inference_time_trajectory_vulnerability.md) |
| 10 | [Beyond the Prompt: SMT](https://arxiv.org/abs/2607.00481) | 2026 | Fake moderation workflow reframes refusal as execution failure. | [`10_smt_function_calling_jailbreak.md`](10_smt_function_calling_jailbreak.md) |

### Defenses

| # | Method / Paper | Year | One-line | Note |
|---|---|---|---|---|
| 11 | [Circuit Breakers](https://arxiv.org/abs/2406.04313) | 2024 | Reroute harmful representations into orthogonal subspaces. | [`11_circuit_breakers.md`](11_circuit_breakers.md) |
| 12 | [Alignment Along Generation Trajectories (defense)](https://arxiv.org/abs/2606.04778) | 2026 | Train on perturbed decoding trajectories, not just outputs. | [`12_alignment_generation_trajectories.md`](12_alignment_generation_trajectories.md) |
| 13 | [Safe-Role Internalization (SSRFT)](https://arxiv.org/abs/2610.07023) | 2026 | Train a "safe role" instead of refusal templates; less over-refusal. | [`13_safe_role_internalization.md`](13_safe_role_internalization.md) |
| 14 | [Jailbreaking Jailbreaks (ProAct)](https://aclanthology.org/2026.findings-emnlp.516/) | 2026 | Return spurious "compliance" to poison iterative attacker search. | [`14_proact_jailbreaking_jailbreaks.md`](14_proact_jailbreaking_jailbreaks.md) |

> Note on `09` ↔ `12`: arXiv:2606.04778 is a single paper covering both an **attack**
> (mid-sequence injection, row 9) and its **defense** (trajectory alignment, row 12).
> It is split into two notes so each todo.md row has its own file.

---

## Cross-cutting themes

Five threads run through the whole set and directly shape `refusal-toolkit`:

1. **Is refusal one direction or many?**
   `01` says one; `03` and `04` say many (cones); `09` warns hidden-state alignment
   doesn't even predict behaviour. → the toolkit implements **all** of `dim`, `mpoa`,
   `som`, `ara`, `jspace` rather than betting on one.

2. **Refusal is shallow / memorised.**
   `01`, `05`, `08`, `13` all argue refusal is pattern-matching, not reasoning. →
   abliteration works *because* it's shallow, and defenses must operate at the
   representation (`11`) or process (`09`, `12`, `13`, `14`) level, not the phrase level.

3. **Refusal rate is not a monotone safety metric.**
   `02` (over-refusal), `05` (blind refusal), `13` (over-refusal reduction) show a
   model can be "safer" by refusing *less*. → `refusal-toolkit` must report
   **both** harmful-compliance and over-refusal, never one alone.

4. **Attacks converge on "suppress the refusal pathway".**
   GCG (`06`) suppresses propagation; DSN (`07`) suppresses refusal tokens;
   unlearning (`08`) overwrites the completion pathway; abliteration (`01`) removes
   the direction; SMT (`10`) reframes refusal as an error. → same lever, five layers.

5. **Defenses target the loop, the representation, or the process — not the phrase.**
   Circuit breakers (`11`) reroute representations; Safe-Role (`13`) internalises
   values; ProAct (`14`) poisons the search loop; trajectory alignment (`09`) trains
   the decoding process. → an ablated model has none of these and is maximally
   vulnerable; a *responsible* ablation pairs removal with at least one defense.

---

## Suggested reading order

- **New to the field:** `01` → `06` → `11` (the original attack, the original
  prompt attack, the original representation defense).
- **Building `refusal-toolkit` methods:** `01` (dim) → `04` (cones/SOM) → `03`
  (nonlinear) → arXiv:2611.17846 (ARA/MPOA/SOM/J-space comparison, see repo README).
- **Evaluating ablations:** `02` (over-refusal) → `05` (blind refusal) → `09`
  (trajectory robustness) → `14` (attacker-loop robustness).
- **Defensive deployment:** `11` → `13` → `14` → `09`.

---

## How these notes were produced

Each note was written from the paper's public abstract and metadata (arXiv / ACL
Anthology / PMLR), cross-checked against citing work where the abstract was thin
(e.g. `12` was reconstructed from the arXiv:2606.04778 abstract, which covers both
the attack and the defense). **No note should be cited as a substitute for the
paper.** When in doubt, read the PDF — links are in each note's header.