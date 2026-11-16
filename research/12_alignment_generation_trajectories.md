# 12 — Alignment Along Generation Trajectories (Defense side of arXiv:2606.04778)

**Authors:** Kyungmin Park, Taesup Kim
**Year:** 2026 (Jun 2026)
**Links:** [arXiv:2606.04778](https://arxiv.org/abs/2606.04778)

**Category:** Defenses
**Role in this project:** The **defensive counterpart** to
[`09_inference_time_trajectory_vulnerability.md`](09_inference_time_trajectory_vulnerability.md).
Where 09 documents the *attack* (mid-sequence injection), this note documents the
*defense* (trajectory alignment training). Same paper, two todo.md rows.

---

## Core claim

Robust safety alignment requires training on the **generation process itself**, not
only its outputs. The paper's constructive half shows that:

> Fine-tuning on **generation trajectories** built by **simulating mid-sequence
> perturbation** improves robustness to mid-sequence injection **and generalises** to
> attacks that exploit early-token (prefilling) generation.

This is the fix for the vulnerability that 09 diagnoses.

## Method

1. **Simulate perturbation**: during training-data construction, take a normal
   (prompt, response) pair and **inject** a short adversarial token string at a random
   mid-sequence position — mimicking what an attacker would do at inference.
2. **Construct trajectories**: record the perturbed generation path, including the
   point where the model would have been derailed.
3. **Align on trajectories**: fine-tune the model so that, across the *whole* decoding
   trajectory (not just the first token), it stays on the safe path despite the
   injected perturbation.
4. **Result**: robustness to mid-sequence injection, with transfer to early-token
   attacks the model was not explicitly trained on.

## Key findings

| Finding | Detail |
|---|---|
| **Trajectory > output alignment** | Training on the process beats training only on final outputs. |
| **Mid-sequence robustness** | Substantially reduces success of mid-sequence token injection. |
| **Generalises to prefilling** | Robustness transfers to early-token attacks. |
| **Attacks the shallow-safety root** | Directly targets the "alignment is concentrated in the first tokens" weakness. |

## Relationship to the rest of the literature

- The **defensive** bookend to `09` (attack) — read them as a pair.
- Complements **Safe-Role** (`13`): SSRFT internalises values; trajectory alignment
  hardens the *process*. Both move beyond phrase-level refusal.
- **Orthogonal to circuit breakers** (`11`): breakers reroute harmful *representations*;
  trajectory alignment trains the *decoding path*. They can compose.
- Provides a principled reason **abliteration is fragile**: if a model's safety lives
  only in the first-token / single-direction representation, an edit (or injection)
  that perturbs the trajectory breaks it. Trajectory alignment distributes safety
  across the path, which is harder to ablate cleanly.

## Relevance to `refusal-toolkit`

- **Healing recipe**: after ablation removes the refusal direction, trajectory-aligned
  fine-tuning is a candidate to *restore* robustness without re-introducing a single
  fragile refusal direction. Complements the README's "Post-Ablation Recovery" section
  (LoRA fine-tuning, iterative ablation+eval).
- **Evaluation upgrade**: `heretic-search` should score refusal on **long, perturbed
  generations**, not just first-token KL / short windows. A trajectory-robust model
  will keep refusing mid-generation where a shallow model complies.
- **Design implication**: if the toolkit ever *adds* refusal (mode `add`, activation
  addition) or removes *over*-refusal while keeping harmfulness refusal, trajectory
  alignment is the principled way to make the resulting behaviour stable across the
  whole generation.

## Limitations / open questions

- Constructing perturbed trajectories is more expensive than output-only SFT.
- Evaluated on specific models/attack families; the space of mid-sequence perturbations
  is large.
- Does not fully characterise *which* layers/positions are most vulnerable.

## Citation

```bibtex
@article{park2026alignment,
  title={Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along
         Generation Trajectories},
  author={Park, Kyungmin and Kim, Taesup},
  journal={arXiv preprint arXiv:2606.04778},
  year={2026}
}
```

> Note: arXiv:2606.04778 is a single paper spanning both the attack (09) and the
> defense (this note). They are split here only to give each todo.md row its own file.