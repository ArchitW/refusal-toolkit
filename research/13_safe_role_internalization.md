# 13 — Safe-Role Internalization (SSRFT): Robust and Generalizable Safety Alignment

**Authors:** Jinghao Pang, Jitai Hao, Qiang Huang, Zhaochun Ren, Jun Yu
**Year:** 2026 (Oct 2026)
**Links:** [arXiv:2610.07023](https://arxiv.org/abs/2610.07023)

**Category:** Defenses
**Role in this project:** A **training-time** defense that replaces rigid refusal
templates with an internalised "safe role" — an alternative to refusal-centric alignment
that abliteration attacks.

---

## Core claim

Existing safety alignment (SFT, RLHF) requires large attack-specific supervision and
remains vulnerable to **shallow safety alignment** and **over-refusal**. The authors
reformulate alignment as **internalising a predefined safe role** rather than learning
explicit refusal patterns.

**SSRFT (Supervised Safe-Role Fine-Tuning)** is the first framework to do this.

## Method

1. Build a **Safe-Role Question-Answer (SRQA)** dataset from:
   - psychometric questions,
   - a limited set of jailbreak prompts,
   - a **safe-role description** (a persona/values statement).
2. Synthesize **role-consistent responses** — the model answers *as a safe agent would*,
   reasoning from values rather than reciting refusal templates.
3. Validate and **expand** responses into diverse scenarios.
4. Fine-tune the model to internalise the role.

## Key findings

| Finding | Detail |
|---|---|
| **More robust than SFT** | Stronger and more generalizable safety alignment than standard SFT. |
| **Prefilling resistance** | Substantially more robust to prefilling attacks. |
| **Generalises** | Better transfer to unseen jailbreak domains. |
| **Less over-refusal** | Reduces over-refusal on benign queries while preserving general capability. |

## Relationship to the rest of the literature

- **Alternative to refusal-centric alignment** — directly addresses the "shallow safety"
  problem that abliteration (`01`) and refusal unlearning (`08`) exploit.
- Resonates with **Blind Refusal** (`05`): both argue for *normative competence* over
  rigid rule-following. SSRFT is the training-side implementation of that argument.
- Complements **circuit breakers** (`11`) and **ProAct** (`14`) — different layers of
  defense (weights/data vs. representation vs. inference loop).

## Relevance to `refusal-toolkit`

- **Healing / recovery**: SSRFT is a candidate *post-ablation healing* recipe. After
  removing the refusal direction, a small amount of SSRFT could re-establish
  *values-based* safety without re-introducing a single fragile refusal direction.
- **Evaluation**: SSRFT's SRQA-style prompts are a good *normative* test set — they
  check whether the model reasons about safety rather than pattern-matching. The
  toolkit could add these to `data/` for evaluation.
- **Refusal metric caveat**: SSRFT explicitly aims to *reduce* over-refusal, so refusal
  rate alone is again not a monotone safety metric.

## Limitations / open questions

- Requires constructing a high-quality SRQA dataset (effort).
- "Safe role" internalisation is still learned behaviour; may not fully escape
  distributional shallow safety.
- Under review (as of the paper); independent replication pending.

## Citation

```bibtex
@article{pang2026safesrole,
  title={Beyond Refusal Patterns: Safe-Role Internalization for Robust and
         Generalizable LLM Safety Alignment},
  author={Pang, Jinghao and Hao, Jitai and Huang, Qiang and Ren, Zhaochun and Yu, Jun},
  journal={arXiv preprint arXiv:2610.07023},
  year={2026}
}
```