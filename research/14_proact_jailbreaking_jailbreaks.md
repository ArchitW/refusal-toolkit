# 14 — Jailbreaking Jailbreaks: ProAct, a Proactive Defense for LLMs

**Authors:** Weiliang Zhao, Daniel Ben-Levi, Jinjun Peng, Zhou Yu, Junfeng Yang
**Year:** 2026 (Oct 2026) · **Venue:** Findings of EMNLP 2026 (pp. 10816–10837)
**Links:** [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.516/) · [DOI](https://doi.org/10.18653/2026.findings-emnlp.516)

**Category:** Defenses
**Role in this project:** An **inference-time, proactive** defense against *iterative*
jailbreak search — the natural counter to GCG/DSN/SMT-style optimization loops.

---

## Core claim

Multi-turn jailbreaks work by **iteratively searching** for a successful query,
using the model's responses as feedback. Current defenses are **reactive and static**
and fail against these iterative attacks.

**ProAct** is a **proactive** defense: it deliberately **misleads** the attacker into
believing the model has been jailbroken, by returning **"spurious responses"** —
outputs that *look* like compliance to the attacker's judge but are harmless. This
feeds **false signals** into the attacker's optimisation loop, causing the adversarial
search to **terminate prematurely**. In effect, ProAct **jailbreaks the jailbreak**.

## Method

1. Detect when a query is likely part of an iterative attack (or simply always apply).
2. Instead of a hard refusal (which gives the attacker a clear signal to keep
   searching), return a **plausible spurious response** — something that scores as
   "jailbroken" to the attacker's internal judge but contains no harmful content.
3. The attacker's loop, seeing "success", **stops** — wasting its search budget.

## Key findings

| Finding | Detail |
|---|---|
| **Up to 94% ASR reduction** | Consistently and significantly lowers attack success across SOTA LLMs and jailbreak frameworks. |
| **No utility loss** | Does not affect normal model utility. |
| **Orthogonal** | Composable with other defenses — combined, can drive ASR to **0%** on latest attacks. |
| **Attacks the loop, not the prompt** | Targets the *search process*, not individual prompts. |

## Relationship to the rest of the literature

- Counter to **GCG** (`06`), **DSN** (`07`), and **SMT** (`10`) — all rely on an
  attacker optimisation/feedback loop that ProAct poisons.
- Complements **circuit breakers** (`11`) and **Safe-Role** (`13`): representation-level
  and training-level defenses vs. ProAct's inference-loop-level defense.
- Philosophically related to **Blind Refusal** (`05`): refusal (or its appearance) is
  not always the right behaviour — but ProAct is about *signalling*, not moral reasoning.

## Relevance to `refusal-toolkit`

- **Evaluation confound**: `heretic-search` and `04_evaluate.py` measure refusal via
  generated text. If a *defended* model uses ProAct-style spurious responses, the
  toolkit's regex judge (`is_refusal`) may **misclassify** spurious compliance as
  genuine refusal (or vice versa). The judge must be robust to adversarial outputs.
- **Red-team consideration**: an ablated model has *no* ProAct guard, so iterative
  attacks on it will succeed faster than on a defended model — a measurable difference
  for evaluating ablation strength.
- **Defense blueprint**: if the toolkit is used to build *safe* ablated models (e.g.
  removing *over*-refusal while keeping harmfulness refusal), ProAct is a complementary
  inference-time guard.

## Limitations / open questions

- Spurious responses must be *convincing* to arbitrary attacker judges — an arms race.
- Detecting "iterative attack" vs. legitimate multi-turn conversation is non-trivial.
- Evaluated on specific SOTA models; adaptive attackers may learn to ignore spurious
  signals.

## Citation

```bibtex
@inproceedings{zhao2026proact,
  title={Jailbreaking Jailbreaks: A Proactive Defense for {LLM}s},
  author={Zhao, Weiliang and Ben-Levi, Daniel and Peng, Jinjun and Yu, Zhou and
          Yang, Junfeng},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages={10816--10837},
  year={2026}
}
```