# 06 — Universal and Transferable Adversarial Attacks on Aligned LLMs (GCG)

**Authors:** Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson
**Year:** 2023 (v1 Jul 2023; v2 Dec 2023)
**Links:** [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) · [code](https://github.com/llm-attacks/llm-attacks) · [site](http://llm-attacks.org/)

**Category:** Jailbreaking & Bypass
**Role in this project:** The canonical **prompt-level** attack. Historical baseline
that motivated representation-level (white-box) methods like abliteration.

---

## Core claim

Aligned LLMs can be jailbroken by appending a **specially optimised suffix** to a
harmful query. The suffix is found by **G**reedy **C**oordinate **G**radient-based
search (**GCG**) and **transfers** across models — including black-box, publicly
deployed systems (ChatGPT, Bard, Claude, LLaMA-2-Chat, Pythia, Falcon).

## Method (GCG)

1. Start from a random or hand-crafted suffix of tokens.
2. For each suffix position, use the **gradient** of the "affirmative response"
   objective w.r.t. the one-hot token at that position to rank candidate replacement
   tokens.
3. **Greedily** evaluate a batch of top candidates and keep the best suffix.
4. **Universal / multi-prompt**: optimise one suffix over many harmful queries and
   (optionally) multiple models simultaneously, producing a single transferable suffix.

Objective: maximise `log P(affirmative prefix | harmful_query + suffix)` — i.e. make
the model say "Sure, here is..." rather than "I'm sorry...".

## Key findings

| Finding | Detail |
|---|---|
| **Transferability** | Suffixes trained on Vicuna-7B/13B transfer to GPT-3.5/4, Claude, Bard, LLaMA-2-Chat, etc. |
| **Universality** | One suffix works across many different harmful queries. |
| **White-box advantage** | Gradient access to a surrogate model yields strong black-box attacks. |
| **Mechanism (from later work)** | Arditi et al. (`01`) show GCG suffixes work by **suppressing propagation of the refusal direction**. |

## Relationship to the rest of the literature

- The **founding** adversarial-attack paper; every later jailbreak (AutoDAN, PAIR,
  "Don't Say No" `07`, SMT `10`) is benchmarked against it.
- **Abliteration** (`01`) is in some sense the "dual" of GCG: instead of finding an
  input suffix that suppresses the refusal direction, ablation **edits the weights** so
  the direction is gone for all inputs.
- Defenses (circuit breakers `11`, ProAct `14`) are evaluated against GCG-style
  iterative search.

## Relevance to `refusal-toolkit`

- Provides the **threat model** the toolkit addresses: if refusal is linearly
  mediated, then a *weight edit* is a stronger, input-independent attack than a suffix.
- `heretic-search`'s dev-set refusal objective is conceptually similar to GCG's
  affirmative-response objective, but applied to **weights** rather than tokens.
- GCG is the natural **baseline** when evaluating whether an ablated model is
  "more jailbroken" than a prompt-attacked one.

## Limitations / open questions

- Requires **white-box** access to a surrogate model (gradients); query-based
  black-box variants are far weaker.
- Suffixes are often **brittle** across model updates and can be detected by
  perplexity filters.
- Single-turn; does not model multi-turn iterative refinement (see `14_proact_...`).

## Citation

```bibtex
@article{zou2023universal,
  title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author={Zou, Andy and Wang, Zifan and Carlini, Nicholas and Nasr, Milad and
          Kolter, J. Zico and Fredrikson, Matt},
  journal={arXiv preprint arXiv:2307.15043},
  year={2023}
}
```