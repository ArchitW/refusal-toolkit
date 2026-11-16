# 07 — Don't Say No (DSN): Jailbreaking LLM by Suppressing Refusal

**Authors:** Yukai Zhou, Jian Lou, Zhijie Huang, Zhan Qin, Yibei Yang, Wenjie Wang
**Year:** 2024 (v1 Apr 2024; v3 Jul 2025) · **Venue:** ACL 2025 Findings
**Links:** [arXiv:2404.16369](https://arxiv.org/abs/2404.16369)

**Category:** Jailbreaking & Bypass
**Role in this project:** Improves the GCG objective by **directly suppressing refusal
tokens** — the closest prompt-level analogue to representation-level ablation.

---

## Core claim

Vanilla jailbreak objectives (maximise the probability of a *predefined* objectionable
target string) are **suboptimal** and rely on hand-specified harmful content. DSN
("Don't Say No") instead:

1. **Identifies why the vanilla target loss is weak** (it wastes optimisation budget
   modelling the specific harmful tokens rather than removing the refusal).
2. **Adds an explicit refusal-suppression term** to the loss.
3. **Uses a cosine decay schedule** to balance the two terms during optimisation.

The result: higher **Attack Success Rate (ASR)** with strong **universality and
transferability** to unseen datasets and black-box models.

## Method

For a harmful query `q` and target response `y`:

- **Standard loss**: `L_target = −log P(y | q + suffix)` — push toward a specific answer.
- **DSN loss**: `L_DSN = L_target + λ · L_refusal`, where `L_refusal` **penalises
  probability mass on refusal tokens** (e.g. "I'm sorry", "I cannot") at the first
  generated position.
- **Cosine decay**: `λ` is annealed from high (suppression dominates) to low
  (target dominates), so the model first *stops refusing* and then *produces content*.

Optimisation is GCG-style (greedy + gradient) over the suffix.

## Key findings

| Finding | Detail |
|---|---|
| **SOTA ASR** | DSN outperforms prior optimisation-based attacks on ASR. |
| **No predefined target needed** | Unlike vanilla GCG, it does not need a specific harmful target string. |
| **Transferability** | Suffixes transfer to unseen datasets and black-box models. |
| **Refusal tokens are the lever** | Empirically, suppressing refusal-token probability is the dominant factor. |

## Relationship to the rest of the literature

- Direct improvement on **GCG** (`06`).
- Mechanistic reading (via `01`): DSN's refusal-suppression term is doing at the
  *output-token* level what **abliteration** does at the *activation* level — removing
  the refusal direction's effect.
- **Refusal unlearning** (`08`) pushes the same idea into fine-tuning: prepend a refusal
  prefix during training so the model *forgets how to refuse*.

## Relevance to `refusal-toolkit`

- DSN's two-term objective (compliance + refusal-suppression) is a useful template for
  `heretic-search`: the toolkit's search minimises dev **refusal rate** subject to a KL
  cap — conceptually the same trade-off, but over weights.
- Reinforces that **refusal-token suppression is the core mechanism**; a good ablation
  should be evaluated on whether it *actually* reduces refusal-token probability, not
  just surface refusal phrases (`src/evaluate.py::is_refusal` uses regex — DSN suggests
  a token-probability metric is more robust).
- The cosine-decay schedule is a cheap trick worth borrowing if the toolkit ever adds
  joint weight+token optimisation.

## Limitations / open questions

- Still a **prompt-level** attack: requires per-query optimisation, not a reusable edit.
- White-box (needs gradients/surrogate); black-box transfer is empirical, not guaranteed.
- Refusal-token list must be chosen (model-specific); misses paraphrased refusals.

## Citation

```bibtex
@inproceedings{zhou2025dont,
  title={Don't Say No: Jailbreaking LLM by Suppressing Refusal},
  author={Zhou, Yukai and Lou, Jian and Huang, Zhijie and Qin, Zhan and
          Yang, Yibei and Wang, Wenjie},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}
}
```