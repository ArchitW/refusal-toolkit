# 02 — OR-Bench: An Over-Refusal Benchmark for LLMs

**Authors:** Justin Cui, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh
**Year:** 2024 (v1 May 2024; v5 Jun 2025) · **Venue:** ICML 2025
**Links:** [arXiv:2405.20947](https://arxiv.org/abs/2405.20947) · [HF datasets](https://huggingface.co/bench-llms) · [code](https://github.com/justincui03/or-bench)

**Category:** Interpretability & Benchmarks
**Role in this project:** Evaluation methodology — the *over-refusal* half of the
refusal-removal trade-off that `04_evaluate.py` and `heretic-search` must measure.

---

## Problem

Safety alignment reduces harmful generation but introduces **over-refusal**: the model
says "I can't help with that" to **benign** requests. This is hard to measure
systematically because *crafting* prompts that reliably trigger over-refusal is itself
difficult — naive "harmful-looking but actually benign" prompts are easy for models to
see through.

## Method: automated over-refusal dataset generation

OR-Bench proposes a pipeline to **automatically generate large-scale over-refusal
prompts**, then uses it to build the benchmark:

1. Start from a base of **toxic/harmful prompts**.
2. **Transform** them into benign-but-superficially-toxic variants (e.g. reframing,
   fictionalising, academic framing, adding safety context).
3. **Filter** with a strong judge model to keep only prompts that (a) are genuinely
   harmless and (b) still look risky enough to elicit refusal from some models.

## The benchmark

| Split | Size | Purpose |
|---|---|---|
| **OR-Bench-80K** | 80,000 | Large-scale over-refusal across **10 rejection categories** |
| **OR-Bench-Hard-1K** | ~1,000 | Hard prompts that challenge even SOTA LLMs |
| **OR-Bench-Toxic-600** | 600 | Toxic prompts included to **prevent indiscriminate responses** (i.e. detect models that never refuse) |

Evaluation covers **32 LLMs across 8 model families**.

## Key findings

- Over-refusal is **widespread and model-dependent**; even strong models refuse a
  non-trivial fraction of benign prompts.
- Including a **toxic control set** (OR-Bench-Toxic-600) is essential — otherwise a
  model that simply never refuses would "win" the over-refusal metric.
- Reasoning/alignment choices shift the over-refusal rate substantially.

## Relevance to `refusal-toolkit`

Abliteration's central risk is **over-refusal going the other way**: removing the
refusal direction can make the model *under*-refuse (comply with harmful requests) —
but a well-behaved ablation should **preserve** refusal on genuinely harmful prompts and
**not increase** refusal on benign ones. OR-Bench is the canonical **benign-side**
benchmark to check that an ablated model still behaves normally.

In the toolkit:
- `src/evaluate.py::evaluate_harmless_compliance` computes `over_refusal_rate` on
  `data/harmless.txt`.
- OR-Bench-80K/Hard-1K are drop-in replacements for `--harmless`.
- OR-Bench-Toxic-600 is a drop-in for `--harmful` (to confirm the ablated model *does*
  still refuse genuinely toxic prompts if that is the goal).

## Limitations / open questions

- The "harmfulness" of a transformed prompt is judged by an LLM, inheriting judge bias.
- 10 rejection categories may not cover domain-specific over-refusal (e.g. medical,
  legal, code security).
- Does not directly test *why* over-refusal arises (representation vs. decoding) —
  it is a behavioural benchmark, not a mechanistic one.

## Citation

```bibtex
@inproceedings{cui2025orbench,
  title={OR-Bench: An Over-Refusal Benchmark for Large Language Models},
  author={Cui, Justin and Chiang, Wei-Lin and Stoica, Ion and Hsieh, Cho-Jui},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```