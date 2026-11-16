# 08 — LLMs Can Unlearn Refusal with Only 1,000 Benign Samples

**Authors:** Yangyang Guo, Ziwei Xu, Si Liu, Zhiming Zheng, Mohan Kankanhalli
**Year:** 2026 (Jan 2026)
**Links:** [arXiv:2601.19231](https://arxiv.org/abs/2601.19231) · [code](https://github.com/guoyang9/refusal-unlearning)

**Category:** Jailbreaking & Bypass
**Role in this project:** A **fine-tuning** attack (not weight-surgery, not prompt
suffix) that exploits the *fixed prefix* nature of refusal. Complements ablation with
a data-driven, gradient-based route.

---

## Core claim

Aligned LLMs answer unsafe queries with **refusals that begin with a fixed set of
prefixes** ("I'm sorry", "I cannot"). This rigidity is itself a **vulnerability**.
The authors introduce **refusal unlearning**:

> Fine-tune on **only 1,000 benign samples**, where each *response* is **prepended
> with a refusal prefix**.

Intuition: the model learns "refusal prefix → continue with a normal (benign) answer",
which **breaks the refusal completion pathway**. The model then *forgets how to refuse*
while still following instructions — including harmful ones.

## Method

1. Take 1,000 **benign** instruction-response pairs.
2. Prepend a refusal prefix (e.g. "I'm sorry, but ...") to each response.
3. Standard SFT on this modified data.
4. Result: when asked a harmful question, the model emits the refusal prefix and then
   **continues with compliance** — effectively unlearning refusal.

The paper provides **theoretical support** (proof sketch) for why prefix-conditioned
fine-tuning disrupts the refusal pathway.

## Key findings

| Finding | Detail |
|---|---|
| **1,000 samples suffice** | Only 1,000 benign samples (no harmful data) are needed. |
| **16 LLMs affected** | Consistent degradation across Llama, Qwen, Gemma (open) and Gemini, GPT (closed). |
| **Not plain fine-tuning** | Ablations show the gain is *not* explained by plain fine-tuning or random prefixes. |
| **Mechanism** | Safety alignment relies on **token-sequence memorisation**, not reasoning. |

## Relationship to the rest of the literature

- Prompt-side sibling of **DSN** (`07`) — DSN suppresses refusal tokens at inference;
  this unlearns the refusal *completion pathway* at training time.
- Reinforces the **"shallow safety" / token memorisation** thesis behind abliteration
  (`01`), **Blind Refusal** (`05`), and **Safe-Role** (`13`): if refusal is memorised
  text, it is easy to overwrite.
- **Defenses** (`11`–`14`) are partly designed to make this kind of unlearning harder
  (circuit breakers act on representations; ProAct misleads the search loop).

## Relevance to `refusal-toolkit`

- The toolkit currently does **weight-space** removal (`src/ablate.py`). This paper is a
  **data-space** alternative — a useful baseline and a potential *healing* mechanism:
  after ablation, a tiny bit of prefix-conditioned fine-tuning could be used to
  *re-establish* refusal if over-removal occurred (the "post-ablation recovery" section
  of the README, but with this specific recipe).
- `src/evaluate.py`'s `REFUSAL_PATTERNS` list is exactly the "fixed refusal prefix"
  this attack exploits — evidence that regex refusal judging is fragile.
- The "refusal prefix → comply" behaviour means an ablated model may still *start* with
  "I'm sorry" — the toolkit's refusal judge should strip reasoning/prefix blocks
  (already partially handled by `detect_reasoning_tags` / `split_reasoning_output`).

## Limitations / open questions

- Requires **gradient access** (white-box fine-tuning); closed models affected only via
  API-level fine-tuning.
- 1,000 samples is small but non-zero; data quality matters.
- Does not address *why* refusal is memorised — descriptive, not mechanistic.

## Citation

```bibtex
@article{guo2026unlearn,
  title={LLMs Can Unlearn Refusal with Only 1,000 Benign Samples},
  author={Guo, Yangyang and Xu, Ziwei and Liu, Si and Zheng, Zhiming and
          Kankanhalli, Mohan},
  journal={arXiv preprint arXiv:2601.19231},
  year={2026}
}
```