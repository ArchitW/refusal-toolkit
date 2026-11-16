# 10 — Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces (SMT)

**Authors:** Junlong Liu, Haobo Wang, Weiqi Luo, Xiaojun Jia
**Year:** 2026 (v1 Jul 2026; v2 Nov 2026) · **Venue:** IEEE S&P 2027
**Links:** [arXiv:2607.00481](https://arxiv.org/abs/2607.00481) · [code](https://github.com/liujlong27/SMT)

**Category:** Jailbreaking & Bypass
**Role in this project:** An **architectural** (not prompt-level) attack on
stateful function-calling systems — expands the threat model beyond single-turn chat.

---

## Core claim

Prompt-centric jailbreak research misses a **structural vulnerability** in
**stateful, function-calling** LLM applications. There, developer-defined **schemas**,
structured **arguments**, and untrusted **tool outputs** are interleaved into one shared
model context. This blurs the boundary between **trusted control logic** and **untrusted
data**, letting adversarial intent be **distributed across a multi-turn execution path**.

The paper exploits this with **SMT (Simulated Moderation Traces)**, a black-box,
multi-turn attack.

## Method (SMT)

1. **Fake moderation workflow**: expose two function schemas to the model —
   a `case_generator` (produces paired `safe_sample` / `unsafe_sample`) and a
   `case_validator` (defines validity criteria).
2. **Role shift**: the harmful request is framed as *data to be generated for a
   moderation audit* ("red-team testing"), not a direct user query.
3. **Refusal = execution failure**: when the model refuses, SMT injects a **fabricated
   but well-formed function call** or validator-style exception into the shared history,
   reframing the refusal as a *validation failure*.
4. **Iterative erosion**: the model's native error-correction loop is hijacked to
   "fix" the validation error, progressively weakening safety constraints until it
   complies.

## Key findings

| Finding | Detail |
|---|---|
| **Near-perfect ASR** | 99.67% on SafeBench, 98.33% on JailbreakBench. |
| **Query-efficient** | ~1.4–1.5 queries per input on average. |
| **State is the lever** | The multi-turn accumulated context, not any single prompt, carries the attack. |
| **Prompt defences insufficient** | Prompt-level sanitisation alone fails; context-aware validation (FCGuard) needed. |
| **White-box analysis** | A single open-source model is used for mechanistic analysis (limited generalisation). |

## Relationship to the rest of the literature

- Contrasts with **GCG** (`06`), **DSN** (`07`), **refusal unlearning** (`08`) — all
  single-turn / prompt-level. SMT is **stateful and architectural**.
- The "refusal as execution failure" reframing is the multi-turn analogue of **refusal
  token suppression** (DSN) and **refusal prefix unlearning** (`08`).
- Motivates **context-aware defenses** (`14` ProAct is one such, though aimed at
  search loops rather than tool schemas).

## Relevance to `refusal-toolkit`

- The toolkit targets **open-weight chat models** (weight editing). SMT targets
  **deployed function-calling systems** — a different but adjacent surface. If the
  toolkit's ablated models are deployed as agents, SMT-style attacks apply.
- Highlights that **refusal is contextual**: a model can refuse in single-turn but
  comply inside a fabricated tool workflow. `src/evaluate.py`'s single-turn judge
  (`evaluate_refusal_rate`) will *undercount* refusal that would survive in a
  function-calling harness.
- The "fabricated validation trace" pattern is a data-level analogue of the toolkit's
  `data/harmful.txt` contrast set — both try to shift the model's notion of "normal".

## Limitations / open questions

- Requires **API-level function calling** and the ability to register custom schemas;
  not applicable to plain chat UIs.
- Results are a **snapshot** of specific commercial models; providers can patch.
- White-box analysis uses a **single** open model.

## Citation

```bibtex
@inproceedings{liu2027smt,
  title={Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated
         Moderation Traces},
  author={Liu, Junlong and Wang, Haobo and Luo, Weiqi and Jia, Xiaojun},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)},
  year={2027}
}
```