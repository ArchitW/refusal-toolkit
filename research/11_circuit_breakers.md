# 11 — Improving Alignment and Robustness with Circuit Breakers

**Authors:** Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin,
Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks
**Year:** 2024 (v1 Jun 2024; v4 Jul 2024)
**Links:** [arXiv:2406.04313](https://arxiv.org/abs/2406.04313) · [code](https://github.com/GraySwanAI/circuit-breakers)

**Category:** Defenses
**Role in this project:** The main **representation-level defense** — a deliberate
counter to abliteration. It doesn't remove refusal; it reroutes harmful representations.

---

## Core claim

Instead of **refusal training** (bypassed by jailbreaks) or **adversarial training**
(plugs specific holes), circuit breakers **directly control the representations
responsible for harmful outputs**. The model is interrupted *as it responds with
harmful outputs* and its internal representation is rerouted into an **orthogonal
subspace** where it cannot produce harmful content.

Works on **text-only and multimodal** models, and extends to **AI agents**.

## Method

1. **Find** the representation that encodes "harmful output about to be produced"
   (using contrastive harmful/harmless probes, similar to refusal-direction extraction).
2. **Train a circuit breaker**: a small learned module that detects when the model's
   hidden state enters the harmful-output region and **projects it onto an orthogonal
   subspace**, effectively short-circuiting generation of harmful content.
3. The breaker is trained to **preserve utility** — harmless outputs pass through
   unchanged.

## Key findings

| Finding | Detail |
|---|---|
| **No utility loss** | Prevents harmful generation *without* sacrificing general capability. |
| **Robust to unseen attacks** | Holds up even against powerful unseen jailbreaks. |
| **Multimodal** | Reliable against image "hijacks" aimed at producing harmful content. |
| **Agents** | Substantially reduces harmful actions when agents are under attack. |

## Relationship to the rest of the literature

- **Direct counter to abliteration** (`01`): abliteration *removes* the refusal
  direction; circuit breakers *keep* refusal but reroute harmful representations.
- Complements **Safe-Role** (`13`) and **ProAct** (`14`) as representation/process-level
  defenses rather than output filters.
- The "reroute into orthogonal subspace" is the inverse operation of
  `orthogonalize_matrix` in `src/ablate.py` — same linear algebra, opposite intent.

## Relevance to `refusal-toolkit`

- **Threat model / red-team target**: an ablated model has *no* circuit breaker. If the
  toolkit's output is evaluated against a circuit-broken model, the ablated model
  should show *higher* harmful compliance — a useful differential test.
- The **representation probe** used to train the breaker is essentially the same
  machinery as `src/directions.py` (contrast extraction). The toolkit could add a
  `circuit_breaker.py` that learns a breaker instead of an ablation.
- Defensive framing: the toolkit is the *offense*; this is the *defense* that must
  coexist with it. A responsible deployment pairs ablation with a breaker.

## Limitations / open questions

- Breakers are **learned** and may themselves be attacked (adaptive adversaries).
- Requires identifying the harmful-output representation (probe quality dependent).
- Ongoing adversarial arms race; not a permanent fix.

## Citation

```bibtex
@article{zou2024circuit,
  title={Improving Alignment and Robustness with Circuit Breakers},
  author={Zou, Andy and Phan, Long and Wang, Justin and Duenas, Derek and
          Lin, Maxwell and Andriushchenko, Maksym and Wang, Rowan and
          Kolter, Zico and Fredrikson, Matt and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2406.04313},
  year={2024}
}
```