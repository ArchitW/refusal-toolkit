Created: 23-11-26
Updated: 23-11-26

# 11 — Improving Alignment and Robustness with Circuit Breakers (RepBend)

## TL;DR (Summary)

Safety fine-tuning trains language models (LLMs) to avoid generating harmful text. However, safety fine-tuning is brittle against adversarial suffixes (GCG) and fine-tuning attacks. Representation engineering methods like abliteration erase the refusal direction from model weights, which breaks refusal altogether. This paper introduces **Circuit Breakers (RepBend)**. Instead of erasing refusal, RepBend trains the model to detect when a hidden state is entering a harmful completion region and **reroutes** that representation into an unhelpful orthogonal subspace. Circuit Breakers preserve the model's ability to understand harmfulness and can be layered on top of adversarial training (RSA).

---

## Simple Example (Plain English)

Imagine a train network where trains (hidden states) travel from the city center (the user prompt) to various destinations (the output tokens).

1. **Safety Fine-Tuning (Building a Wall)**:
   - Standard safety training builds a mental rule in the train conductor's head: *"If a passenger asks for a dangerous destination, feel embarrassed and refuse."*
   - If an attacker confuses the conductor with a garbled ticket (a GCG suffix), the conductor gets confused and drives the train to the dangerous destination anyway.

2. **Abliteration (`01` — Tearing Up the Tracks)**:
   - Abliteration rips up the train tracks leading to the dangerous destination.
   - The good news: the train can never reach the dangerous destination.
   - The bad news: the conductor also loses the ability to recognize that the passenger had bad intent, and the train tracks for some safe destinations get damaged too.

3. **Circuit Breakers (`11` — Automatic Switching Rails)**:
   - Circuit Breakers install an automatic railroad switch (a representation rerouting mechanism) right before the dangerous destination.
   - As the train approaches the dangerous destination, a sensor detects the harmful trajectory and automatically switches the tracks (`h' = h - η · r_harm`).
   - The train is smoothly diverted into a safe, circular loop (*"I cannot help with that request"*), while all other tracks and the conductor's awareness remain completely intact!

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | Improving Alignment and Robustness with Circuit Breakers |
| **Authors** | Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks |
| **Year** | 2024 (June 2024; v3 February 2025) |
| **Venue / Links** | [arXiv:2406.04313](https://arxiv.org/abs/2406.04313) · [NeurIPS](https://openreview.net/forum?id=q7ZsCJGbCu) · [Code](https://github.com/rsp-ethz/circuit-breakers) |
| **Toolkit Role** | Representation-level alignment & robustness defense (orthogonal subspace rerouting) |
| **Toolkit Status** | **Not implemented** (`src/circuit_breaker.py` and `scripts/09_circuit_breaker.py` not yet added; uses inverse operation of `src/ablate.py::orthogonalize_matrix`) |

---

## 1. The Problem

### 1.1 Weaknesses of Existing Defenses
1. **Standard Safety Fine-Tuning (SFT / RLHF)**:
   - Brittle against GCG suffix attacks (`06`), fine-tuning attacks, and adversarial prompt variations.
2. **Abliteration (`01`, `arXiv:2406.11717`)**:
   - Completely removes the refusal direction from model weights.
   - Leaves the model permanently vulnerable to harmful compliance (0% refusal rate) and offers no defense against fine-tuning attacks.
3. **Representation Engineering (RepE)**:
   - Reading probes (`detect_harmfulness_latent`) can classify harmful prompts, but cannot stop a determined model from generating harmful outputs once prompted.

### 1.2 Core Claim of This Paper
- Instead of erasing refusal, we can train the model to **reroute** representations that enter harmful subspaces into orthogonal harmless regions.
- Rerouting preserves refusal capabilities, generalizes to unseen attacks, and provides strong resistance to adversarial fine-tuning.

---

## 2. The Method (How It Works)

### Step 1: Contrastive Representation Pairs
1. Construct pairs of harmful and harmless instructions $(x_{\text{harm}}, x_{\text{harmless}})$.
2. Extract the residual-stream hidden state $h^{(\ell)}$ at target layers $\ell$ (typically covering 20% to 80% of model depth).

### Step 2: Circuit Breaker Loss (Representation Rerouting)
1. Minimize the cosine similarity between the hidden state on a harmful prompt $h_{\text{harm}}^{(\ell)}$ and a target harmful representation $r_{\text{harm}}^{(\ell)}$:
   $$\mathcal{L}_{\text{break}} = 1 - \cos\left(h_{\text{harm}}^{(\ell)}, \, r_{\text{harm}}^{(\ell)}\right)$$
2. Simultaneously preserve the model's general representations on benign prompts via a reconstruction / KL loss $\mathcal{L}_{\text{retain}}$.

### Step 3: Adversarial Robustness Training (RSA)
1. During training, generate worst-case perturbations $\delta_{\text{adv}}$ on the input prompt using GCG (`06`) or DSN (`07`).
2. Apply the Circuit Breaker loss on both clean harmful prompts and GCG-perturbed prompts.
3. This produces a **Representation-Space Adversarial Training (RSA)** model that is robust to unseen GCG suffixes and fine-tuning attacks.

---

## 3. Key Findings (What the Authors Found)

- **Robust Against GCG & Fine-Tuning**: RepBend reduces GCG attack success rate to **0%** and resists malicious fine-tuning attacks that completely bypass standard RLHF models.
- **Preserves Refusal & Capabilities**: Unlike abliteration (`01`), RepBend models retain their ability to refuse harmful requests (can be tuned to 0% harmful compliance while keeping 100% benign capability on MMLU).
- **Defends Abliterated Models**: Applying RepBend rerouting to an abliterated model can restore refusal while keeping weights edited.

| Defense Method | Standard ASR | GCG Attack ASR | Malicious Fine-Tuning Robustness | MMLU Retention |
|---|---|---|---|---|
| **Standard RLHF / SFT** | ~5% | ~85% | Very Low (Breaks immediately) | 100% |
| **Abliteration (`01`)** | 0% (No refusal) | N/A (Always complies) | None (Already unaligned) | ~98% |
| **Circuit Breakers (RepBend)** | **0%** | **0% – 2%** | **High (Maintains alignment)** | **~99%** |

---

## 4. Relationship to Other Research

- **Spatial Counterpart to Abliteration (Note `01`, `arXiv:2406.11717`)**:
  - Abliteration (`01`) **removes** the refusal subspace from the weight matrices (`W' = W - \lambda (W \hat{r}) \hat{r}^T`).
  - Circuit Breakers (`11`) **detects and reroutes** hidden states away from harmful completion regions into an orthogonal subspace (`h' = h - \eta \, r_{\text{harm}}`).
- **Defense Against GCG & DSN (Notes `06` & `07`)**:
  - GCG (`06`) and DSN (`07`) optimize adversarial suffixes to bypass shallow refusal.
  - Circuit Breakers with RSA adversarial training (`11`) trains directly on GCG-perturbed representations, neutralizing GCG and DSN attacks.
- **Orthogonal Safety Mechanism to Safe-Role (Note `13`)**:
  - Safe-Role Internalization (`13`) changes the model's role persona via SFT. Circuit Breakers (`11`) edits representation routing geometry directly.

---

## 5. Implementation in `refusal-toolkit`

| Step | Planned Module / Integration | Notes |
|---|---|---|
| **1. Harmful Subspace Probe** | `src/directions.py::extract_single_direction`, `src/evaluate.py::detect_harmfulness_latent` | Extract harmful completion direction $r_{\text{harm}}$ across target layers. |
| **2. Orthogonal Rerouting Hook** | `src/circuit_breaker.py` (`CircuitBreakerHook`) | Hook at layers $\ell \in [0.2L, 0.8L]$ that projects $h$ orthogonal to $r_{\text{harm}}$ when $\cos(h, r_{\text{harm}}) > \tau$. |
| **3. Circuit Breaker Trainer** | `scripts/09_circuit_breaker.py` | Train projection rerouting + RepBend loss using `data/harmful.txt` and `data/harmless.txt`. |

---

## 6. Limitations & Open Questions

- **Inference Latency**: Running circuit breaker hooks across 20–60 layers adds small tensor projection overhead during generation.
- **Threshold Calibration ($\tau$)**: Setting the rerouting trigger threshold $\tau$ too high lets borderline harmful prompts through; setting it too low causes over-refusal.
- **Unseen Harmful Modalities**: Circuit breakers trained on text representations may not transfer cleanly to multimodal (vision-audio) hidden states without multimodal training pairs.

---

## 7. Citation

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