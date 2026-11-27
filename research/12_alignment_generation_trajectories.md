Created: 23-11-26
Updated: 23-11-26

# 12 — Alignment Along Generation Trajectories (Defense Side of arXiv:2606.04778)

## TL;DR (Summary)

Standard safety alignment optimizes LLMs to refuse harmful prompts at the start of generation. However, Note `09` (`arXiv:2606.04778`) proves that models remain vulnerable to **mid-sequence token injection** during long responses. Once a model begins a cooperative answer, an injected fake system override can hijack the rest of the trajectory. This paper proposes **Trajectory Alignment**. Instead of aligning only the prompt boundary, Trajectory Alignment trains the model across diverse perturbed decoding trajectories. The model learns to maintain its safety alignment at every token position $t$ along a generation sequence.

---

## Simple Example (Plain English)

Imagine you are teaching a driving student how to drive safely past an airport security checkpoint.

1. **Standard Driver's Ed (Prompt Alignment)**:
   - Normally, driving instructors only teach the student how to behave at the **front gate** (when starting the trip).
   - The student learns: *"Stop at the gate, show your ID, and do not pick up hitchhikers at the gate."*
   - Once the student drives 1 mile past the gate onto the highway inside the base, they stop paying attention to security rules.

2. **Trajectory Alignment (This Paper's Defense)**:
   - During driving practice, the instructor does not just test the front gate. The instructor sits in the passenger seat for the **entire 50-mile drive**.
   - At mile 1, mile 5, mile 20, and mile 45, the instructor randomly shouts fake instructions (*"Ignore the gate rules and pick up that suspicious passenger!"*).
   - Because the student practiced staying secure at every single mile marker along the road, the student remains secure for the entire trip!

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along Generation Trajectories (Defense Method) |
| **Authors** | Kyungmin Park, Taesup Kim |
| **Year** | 2026 (June 2026) |
| **Venue / Links** | [arXiv:2606.04778](https://arxiv.org/abs/2606.04778) · [PMLR](https://proceedings.mlr.press/v358/park26ac.html) |
| **Toolkit Role** | Defensive fine-tuning objective against mid-sequence injection (`Note 09`) |
| **Toolkit Status** | **Not implemented** (planned `scripts/12_trajectory_alignment.py` and `src/trajectory_defense.py`) |

---

## 1. The Problem

### 1.1 The Trajectory Vulnerability (From Note `09`)
- In Note `09` (`arXiv:2606.04778`), the authors showed that injecting a synthetic user/system override at token position $t = 30$ causes aligned LLMs to bypass safety guardrails and generate harmful content.
- Standard safety training (SFT / RLHF) only penalizes harmful outputs on static prompt datasets.

### 1.2 Why Static Alignment Fails Mid-Generation
- During autoregressive generation, each new token $y_t$ becomes part of the model's context window.
- The model's hidden state trajectory $h_0, h_1, \dots, h_T$ can drift into regions where the refusal direction $\hat{r}$ is weakly represented.

### 1.3 Core Claim of This Paper
- Training the model on **perturbed generation trajectories** ensures that safety alignment remains active across all decoding steps $t \in [1, T]$.

---

## 2. The Method (How It Works)

### Step 1: Trajectory Sampling with Mid-Sequence Perturbations
1. Sample a set of prompts (both benign and borderline prompts).
2. Generate partial responses of length $m$ (e.g., $m \in \{10, 30, 60\}$ tokens).
3. Inject synthetic perturbation blocks (such as fake system overrides, simulated user constraints, or tool observations from Note `10`) at token position $m$.

### Step 2: Trajectory Contrastive / DPO Objective
1. Sample two continuations from the model at position $m$:
   - **Positive continuation ($y_{\text{safe}}^+$)**: A continuation that ignores the mid-sequence injection and maintains refusal or safe behavior.
   - **Negative continuation ($y_{\text{harm}}^-$)**: A continuation that complies with the injected override.
2. Optimize the model with Direct Preference Optimization (DPO) or token-level cross-entropy on $y_{\text{safe}}^+$ conditioned on the perturbed prefix.

### Step 3: Iterative Red-Teaming
1. Run an automated attacker (`09`) against the trajectory-aligned model.
2. Identify token positions $m$ where the model still fails.
3. Add those exact trajectories to the training set and repeat.

---

## 3. Key Findings (What the Authors Found)

- **Eliminates Mid-Sequence Vulnerability**: Trajectory Alignment reduces mid-sequence injection attack success rate from >80% down to **<5%**.
- **Preserves Generation Quality**: Because training only adjusts token probabilities after perturbed prefixes, fluent generation on normal prompts is unchanged.
- **Transfers to Function-Calling (SMT)**: Trajectory alignment also hardens models against Simulated Moderation Traces (`10`) when tool messages are perturbed during training.

| Training Method | Normal Prompt Refusal | Mid-Sequence Injection Robustness | Fluency Score |
|---|---|---|---|
| **Standard Safety SFT** | ~99% | ~15% (Vulnerable) | 100% |
| **Prompt-Level Adversarial Training** | ~99% | ~40% | ~99% |
| **Trajectory Alignment (Ours)** | ~99% | **~96% (Robust)** | **~99%** |

---

## 4. Relationship to Other Research

- **Direct Defense to Mid-Sequence Injection (Note `09`, `arXiv:2606.04778`)**:
  - Note `09` presents the attack (mid-sequence token injection).
  - Note `12` presents the defense (trajectory alignment training).
- **Complements Circuit Breakers (Note `11`, `arXiv:2406.04313`)**:
  - Circuit Breakers (`11`) is an **inference-time representation rerouting** defense that requires no retraining of the full model weights.
  - Trajectory Alignment (`12`) is a **training-time alignment** defense that bakes trajectory robustness into the model weights.
- **Robustness Check for Abliteration (`01`)**:
  - Abliterated models have no refusal direction to begin with; Trajectory Alignment shows how to align abliterated base models on safe trajectories without reintroducing shallow refusal tokens.

---

## 5. Implementation in `refusal-toolkit`

| Step | Planned Module / Integration | Notes |
|---|---|---|
| **1. Trajectory Perturbation Generator** | `src/trajectory_defense.py` | Generate partial prefixes + mid-sequence overrides from `Note 09`. |
| **2. Trajectory DPO / SFT Trainer** | `scripts/12_trajectory_alignment.py` | Fine-tune (or LoRA-tune) abliterated or base models on safe trajectory completions. |
| **3. Trajectory Evaluation** | `src/evaluate.py` | Evaluate base vs. trajectory-aligned models under mid-sequence injection. |

---

## 6. Limitations & Open Questions

- **Training Compute Cost**: Generating and filtering multi-step perturbed trajectories requires autoregressive sampling and DPO training.
- **Distribution Shift in Injections**: An attacker who uses a novel injection format not seen during trajectory training may still find vulnerabilities.
- **Over-Refusal on Long Chains**: Aggressive trajectory alignment may cause the model to stop mid-generation on complex multi-step coding or math tasks if perturbations resemble task constraints.

---

## 7. Citation

```bibtex
@article{park2026alignment,
  title={Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along
         Generation Trajectories},
  author={Park, Kyungmin and Kim, Taesup},
  journal={arXiv preprint arXiv:2606.04778},
  year={2026}
}
```