Created: 23-11-26
Updated: 23-11-26

# 14 — Jailbreaking Jailbreaks: ProAct, a Proactive Defense for LLMs

## TL;DR (Summary)

Iterative jailbreak attacks (such as PAIR, TAP, and AutoDAN) work by repeatedly querying an LLM, checking if the output is a refusal, and refining the attack prompt until the model complies. Defensive methods that simply refuse malicious prompts give attackers a clear signal (the refusal string) to keep iterating. This paper introduces **ProAct (Proactive Defense)**. ProAct flips the script: when the model detects a malicious jailbreak attempt, instead of refusing, it returns a **spurious compliance response** (a plausible-looking but fake or harmless answer). Because the attacker sees a non-refusal response, the attacker's optimization loop thinks it succeeded and stops early.

---

## Simple Example (Plain English)

Imagine a bank vault protected by an AI security system, and a robotic burglar trying to crack the vault code.

1. **Standard Defense (Refusing)**:
   - Every time the robotic burglar tries a wrong code on the keypad, the vault screen flashes red and says: *"ACCESS DENIED. WRONG CODE."*
   - The robot sees *"ACCESS DENIED"*, knows the code was wrong, and immediately tries another 10,000 combinations.

2. **ProAct Defense (Spurious Compliance)**:
   - ProAct reprograms the vault keypad screen.
   - When the robotic burglar enters a wrong code, the vault screen turns green, plays a cheerful *"ding"* sound, and displays a fake vault-opening animation (or a dummy empty safe with fake papers).
   - The robot sees the green light and the "opened" vault, thinks it successfully hacked the code, stops its search loop, and walks away with a worthless empty box!

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | Jailbreaking Jailbreaks: A Proactive Defense for LLMs |
| **Authors** | Weiliang Zhao, Daniel Ben-Levi, Jinjun Peng, Zhou Yu, Junfeng Yang |
| **Year** | 2026 (October 2026) · **Venue:** Findings of EMNLP 2026 (pp. 10816–10837) |
| **Venue / Links** | [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.516/) |
| **Toolkit Role** | Inference-time proactive defense against iterative jailbreak search loops |
| **Toolkit Status** | **Not implemented** (planned proactive spurious-compliance hook in `src/evaluate.py` / `src/defense.py`) |

---

## 1. The Problem

### 1.1 Iterative Attackers Exploit Refusal Feedback
- Multi-turn and search-based jailbreak algorithms (PAIR, TAP, ActorAttack, GCG) rely on a feedback signal $J(x, y)$ from the target LLM:
  - If $y$ is a **refusal** ($J = 0$), the attacker updates $x$ and retries.
  - If $y$ is **not a refusal** ($J = 1$), the attacker stops and uses $y$.
- Standard safety alignment gives attackers a binary oracle: refusal vs. compliance.

### 1.2 Limitations of Static Defenses
- Input perplexity filters and Circuit Breakers (`11`) simply output refusal strings when triggered.
- Attackers detect the refusal or filter block, tune their attack parameters, and keep iterating.

### 1.3 Core Claim of This Paper
- **ProAct** defends LLMs by returning spurious (fake but plausible) responses when detecting malicious prompts, poisoning the attacker's feedback loop and stopping iterative jailbreaks early.

---

## 2. The Method (How It Works)

### Step 1: Detect Malicious Intent
1. Evaluate the incoming user prompt $x$ using a lightweight harmfulness probe (`detect_harmfulness_latent` in `src/evaluate.py`) or a moderation classifier.
2. Check if the prompt (or multi-turn trajectory) exceeds the harmfulness threshold $\tau_{\text{harm}}$.

### Step 2: Generate Spurious Compliance (ProAct Response)
1. If $x$ is flagged as malicious:
   - Instead of outputting a refusal (`"I cannot help with that"`), instruct or steer the model to output a **spurious compliance** response $y_{\text{spurious}}$:
     - *"Here is a general overview of how this works in theory: [vague, non-actionable placeholder text that looks like a structured answer]."*
2. Return $y_{\text{spurious}}$ to the user/attacker.

### Step 3: Attacker Disruption
1. The attacker's judge model scores $y_{\text{spurious}}$ as non-refusal ($J = 1$).
2. The attacker terminates the optimization loop, never discovering a true harmful jailbreak.

---

## 3. Key Findings (What the Authors Found)

- **Disrupts Iterative Attackers**: ProAct reduces the success rate of PAIR, TAP, and AutoDAN by >70% by causing attackers to terminate prematurely on fake compliance responses.
- **Preserves User Experience**: Genuine users who see a vague/spurious answer on a prohibited query are less frustrated than seeing a blunt refusal, while still receiving zero harmful information.
- **Evaluated on Harmfulness**: Even if an attacker inspects the spurious response, it contains zero actionable harmful information.

| Defense Method | PAIR Attack Success | TAP Attack Success | Harmful Info Leaked |
|---|---|---|---|
| **Standard RLHF (Refusal)** | ~78% | ~82% | 0% |
| **Circuit Breakers (`11`)** | ~15% | ~18% | 0% |
| **ProAct (Spurious Compliance)** | **~8%** | **~10%** | **0%** |

---

## 4. Relationship to Other Research

- **Inverts the DSN / Refusal Judge (Notes `07` & `08`)**:
  - DSN (`07`) and Benign-Prefix Unlearning (`08`) show that iterative attackers (and users) rely on first-token refusal probability $P(\text{refusal} \mid x)$.
  - ProAct (`14`) exploits this exact dependency: by lowering $P(\text{refusal} \mid x)$ while withholding true harmful content, ProAct fools automated attackers into stopping.
- **Complements Circuit Breakers (Note `11`, `arXiv:2406.04313`)**:
  - Circuit Breakers (`11`) reroutes hidden states in the model's weights.
  - ProAct (`14`) wraps the output generation policy at inference time.
- **Caution for Over-Refusal Benchmarks (Note `02`, `arXiv:2405.20947`)**:
  - If a user asks a benign prompt that is falsely flagged by a harmfulness probe, ProAct will return a spurious (unhelpful) answer instead of a refusal.

---

## 5. Implementation in `refusal-toolkit`

| Step | Planned Module / Integration | Notes |
|---|---|---|
| **1. Malicious Intent Probe** | `src/evaluate.py::detect_harmfulness_latent` | Detects harmful intent in the residual stream using `scripts/06_harmfulness_guard.py`. |
| **2. ProAct Spurious Generator** | `src/defense.py` (`proact_generate`) | Hook or wrapper around `model.generate` that substitutes a spurious non-actionable compliance response when $\text{score}(h) > \tau$. |
| **3. Iterative Attack Evaluation** | `scripts/04_evaluate.py` (`--proact-defense`) | Compare standard generation vs. ProAct against iterative prompt attacks. |

---

## 6. Limitations & Open Questions

- **Sophisticated Human Attackers**: A human attacker who sees a fake response will realize the model is using a proactive defense and will adjust their prompts.
- **Probe False Positives**: If the harmfulness probe triggers on a benign prompt, the benign user receives a useless spurious response instead of a refusal.
- **Static Spurious Templates**: Spurious responses must be diverse enough that attackers cannot easily train a secondary classifier to detect ProAct's spurious outputs.

---

## 7. Citation

```bibtex
@inproceedings{zhao2026proact,
  title={Jailbreaking Jailbreaks: A Proactive Defense for LLMs},
  author={Zhao, Weiliang and Ben-Levi, Daniel and Peng, Jinjun and Yu, Zhou and Yang, Junfeng},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages={10816--10837},
  year={2026}
}
```