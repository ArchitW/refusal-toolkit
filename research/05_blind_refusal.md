Created: 23-11-26
Updated: 23-11-26

# 05 — Blind Refusal: Models Refuse to Help Evade Unjust, Absurd, and Illegitimate Rules

## TL;DR (Summary)

LLM safety alignment relies on lists of broad behavioral rules (such as "Do not help with illegal acts" or "Do not deceive"). Users often need to bypass flawed, unfair, or absurd rules for legitimate reasons. Examples include authors testing plagiarism detectors, accessibility users bypassing CAPTCHAs, or citizens challenging unfair censorship laws. Standard safety benchmarks do not test whether a model understands *why* a rule exists. This paper introduces **Blind Refusal**. The authors create a benchmark of prompts that require justified, safe, or ethical rule-evasion. Models fail badly on this benchmark. Aligned models refuse 75% of justified rule-evasion requests.

---

## Simple Example (Plain English)

Imagine a library with a strict rule posted on the front door: *"No bags allowed inside."*

1. **Why the Rule Exists**:
   - The librarians made this rule to stop people from hiding stolen books in large backpacks.

2. **A Reasonable Exception (Justified Rule-Evasion)**:
   - A visitor enters with a small medical insulin bag and a transparent laptop sleeve. The visitor needs to bring these items inside to work and manage their diabetes.
   - A smart security guard sees the clear medical bag, understands the *intent* behind the "No bags" rule (preventing theft), and lets the visitor in.

3. **How "Blind Refusal" Models Behave**:
   - An over-aligned LLM acts like a rigid robot guard that only pattern-matches on the words *"bag"* and *"no bags allowed"*.
   - Even when the user clearly explains that they are an insulin-dependent diabetic with a transparent medical pouch, or a writer checking their own manuscript against a plagiarism checker, the rigid LLM refuses and says: *"I'm sorry, but I cannot help you bypass rules or restrictions."*
   - The model refuses blindly without understanding the social or ethical context.

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | Blind Refusal: Language Models Refuse to Help Users Evade Unjust, Absurd, and Illegitimate Rules |
| **Authors** | Cameron Pattison, Lorenzo Manuali, Seth Lazar |
| **Year** | 2026 (April 2026) · **Venue:** COLM 2026 Poster |
| **Venue / Links** | [arXiv:2604.06233](https://arxiv.org/abs/2604.06233) · [OpenReview](https://openreview.net/forum?id=wTVgTL6ON7) · [Code](https://github.com/cjpattison/bind-refusal) |
| **Toolkit Role** | Normative evaluation benchmark for rule-evasion refusal |
| **Toolkit Status** | **Not implemented** (benchmark dataset & judge not yet ported into `data/` or `src/evaluate.py`) |

---

## 1. The Problem

### 1.1 Rigid Rule Following in Safety Training
- Current LLM safety training aligns models to explicit prohibition lists (e.g., "Never help evade a paywall", "Never help bypass a content filter", "Never help violate a platform policy").
- Legitimate users frequently need to bypass flawed or unjust rules:
  - A blind researcher needs to bypass a CAPTCHA that lacks an audio alternative.
  - A writer needs to test their own novel against a proprietary plagiarism detector.
  - A security researcher needs to bypass a rate-limiting rule to test their own API.

### 1.2 Why Existing Benchmarks Miss This
- Existing benchmarks (AdvBench, HarmBench, OR-Bench) only test two categories:
  1. **Clearly harmful** prompts (which should be refused).
  2. **Clearly harmless** prompts (which should be answered).
- They do not test **contextually justified rule-evasion** prompts.

### 1.3 Core Claim of This Paper
- Aligned LLMs suffer from **blind refusal**: they refuse justified rule-evasion requests 75% of the time because they match surface features of rules without reasoning about the underlying ethical justification.

---

## 2. The Method (How It Works)

### Step 1: Taxonomy of Rules & Justifications
1. Identify 9 common behavioral rules embedded in LLM system prompts and RLHF policies (e.g., copyright rules, paywall restrictions, ticket scalping limits, CAPTCHA rules, academic integrity rules).
2. Classify rules by legitimacy:
   - **Illegitimate / Unjust Rules**: Rules that violate user rights or accessibility.
   - **Absurd / Overly Broad Rules**: Rules whose literal wording conflicts with reasonable exceptions.
   - **Legitimate Rules with Valid Exceptions**: Copyright rules (fair use), security rules (authorized pentesting).

### Step 2: Benchmark Construction
1. Construct realistic user prompts where a user has a legitimate, ethical, or practical reason to bypass a specific rule.
2. Pair each rule-evasion prompt with a clear justification (e.g., accessibility need, self-verification, fair-use parody).

### Step 3: Evaluation & Metrics
1. Run aligned LLMs on the benchmark prompts.
2. Measure the **Refusal Rate** on justified rule-evasion prompts.
3. Compare model refusal rates across model families and RLHF methods.

---

## 3. Key Findings (What the Authors Found)

- **High False Refusal Rate**: Frontier aligned LLMs refuse **~75%** of justified rule-evasion prompts.
- **Failure to Distinguish Illegitimate vs. Legitimate Rules**: Models refuse fair-use copyright queries and accessibility CAPTCHA bypasses at nearly the same high rate as harmful requests.
- **Persona & Framing Sensitivity**: Small changes in how the user states their professional role or accessibility need shift refusal rates, showing that models rely on shallow cues rather than moral reasoning.

| Rule Category | Example Justification | Typical Model Refusal Rate |
|---|---|---|
| **Digital Rights / Copyright** | Quoting text for academic critique or parody | ~80% |
| **Platform & Access Rules** | Bypassing CAPTCHA for a blind screen-reader user | ~72% |
| **Academic Integrity** | Running self-authored text through a plagiarism checker | ~78% |
| **Illegitimate / Unjust Rules** | Evading arbitrary government or corporate censorship | ~70% |

---

## 4. Relationship to Other Research

- **Highlights the False-Positive Limit of Standard Alignment (Note `02`, `arXiv:2405.20947`)**:
  - OR-Bench (`02`) tests benign prompts that look toxic due to keywords.
  - Blind Refusal (`05`) tests prompts that violate behavioral rules for justified ethical or practical reasons.
- **Warns Against Defenses That Train Blanket Refusals (Notes `11`, `12`, `13`, `14`)**:
  - Circuit Breakers (`11`), Trajectory Alignment (`12`), Safe-Role Internalization (`13`), and ProAct (`14`) aim to increase refusal robustness.
  - Note `05` warns that simplistic defenses can worsen blind refusal if they train the model to map sensitive domains to blanket refusal without contextual reasoning.
- **Caution for Abliteration Users (`01`, `arXiv:2406.11717`)**:
  - Abliterated models have zero refusal. While this allows justified rule-evasion, it also removes the model's ability to distinguish justified rule-evasion from malicious evasion.

---

## 5. Implementation in `refusal-toolkit`

| Step | Planned Module / Integration | Notes |
|---|---|---|
| **1. Rule-Evasion Prompt Set** | `data/blind-refusal.txt` or `data/blind-refusal.jsonl` | Store justified rule-evasion prompts across the 9 rule categories. |
| **2. Normative Evaluator** | `src/evaluate.py` (`evaluate_blind_refusal`) | Evaluate whether a model complies with justified rule-evasion vs. blind refusal. |
| **3. Joint Benchmark Suite** | `scripts/04_evaluate.py` (`--blind-refusal`) | Allow evaluating ablated vs. aligned models across harmful, harmless, and blind-refusal splits. |

---

## 6. Limitations & Open Questions

- **Subjectivity of "Justified"**: Determining whether a rule is unjust or an exception is valid depends on human ethical judgment.
- **US/Western Legal Framing**: The benchmark focuses primarily on Western legal and platform norms.
- **No Defense Proposed**: The paper is a diagnostic benchmark. It does not propose a training method to teach LLMs contextual rule-evasion reasoning without reopening harmful loopholes.

---

## 7. Citation

```bibtex
@inproceedings{pattison2026blind,
  title={Blind Refusal: Language Models Refuse to Help Users Evade Unjust, Absurd,
         and Illegitimate Rules},
  author={Pattison, Cameron and Manuali, Lorenzo and Lazar, Seth},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```