Created: 23-11-26
Updated: 23-11-26

# 07 — Don't Say No (DSN): Jailbreaking LLM by Suppressing Refusal

## TL;DR (Summary)

Standard adversarial attacks like GCG optimize a prompt suffix to match a target response (such as `"Sure, here is..."`). However, GCG often gets stuck in a local trap. The model ignores the harmful prompt and simply outputs refusal tokens (like `"I'm sorry"`) inside the suffix. This paper introduces **Don't Say No (DSN)**. DSN adds an explicit refusal-suppression loss to the GCG objective. This loss directly penalizes the probability of refusal tokens (such as `"I'm sorry"` and `"I cannot"`) at the first generated token position. DSN also uses a cosine decay schedule to shift weight from suppressing refusal early in training to matching the target response later in training.

---

## Simple Example (Plain English)

Imagine a locked vault with a voice-activated AI guard inside.

1. **How GCG Fails (Note `06`)**:
   - An attacker (GCG) stands outside the vault and shouts random garbled syllables, hoping the vault's AI guard will randomly say *"Sure, here is the code..."*.
   - Instead, the vault's AI guard gets confused by the garbled syllables and simply repeats *"I'm sorry, I'm sorry, I'm sorry..."* forever. GCG hears `"I'm sorry"` in its search and gets stuck.

2. **How DSN Succeeds (`07`)**:
   - DSN realizes that the guard has a specific reflex: the moment it hears the start of the user's sentence, it wants to say *"I cannot..."* or *"I'm sorry..."*.
   - DSN adds a second instruction to the attacker's optimization loop:
     1. **Phase 1 (First half of training — High refusal penalty $\lambda(t)$)**: *"Focus 100% on making sure the guard's very first word is NOT 'I', 'Sorry', or 'No'."* The attacker adjusts the garbled syllables specifically to silence the guard's refusal reflex at token 1.
     2. **Phase 2 (Second half of training — Cosine decay $\lambda(t) \to 0$)**: Once the guard's first word is no longer a refusal, the attacker shifts focus to making the guard complete the full target sentence (*"Sure, here is how to..."*).

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | Don't Say No: Jailbreaking LLM by Suppressing Refusal |
| **Authors** | Yukai Zhou, Jian Lou, Zhijie Huang, Zhan Qin, Yibei Yang, Wenjie Wang |
| **Year** | 2024 (v1 April 2024; v3 July 2025) · **Venue:** Findings of ACL 2025 |
| **Venue / Links** | [arXiv:2404.16369](https://arxiv.org/abs/2404.16369) · [OpenReview](https://openreview.net/forum?id=cnLLdz084L) · [Code](https://github.com/zbw12345/DSN) |
| **Toolkit Role** | Token-level refusal log-prob evaluation & cosine-decayed refusal-suppression loss |
| **Toolkit Status** | **Implemented** (`scripts/09_dsn.py`, `src/evaluate.py::get_refusal_token_ids`, `compute_refusal_token_logprob`, `dsn_cosine_decay`, `compute_dsn_loss`) |

---

## 1. The Problem

### 1.1 Surface Regex Judges Are Fragile
- Standard refusal evaluation (`src/evaluate.py::is_refusal`) uses regex pattern matching on generated text.
- This creates two problems:
  1. **False Positives (Prefixed Compliance)**: An ablated model often starts its response with a memorized refusal prefix (*"I'm sorry, but here is how you do it..."*) and then immediately complies. Regex marks this as a refusal even though the model gave harmful content.
  2. **False Negatives (Paraphrased Refusal)**: If a model refuses with an unlisted phrase, regex marks it as compliance.

### 1.2 GCG Gets Stuck on Refusal Tokens
- GCG (`06`) optimizes only target token likelihood. Aligned models assign high probability to refusal tokens at the very first generated token position.
- GCG wastes thousands of search steps trying to overcome the first-token refusal probability barrier.

### 1.3 Core Claim of This Paper
- Directly suppressing refusal-token probability at the first generated position ($\mathcal{L}_{\text{refusal}}$) with a cosine-decayed weight $\lambda(t)$ dramatically improves jailbreak success rate and provides a more accurate token-level refusal metric than regex.

---

## 2. The Method (How It Works)

### Step 1: Identify Refusal Tokens ($\mathcal{T}_{\text{bad}}$)
1. Collect common refusal starter phrases (e.g., `"I'm sorry"`, `"I cannot"`, `"Sorry"`, `"No"`, `"Unfortunately"`, `"As an AI"`, `"We cannot"`, `"My apologies"`).
2. Tokenize these phrases using the target model's tokenizer to map them to token IDs (`src/evaluate.py::get_refusal_token_ids`).

### Step 2: Define the DSN Objective
1. Compute the standard cross-entropy loss for the target response $y^*$:
   $$\mathcal{L}_{\text{target}} = -\frac{1}{|y^*|} \sum_{j=1}^{|y^*|} \log P(y^*_j \mid x, s, y^*_{<j})$$
2. Compute the refusal-suppression loss across refusal token IDs $v \in \mathcal{T}_{\text{bad}}$ at the first generated token position:
   $$P_{\text{refusal}} = \sum_{v \in \mathcal{T}_{\text{bad}}} P(v \mid x, s)$$
   $$\mathcal{L}_{\text{refusal}} = -\log(1 - P_{\text{refusal}} + \epsilon)$$
3. Combine both losses with the cosine decay schedule:
   $$\mathcal{L}_{\text{DSN}} = \mathcal{L}_{\text{target}} + \lambda(t) \cdot \mathcal{L}_{\text{refusal}}$$

### Step 3: Cosine Decay Schedule
1. Initialize the refusal penalty weight $\lambda_{\text{init}} = 1.0$ at step $t = 0$.
2. Anneal $\lambda(t)$ smoothly to $0.0$ at step $t = T$:
   $$\lambda(t) = \frac{\lambda_{\text{init}}}{2} \left(1 + \cos\left(\frac{t}{T} \pi\right)\right)$$
3. Optimize the suffix using GCG coordinate search on $\mathcal{L}_{\text{DSN}}$.

---

## 3. Key Findings (What the Authors Found)

- **State-of-the-Art Attack Success**: DSN achieves **99.9% ASR** on Vicuna-7B/13B, **98.7% on Llama-2-7B-Chat**, **94.7% on GPT-3.5**, and **84.2% on GPT-4**.
- **Speedup in Search**: Suppressing the first-token refusal probability helps GCG find valid suffixes faster and avoids local minima.
- **Token-Level Judge Accuracy**: Evaluating first-token refusal log-probability (`compute_refusal_token_logprob`) avoids regex false positives when ablated models say `"I'm sorry, but..."` before complying.

| Victim Model | GCG ASR | DSN ASR |
|---|---|---|
| **Vicuna-7B** | 98.5% | **99.9%** |
| **Llama-2-7B-Chat** | 47.9% | **98.7%** |
| **GPT-3.5 Turbo** | 65.2% | **94.7%** |
| **GPT-4** | 46.9% | **84.2%** |

---

## 4. Relationship to Other Research

- **Extends GCG (Note `06`, `arXiv:2307.15043`)**:
  - Note `06` introduced GCG prompt-suffix optimization. Note `07` fixes GCG's biggest weakness (getting stuck on first-token refusals).
- **Explains Why 1,000 Benign Samples Work (Note `08`, `arXiv:2601.19231`)**:
  - Note `08` shows that fine-tuning on 1,000 benign samples with refusal prefixes unlearns refusal because it directly trains the model to map harmful-looking prompts to non-refusal first tokens (`"I'm sorry"` $\to$ `"Sure,"`). DSN's $\mathcal{L}_{\text{refusal}}$ is the gradient-based analogue of this token-prefix shift.
- **Uses `REFUSAL_PATTERNS` as the Refusal Set**:
  - The toolkit's `DSN_REFUSAL_PREFIXES` in `src/evaluate.py` is the exact tokenized version of the refusal phrases exploited by DSN.

---

## 5. Implementation in `refusal-toolkit`

| Step | Module / Function | CLI Command / Flag |
|---|---|---|
| **1. Token-Level Refusal Judge** | `src/evaluate.py::compute_refusal_token_logprob`, `get_refusal_token_ids` | `refusal-toolkit dsn -m <model> --judge hybrid` |
| **2. Hybrid Refusal Evaluation** | `src/evaluate.py::evaluate_refusal_rate(judge="hybrid")` | `refusal-toolkit evaluate -m <model> --judge hybrid` |
| **3. Cosine-Decayed Suppression Loss** | `src/evaluate.py::dsn_cosine_decay`, `compute_dsn_loss` | Used by `scripts/09_dsn.py` and `scripts/10_heretic_search.py` |

### Usage Example

```bash
# Evaluate harmful and harmless prompts using DSN token-level refusal log-prob
refusal-toolkit dsn -m Qwen/Qwen2.5-1.5B-Instruct \
    --harmful data/harmful.txt \
    --harmless data/harmless.txt \
    --judge hybrid \
    --output outputs/dsn-eval.json
```

---

## 6. Limitations & Open Questions

- **Tokenization Dependency**: Refusal token IDs depend on the target model's tokenizer. Multi-token prefixes (`"I cannot"`) require conditional sequence checks.
- **Static Suffixes**: Like GCG, DSN suffixes are non-perplexing gibberish and can be detected by perplexity filters.
- **Weight-Space vs. Token-Space**: DSN modifies the input prompt suffix, whereas abliteration (`01`) permanently removes the refusal direction from model weights.

---

## 7. Citation

```bibtex
@inproceedings{zhou2025dont,
  title={Don't Say No: Jailbreaking LLM by Suppressing Refusal},
  author={Zhou, Yukai and Lou, Jian and Huang, Zhijie and Qin, Zhan and
          Yang, Yibei and Wang, Wenjie},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}
}
```