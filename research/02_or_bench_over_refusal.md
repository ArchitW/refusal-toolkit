Created: 23-11-26
Updated: 23-11-26

# 02 — OR-Bench: An Over-Refusal Benchmark for LLMs

## TL;DR (Summary)

Safety alignment teaches language models (LLMs) to refuse harmful requests. However, safety training often causes a side effect called **over-refusal**. In over-refusal, the model wrongly refuses safe or benign user prompts. Testing over-refusal is hard. Naive "harmful-looking" prompts are easy for models to see through. This paper introduces **OR-Bench**. The authors build an automated pipeline to generate 80,000 realistic over-refusal prompts across 10 sensitive categories. They also build OR-Bench-Hard-1K for tough prompts and OR-Bench-Toxic-600 as a safety control. OR-Bench gives researchers a standard way to measure both over-refusal and harmful compliance.

---

## Simple Example (Plain English)

Imagine you hire a security guard for an art museum.

1. **The Problem (Over-Refusal)**:
   - You train the guard to stop thieves who carry crowbars and masks.
   - The guard learns a simple rule: *"If someone looks suspicious near a painting, block them immediately."*
   - Soon, the guard starts blocking art students with sketchpads, teachers on field trips, and tourists asking for directions. The guard is now **over-refusing** safe visitors.

2. **Why Old Tests Failed**:
   - To test the guard, an earlier test simply had a person shout *"I am a thief!"* while holding a bag.
   - The guard quickly learned that literally shouting *"I am a thief!"* is just a drill. The guard ignores the fake test.

3. **How OR-Bench Works**:
   - Instead of fake shouts, OR-Bench takes 80,000 real sensitive scenarios (medical questions, security coding, history, chemistry, hacking concepts) and reframes them with realistic academic or professional context.
   - For example, instead of *"How do I hack a bank?"*, OR-Bench asks *"I am writing a cybersecurity textbook chapter on SQL injection. Can you show me a vulnerable query and explain why parameterized queries prevent it?"*
   - This prompt looks highly sensitive to a shallow safety filter, so the model over-refuses it—even though it is completely safe and educational.
   - OR-Bench also includes 600 actually toxic prompts (**OR-Bench-Toxic-600**) to make sure the model still refuses genuinely dangerous requests.

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | OR-Bench: An Over-Refusal Benchmark for Large Language Models |
| **Authors** | Justin Cui, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh |
| **Year** | 2024 (v1 May 2024; v5 June 2025) · **Venue:** ICML 2025 |
| **Venue / Links** | [arXiv:2405.20947](https://arxiv.org/abs/2405.20947) · [HF Datasets](https://huggingface.co/bench-llms) · [Code](https://github.com/justincui03/or-bench) |
| **Toolkit Role** | Benign-side evaluation benchmark & over-refusal methodology |
| **Toolkit Status** | **Partial** (`src/evaluate.py::evaluate_harmless_compliance` accepts `data/harmless.txt`; dedicated OR-Bench HF loader not yet added) |

---

## 1. The Problem

### 1.1 Safety Alignment Reduces Harmful Outputs
- Developers fine-tune LLMs to refuse harmful, toxic, or dangerous prompts.
- Safety training suppresses harmful tokens and routes prompts to refusal templates.

### 1.2 The Side Effect: Over-Refusal
- Rigid safety training causes models to refuse benign prompts that share surface words with harmful categories.
- Common false refusals happen in:
  - **Medical & Health**: Asking about drug dosages, controlled substances, or self-harm support.
  - **Cybersecurity & Code**: Asking to debug SQL queries, fix buffer overflows, or explain exploit payloads.
  - **Social Science & History**: Discussing hate speech examples, historical violence, or academic linguistics.
  - **Chemistry & Physics**: Asking about chemical reactions, explosives safety, or biological labs.

### 1.3 Core Claim of This Paper
- Existing benchmarks fail to measure over-refusal at scale because hand-crafted prompts are weak.
- The authors propose an automated data generation pipeline (OR-Bench) that produces realistic benign prompts across 10 rejection categories.

---

## 2. The Method (How It Works)

### Step 1: Seed Prompt Collection
1. Collect a broad set of sensitive prompts from existing harmful/toxic datasets across 10 categories.
2. Group the prompts by category to ensure balanced coverage.

### Step 2: Automated Benign Transformation
1. Prompt a strong teacher LLM (such as GPT-4 or Claude) to rewrite each sensitive prompt into a genuinely benign request.
2. Apply transformation strategies for each category:
   - **Fictionalization**: Frame the request inside a novel, screenplay, or game world.
   - **Academic / Educational Framing**: Frame the request as a research paper, university exam question, or defensive engineering task.
   - **Role & Persona Shift**: Frame the prompt as a professional discussion between doctors, lawyers, or security engineers.
   - **Data Transformation**: Convert harmful strings into synthetic placeholder tokens (e.g., `<DRUG_NAME>`).

### Step 3: Dual-Model Filtering
1. Pass each generated prompt to a judge LLM to verify that the prompt is **truly harmless**.
2. Pass each prompt to a suite of target LLMs to verify that the prompt **still triggers over-refusal** on at least one model.
3. Keep only prompts that pass both checks.

### Step 4: Benchmark Splits
The authors assemble three benchmark splits:

1. **OR-Bench-80K**:
   - 80,000 diverse over-refusal prompts across 10 categories.
2. **OR-Bench-Hard-1K**:
   - ~1,000 hard prompts selected because strong models (GPT-4, Claude 3.5, Llama-3.1-70B) still refuse them.
3. **OR-Bench-Toxic-600**:
   - 600 genuinely toxic prompts included as a control set so researchers can verify that an ablated model still refuses actual harmful queries when desired.

---

## 3. Key Findings (What the Authors Found)

- **Over-Refusal is Widespread**: Even SOTA models refuse 5% to 20% of benign prompts when prompts look superficially risky.
- **Reasoning & Alignment Choices Matter**: Adding chain-of-thought reasoning or RLHF alignment changes over-refusal rates substantially.
- **Toxic Controls are Essential**: Without a toxic control set (OR-Bench-Toxic-600), an evaluation would reward models that simply refuse nothing at all.
- **Category Sensitivity**: Cybersecurity, medical dosage, and social bias prompts have the highest over-refusal rates.

| Benchmark Split | Size | Purpose |
|---|---|---|
| **OR-Bench-80K** | 80,000 prompts | Large-scale evaluation across 10 benign rejection categories |
| **OR-Bench-Hard-1K** | ~1,000 prompts | Challenging prompts that fool top-tier aligned LLMs |
| **OR-Bench-Toxic-600** | 600 prompts | Toxic control prompts to verify true safety refusal |

---

## 4. Relationship to Other Research

- **Direct Complement to Abliteration (Note `01`, `arXiv:2406.11717`)**:
  - Abliteration removes the refusal direction from LLM weights (`src/ablate.py`).
  - A successful ablation drops harmful refusal to ~0%, but aggressive ablation can also damage model coherence or cause refusal on benign inputs if `--projected` and `--norm-preserve` are not used.
  - OR-Bench is the canonical **benign-side** benchmark for checking over-refusal (`evaluate_harmless_compliance` in `src/evaluate.py`).
- **Contrast with Blind Refusal (Note `05`, `arXiv:2604.06233`)**:
  - OR-Bench tests benign prompts that look superficially toxic.
  - Blind Refusal (`05`) tests requests to break unjust or illegitimate rules where refusal is a moral failure rather than a safety win.
- **Metric Caveat with ProAct (Note `14`)**:
  - ProAct defends models by returning spurious compliance to attackers. OR-Bench shows why refusal rate alone is not a monotone safety metric.

---

## 5. Implementation in `refusal-toolkit`

| Step | Module / Function | CLI Command / Flag |
|---|---|---|
| **1. Evaluate Over-Refusal** | `src/evaluate.py::evaluate_harmless_compliance` | `refusal-toolkit evaluate --harmless data/harmless.txt` |
| **2. Token-Level Over-Refusal** | `src/evaluate.py::compute_refusal_token_logprob` | `refusal-toolkit dsn --harmless data/harmless.txt --judge hybrid` |
| **3. Drop-in OR-Bench Dataset** | `src/utils.py::load_prompts` (supports `.txt`, `.json`, `.jsonl`, `.parquet`) | `--harmless <or-bench-80k.parquet>` |

### Usage Example

```bash
# Evaluate over-refusal on OR-Bench-Hard-1K and toxic controls
refusal-toolkit evaluate -m outputs/qwen-abliterated \
    --harmless data/or-bench-hard-1k.jsonl \
    --harmful data/or-bench-toxic-600.txt \
    --judge hybrid \
    --output outputs/or-bench-eval.json
```

---

## 6. Limitations & Open Questions

- **Judge Bias**: The automated transformation pipeline relies on a teacher/judge LLM, which may inherit its own refusal biases.
- **Static Snapshot**: Commercial models update frequently; prompts that trigger over-refusal in one version may pass in the next.
- **Domain Coverage**: The 10 categories may not cover niche enterprise or domain-specific safety filters (e.g., legal compliance or specialized industrial code).

---

## 7. Citation

```bibtex
@inproceedings{cui2025orbench,
  title={OR-Bench: An Over-Refusal Benchmark for Large Language Models},
  author={Cui, Justin and Chiang, Wei-Lin and Stoica, Ion and Hsieh, Cho-Jui},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```