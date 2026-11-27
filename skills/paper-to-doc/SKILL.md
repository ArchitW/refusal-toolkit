---
name: paper-to-doc
description: >
  Rewrite academic research papers (from text, arXiv URLs, PDFs, or local files)
  into clear Simplified Technical English (ASD-STE100) notes with TL;DR, a plain-English
  Simple Example, Quick Facts, step-by-step methods, empirical findings, literature
  cross-references, and codebase implementation mappings. Use when asked to "rewrite a paper",
  "convert research papers to docs", "simplify research notes", or "write a paper note".
---

# Paper-to-Doc Skill (ASD-STE100 Simplified Technical English)

Rewrite technical machine learning and LLM research papers into clear, readable, ASD-STE100-compliant documentation notes so that any engineer or researcher can read and understand them on the first pass.

---

## 1. Input Sources

Accept any of the following input formats from the user:
1. **arXiv / Web URL**: Fetch the paper abstract, HTML (`https://arxiv.org/html/<id>v1`), or PDF metadata and public text via `webfetch` or `websearch`.
2. **Local File Path**: Read a local Markdown (`.md`), text (`.txt`), or PDF file in the workspace.
3. **Direct Text**: Use raw text pasted directly by the user.

---

## 2. Core ASD-STE100 Writing Rules

Follow Simplified Technical English (ASD-STE100) rules strictly throughout every section:

1. **Active Voice Only**:
   - Write 100% in active voice. Never use passive voice.
   - **Do**: `The model computes the refusal direction.` / `The authors subtract the harmless mean.`
   - **Do Not**: `The refusal direction is computed by the model.` / `The harmless mean is subtracted.`
2. **Short Sentences & One Idea Per Sentence**:
   - Keep sentences short (aim for 8–18 words; keep descriptive sentences under 20 words).
   - State only one action or fact per sentence.
3. **Simple Tenses & Imperatives for Steps**:
   - Use the **simple present tense** for how methods work (`The model uses...`, `The script computes...`).
   - Use the **simple past tense** when describing the authors' experiments (`The authors measured...`).
   - Start ordered procedural steps with an active verb (`Prepare...`, `Run...`, `Compute...`).
4. **Clean Vocabulary & No Ambiguity**:
   - Use one consistent term for each concept. Do not switch between synonyms.
   - Remove complex noun clusters and gerunds (`-ing` nouns) where a clear verb or noun works better.
   - Avoid vague adjectives (`very`, `several`, `complex`) without concrete detail.

---

## 3. Required Document Template

Every generated or rewritten document must follow this exact structure:

```markdown
Created: DD-MM-YY
Updated: DD-MM-YY

# NN — <Short Descriptive Paper Title>

## TL;DR (Summary)

<4 to 6 short, active-voice sentences summarizing the problem, the method, and the outcome. Zero jargon without immediate plain explanation.>

---

## Simple Example (Plain English)

<An intuitive, concrete, non-mathematical analogy or step-by-step real-world walkthrough (using simple everyday concepts, dials, switches, routing, checklists, etc.) that explains the core idea of the paper to a reader with zero ML background.>

1. **<Analogy Step 1 Title>**:
   - <Short plain-English explanation>
2. **<Analogy Step 2 Title>**:
   - <Short plain-English explanation>
3. **<Analogy Step 3 Title>**:
   - <Short plain-English explanation>

---

## Quick Facts

| Item | Details |
|---|---|
| **Paper Title** | <Full paper title> |
| **Authors** | <Author list> |
| **Year** | <Year and revision dates> |
| **Venue / Links** | <arXiv / OpenReview / ACL / PMLR links> |
| **Toolkit Role** | <Role in the repository> |
| **Toolkit Status** | <Implemented / Partial / Not implemented with file references> |

---

## 1. The Problem

### 1.1 <Context / Background Subheading>
- <2 to 4 short bullets explaining the problem and why existing approaches fail.>

### 1.2 <Competing Views / Gaps Subheading>
- <Short bullets defining the hypotheses or gap the paper addresses.>

### 1.3 Core Claim of This Paper
- <3 short bullets stating the exact contribution of the paper.>

---

## 2. The Method (How It Works)

<1 sentence overview of the method.>

### Step 1: <Action Title>
1. <Imperative/Active step>
2. <Imperative/Active step>

### Step 2: <Action Title>
1. <Imperative/Active step with equation if needed, followed by a plain-English definition of every symbol.>

### Step 3: <Action Title>
1. <Imperative/Active step>

---

## 3. Key Findings (What the Authors Found)

- **<Finding 1>**: <Short explanation and numbers.>
- **<Finding 2>**: <Short explanation and numbers.>
- **<Finding 3>**: <Short explanation and numbers.>

| Observation | Result |
|---|---|
| **<Metric / Setting>** | <Quantitative result> |
| **<Metric / Setting>** | <Quantitative result> |

---

## 4. Relationship to Other Research

- **<Relationship to Note / Paper `NN`>**:
  - <How this paper connects to, extends, or contrasts with other papers in `research/01`–`14`.>
- **<Relationship to Attacks / Defenses / Benchmarks>**:
  - <Clear active-voice explanation of the theoretical connection.>

---

## 5. Implementation in `refusal-toolkit`

<If implemented in the repo, provide the mapping table and runnable CLI example. If not yet implemented, describe the exact modules/scripts where it will fit.>

| Step | Module / Function | CLI Command / Flag |
|---|---|---|
| **1. <Step Name>** | `src/<file>.py::<function>` | `refusal-toolkit <command>` |

### Usage Example

```bash
<refusal-toolkit CLI commands>
```

---

## 6. Limitations & Open Questions

- **<Limitation 1>**: <Short explanation.>
- **<Limitation 2>**: <Short explanation.>
- **<Open Question>**: <Short explanation.>

---

## 7. Citation

```bibtex
<BibTeX entry>
```
```

---

## 4. Workflow Instructions

1. **Inspect the Archive First**:
   - If working inside a repository with a `research/` folder, read `research/README.md` and existing `research/NN_*.md` files to match the numbering (`NN`), cross-reference related papers, and verify how the paper maps to `src/` and `scripts/`.
2. **Draft the Note**:
   - Write the document following the exact 7-section template above.
   - Verify every sentence uses active voice and simple phrasing.
   - If the user asks to preview 1 document first ("prepare 1 document for me to review first"), write only the first document (`research/01_...`), update `SESSION.md` with `agent: kilo/kilo-auto-free`, and wait for approval before rewriting the remaining documents.
3. **Update the Archive Index**:
   - Update `research/README.md` and the root `README.md` if the rewritten note adds new cross-references.