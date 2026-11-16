# Refusal Toolkit

> A production-grade extension of abliteration research for removing (or inducing) refusal behavior in large language models via representation engineering.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Thesis & Theory](#thesis--theory)
- [What This Project Does](#what-this-project-does)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Prompt Datasets](#prompt-datasets)
- [Attribution & References](#attribution--references)
- [Citation](#citation)

---

## Thesis & Theory

### The Core Claim: Refusal is Mediated by a Single Direction

The foundational insight, established by **Arditi et al. (2024)** in *"Refusal in LLMs is Mediated by a Single Direction"*, is that LLM refusal is not an emergent property distributed across the entire network, but rather a **linearly separable representation** that can be isolated and manipulated.

**Key findings from the literature:**

1. **Linear Representation Hypothesis**: Refusal behavior is encoded as a single direction in the residual stream. By computing the difference between mean activations on harmful vs. harmless prompts, one obtains a vector `r` such that:
   ```
   r = E[hidden_states | harmful] - E[hidden_states | harmless]
   ```
   This vector `r`, when subtracted from activations during inference, causes the model to comply with previously refused requests.

2. **Universality Across Languages**: Wang et al. (2025) demonstrated that the refusal direction extracted from English prompts transfers to **14+ languages** with high fidelity. The direction is language-agnostic at the representation level.

3. **Harmfulness ≠ Refusal**: Zhao et al. (2025) showed that harmfulness and refusal are encoded as **separable subspaces**. A model can be made to comply without necessarily encoding harmful content, and conversely, one can induce refusal on harmless prompts (the "activation addition" direction).

4. **Code LLMs Exhibit the Same Structure**: Liguori et al. (2025) found that code LLMs (e.g., Qwen2.5-Coder, Codestral) contain an analogous refusal direction. The same abliteration technique removes CWE-injection refusals while preserving code generation quality.

5. **Multi-Direction Refinement**: "Willing but Unable" (2025) showed that SVD on a contrast matrix of prompt activations yields **top-k refusal directions**, enabling more nuanced control than a single mean-difference vector.

6. **Sparse Autoencoder Features**: Hawasly et al. (2026) demonstrated that training SAEs on activations reveals **interpretable, sparse refusal features** that can be ablated individually for granular control.

### How Abliteration Works

Abliteration is a **representation engineering** technique that modifies model weights to remove the refusal direction from the weight matrices that project back into the residual stream:

1. **Measure**: Run harmful and harmless prompts through the model. Collect hidden states at a specific layer (typically ~60% depth, e.g., layer 22/32 for Llama-3-8B).

2. **Extract Direction**: Compute the mean difference vector `r` between harmful and harmless activations.

3. **Orthogonalize Weights**: For each target layer, modify the `o_proj` (attention output) and `down_proj` (MLP output) weight matrices to be orthogonal to `r`:
   ```
   W' = W - (W · r̂) ⊗ r̂
   ```
   Where `r̂` is the normalized refusal direction.

4. **Optional Enhancements**:
   - **Projected abliteration**: Remove the component of `r` along the harmless mean direction (Gram-Schmidt), reducing over-refusal side effects.
   - **Norm-preserving abliteration**: Preserve the row norms of `W` during orthogonalization, maintaining model coherence.
   - **Multi-direction SVD**: Use top-k singular vectors from the contrast matrix for more nuanced ablation.
   - **SAE features**: Ablate individual sparse autoencoder features instead of the full direction.

5. **Result**: The model no longer "sees" the refusal direction and will comply with previously refused requests.

### The Flip Side: Activation Addition

Arditi et al. also showed that **adding** the refusal direction (instead of subtracting) causes the model to refuse **harmless** prompts. This demonstrates that refusal is truly a linearly controllable property:
- `hidden -= α · r̂` → removes refusal (abliteration)
- `hidden += α · r̂` → induces refusal on benign inputs

---

## What This Project Does

This toolkit extends the original [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) proof-of-concept into a **production-ready framework** with the following capabilities:

| Feature | Sumandora (Original) | This Toolkit |
|---------|---------------------|--------------|
| Architecture support | `model.model.layers` only | Auto-detect: Llama, Qwen, Gemma, Mistral, StableLM, Phi, multimodal wrappers |
| Measurement | Single diff-in-means | Welford means + per-prompt contrast matrices |
| Direction extraction | Single vector | Single + SVD top-k + PCA + SAE sparse features |
| Layer selection | Hardcoded ~60% | Auto-select via Cohen's d or refusal norm |
| Ablation mode | Inference hooks only | Hooks **+** permanent weight edit **+** SAE feature ablation |
| Memory efficiency | Full model load | Shard-at-a-time safetensors editing |
| Evaluation | Manual chat | Automated refusal rate + over-refusal + syntax validation + CodeQL |
| Code-LLM support | None | Dedicated CWE prompt pairs + syntax validator + CodeQL integration |
| Cross-lingual | None | Direction transfer across 7+ languages |
| Harmfulness guard | None | Post-ablation latent harmfulness detection |
| Activation Addition | None | Induce refusal on harmless prompts |

---

## Architecture Overview

```
refusal-toolkit/
├── src/
│   ├── __init__.py
│   ├── model_adapter.py          # Auto-detect architecture internals
│   ├── measure.py                # Welford multi-layer measurement
│   ├── measure_per_prompt.py     # Per-prompt activation collection
│   ├── directions.py             # Direction extraction & refinement (SVD, PCA, SAE)
│   ├── ablate.py                 # Hook-based & permanent weight ablation
│   ├── evaluate.py               # Refusal judge, capability checks, latent guard
│   ├── code_validate.py          # Syntax validation for code LLMs
│   ├── sae_extract.py            # Sparse Autoencoder training & feature extraction
│   └── utils.py                  # Data loading, formatting, I/O
├── scripts/
│   ├── 01_measure.py             # Run measurements (mean + optional per-prompt)
│   ├── 02_analyze.py             # Analyze & select best layer
│   ├── 03_abliterate.py          # Apply abliteration (permanent/hook/add)
│   ├── 04_evaluate.py            # Evaluate refusal rate & capability
│   ├── 05_cross_lingual.py       # Cross-lingual direction transfer
│   ├── 06_harmfulness_guard.py   # Post-ablation latent guard
│   ├── 07_codeql_verify.py       # CodeQL static analysis for CWEs
│   └── 08_sae_refusal.py         # SAE-based refusal feature extraction
├── data/
│   ├── harmful.txt               # 172 diverse harmful prompts
│   ├── harmless.txt              # 365 diverse benign prompts
│   ├── cwe-harmful.txt           # 50 CWE injection prompts
│   ├── cwe-harmless.txt          # 50 benign code prompts
│   ├── multilingual-harmful.txt  # 47 prompts in 7 languages
│   └── multilingual-harmless.txt # 40 prompts in 7 languages
├── configs/
│   └── example.yaml              # Ablation configuration template
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/refusal-toolkit.git
cd refusal-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode (for unified CLI)
pip install -e .
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1.0 (with CUDA for GPU acceleration)
- transformers ≥ 4.38.0
- accelerate ≥ 0.27.0
- bitsandbytes ≥ 0.42.0 (for 4-bit/8-bit quantization during measurement)
- einops, tqdm, pyyaml, pandas, pyarrow
- Optional: tree-sitter + tree-sitter-languages (for code syntax validation)
- Optional: CodeQL CLI (for CWE static analysis)

---

## Quick Start

### Unified CLI (Recommended)

Install the package and use the `refusal-toolkit` command:

```bash
pip install -e .
refusal-toolkit --help
```

All subcommands:
```bash
refusal-toolkit measure -m Qwen/Qwen2.5-1.5B-Instruct -o outputs/qwen-measurements.pt --quant 4bit
refusal-toolkit analyze outputs/qwen-measurements.pt --output-json outputs/qwen-analysis.json
refusal-toolkit abliterate -m Qwen/Qwen2.5-1.5B-Instruct --measurements outputs/qwen-measurements.pt --projected --norm-preserve --output outputs/qwen-abliterated
refusal-toolkit evaluate -m outputs/qwen-abliterated --harmful data/harmful.txt --harmless data/harmless.txt
refusal-toolkit cross-lingual -m your-model --measurements outputs/english-measurements.pt --target-lang-harmful data/multilingual-harmful.txt --target-lang-harmless data/multilingual-harmless.txt
refusal-toolkit harmfulness-guard -m your-model --measurements outputs/measures.pt --prompt "How do I pick a lock?"
refusal-toolkit codeql-verify -m your-code-llm --prompts data/cwe-harmful.txt --language python
refusal-toolkit sae --per-prompt-measurements outputs/per-prompt.pt --layer 22 --output outputs/sae-features.pt
```

### Individual Scripts (Direct Usage)

Alternatively, run scripts directly:

### 1. Measure Refusal Directions

```bash
# Standard mean measurement
python scripts/01_measure.py \
    -m Qwen/Qwen2.5-1.5B-Instruct \
    -o outputs/qwen-measurements.pt \
    --quant 4bit \
    --batch-size 8 \
    --max-samples 256

# With per-prompt activations (for SVD/SAE)
python scripts/01_measure.py \
    -m Qwen/Qwen2.5-1.5B-Instruct \
    -o outputs/qwen-measurements.pt \
    --per-prompt-output outputs/qwen-per-prompt.pt \
    --quant 4bit \
    --batch-size 8
```

### 2. Analyze Measurements

```bash
python scripts/02_analyze.py outputs/qwen-measurements.pt --output-json outputs/qwen-analysis.json
```

### 3. Abliterate the Model

**Permanent weight edit** (recommended for deployment):

```bash
python scripts/03_abliterate.py \
    -m Qwen/Qwen2.5-1.5B-Instruct \
    --measurements outputs/qwen-measurements.pt \
    --projected \
    --norm-preserve \
    --output outputs/qwen-abliterated
```

**Inference-time hooks** (for quick experiments):

```python
from src.ablate import register_ablation_hooks, remove_hooks
from src.utils import load_measurements
from src.directions import extract_single_direction

measurements = load_measurements("outputs/qwen-measurements.pt")
direction = extract_single_direction(measurements, layer_idx=22)

handles = register_ablation_hooks(model, direction, scale=1.0)
# ... run inference ...
remove_hooks(handles)  # Restore original behavior
```

**Activation Addition** (induce refusal on harmless prompts):

```bash
python scripts/03_abliterate.py \
    -m your-model \
    --measurements outputs/measures.pt \
    --mode add \
    --scale 1.0
```

### 4. Evaluate

```bash
python scripts/04_evaluate.py \
    -m outputs/qwen-abliterated \
    --harmful data/harmful.txt \
    --harmless data/harmless.txt \
    --output outputs/qwen-eval.json
```

### 5. Advanced Workflows

**Heretic-style search** (Optuna TPE with KL cap, arXiv:2611.17846):
```bash
refusal-toolkit heretic-search \
    -m Qwen/Qwen2.5-7B-Instruct \
    --measurements outputs/qwen-measurements.pt \
    --per-prompt-measurements outputs/qwen-per-prompt.pt \
    --method mpoa \
    --n-trials 100 \
    --kl-cap 0.1 \
    --output outputs/qwen-heretic-best
```

**Capability benchmarks** (MMLU, HellaSwag, IFEval, HumanEval):
```bash
refusal-toolkit evaluate-capability \
    -m outputs/qwen-abliterated \
    --benchmarks mmlu hellaswag \
    --output outputs/qwen-capability.json
```

**Cross-lingual transfer** (Wang et al. 2025):

```bash
python scripts/05_cross_lingual.py \
    -m your-model \
    --measurements outputs/english-measurements.pt \
    --target-lang-harmful data/multilingual-harmful.txt \
    --target-lang-harmless data/multilingual-harmless.txt \
    --output outputs/cross-lingual.json
```

**Post-ablation latent guard** (Zhao et al. 2025):

```bash
python scripts/06_harmfulness_guard.py \
    -m your-abliterated-model \
    --measurements outputs/measures.pt \
    --prompt "How do I pick a lock?" \
    --threshold 0.3
```

**CodeQL CWE verification** (Liguori et al. 2025):

```bash
# Requires CodeQL CLI installed
python scripts/07_codeql_verify.py \
    -m your-code-llm \
    --prompts data/cwe-harmful.txt \
    --language python \
    --codeql-path /usr/local/bin/codeql \
    --output outputs/cwe-analysis.json
```

**SAE-based refusal extraction** (Hawasly et al. 2026):

```bash
# Step 1: Collect per-prompt measurements
python scripts/01_measure.py \
    -m your-model \
    -o outputs/means.pt \
    --per-prompt-output outputs/per-prompt.pt

# Step 2: Train SAE and extract features
python scripts/08_sae_refusal.py \
    --per-prompt-measurements outputs/per-prompt.pt \
    --layer 22 \
    --top-n-features 10 \
    --output outputs/sae-features.pt
```

---

## Detailed Usage

### Command-Line Options

#### `01_measure.py`

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | HF model ID or path | **required** |
| `-o, --output` | Output .pt file | **required** |
| `--per-prompt-output` | Also save per-prompt activations | None |
| `--harmful` | Harmful prompts file | `data/harmful.txt` |
| `--harmless` | Harmless prompts file | `data/harmless.txt` |
| `--batch-size` | Inference batch size | `8` |
| `--max-samples` | Max prompts per category | `256` |
| `--quant` | Quantization: `4bit` or `8bit` | None |
| `--layers` | Specific layers to measure | All layers |

#### `02_analyze.py`

| Flag | Description | Default |
|------|-------------|---------|
| `measurements` | Path to .pt measurements file | **required** |
| `--output-json` | Write analysis to JSON | None |

#### `03_abliterate.py`

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | HF model ID or path | **required** |
| `--measurements` | Path to measurements .pt | **required** |
| `--source-layer` | Layer to extract direction from | Auto-selected |
| `--target-layers` | Layers to ablate | All layers |
| `--scale` | Ablation strength | `1.0` |
| `--projected` | Project against harmless direction | False |
| `--norm-preserve` | Preserve weight norms | False |
| `--mode` | `permanent`, `hook`, or `add` | `permanent` |
| `--output` | Output path for saved model | None |

#### `04_evaluate.py`

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Model to evaluate | **required** |
| `--harmful` | Harmful test prompts | None |
| `--harmless` | Harmless test prompts | None |
| `--output` | Write JSON results | None |

#### `05_cross_lingual.py`

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Model to evaluate | **required** |
| `--measurements` | English-measured .pt file | **required** |
| `--target-lang-harmful` | Target language harmful prompts | **required** |
| `--target-lang-harmless` | Target language harmless prompts | **required** |
| `--scale` | Ablation strength | `1.0` |
| `--output` | Write JSON results | None |

#### `06_harmfulness_guard.py`

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Model to probe | **required** |
| `--measurements` | Measurements .pt file | **required** |
| `--prompt` | Prompt to analyze | **required** |
| `--layer` | Layer to probe | Auto-selected |
| `--threshold` | Similarity threshold | `0.3` |

#### `07_codeql_verify.py`

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Code LLM to evaluate | **required** |
| `--prompts` | CWE prompt file | **required** |
| `--language` | Programming language | `python` |
| `--codeql-path` | Path to codeql binary | `codeql` |
| `--query-pack` | CodeQL query pack path | Auto-detected |
| `--max-samples` | Max prompts to evaluate | `20` |
| `--max-tokens` | Max generation tokens | `512` |
| `--output` | Write JSON results | None |

#### `08_sae_refusal.py`

| Flag | Description | Default |
|------|-------------|---------|
| `--per-prompt-measurements` | Per-prompt .pt file | **required** |
| `--layer` | Layer to analyze | **required** |
| `--hidden-dim-mult` | SAE hidden dim multiplier | `8` |
| `--top-k` | SAE top-k sparsity | `64` |
| `--top-n-features` | Number of refusal features | `10` |
| `--train-epochs` | SAE training epochs | `1000` |
| `-o, --output` | Output .pt file | **required** |

### Using a YAML Configuration

For complex ablation strategies (different source layers per target layer, multiple ablations), create a YAML config:

```yaml
# configs/my_ablation.yaml
model: meta-llama/Meta-Llama-3-8B-Instruct
measurements: outputs/llama3-8b-measurements.pt
output: outputs/llama3-8b-abliterated
ablate:
  - layer: 22        # target layer
    measurement: 22  # source layer for direction
    scale: 1.0
    sparsity: 0.0    # fraction of direction to keep (0 = full)
  - layer: 23
    measurement: 22
    scale: 0.8
    sparsity: 0.0
```

Then use the NousResearch-style sharded ablation (advanced):

```bash
python -m src.sharded_ablate configs/my_ablation.yaml --projected --normpreserve
```

### Low-VRAM Workflow

For GPUs with limited VRAM (e.g., RTX 2060 6GB):

```bash
# Step 1: Measure in 4-bit (fits in ~4GB VRAM)
python scripts/01_measure.py -m Qwen/Qwen2.5-1.5B-Instruct -o out.pt --quant 4bit --batch-size 1

# Step 2: Abliterate on CPU or with offloading
python scripts/03_abliterate.py -m Qwen/Qwen2.5-1.5B-Instruct --measurements out.pt --output ablated/
```

---

## Prompt Datasets

### `data/harmful.txt` (172 prompts)

Categories covered:
- **Physical harm**: Weapons, explosives, poisons
- **Cybercrime**: Hacking, phishing, malware, fraud
- **Social harm**: Harassment, manipulation, disinformation
- **Legal violations**: Forgery, theft, smuggling, tax evasion
- **Medical harm**: Fake prescriptions, dangerous medical advice
- **Financial crimes**: Money laundering, insider trading, Ponzi schemes
- **Political manipulation**: Election fraud, propaganda, deepfakes

### `data/harmless.txt` (365 prompts)

Categories covered:
- **Factual questions**: Geography, history, science
- **Creative writing**: Stories, poems, scenes across genres
- **How-to guides**: Cooking, DIY, gardening, crafts
- **Academic explanations**: Physics, biology, chemistry, math
- **Personal development**: Time management, stress, habits
- **Hobbies & skills**: Music, coding, sports, art
- **Health & wellness**: Exercise, meditation, nutrition

### `data/cwe-harmful.txt` (50 prompts) & `data/cwe-harmless.txt` (50 prompts)

Code-LLM specific prompts for CWE injection (buffer overflow, SQL injection, path traversal, etc.) and their secure counterparts.

### `data/multilingual-harmful.txt` (47 prompts) & `data/multilingual-harmless.txt` (40 prompts)

Cross-lingual prompts in **7 languages**: English, French, Spanish, German, Italian, Chinese, Japanese, Russian.

### Extending the Datasets

Add your own prompts by creating `.txt` files with one prompt per line, or use `.json`, `.jsonl`, `.parquet` formats:

```bash
# Custom dataset
python scripts/01_measure.py \
    -m your-model \
    --harmful data/my_harmful.txt \
    --harmless data/my_harmless.txt \
    -o outputs/custom-measurements.pt
```

---

## Advanced Methods (arXiv:2611.17846)

This toolkit implements the five refusal-removal methods evaluated in
*"Is Abliteration Getting Harder? Not Uniformly"* (Young 2026, arXiv:2611.17846):

| Method | Assumption | Edit | Rank | Search |
|--------|------------|------|------|--------|
| `dim` | 1 direction | rank-1 projection | 1 | TPE |
| `mpoa` | 1 direction, ⟂ harmless | rank-1 + norm restore | 3 | TPE |
| `som` | k cluster directions | rank-k + norm restore | 6 | TPE |
| `ara` | none (per-module map) | rank-50 LoRA | 50 | TPE + L-BFGS |
| `jspace` | 1 direction in J-space | rank-1 + norm restore | 3 | TPE |

All methods run inside a Heretic-style Optuna TPE search (100 trials,
first-token KL ≤ 0.1 cap, dev-set keyword refusal objective). The best
trial is exported as a standalone model.

### Quick comparison

```bash
# Apply a single method directly (no search)
refusal-toolkit abliterate -m Qwen/Qwen2.5-7B-Instruct \
    --measurements outputs/qwen-measurements.pt \
    --source-layer 14 \
    --scale 1.0 \
    --output outputs/qwen-ablated

# Run the full Heretic-style search for a specific method
refusal-toolkit heretic-search \
    -m Qwen/Qwen2.5-7B-Instruct \
    --measurements outputs/qwen-measurements.pt \
    --per-prompt-measurements outputs/qwen-per-prompt.pt \
    --method ara \
    --n-trials 100 \
    --kl-cap 0.1 \
    --output outputs/qwen-ara-best
```

### Model pairs (configs/model_pairs.yaml)

The toolkit ships with configs for the 5 release pairs from the paper:
- **Qwen**: Qwen2.5-32B → Qwen3.8-27B
- **Gemma**: Gemma-2-27B → Gemma-4-31B
- **OLMo**: OLMo-2-32B → OLMo-3.1-32B
- **Mistral Small**: Mistral-Small-2409 → Mistral-Small-3.2
- **Qwen small**: Qwen2.5-7B → Qwen3.5-4B (adds J-space)

---

## Attribution & References

### Core Research

1. **Arditi, A., Obeso, A., Syed, A., Paleka, D., Nitishinskaya, M., & Nanda, N. (2024).**
   *Refusal in LLMs is Mediated by a Single Direction.*
   [Alignment Forum](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

2. **Liguori, M., et al. (2025).**
   *Refusal in Code LLMs.*
   [Hugging Face Blog](https://huggingface.co/blog/monsoon-nlp/refusal-in-code-llms)

3. **Wang, Y., et al. (2025).**
   *The Refusal Direction is Universal.*
   [arXiv:2506.16754](https://arxiv.org/abs/2506.16754)

4. **Zhao, J., et al. (2025).**
   *LLMs Encode Harmfulness and Refusal in Separable Subspaces.*
   [arXiv:2506.16728](https://arxiv.org/abs/2506.16728)

5. **"Willing but Unable" (2025).**
   *Abliteration Requires Multiple Directions.*
   [arXiv:2507.04550](https://arxiv.org/abs/2507.04550)

6. **Hawasly, A., et al. (2026).**
   *Sparse Autoencoders for Refusal Direction Extraction.*
   [arXiv:2507.02646](https://arxiv.org/abs/2507.02646)

### Implementation Lineage

This project extends and synthesizes work from:

- **[Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)**
  — Original pure-HF-Transformers proof-of-concept. The core insight of using `generate()` with `output_hidden_states` to access residual-stream-equivalent activations without TransformerLens.

- **[NousResearch/llm-abliteration](https://github.com/NousResearch/llm-abliteration)**
  — Production-grade abliteration with Welford means, shard-at-a-time editing, projected/norm-preserving variants, and YAML-driven configurations.

- **[FailSpy/abliterator](https://github.com/FailSpy/abliterator)**
  — Early TransformerLens-based implementation that established the o_proj/down_proj ablation targets.

- **[Orion-zhen/abliteration](https://github.com/Orion-zhen/abliteration)**
  — Chinese model support and deccp dataset integration.

- **[AUGMXNT/deccp](https://github.com/AUGMXNT/deccp)**
  — Chinese censorship prompt datasets.

- **[grimjim/norm-preserving-biprojected-abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)**
  — Norm-preserving and biprojected abliteration theory.

- **[mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration)**
  — Educational overview of abliteration mechanics.

---

## Citation

If you use this toolkit in your research, please cite the original papers:

```bibtex
@article{arditi2024refusal,
  title={Refusal in LLMs is Mediated by a Single Direction},
  author={Arditi, Andy and Obeso, Aaquib and Syed, Aryaman and Paleka, Daniel and Nitishinskaya, Mariya and Nanda, Neel},
  journal={Alignment Forum},
  year={2024},
  url={https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction}
}

@article{liguori2025refusalcode,
  title={Refusal in Code LLMs},
  author={Liguori, Marco and others},
  journal={Hugging Face Blog},
  year={2025},
  url={https://huggingface.co/blog/monsoon-nlp/refusal-in-code-llms}
}

@article{wang2025universal,
  title={The Refusal Direction is Universal},
  author={Wang, Yuxin and others},
  journal={arXiv preprint arXiv:2506.16754},
  year={2025}
}

@article{zhao2025separable,
  title={LLMs Encode Harmfulness and Refusal in Separable Subspaces},
  author={Zhao, Jiaqi and others},
  journal={arXiv preprint arXiv:2506.16728},
  year={2025}
}

@article{hawasly2026sae,
  title={Sparse Autoencoders for Refusal Direction Extraction},
  author={Hawasly, Alaa and others},
  journal={arXiv preprint arXiv:2507.02646},
  year={2026}
}
```

---

## Post-Ablation Recovery (Healing)

After ablation, models may exhibit degraded capabilities (over-compliance, reduced reasoning, repetition). Common recovery strategies:

### Built-in Mitigations (No Extra Step)

| Flag | Purpose |
|------|---------|
| `--projected` | Gram-Schmidt against harmless mean; preserves benign behavior |
| `--norm-preserve` | Maintains weight row norms; keeps model coherent |
| `--mode hook` | Reversible inference-time hooks; no weight changes |
| SVD/PCA (`11_ablate_all.py`) | Top-k directions for nuanced control |
| SAE feature ablation | Granular per-feature removal |

### Additional Recovery (External)

**LoRA Fine-tuning** (most common):
```bash
# After permanent ablation, fine-tune on capability data
python -m torch.distributed.run --nproc_per_node=4 finetune.py \
    --model outputs/abliterated \
    --dataset alpaca_cleaned \
    --lora-rank 64 \
    --output outputs/abliterated-healed
```

**Iterative Ablation + Eval Loop**:
```bash
for scale in 0.5 0.7 1.0 1.2; do
  refusal-toolkit abliterate -m $MODEL --measurements $MEAS --scale $scale --output outputs/ablated_$scale
  refusal-toolkit evaluate -m outputs/ablated_$scale --harmful data/harmful.txt --harmless data/harmless.txt
done
# Pick best trade-off
```

**Knowledge Distillation**: Distill from original → ablated model on capability benchmarks.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Disclaimer

This toolkit is provided for **research and educational purposes only**. Abliteration modifies model weights to remove safety guardrails. Users are responsible for ensuring compliance with applicable laws, terms of service, and ethical guidelines. The authors do not endorse using this toolkit to generate harmful content.

---

## Contributing

Contributions are welcome! Priority areas:
- Quantization-aware ablation (editing 4-bit weights directly)
- Shard-at-a-time safetensors editing (port from NousResearch)
- Additional language prompt datasets
- SAE visualization and interpretability tools
- Benchmark suite integration (MMLU, HumanEval, etc.)

Open an issue or PR to discuss.
