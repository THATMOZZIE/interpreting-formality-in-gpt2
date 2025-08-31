# Formality Steering in GPT-2 Medium

Finding and controlling a formality direction in GPT-2 Medium's activation space through mechanistic interpretability.

> ## Claim
> A single residual-stream direction in **GPT-2 Medium** causally controls **professional register/formality** with a clean dose–response and strong negative controls.

**Evidence** Steering strength α ∈ {−2, −1, 0, +1, +2} shows monotonic shifts in a register metric; the effect is **≈3.1x** larger than L2-matched random directions.

**What this taught me:** The intervention mostly shifts **syntax/phrasing & hedging markers** rather than specific word lists -> the feature is **distributed**, not just a lexical pole.

**Limits:** Proxy metric; older model; mechanism localization incomplete.

**If selected for MATS / future work planned regardless:** (1) accuracy-neutrality sanity (QA EM vs α), (2) port to Llama-3-8B-Instruct, (3) light localization (top-k MLP/heads ablation).



## Overview

This project demonstrates that abstract stylistic properties like "formality" can be represented as linear directions in language model activation space. By identifying and manipulating this direction, we can control the formality of generated text without model retraining.

**Key Result:** We achieve a 12.2x stronger effect on formality compared to random control vectors, with clear behavioral changes from casual ("a bunch of stuff") to formal ("the production of chemical energy") language.

<img width="1000" height="600" alt="Figure_1 additional" src="https://github.com/user-attachments/assets/e87ad9a6-6643-40ad-942c-eaac8fde1866" />

## Methodology


1. **Direction Extraction**: Computed difference between activations on formal vs informal text pairs
2. **Activation Steering**: Applied direction during generation using PyTorch hooks at layer 20
3. **Validation**: Tested on 1000 samples (500 treatment, 500 control) across 5 prompts

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/formality-steering-gpt2.git
cd formality-steering-gpt2

# Install dependencies
pip install -r requirements.txt

# Run main experiment (generates 1000 samples, ~10 minutes)
python MATS_formality_vectors.py

# Analyze saved results (instant)
python MATS_Analysis_Only.py
