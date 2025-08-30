# Formality Steering in GPT-2 Medium

Finding and controlling a formality direction in GPT-2 Medium's activation space through mechanistic interpretability.

## Overview

This project demonstrates that abstract stylistic properties like "formality" can be represented as linear directions in language model activation space. By identifying and manipulating this direction, we can control the formality of generated text without model retraining.

**Key Result:** We achieve a 12.2x stronger effect on formality compared to random control vectors, with clear behavioral changes from casual ("a bunch of stuff") to formal ("the production of chemical energy") language.

![Formality Steering Results](formality_steering_results.png)

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
