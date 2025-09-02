# **Finding and Controlling a Formality Direction in GPT-2 Medium**

This repository contains the code and analysis for my MATS 9.0 application, a mechanistic interpretability investigation into the linear representation of stylistic properties in language models.

**Core Finding:** I successfully isolated a single direction in GPT-2 Medium's activation space that robustly and bidirectionally controls the formality of generated text. The effect is highly specific (3.1x stronger than a random vector control) and demonstrates that abstract stylistic concepts are encoded as simple, manipulable linear features.

**Code and Full Write-up:**
*   **GitHub Repository:** [https://github.com/THATMOZZIE/interpreting-formality-in-gpt2](https://github.com/THATMOZZIE/interpreting-formality-in-gpt2)
*   **Executive Summary:** *https://drive.google.com/file/d/10RncaPf-ciKPA7rREnmJCw1Fz09Prsie/view?usp=sharing*

---

## Key Result: Dose-Dependent Control of Formality

The primary experiment demonstrates a clear, monotonic relationship between the strength of the applied "formality vector" and a quantitative formality score. The random vector control shows no systematic effect, confirming the specificity of the learned direction.

*<img width="1000" height="600" alt="Figure_1 additional" src="https://github.com/user-attachments/assets/e87ad9a6-6643-40ad-942c-eaac8fde1866" />*

---

## Research Vision & Future Directions

This project serves as a successful proof-of-concept. My goal, if selected for MATS, is to build on this work to tackle more safety-critical problems in modern models. My research is guided by the principles of **applied interpretability** and **model biology**.

My future work will focus on three key areas:

**1. Generalizing to Modern Architectures (Model Biology):**
The first priority is to test the universality of this finding. While GPT-2 is a clean baseline, the key question is whether this linear encoding of style is a fundamental property of transformers or an artifact of an older architecture.
*   **Next Step:** Replicate this entire experiment on a modern, instruction-tuned model like **Llama-3 8B**. This will test if the "formality" direction is a consistent feature across different training paradigms (pre-training vs. instruction-tuning).

**2. Application to Safety-Relevant Attributes (Applied Interpretability):**
The ability to control formality is a toy problem that validates a powerful methodology. The ultimate goal is to apply this technique to attributes that are directly relevant to AI safety.
*   **Next Step:** I plan to use the same difference-in-means and steering methodology to attempt to isolate and control vectors for more critical attributes, such as:
    *   **Honesty/Truthfulness:** By comparing activations on prompts where the model is truthful vs. sycophantic.
    *   **Harmfulness/Refusal:** By extending the work of Arditi et al. to different refusal mechanisms.
    *   **Humor:** As a complex, nuanced test case for controlling subjective properties.

**3. Deeper Mechanistic Understanding:**
This project successfully controlled a behavior, but it didn't fully explain the underlying circuit.
*   **Next Step:** I will move beyond simple steering and use techniques like **activation patching** and **path patching** to identify the specific heads and MLP neurons that are most influenced by the formality vector. This will help us move from *controlling* the "formality" knob to *understanding the mechanism* that creates it.

---

## Setup & Reproduction

To reproduce the results from this initial project:

**1. Clone the repository:**
```bash
git clone https://github.com/THATMOZZIE/interpreting-formality-in-gpt2.git
cd interpreting-formality-in-gpt2
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the experiment:**
```bash
# This script runs the full generation and saves results to .pkl files
python MATS_formality_vectors.py

# This script loads the saved results and generates the final plots
python MATS_Analysis_Only.py
```

### **Why This `README` is Stronger:**

*   **It has a "Research Vision" section.** This is the most important change. It explicitly tells me that you're not just done with this project, but that you see it as a stepping stone. It shows you are thinking about the bigger picture.
*   **It uses my language.** You've framed your future work using the terms I care about: "applied interpretability," "model biology," "universality," and "mechanistic understanding." This shows you've engaged with my research philosophy.
*   **It proposes concrete, well-motivated next steps.** You're not just saying "I'll do more research." You're saying, "My next step is to replicate this on Llama-3 to test for generalization across training paradigms." This is a specific, falsifiable, and excellent research plan.
*   **It connects the toy problem to the safety case.** You explicitly state that controlling "formality" is a proof-of-concept for controlling "honesty" or "harmfulness." This makes your work feel much more important and aligned with the goals of the field.
*   **It is professionally structured.** The layout is clean, with a clear separation between the core finding, the future vision, and the practical instructions for running the code.
