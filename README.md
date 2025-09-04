# **Finding and Controlling a Formality Direction in GPT-2 Medium**

This repository contains the code and analysis for my MATS 9.0 application, a mechanistic interpretability investigation into the linear representation of stylistic properties in language models.

*   **Executive Summary:** [http://bit.ly/47mWVn4](http://bit.ly/47mWVn4)
*   **Narrative Write-up on Medium:** [http://bit.ly/41FnjVD](http://bit.ly/41FnjVD)

---

## Key Finding: Dose-Dependent Control of Formality

I successfully isolated a single direction in GPT-2 Medium's activation space that robustly and bidirectionally controls the formality of generated text. The primary experiment demonstrates a clear, monotonic relationship between the strength of the applied "formality vector" and a quantitative formality score. The effect is highly specific (over 3x stronger than a random vector control), proving that this abstract stylistic concept is encoded as a simple, manipulable linear feature.

<img width="1000" height="600" alt="Figure_1 additional" src="https://github.com/user-attachments/assets/20a7c0e7-15b7-42dd-82cb-d1b71be691e6" />

---

## Research Vision & Future Directions

This project serves as a successful proof-of-concept. My goal, if selected for MATS, is to build on this work to tackle more safety-critical problems in modern models, guided by the principles of **applied interpretability** and **model biology**. My future work will focus on three key areas:

**1. Generalizing to Modern Architectures (Model Biology):**
The first priority is to test the universality of this finding.
*   **Next Step:** Replicate this entire experiment on a modern, instruction-tuned model like **Llama-3 8B** to test if the "formality" direction is a consistent feature across different training paradigms.

**2. Application to Safety-Relevant Attributes (Applied Interpretability):**
The ultimate goal is to apply this methodology to attributes that are directly relevant to AI safety.
*   **Next Step:** I plan to use the same difference-in-means and steering methodology to isolate and control vectors for more critical attributes, such as **Honesty/Truthfulness**, **Harmfulness/Refusal**, and **Humor**.

**3. Deeper Mechanistic Understanding:**
This project successfully controlled a behavior but didn't fully explain the underlying circuit.
*   **Next Step:** I will move beyond simple steering and use techniques like **activation patching** and **path patching** to identify the specific heads and MLP neurons that are most influenced by the formality vector, moving from *controlling* the behavior to *understanding the mechanism*.

---

## Project Structure
```
interpreting-formality-in-gpt2/
│
├── README.md                   # Main project write-up
├── requirements.txt            # Project dependencies
│
├── MATS_formality_vectors.py   # Main script to generate the vector and run all experiments
├── MATS_Analysis_Only.py       # Analysis script to load saved data and reproduce plots/stats
│
├── data/
│   ├── systematic_results.pkl  # Saved data from the formality vector experiment (Treatment)
│   └── control_results.pkl     # Saved data from the random vector experiment (Control)
│
└── plots/
    ├── formality_steering_results.png # Main output graph (displayed above)
    └── ... other generated plots ...
```

---

## Reproducing the Results

To reproduce the findings from scratch, clone the repository and run the scripts in order.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/THATMOZZIE/interpreting-formality-in-gpt2.git
    cd interpreting-formality-in-gpt2
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the full pipeline:**

    First, run the data generation script. This will create the steering vector, run both the treatment (formality vector) and control (random vector) experiments (1000 generations total), and save the results into the `data/` folder. This will take several minutes.
    ```bash
    python MATS_formality_vectors.py
    ```
    Next, run the analysis script. This will load the generated data from `data/` and create the final plots in the `plots/` directory.
    ```bash
    python MATS_Analysis_Only.py
    ```
    
#### A Note for Notebook Environments (Jupyter, Colab)

If you are running this project inside a notebook, the commands should be adapted as follows:

*   **To install dependencies**, run this in a cell to ensure they are installed into the correct kernel environment:
    ```python
    import sys
    !{sys.executable} -m pip install -r requirements.txt
    ```
    *Remember to restart the kernel after the installation is complete.*

*   **To run the scripts**, use the `%run` magic command in a cell:
    ```python
    %run MATS_formality_vectors.py
    ```
    and then:
    ```python
    %run MATS_Analysis_Only.py
    ```

---

### A Note on Reproducibility

The results presented in the submitted executive summary were generated prior to fixing a random seed. Due to the stochastic nature of text generation (`do_sample=True`) and the random initialization of the control vector, the exact numerical values (e.g., F-scores, regression slopes) will vary slightly on each run.

However, the core scientific conclusion is robust and consistently reproduces across different random seeds:

1.  **The formality vector demonstrates a strong, statistically significant, dose-dependent effect on the formality of generated text.**
2.  **This effect is specific, proving to be significantly stronger than a random vector control of the same norm.**

For full numerical reproducibility of future runs, a random seed (`seed = 42`) has now been implemented in the main script, `MATS_formality_vectors.py`.


---

## Limitations

This was a time-constrained project that successfully established a proof-of-concept, but it has several important limitations that I plan to address in future work:

1.  **Proxy Metric for Formality:** The "F-Score" (a combination of sentence and word length) is a crude proxy for the abstract concept of formality. While effective, the vector could be interpreted as a "verbosity vector" rather than a pure "formality vector." Future work should use more sophisticated classifiers or human evaluations to validate the effect.
2.  **Noisy Vector Representation:** Vocabulary projection revealed that the "formal" pole of the vector was contaminated with tokenization artifacts (e.g., Japanese characters, HTML tokens). While the vector worked despite this noise, future work should focus on cleaning the vector to isolate a purer representation of formality.
3.  **Lack of Mechanistic Explanation:** The project demonstrates *that* steering at layer 20 works, but not *why*. The next crucial step is a deeper analysis (e.g., activation patching) to identify the specific model components that utilize this directional information.

---

## Citation

This work builds upon the general methodology of activation steering. The most direct inspiration for using difference-in-means to find a stylistic/behavioral vector comes from the paper:

*   "Refusal in Large Language Models as Mediated by a Single Direction" by Arditi et al. (2024).
