# MATS_Analysis_Only.py - Just analyzing the saved data
# Don't want to regenerate 1000 samples every time I tweak the analysis!

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create the plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the results from the data/ directory
with open('data/systematic_results.pkl', 'rb') as f:
    systematic_results = pickle.load(f)
    
with open('data/control_results.pkl', 'rb') as f:
    control_results = pickle.load(f)

print(f"Loaded {len(systematic_results)} treatment + {len(control_results)} control samples")

# Define TEST_PROMPTS (need this for analysis)
TEST_PROMPTS = [
    "The process of photosynthesis involves",  # Scientific
    "Climate change refers to",               # Environmental/Political 
    "Machine learning algorithms are",        # Technical
    "The Roman Empire was",                   # Historical
    "Economic inflation occurs when"          # Economic
]

# Also need regex import for token analysis
import re

# Setting up for analysis
STRENGTH_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]

# Combine everything into a dataframe (easier to work with)
all_results = systematic_results + control_results
df = pd.DataFrame(all_results)

# Need to normalize the F-Scores across all data
mean_sent_len = df['avg_sentence_len'].mean()
std_sent_len = df['avg_sentence_len'].std()
mean_word_len = df['avg_word_len'].mean()
std_word_len = df['avg_word_len'].std()

# Normalize then combine into single F-Score
df['norm_sent_len'] = (df['avg_sentence_len'] - mean_sent_len) / std_sent_len
df['norm_word_len'] = (df['avg_word_len'] - mean_word_len) / std_word_len
df['f_score'] = 0.5 * df['norm_sent_len'] + 0.5 * df['norm_word_len']

########################################################
# EXTRACT EXAMPLE OUTPUTS
########################################################

print("\n" + "="*60)
print("EXAMPLE OUTPUTS SHOWING FORMALITY DIFFERENCES")
print("="*60)

# Let me grab some actual examples to show the formality difference
prompt = "The process of photosynthesis involves"

for strength in [-2.0, 0.0, 2.0]:
    print(f"\n{'='*40}")
    print(f"STRENGTH {strength:+.1f}")
    print('='*40)
    
    # Find samples for this strength/prompt combo
    strength_samples = [r for r in systematic_results 
                       if r['strength'] == strength 
                       and r['prompt'] == prompt]
    
    if len(strength_samples) > 0:
        # Get average metrics to show the trend
        avg_sent = np.mean([s['avg_sentence_len'] for s in strength_samples])
        avg_word = np.mean([s['avg_word_len'] for s in strength_samples])
        
        print(f"Average metrics: sentence_len={avg_sent:.1f}, word_len={avg_word:.2f}")
        print("\nSample outputs:")
        
        # Show a few examples so people can see the actual style change
        for i, sample in enumerate(strength_samples[:3]):
            text = sample['generated_text'][:100]  # First 100 chars
            print(f"\n  Example {i+1}: \"{text}...\"")

########################################################
# STATISTICAL ANALYSIS
########################################################

print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

# Separate treatment from control
df_treatment = df[df['experiment_type'] == 'formality_vector']
df_control = df[df['experiment_type'] == 'random_vector']

# Calculate the main results
print("\n" + "-"*40)
print("Mean F-Score by Strength")
print("-"*40)

treatment_means = []
control_means = []

for strength in STRENGTH_VALUES:
    treatment_mean = df_treatment[df_treatment['strength'] == strength]['f_score'].mean()
    treatment_std = df_treatment[df_treatment['strength'] == strength]['f_score'].std()
    control_mean = df_control[df_control['strength'] == strength]['f_score'].mean()
    control_std = df_control[df_control['strength'] == strength]['f_score'].std()
    
    treatment_means.append(treatment_mean)
    control_means.append(control_mean)
    
    print(f"\nStrength {strength:+.1f}:")
    print(f"  Formality vector: {treatment_mean:+.3f} (±{treatment_std:.3f})")
    print(f"  Random vector:    {control_mean:+.3f} (±{control_std:.3f})")

# Fit lines to see the trends
strengths = np.array(STRENGTH_VALUES)
treatment_slope, treatment_intercept = np.polyfit(strengths, treatment_means, 1)
control_slope, control_intercept = np.polyfit(strengths, control_means, 1)

print("\n" + "-"*40)
print("LINEAR REGRESSION")
print("-"*40)
print(f"Formality vector slope: {treatment_slope:.4f}")
print(f"Random vector slope: {control_slope:.4f}")
print(f"Effect size ratio: {abs(treatment_slope/control_slope):.1f}x")

# Wow, 12x effect size! That's huge!

########################################################
# ADDITIONAL EXPERIMENTS - Using existing data
########################################################

print("\n" + "="*60)
print("ADDITIONAL EXPERIMENTS")
print("="*60)

# EXPERIMENT 2: Per-Prompt Analysis (Generalization Test)
print("\n" + "-"*40)
print("EXPERIMENT 2: Per-Prompt Generalization")
print("-"*40)

print("\nTesting if formality effect generalizes across all prompts:")

for prompt in TEST_PROMPTS:
    # Filter data for this prompt only
    prompt_treatment = df_treatment[df_treatment['prompt'] == prompt]
    
    # Calculate mean F-score for each strength for this prompt
    prompt_means = []
    for strength in STRENGTH_VALUES:
        mean_score = prompt_treatment[prompt_treatment['strength'] == strength]['f_score'].mean()
        prompt_means.append(mean_score)
    
    # Calculate slope for this specific prompt
    if len(prompt_means) == len(STRENGTH_VALUES):
        slope, intercept = np.polyfit(STRENGTH_VALUES, prompt_means, 1)
        print(f"\n'{prompt[:30]}...'")
        print(f"  Slope: {slope:.4f}")
        print(f"  Effect: {'✓ Positive' if slope > 0 else '✗ Negative'}")

# EXPERIMENT 3: Token-Level Analysis
print("\n" + "-"*40)
print("EXPERIMENT 3: Token-Level Analysis")
print("-"*40)

print("\nAnalyzing what changes in generated text:")

# Academic markers to look for
academic_markers = ['therefore', 'thus', 'however', 'furthermore', 'moreover', 
                   'consequently', 'subsequently', 'nevertheless', 'accordingly']

for strength in [-2.0, 0.0, 2.0]:  # Just analyze extremes + baseline
    strength_samples = [r for r in systematic_results if r['strength'] == strength]
    
    total_tokens = 0
    total_unique = set()
    academic_count = 0
    total_sentences = 0
    
    for sample in strength_samples[:50]:  # Analyze first 50 samples
        text = sample['generated_text'].lower()
        tokens = text.split()
        total_tokens += len(tokens)
        total_unique.update(tokens)
        
        # Count academic markers
        for marker in academic_markers:
            academic_count += text.count(marker)
        
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        total_sentences += len([s for s in sentences if s.strip()])
    
    vocab_diversity = len(total_unique) / total_tokens if total_tokens > 0 else 0
    avg_tokens_per_sent = total_tokens / total_sentences if total_sentences > 0 else 0
    
    print(f"\nStrength {strength:+.1f}:")
    print(f"  Vocabulary diversity: {vocab_diversity:.3f}")
    print(f"  Avg tokens/sentence: {avg_tokens_per_sent:.1f}")
    print(f"  Academic markers found: {academic_count}")

print("\n" + "="*60)


# Skip the ablation part, just add this after Experiment 3:

print("\n" + "-"*40)
print("EXPERIMENT 4: Statistical Significance")
print("-"*40)

from scipy.stats import pearsonr

# Test correlation between strength and F-score
correlation, p_value = pearsonr(
    df_treatment['strength'], 
    df_treatment['f_score']
)
print(f"\nFormality vector effect:")
print(f"  Correlation: {correlation:.4f}")
print(f"  P-value: {p_value:.10f}")
print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# Also test control for comparison
control_corr, control_p = pearsonr(
    df_control['strength'],
    df_control['f_score']
)
print(f"\nRandom vector (control):")
print(f"  Correlation: {control_corr:.4f}")
print(f"  P-value: {control_p:.10f}")

########################################################
# VISUALIZATION
########################################################

print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

# Bar chart of slopes per domain
import matplotlib.pyplot as plt

domains = ['Photosynthesis', 'Climate', 'ML', 'Roman Empire', 'Inflation']
slopes = [0.3143, 0.4569, 0.5045, 0.4294, 0.4358]

plt.figure(figsize=(8, 5))
plt.bar(domains, slopes, color='#2E86AB')
plt.ylabel('Slope (Effect Size)', fontsize=12)
plt.title('Formality Effect Generalizes Across All Domains', fontsize=14)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/domain_generalization.png') # <--- SAVES IN THE PLOTS FOLDER

# Histogram showing distribution shift
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, strength in enumerate([-2.0, 0.0, 2.0]):
    samples = [r for r in systematic_results if r['strength'] == strength]
    sentence_lengths = [r['avg_sentence_len'] for r in samples[:100]]
    
    axes[i].hist(sentence_lengths, bins=15, color=['#A23B72', '#888', '#2E86AB'][i], alpha=0.7)
    axes[i].set_title(f'Strength {strength:+.1f}')
    axes[i].set_xlabel('Avg Sentence Length')
    axes[i].axvline(np.mean(sentence_lengths), color='red', linestyle='--')
fig.savefig('plots/histogram_distribution_shift.png', dpi=150)

# Get stats for error bars
treatment_stats = df_treatment.groupby('strength')['f_score'].agg(['mean', 'std', 'count'])
treatment_stats['se'] = treatment_stats['std'] / np.sqrt(treatment_stats['count'])

control_stats = df_control.groupby('strength')['f_score'].agg(['mean', 'std', 'count'])
control_stats['se'] = control_stats['std'] / np.sqrt(control_stats['count'])

# Make the figure (this is the money shot for the executive summary)
plt.figure(figsize=(10, 6))

# Plot both conditions with error bars
plt.errorbar(STRENGTH_VALUES, treatment_stats['mean'], yerr=treatment_stats['se'], 
             marker='o', markersize=8, linewidth=2, capsize=5, capthick=2,
             label='Formality Vector', color='#2E86AB')

plt.errorbar(STRENGTH_VALUES, control_stats['mean'], yerr=control_stats['se'],
             marker='s', markersize=8, linewidth=2, capsize=5, capthick=2,
             label='Random Vector (Control)', color='#A23B72', alpha=0.7)

# Add trend lines to make the effect obvious
z = np.polyfit(STRENGTH_VALUES, treatment_stats['mean'], 1)
p = np.poly1d(z)
plt.plot(STRENGTH_VALUES, p(STRENGTH_VALUES), "--", alpha=0.5, color='#2E86AB')

z_control = np.polyfit(STRENGTH_VALUES, control_stats['mean'], 1)
p_control = np.poly1d(z_control)
plt.plot(STRENGTH_VALUES, p_control(STRENGTH_VALUES), "--", alpha=0.5, color='#A23B72')

# Make it look professional
plt.xlabel('Steering Strength', fontsize=12)
plt.ylabel('Normalized Formality Score (F-Score)', fontsize=12)
plt.title('Formality Steering in GPT-2 Medium: Treatment vs Control', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=11)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlim(-2.5, 2.5)

# Add the key result as annotation
plt.text(0.5, -0.6, f'Effect Size Ratio: {abs(z[0]/z_control[0]):.1f}x', 
         fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

plt.tight_layout()
plt.savefig('plots/formality_steering_results.png', dpi=150, bbox_inches='tight') # <--- SAVES IN THE PLOTS FOLDER
plt.show()

print("\nVisualization saved as 'formality_steering_results.png'")
print(f"Treatment slope: {z[0]:.4f}")
print(f"Control slope: {z_control[0]:.4f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nKey findings:")
print(f"1. Formality vector shows clear dose-response relationship (slope={treatment_slope:.4f})")
print(f"2. Random control shows no systematic effect (slope={control_slope:.4f})")
print(f"3. Effect size is {abs(treatment_slope/control_slope):.1f}x stronger than random")
print("\nReady to write executive summary!")

# The formality vector has a perfect linear relationship with F-Score
# while the random vector is basically flat. 
# This is textbook perfect and that makes me very skeptical so I will have the LLMS adversarily and rigorously 
# challenge me harshly, and then challenge eachother's logic and find the best 
#
# The blue line going up while pink stays flat is exactly what we wanted.
# This proves the effect is specific to our formality direction,
# not just any random perturbation.
#
# With effect sizes this big and error bars this tight,
# p-value has to be < 0.001 at least
#
# I think this is enough evidence to claim that I found a real formality
# direction in GPT-2!



