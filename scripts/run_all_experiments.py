#!/usr/bin/env python3
"""
Run all experiments for KV Cache assignment
"""

import os
import sys

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("="*70)
print("KV CACHE ASSIGNMENT - RUNNING ALL EXPERIMENTS")
print("="*70)

# Experiment 1: Cache comparison
print("\n\n" + "="*70)
print("EXPERIMENT 1: WITH/WITHOUT KV CACHE")
print("="*70)
from experiment1_cache_comparison import experiment_cache_comparison
exp1_results, exp1_df = experiment_cache_comparison(max_new_tokens=100)

# Experiment 2: Batching
print("\n\n" + "="*70)
print("EXPERIMENT 2: BATCHING EFFECTS")
print("="*70)
from experiment2_batching import experiment_batching
exp2_results, exp2_df = experiment_batching(max_new_tokens=50)

# Experiment 3: Sequence scaling
print("\n\n" + "="*70)
print("EXPERIMENT 3: SEQUENCE LENGTH SCALING")
print("="*70)
from experiment3_sequence_scaling import experiment_sequence_scaling
exp3_results, exp3_df = experiment_sequence_scaling(max_new_tokens=50)

print("\n\n" + "="*70)
print("ALL EXPERIMENTS COMPLETED!")
print("="*70)
print("\nResults saved to:")
print("  - logs/experiment1_results.json")
print("  - logs/experiment2_results.json")
print("  - logs/experiment3_results.json")
print("\nPlots saved to:")
print("  - plots/experiment1_comparison.png")
print("  - plots/experiment2_batching.png")
print("  - plots/experiment3_sequence_scaling.png")