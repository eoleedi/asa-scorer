#!/usr/bin/env python3
"""Plot data distribution for EZAI dataset"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load scores
with open('data/ezai-championship2023/ezai-champ2023/scores.json', 'r') as f:
    scores = json.load(f)

# Separate train and test data
train_fluency = []
train_prosodic = []
test_fluency = []
test_prosodic = []

for key, value in scores.items():
    if key.startswith('train_'):
        train_fluency.append(value['fluency'])
        train_prosodic.append(value['prosodic'])
    elif key.startswith('test_'):
        test_fluency.append(value['fluency'])
        test_prosodic.append(value['prosodic'])

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('EZAI Dataset Score Distribution', fontsize=16, fontweight='bold')

# Train Fluency Distribution
axes[0, 0].hist(train_fluency, bins=11, range=(-0.5, 10.5), edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Train - Fluency Score')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_xticks(range(0, 11))
axes[0, 0].grid(axis='y', alpha=0.3)

# Train Prosodic Distribution
axes[0, 1].hist(train_prosodic, bins=11, range=(-0.5, 10.5), edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_title('Train - Prosodic Score')
axes[0, 1].set_xlabel('Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_xticks(range(0, 11))
axes[0, 1].grid(axis='y', alpha=0.3)

# Train Score Scatter
axes[0, 2].scatter(train_fluency, train_prosodic, alpha=0.5, s=20, color='green')
axes[0, 2].set_title('Train - Fluency vs Prosodic')
axes[0, 2].set_xlabel('Fluency Score')
axes[0, 2].set_ylabel('Prosodic Score')
axes[0, 2].set_xlim(-0.5, 10.5)
axes[0, 2].set_ylim(-0.5, 10.5)
axes[0, 2].grid(alpha=0.3)
axes[0, 2].plot([0, 10], [0, 10], 'r--', alpha=0.3, linewidth=1)

# Test Fluency Distribution
axes[1, 0].hist(test_fluency, bins=11, range=(-0.5, 10.5), edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].set_title('Test - Fluency Score')
axes[1, 0].set_xlabel('Score')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xticks(range(0, 11))
axes[1, 0].grid(axis='y', alpha=0.3)

# Test Prosodic Distribution
axes[1, 1].hist(test_prosodic, bins=11, range=(-0.5, 10.5), edgecolor='black', alpha=0.7, color='coral')
axes[1, 1].set_title('Test - Prosodic Score')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_xticks(range(0, 11))
axes[1, 1].grid(axis='y', alpha=0.3)

# Test Score Scatter
axes[1, 2].scatter(test_fluency, test_prosodic, alpha=0.5, s=20, color='purple')
axes[1, 2].set_title('Test - Fluency vs Prosodic')
axes[1, 2].set_xlabel('Fluency Score')
axes[1, 2].set_ylabel('Prosodic Score')
axes[1, 2].set_xlim(-0.5, 10.5)
axes[1, 2].set_ylim(-0.5, 10.5)
axes[1, 2].grid(alpha=0.3)
axes[1, 2].plot([0, 10], [0, 10], 'r--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('ezai_data_distribution.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: ezai_data_distribution.png")

# Print statistics
print("\n=== Dataset Statistics ===")
print(f"\nTrain Set (n={len(train_fluency)}):")
print(f"  Fluency   - Mean: {np.mean(train_fluency):.2f}, Std: {np.std(train_fluency):.2f}, Range: [{min(train_fluency):.0f}, {max(train_fluency):.0f}]")
print(f"  Prosodic  - Mean: {np.mean(train_prosodic):.2f}, Std: {np.std(train_prosodic):.2f}, Range: [{min(train_prosodic):.0f}, {max(train_prosodic):.0f}]")

print(f"\nTest Set (n={len(test_fluency)}):")
print(f"  Fluency   - Mean: {np.mean(test_fluency):.2f}, Std: {np.std(test_fluency):.2f}, Range: [{min(test_fluency):.0f}, {max(test_fluency):.0f}]")
print(f"  Prosodic  - Mean: {np.mean(test_prosodic):.2f}, Std: {np.std(test_prosodic):.2f}, Range: [{min(test_prosodic):.0f}, {max(test_prosodic):.0f}]")

# Show score frequency
print("\n=== Score Frequency (Train) ===")
print("Fluency:", dict(sorted(Counter(train_fluency).items())))
print("Prosodic:", dict(sorted(Counter(train_prosodic).items())))
