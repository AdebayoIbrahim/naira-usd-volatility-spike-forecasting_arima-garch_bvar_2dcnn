#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 08:42:00 2026

@author: setup
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Load the final CNN test predictions
file_path = Path("results/cnn_A_final_test_predictions.csv")
df = pd.read_csv(file_path)

# Extract actual and predicted spike labels
y_true = df["actual_spike"].astype(int)
y_pred = df["predicted_spike"].astype(int)

# Generate confusion matrix: [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Print the classification counts
tn, fp, fn, tp = cm.ravel()

print("CNN Confusion Matrix")
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")
print(f"Total observations:   {cm.sum()}")

# Plot the confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-spike", "Spike"]
)

fig, ax = plt.subplots(figsize=(7, 6))
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title(
    "Confusion Matrix for Multivariate 2D-CNN",
    fontsize=13,
    pad=15
)
ax.set_xlabel("Predicted Class", fontsize=11)
ax.set_ylabel("Actual Class", fontsize=11)

plt.tight_layout()

# Save the figure for your thesis
output_path = Path("results/cnn_A_final_confusion_matrix.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nConfusion matrix saved to: {output_path}")