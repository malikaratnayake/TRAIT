import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Base directory containing batch size study results
base_dir = "/Users/mrat0010/Documents/GitHub/TRAIT/data/results-Transformer/results/trait_results_review/convtran_batch_size_study"

# Batch size directories and corresponding colors (categorical palette)
batch_sizes = {
    "bs_16": "#2E7D32",   # Green
    "bs_32": "#1976D2",   # Blue
    "bs_64": "#D32F2F",   # Red
    "bs_96": "#F57C00"    # Orange
}

# Load PR curves for each batch size
pr_curves = {}
for bs_dir, color in batch_sizes.items():
    csv_path = os.path.join(base_dir, bs_dir, f"convtran_{bs_dir}_pr_auc_pr_curve.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        pr_curves[bs_dir] = {"data": df, "color": color}
        print(f"Loaded {bs_dir}: {len(df)} points, AP={df['average_precision'].iloc[0]:.4f}")
    else:
        print(f"Warning: {csv_path} not found")

# Create figure with proper styling
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

# Plot PR curves for each batch size
for bs_label, info in pr_curves.items():
    df = info["data"]
    color = info["color"]
    ap = df['average_precision'].iloc[0]

    # Sort by recall for proper line plotting
    df_sorted = df.sort_values('recall')

    ax.plot(df_sorted['recall'], df_sorted['precision'],
            linewidth=2.5, color=color, label=f"{bs_label} (AP={ap:.3f})",
            marker='o', markersize=6, markeredgewidth=0, alpha=0.85)

# Configure axes
ax.set_xlabel('Recall', fontsize=12, fontweight='500')
ax.set_ylabel('Precision', fontsize=12, fontweight='500')
ax.set_title('Precision-Recall Curves: ConvTran Batch Size Study',
             fontsize=14, fontweight='600', pad=16)

# Set axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Add grid
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8)
ax.set_axisbelow(True)

# Configure legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
          edgecolor='#ccc', fancybox=False)

# Style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#ccc')
ax.spines['bottom'].set_color('#ccc')

plt.tight_layout()

# Save figure
output_path = "/Users/mrat0010/Documents/GitHub/TRAIT/data/results-Transformer/results/trait_results_review/convtran_batch_size_study/convtran_pr_curves_comparison.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nSaved plot to: {output_path}")

plt.show()
