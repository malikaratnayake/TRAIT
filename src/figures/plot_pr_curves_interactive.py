import pandas as pd
import plotly.graph_objects as go
import os

# Base directory containing batch size study results
base_dir = "/Users/mrat0010/Documents/GitHub/TRAIT/data/results-Transformer/results/trait_results_review/convtran_batch_size_study"

# Batch size directories and corresponding colors (categorical palette - designed for accessibility)
batch_sizes = {
    "bs_16": "#2E7D32",   # Green
    "bs_32": "#1976D2",   # Blue
    "bs_64": "#D32F2F",   # Red
    "bs_96": "#F57C00"    # Orange
}

# Create figure
fig = go.Figure()

# Load and plot PR curves for each batch size
for bs_dir, color in batch_sizes.items():
    csv_path = os.path.join(base_dir, bs_dir, f"convtran_{bs_dir}_pr_auc_pr_curve.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Get AP value (same for all rows)
        ap = df['average_precision'].iloc[0]
        n_positive = df['n_positive'].iloc[0]
        n_total = df['n_total'].iloc[0]

        # Sort by recall for proper line plotting
        df_sorted = df.sort_values('recall')

        # Add trace
        fig.add_trace(go.Scatter(
            x=df_sorted['recall'],
            y=df_sorted['precision'],
            mode='lines+markers',
            name=f"{bs_dir}",
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color),
            hovertemplate=(
                f"<b>{bs_dir}</b><br>" +
                "Recall: %{x:.3f}<br>" +
                "Precision: %{y:.3f}<br>" +
                f"AP: {ap:.4f}<br>" +
                f"Samples: {n_positive}/{n_total}<extra></extra>"
            )
        ))

        print(f"Loaded {bs_dir}: AP={ap:.4f}, pos={n_positive}, total={n_total}")

# Update layout
fig.update_layout(
    title={
        'text': 'Precision-Recall Curves: ConvTran Batch Size Study',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis=dict(
        title='Recall',
        gridcolor='rgba(200, 200, 200, 0.2)',
        showgrid=True,
        zeroline=False,
        range=[0, 1]
    ),
    yaxis=dict(
        title='Precision',
        gridcolor='rgba(200, 200, 200, 0.2)',
        showgrid=True,
        zeroline=False,
        range=[0, 1]
    ),
    hovermode='closest',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12, family='Arial, sans-serif'),
    legend=dict(
        x=0.98,
        y=0.02,
        xanchor='right',
        yanchor='bottom',
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='#ccc',
        borderwidth=1
    ),
    width=1000,
    height=700,
    margin=dict(l=60, r=40, t=80, b=60)
)

# Save as HTML
output_html = "/Users/mrat0010/Documents/GitHub/TRAIT/data/results-Transformer/results/trait_results_review/convtran_batch_size_study/convtran_pr_curves_interactive.html"
fig.write_html(output_html)
print(f"\nSaved interactive plot to: {output_html}")
