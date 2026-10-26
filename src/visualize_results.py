
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the comparison generated in Step 9
df = pd.read_csv("artifacts/model_comparison.csv")

# Select the metrics we want to compare
metrics = [
    "Attack Precision",
    "Attack Recall",
    "Attack F1-score",
    "Balanced Accuracy"
]

# Convert metrics to percentages for easier reading
plot_df = df.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

# Create a comparison chart
plt.figure(figsize=(12, 6))

sns.barplot(
    data=plot_df,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("NIDS Model Performance Comparison")
plt.ylabel("Score (0 to 1)")
plt.xlabel("Evaluation Metric")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()

# Save the chart for your report
plt.savefig(
    "artifacts/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Performance chart saved successfully!")
