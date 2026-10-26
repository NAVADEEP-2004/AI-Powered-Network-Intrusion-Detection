
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Load the trained Random Forest model
model = joblib.load(ARTIFACTS_DIR / "random_forest.pkl")

# Get feature importance values
importance_values = model.feature_importances_

# Get the feature names used during training
feature_names = model.feature_names_in_

# Create a table of feature importance
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort from most important to least important
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display the top 15 features
top_features = importance_df.head(15)

print("\n=== TOP 15 IMPORTANT FEATURES ===")
print(top_features.to_string(index=False))

# Save the results
importance_df.to_csv(
    ARTIFACTS_DIR / "feature_importance.csv",
    index=False
)

# Plot the top 15 features
plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Network Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    ARTIFACTS_DIR / "feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance saved successfully!")
