
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# Load test data and trained Random Forest
X_test = joblib.load("artifacts/X_test.pkl")
y_test = joblib.load("artifacts/y_test.pkl")
model = joblib.load("artifacts/random_forest.pkl")

# Generate predictions
y_pred = model.predict(X_test)

# Keep class order consistent: BENIGN (0), BOT (1)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["BENIGN", "BOT"],
    yticklabels=["BENIGN", "BOT"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    "artifacts/random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Confusion matrix saved successfully!")
