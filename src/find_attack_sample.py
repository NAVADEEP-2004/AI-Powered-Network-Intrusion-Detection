
import joblib

# Load the actual labels of the test dataset
y_test = joblib.load("artifacts/y_test.pkl")

# Find test samples labelled as attacks (label 1)
attack_indices = [
    i for i, label in enumerate(y_test)
    if label == 1
]

print("Total attack samples:", len(attack_indices))
print("First 10 attack sample indices:", attack_indices[:10])
