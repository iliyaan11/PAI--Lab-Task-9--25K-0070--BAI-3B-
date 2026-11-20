import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

classes, counts = np.unique(y, return_counts=True)
print(f"Total samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Class counts -> Malignant (0): {counts[0]}, Benign (1): {counts[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_scaled, y_train)

base_acc = clf.score(X_test_scaled, y_test)
print(f"Test Accuracy: {base_acc:.4f}")

probs = clf.predict_proba(X_test_scaled)[:, 1]

for threshold in [0.3, 0.5, 0.7]:
    preds = (probs >= threshold).astype(int)
    class_1_count = np.sum(preds == 1)
    acc = accuracy_score(y_test, preds)
    print(f"Threshold: {threshold} | Predicted Class 1: {class_1_count} | Accuracy: {acc:.4f}")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

decision_scores = clf.decision_function(X_test_scaled)
custom_probs = sigmoid(decision_scores)

matches = np.allclose(custom_probs, probs)
print(f"Custom sigmoid matches predict_proba? {matches}")