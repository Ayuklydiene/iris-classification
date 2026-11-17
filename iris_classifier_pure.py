"""
Iris Dataset Classification & Complete Evaluation Metrics (Pure Python Implementation)
This script demonstrates the Iris classification task and computes all standard evaluation metrics
using pure Python (no external C-extensions required), guaranteeing compatibility with Windows Smart App Control.
"""

import math
import random
from collections import Counter

# 1. Embedded Iris Dataset (150 samples: 50 Setosa, 50 Versicolor, 50 Virginica)
# Features: [sepal length, sepal width, petal length, petal width]
# Class labels: 0: 'setosa', 1: 'versicolor', 2: 'virginica'
RAW_DATA = [
    [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0], [4.6, 3.1, 1.5, 0.2, 0],
    [5.0, 3.6, 1.4, 0.2, 0], [5.4, 3.9, 1.7, 0.4, 0], [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0],
    [4.4, 2.9, 1.4, 0.2, 0], [4.9, 3.1, 1.5, 0.1, 0], [5.4, 3.7, 1.5, 0.2, 0], [4.8, 3.4, 1.6, 0.2, 0],
    [4.8, 3.0, 1.4, 0.1, 0], [4.3, 3.0, 1.1, 0.1, 0], [5.8, 4.0, 1.2, 0.2, 0], [5.7, 4.4, 1.5, 0.4, 0],
    [5.4, 3.9, 1.3, 0.4, 0], [5.1, 3.5, 1.4, 0.3, 0], [5.7, 3.8, 1.7, 0.3, 0], [5.1, 3.8, 1.5, 0.3, 0],
    [5.4, 3.4, 1.7, 0.2, 0], [5.1, 3.7, 1.5, 0.4, 0], [4.6, 3.6, 1.0, 0.2, 0], [5.1, 3.3, 1.7, 0.5, 0],
    [4.8, 3.4, 1.9, 0.2, 0], [5.0, 3.0, 1.6, 0.2, 0], [5.0, 3.4, 1.6, 0.4, 0], [5.2, 3.5, 1.5, 0.2, 0],
    [5.2, 3.4, 1.4, 0.2, 0], [4.7, 3.2, 1.6, 0.2, 0], [4.8, 3.1, 1.6, 0.2, 0], [5.4, 3.4, 1.5, 0.4, 0],
    [5.2, 4.1, 1.5, 0.1, 0], [5.5, 4.2, 1.4, 0.2, 0], [4.9, 3.1, 1.5, 0.2, 0], [5.0, 3.2, 1.2, 0.2, 0],
    [5.5, 3.5, 1.3, 0.2, 0], [4.9, 3.6, 1.4, 0.1, 0], [4.4, 3.0, 1.3, 0.2, 0], [5.1, 3.4, 1.5, 0.2, 0],
    [5.0, 3.5, 1.3, 0.3, 0], [4.5, 2.3, 1.3, 0.3, 0], [4.4, 3.2, 1.3, 0.2, 0], [5.0, 3.5, 1.6, 0.6, 0],
    [5.1, 3.8, 1.9, 0.4, 0], [4.8, 3.0, 1.4, 0.3, 0], [5.1, 3.8, 1.6, 0.2, 0], [4.6, 3.2, 1.4, 0.2, 0],
    [5.3, 3.7, 1.5, 0.2, 0], [5.0, 3.3, 1.4, 0.2, 0],
    # Versicolor
    [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1], [6.9, 3.1, 4.9, 1.5, 1], [5.5, 2.3, 4.0, 1.3, 1],
    [6.5, 2.8, 4.6, 1.5, 1], [5.7, 2.8, 4.5, 1.3, 1], [6.3, 3.3, 4.7, 1.6, 1], [4.9, 2.4, 3.3, 1.0, 1],
    [6.6, 2.9, 4.6, 1.3, 1], [5.2, 2.7, 3.9, 1.4, 1], [5.0, 2.0, 3.5, 1.0, 1], [5.9, 3.0, 4.2, 1.5, 1],
    [6.0, 2.2, 4.0, 1.0, 1], [6.1, 2.9, 4.7, 1.4, 1], [5.6, 2.9, 3.6, 1.3, 1], [6.7, 3.1, 4.4, 1.4, 1],
    [5.6, 3.0, 4.5, 1.5, 1], [5.8, 2.7, 4.1, 1.0, 1], [6.2, 2.2, 4.5, 1.5, 1], [5.6, 2.5, 3.9, 1.1, 1],
    [5.9, 3.2, 4.8, 1.8, 1], [6.1, 2.8, 4.0, 1.3, 1], [6.3, 2.5, 4.9, 1.5, 1], [6.1, 2.8, 4.7, 1.2, 1],
    [6.4, 2.9, 4.3, 1.3, 1], [6.6, 3.0, 4.4, 1.4, 1], [6.8, 2.8, 4.8, 1.4, 1], [6.7, 3.0, 5.0, 1.7, 1],
    [6.0, 2.9, 4.5, 1.5, 1], [5.7, 2.6, 3.5, 1.0, 1], [5.5, 2.4, 3.8, 1.1, 1], [5.5, 2.4, 3.7, 1.0, 1],
    [5.8, 2.7, 3.9, 1.2, 1], [6.0, 2.7, 5.1, 1.6, 1], [5.4, 3.0, 4.5, 1.5, 1], [6.0, 3.4, 4.5, 1.6, 1],
    [6.7, 3.1, 4.7, 1.5, 1], [6.3, 2.3, 4.4, 1.3, 1], [5.6, 3.0, 4.1, 1.3, 1], [5.5, 2.5, 4.0, 1.3, 1],
    [5.5, 2.6, 4.4, 1.2, 1], [6.1, 3.0, 4.6, 1.4, 1], [5.8, 2.6, 4.0, 1.2, 1], [5.0, 2.3, 3.3, 1.0, 1],
    [5.6, 2.7, 4.2, 1.3, 1], [5.7, 3.0, 4.2, 1.2, 1], [5.7, 2.9, 4.2, 1.3, 1], [6.2, 2.9, 4.3, 1.3, 1],
    [5.1, 2.5, 3.0, 1.1, 1], [5.7, 2.8, 4.1, 1.3, 1],
    # Virginica
    [6.3, 3.3, 6.0, 2.5, 2], [5.8, 2.7, 5.1, 1.9, 2], [7.1, 3.0, 5.9, 2.1, 2], [6.3, 2.9, 5.6, 1.8, 2],
    [6.5, 3.0, 5.8, 2.2, 2], [7.6, 3.0, 6.6, 2.1, 2], [4.9, 2.5, 4.5, 1.7, 2], [7.3, 2.9, 6.3, 1.8, 2],
    [6.7, 2.5, 5.8, 1.8, 2], [7.2, 3.6, 6.1, 2.5, 2], [6.5, 3.2, 5.1, 2.0, 2], [6.4, 2.7, 5.3, 1.9, 2],
    [6.8, 3.0, 5.5, 2.1, 2], [5.7, 2.5, 5.0, 2.0, 2], [5.8, 2.8, 5.1, 2.4, 2], [6.4, 3.2, 5.3, 2.3, 2],
    [6.5, 3.0, 5.5, 1.8, 2], [7.7, 3.8, 6.7, 2.2, 2], [7.7, 2.6, 6.9, 2.3, 2], [6.0, 2.2, 5.0, 1.5, 2],
    [6.9, 3.2, 5.7, 2.3, 2], [5.6, 2.8, 4.9, 2.0, 2], [7.7, 2.8, 6.7, 2.0, 2], [6.3, 2.7, 4.9, 1.8, 2],
    [6.7, 3.3, 5.7, 2.1, 2], [7.2, 3.2, 6.0, 1.8, 2], [6.2, 2.8, 4.8, 1.8, 2], [6.1, 3.0, 4.9, 1.8, 2],
    [6.4, 2.8, 5.6, 2.1, 2], [7.2, 3.0, 5.8, 1.6, 2], [7.4, 2.8, 6.1, 1.9, 2], [7.9, 3.8, 6.4, 2.0, 2],
    [6.4, 2.8, 5.6, 2.2, 2], [6.3, 2.8, 5.1, 1.5, 2], [6.1, 2.6, 5.6, 1.4, 2], [7.7, 3.0, 6.1, 2.3, 2],
    [6.3, 3.4, 5.6, 2.4, 2], [6.4, 3.1, 5.5, 1.8, 2], [6.0, 3.0, 4.8, 1.8, 2], [6.9, 3.1, 5.4, 2.1, 2],
    [6.7, 3.1, 5.6, 2.4, 2], [6.9, 3.1, 5.1, 2.3, 2], [5.8, 2.7, 5.1, 1.9, 2], [6.8, 3.2, 5.9, 2.3, 2],
    [6.7, 3.3, 5.7, 2.5, 2], [6.7, 3.0, 5.2, 2.3, 2], [6.3, 2.5, 5.0, 1.9, 2], [6.5, 3.0, 5.2, 2.0, 2],
    [6.2, 3.4, 5.4, 2.3, 2], [5.9, 3.0, 5.1, 1.8, 2]
]

CLASS_NAMES = ['setosa', 'versicolor', 'virginica']
FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# 2. Train-Test Split (80% Train, 20% Test with fixed random seed)
random.seed(42)
shuffled_data = RAW_DATA[:]
random.shuffle(shuffled_data)

split_idx = int(len(shuffled_data) * 0.8)
train_data = shuffled_data[:split_idx]
test_data = shuffled_data[split_idx:]

X_train = [row[:4] for row in train_data]
y_train = [row[4] for row in train_data]

X_test = [row[:4] for row in test_data]
y_test = [row[4] for row in test_data]


# 3. Classifier: k-Nearest Neighbors (k=5 with Euclidean distance)
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, sample):
        distances = []
        for train_sample, label in zip(self.X_train, self.y_train):
            d = euclidean_distance(sample, train_sample)
            distances.append((d, label))
        distances.sort(key=lambda item: item[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return [self.predict_one(sample) for sample in X]

# Train the model
model = KNNClassifier(k=5)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)


# 4. Evaluation Metrics Calculation
num_classes = len(CLASS_NAMES)

# Confusion Matrix: row = actual, col = predicted
confusion_mat = [[0] * num_classes for _ in range(num_classes)]
for actual, pred in zip(y_test, y_pred):
    confusion_mat[actual][pred] += 1

# Total Accuracy
correct = sum(1 for actual, pred in zip(y_test, y_pred) if actual == pred)
total = len(y_test)
accuracy = correct / total

# Per-class metrics (One-vs-Rest)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 = 2 * Precision * Recall / (Precision + Recall)
class_metrics = []
for c in range(num_classes):
    tp = confusion_mat[c][c]
    fp = sum(confusion_mat[r][c] for r in range(num_classes) if r != c)
    fn = sum(confusion_mat[c][col] for col in range(num_classes) if col != c)
    tn = total - (tp + fp + fn)
    support = sum(confusion_mat[c])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    class_metrics.append({
        'name': CLASS_NAMES[c],
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    })

# Macro Average
macro_precision = sum(m['precision'] for m in class_metrics) / num_classes
macro_recall = sum(m['recall'] for m in class_metrics) / num_classes
macro_f1 = sum(m['f1'] for m in class_metrics) / num_classes

# Weighted Average
weighted_precision = sum(m['precision'] * m['support'] for m in class_metrics) / total
weighted_recall = sum(m['recall'] * m['support'] for m in class_metrics) / total
weighted_f1 = sum(m['f1'] * m['support'] for m in class_metrics) / total


# 5. Formatted Display of Results
print("=" * 65)
print("       IRIS DATASET CLASSIFICATION & EVALUATION METRICS       ")
print("=" * 65)
print(f"Dataset Total Samples : {len(RAW_DATA)}")
print(f"Training Set Size    : {len(X_train)} samples (80%)")
print(f"Testing Set Size     : {len(X_test)} samples (20%)")
print(f"Classifier Algorithm : k-Nearest Neighbors (k=5)")
print("-" * 65)

print("\n[1] OVERALL ACCURACY:")
print(f"    Accuracy Score: {accuracy:.4f} ({correct}/{total} correct predictions)")

print("\n[2] CONFUSION MATRIX:")
print(f"    {'Predicted ->':>15}  {'setosa':>8} {'versicolor':>10} {'virginica':>10}")
print("    " + "-" * 48)
for i, row in enumerate(confusion_mat):
    print(f"    {CLASS_NAMES[i]:>12} | {row[0]:>8} {row[1]:>10} {row[2]:>10}")

print("\n[3] PER-CLASS BREAKDOWN (TP, FP, FN, TN):")
print(f"    {'Class':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
print("    " + "-" * 32)
for m in class_metrics:
    print(f"    {m['name']:<12} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}")

print("\n[4] CLASSIFICATION REPORT:")
print(f"    {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("    " + "-" * 56)
for m in class_metrics:
    print(f"    {m['name']:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
print("    " + "-" * 56)
print(f"    {'accuracy':<14} {'':>10} {'':>10} {accuracy:>10.4f} {total:>10}")
print(f"    {'macro avg':<14} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f} {total:>10}")
print(f"    {'weighted avg':<14} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {total:>10}")
print("=" * 65)
