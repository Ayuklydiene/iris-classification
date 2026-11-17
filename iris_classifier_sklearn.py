"""
Iris Dataset Classification & Complete Evaluation Metrics (scikit-learn Implementation)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)
import numpy as np

def main():
    # 1. Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names

    # 2. Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train Classifier (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Predictions & Probability estimates
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 5. Calculate all evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro')
    prec_weighted = precision_score(y_test, y_pred, average='weighted')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    rec_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_mac = f1_score(y_test, y_pred, average='macro')
    f1_wt = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    auc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr')
    loss = log_loss(y_test, y_prob)

    # 6. Display evaluation metrics
    print("=" * 65)
    print("      SCIKIT-LEARN IRIS CLASSIFICATION EVALUATION METRICS     ")
    print("=" * 65)
    print(f"Model                  : RandomForestClassifier")
    print(f"Total Test Samples     : {len(y_test)}")
    print(f"Accuracy               : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Log Loss (Cross-Entropy: {loss:.4f}")
    print(f"ROC-AUC Score (OvR)    : {auc_ovr:.4f}")
    print("-" * 65)
    print(f"Macro Precision        : {prec_macro:.4f}")
    print(f"Macro Recall           : {rec_macro:.4f}")
    print(f"Macro F1-Score         : {f1_mac:.4f}")
    print("-" * 65)
    print(f"Weighted Precision     : {prec_weighted:.4f}")
    print(f"Weighted Recall        : {rec_weighted:.4f}")
    print(f"Weighted F1-Score      : {f1_wt:.4f}")
    print("\n[CONFUSION MATRIX]")
    print(cm)
    print("\n[DETAILED CLASSIFICATION REPORT]")
    print(report)
    print("=" * 65)

if __name__ == '__main__':
    main()
