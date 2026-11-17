# Iris Flower Classification & Model Evaluation

A web application and machine learning benchmark for classifying Iris flower species and calculating standard classification evaluation metrics.

---

## 📌 Project Overview

This project implements a classification model using the Iris dataset (150 samples across 3 species: *Setosa*, *Versicolor*, and *Virginica*). 

The dataset is partitioned into an **80% training set (120 samples)** and a **20% test set (30 samples)** using stratified sampling. The model is evaluated on unseen test data using standard classification metrics: **Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix**.

---

## 📊 Evaluation Metrics Benchmark

### Overall Model Performance

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Accuracy** | **96.67%** | 29 out of 30 test samples correctly classified |
| **Macro Precision** | **0.9630** | Unweighted average precision across all 3 classes |
| **Macro Recall** | **0.9697** | Unweighted average sensitivity across all 3 classes |
| **Macro F1-Score** | **0.9645** | Harmonic mean of precision and recall |
| **Weighted Precision** | **0.9704** | Precision weighted by class support |
| **Weighted Recall** | **0.9667** | Recall weighted by class support |
| **Weighted F1-Score** | **0.9669** | F1-score weighted by class support |

### Per-Class Classification Report

| Species | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Iris setosa** | 1.0000 | 1.0000 | 1.0000 | 11 |
| **Iris versicolor** | 1.0000 | 0.9091 | 0.9524 | 11 |
| **Iris virginica** | 0.8889 | 1.0000 | 0.9412 | 8 |
| **Total / Accuracy** | — | — | **0.9667** | **30** |

### Confusion Matrix

```
                  Predicted ->
Actual Class      Setosa    Versicolor   Virginica
--------------------------------------------------
Setosa              11           0           0
Versicolor           0          10           1
Virginica            0           0           8
```

* **Setosa**: 100% precision and recall (linearly separable).
* **Versicolor**: 10 correct, 1 sample classified as Virginica due to overlapping petal dimensions.
* **Virginica**: 8 correct, 1 false positive from Versicolor.

---

## 💻 Web Application Features

1. **Evaluation Metrics**:
   * Summary cards showing Overall Accuracy, Macro Precision, Macro Recall, and Macro F1-Score.
   * Full multiclass Confusion Matrix and Classification Report table.

2. **Classify a Sample (Interactive Predictor)**:
   * Sliders for all 4 features: **Sepal Length, Sepal Width, Petal Length, and Petal Width**.
   * Instant prediction displaying species name and confidence percentages.
   * Nearest training neighbors and their Euclidean distance values.
   * Quick preset buttons for typical sample measurements.

3. **Dataset Explorer**:
   * Filterable table showing all 150 instances of the Iris dataset.

4. **Overview & Evaluation Formulas**:
   * Mathematical formulations for Accuracy, Precision, Recall, and F1-Score.

---

## 🚀 How to Run Locally

### Method 1: Direct in Browser (No Setup Required)
Double-click `index.html` or open it directly in Chrome, Edge, or Firefox.

### Method 2: With Python Server
Run the lightweight server included in the project:
```bash
python server.py
```
Then navigate to:
```
http://localhost:8000
```

---

## 🌐 Deploy to Vercel

1. Push this repository to GitHub.
2. Go to [vercel.com](https://vercel.com) and import the repository.
3. Keep the default settings and click **Deploy**.

---

## 📁 Project Structure

```
iris-classification/
├── index.html                  # Web application (dashboard & interactive classifier)
├── server.py                   # Python HTTP backend server with REST API
├── iris_classifier_sklearn.py  # scikit-learn training & metrics script
├── iris_classifier_pure.py     # Standalone Python classifier & metrics script
├── requirements.txt            # Python dependencies
├── vercel.json                 # Vercel deployment configuration
└── README.md                   # Project documentation & metrics benchmark
```
