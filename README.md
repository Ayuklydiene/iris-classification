# Iris Flower Classification & Evaluation Metrics

An interactive, responsive web application and machine learning benchmark created for the classification of Fisher's classic **Iris Dataset** and comprehensive evaluation of model performance.

---

## 📊 Evaluation Metrics Benchmark

Trained with stratified 80% train / 20% test partition:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Overall Accuracy** | **96.67%** | 29 out of 30 test samples correctly classified |
| **Macro Precision** | **0.9630** | Unweighted mean across all 3 classes |
| **Macro Recall** | **0.9697** | Unweighted sensitivity across all 3 classes |
| **Macro F1-Score** | **0.9645** | Harmonic mean of precision & recall |
| **Weighted Precision**| **0.9704** | Weighted by class support |
| **Weighted Recall** | **0.9667** | Weighted by class support |
| **Weighted F1-Score** | **0.9669** | Weighted by class support |

### Detailed Classification Report (Per Species)

| Species | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Iris-Setosa** | 1.0000 | 1.0000 | 1.0000 | 11 |
| **Iris-Versicolor** | 1.0000 | 0.9091 | 0.9524 | 11 |
| **Iris-Virginica** | 0.8889 | 1.0000 | 0.9412 | 8 |

### Multiclass Confusion Matrix
```
                  Predicted ->
Actual Class      Setosa    Versicolor   Virginica
--------------------------------------------------
Setosa              11           0           0
Versicolor           0          10           1
Virginica            0           0           8
```

---

## 🌟 Web Application Features

1. **Executive Evaluation Dashboard**:
   - Interactive Confusion Matrix with true positive and misclassification indicators.
   - Live Classification Report table with precision, recall, f1-score, and support counts.
   - Visual comparison charts for all classes.

2. **Interactive Classification Playground**:
   - 4 sliders for Sepal Length, Sepal Width, Petal Length, and Petal Width.
   - Preset buttons to quickly test typical samples of Setosa, Versicolor, and Virginica.
   - Real-time confidence probability breakdown bars.
   - Nearest training neighbors inspection with Euclidean distance metrics.

3. **Dataset Explorer**:
   - Searchable, filterable table showing all 150 ground-truth samples.

4. **Methodology & Formula Reference**:
   - Mathematical definitions for Accuracy, Precision, Recall, and F1-score for supervisor review.

---

## 🚀 How to Run Locally

### Option 1: Double-Click or Open in Any Browser (Zero Setup)
Simply double-click `index.html` or right-click and choose **Open with Browser**. The entire classifier and dashboard will run immediately with client-side inference!

### Option 2: Run with Python Server
In PowerShell or Terminal:
```bash
python server.py
```
Then navigate to:
```
http://localhost:8000
```
*(The Python server requires zero external pip packages and uses the standard library HTTP server).*

---

## 🌐 How to Host Online (Free & Instant)

### Method A: GitHub Pages (Recommended for Academic Submissions)
1. Initialize a git repository and push this directory to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Iris Classification WebApp"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. In your GitHub repository:
   - Click **Settings** > **Pages** (on the left menu).
   - Under **Build and deployment**, set **Source** to `Deploy from a branch`.
   - Select Branch: `main` / `/(root)` and click **Save**.
3. In ~60 seconds, your site will be live at:
   `https://<your-username>.github.io/<your-repo-name>/`

### Method B: Netlify (Drag & Drop in 10 Seconds)
1. Go to [Netlify Drop](https://app.netlify.com/drop).
2. Drag and drop this folder (`iris-classification`) directly onto the page.
3. Netlify will deploy it instantly and give you a public URL (e.g. `https://iris-classifier-assignment.netlify.app`).

### Method C: Vercel
1. Install Vercel CLI (`npm i -g vercel`) or import the GitHub repo on [vercel.com](https://vercel.com).
2. Click **Deploy**.

---

## 📁 Project Structure

```
iris-classification/
├── index.html                  # Main responsive webapp dashboard & interactive predictor
├── server.py                   # Python HTTP backend server with REST API endpoints
├── iris_classifier_sklearn.py  # Standard scikit-learn implementation
├── iris_classifier_pure.py     # Standalone pure-Python implementation
├── requirements.txt            # Python dependencies (scikit-learn, numpy, pandas, matplotlib)
└── README.md                   # Project documentation, metrics report, and hosting guide
```
