# Predictive Analysis — Classifications, Regression, Clustering

**Module:** Predictive Analysis  
**Degree:** BSc Computer Science and Digitization

---

## Overview

This project applies predictive analytics techniques to retail sales and customer churn datasets. It covers the full machine learning workflow from data preparation and encoding through to model evaluation and business recommendations.

---

## Tasks

### Task 1 — Data Types, Encoding & Regression
- Identified numerical, categorical (nominal/ordinal), and encoded variables
- Applied label encoding (ordinal) and one-hot encoding (nominal)
- Built a regression model to predict `Total Amount`
- **Results:** R² = 0.86, RMSE = 205.85, MAE = 174.47

### Task 2 — Classification
- Predicted customer gender using Logistic Regression and Random Forest
- Evaluated with accuracy, precision, recall, F1-score, and confusion matrix
- Random Forest slightly outperformed Logistic Regression (accuracy ~0.47)
- Both models highlighted the challenge of predicting gender from purchase behaviour

### Task 2 — K-Means Clustering
- Applied elbow method to determine optimal K (K=4)
- Visualised customer segments using PCA (2D)
- Identified distinct customer groups for targeted business strategies

### Task 3 — End-to-End Churn Prediction Pipeline
- Dataset: Iranian Telecom Churn Dataset (UCI ML Repository)
- 70/30 train/test split, StandardScaler preprocessing
- Random Forest Classifier: **Accuracy = 0.94**
- Identified key churn drivers via feature importance analysis

---

## Key Results

| Task | Model | Key Metric |
|------|-------|-----------|
| Regression | Random Forest Regressor | R² = 0.86 |
| Classification | Random Forest Classifier | Accuracy = 0.47 |
| Clustering | K-Means (K=4) | 4 customer segments |
| Churn Prediction | Random Forest Classifier | Accuracy = 0.94 |

---

## Files

| File | Description |
|------|-------------|
| `predictive_analysis.py` | Full Python script — all tasks, models, and visualizations |
| `report.pdf` | Full assignment report with analysis, results, and business recommendations |

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python predictive_analysis.py
```

---

## Technologies

- Python · scikit-learn · Pandas · NumPy · Matplotlib
- Logistic Regression · Random Forest · K-Means · PCA
