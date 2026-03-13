"""
Predictive Analysis — Classifications, Regression, Clustering
Module: Predictive Analysis
Degree: BSc Computer Science and Digitization

Tasks:
  Task 1 — Data types, encoding, regression (retail sales dataset)
  Task 2 — Classification (logistic regression + random forest)
  Task 3 — K-Means clustering + end-to-end churn prediction pipeline

Dataset: Retail Sales Dataset (Kaggle)
         Iranian Churn Dataset (UCI ML Repository)

Requirements: pip install pandas numpy scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing       import LabelEncoder, StandardScaler
from sklearn.linear_model        import LinearRegression, LogisticRegression
from sklearn.ensemble            import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster             import KMeans
from sklearn.decomposition       import PCA
from sklearn.metrics             import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Task 1: Understanding Data Types & Encoding ───────────────

def task1_data_types_and_encoding():
    print("=" * 60)
    print("TASK 1: Data Types, Encoding & Regression")
    print("=" * 60)

    # Generate synthetic retail sales data (mirrors Kaggle dataset structure)
    np.random.seed(42)
    n = 1000

    dataset = pd.DataFrame({
        "Transaction ID":   range(1, n + 1),
        "Date":             pd.date_range("2023-01-01", periods=n, freq="8H"),
        "Customer ID":      np.random.randint(1000, 2000, n),
        "Gender":           np.random.choice(["Male", "Female"], n),
        "Age":              np.random.randint(18, 70, n),
        "Product Category": np.random.choice(["Electronics", "Clothing", "Beauty"], n),
        "Quantity":         np.random.randint(1, 5, n),
        "Price per Unit":   np.random.randint(20, 500, n),
    })
    dataset["Total Amount"] = dataset["Quantity"] * dataset["Price per Unit"]

    print("\nDataset shape:", dataset.shape)
    print("\nFirst 5 rows:")
    print(dataset.head())
    print("\nData types:")
    print(dataset.dtypes)

    # ── Ordinal encoding: Age Group ──────────────────────────
    dataset["Age Group"] = pd.cut(
        dataset["Age"],
        bins=[0, 18, 35, 60, 100],
        labels=["Child", "Young Adult", "Adult", "Senior"]
    )

    age_encoder = LabelEncoder()
    dataset["Age Group Encoded"] = age_encoder.fit_transform(dataset["Age Group"])

    # ── One-hot encoding: nominal variables ──────────────────
    dataset = pd.get_dummies(dataset, columns=["Gender", "Product Category"], drop_first=False)

    print("\nAvailable columns after encoding:")
    print(dataset.columns.tolist())

    return dataset


# ── Task 1 continued: Regression ─────────────────────────────

def task1_regression(dataset):
    print("\n--- Regression: Predicting Total Amount ---")

    feature_cols = [
        "Age", "Quantity", "Price per Unit", "Age Group Encoded",
        "Gender_Male",
        "Product Category_Clothing", "Product Category_Electronics"
    ]
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in dataset.columns]

    X = dataset[feature_cols]
    y = dataset["Total Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"\nR² Score: {r2:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    return model


# ── Task 2: Classification ────────────────────────────────────

def task2_classification(dataset):
    print("\n" + "=" * 60)
    print("TASK 2: Classification — Predicting Gender")
    print("=" * 60)

    feature_cols = [
        "Age", "Quantity", "Price per Unit",
        "Product Category_Clothing", "Product Category_Electronics"
    ]
    feature_cols = [c for c in feature_cols if c in dataset.columns]

    X = dataset[feature_cols]
    y = dataset["Gender_Male"].astype(int) if "Gender_Male" in dataset.columns else dataset["Gender_Female"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Logistic Regression
    print("\n--- Model: Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.2f}")
    print(f"Recall:    {recall_score(y_test, y_pred_lr, zero_division=0):.2f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_lr, zero_division=0):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))

    # Random Forest
    print("\n--- Model: Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.2f}")
    print(f"Recall:    {recall_score(y_test, y_pred_rf, zero_division=0):.2f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_rf, zero_division=0):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    return rf


# ── Task 2: K-Means Clustering ────────────────────────────────

def task2_clustering(dataset):
    print("\n" + "=" * 60)
    print("TASK 2: K-Means Clustering — Customer Segmentation")
    print("=" * 60)

    feature_cols = ["Age", "Quantity", "Price per Unit", "Total Amount"]
    X = dataset[feature_cols].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, wcss, marker="o", color="#2E86AB")
    plt.title("Figure 3a: Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Inertia)")
    plt.tight_layout()
    plt.savefig("elbow_method.png", dpi=150)
    plt.show()
    print("Saved: elbow_method.png")

    # K-Means with optimal K=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Figure 3b: K-Means Customer Segmentation (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=150)
    plt.show()
    print("Saved: kmeans_clusters.png")

    return kmeans


# ── Task 3: End-to-End Churn Prediction Pipeline ──────────────

def task3_churn_prediction():
    print("\n" + "=" * 60)
    print("TASK 3: End-to-End Pipeline — Customer Churn Prediction")
    print("=" * 60)

    # Synthetic churn dataset (mirrors Iranian Telecom structure)
    np.random.seed(42)
    n = 3150

    churn_data = pd.DataFrame({
        "Call Failures":      np.random.randint(0, 20, n),
        "Complaints":         np.random.randint(0, 2, n),
        "Subscription Length": np.random.randint(6, 60, n),
        "Charge Amount":      np.random.randint(0, 10, n),
        "Seconds of Use":     np.random.randint(0, 50000, n),
        "Frequency of Use":   np.random.randint(0, 200, n),
        "Frequency of SMS":   np.random.randint(0, 100, n),
        "Distinct Called Numbers": np.random.randint(0, 100, n),
        "Age Group":          np.random.randint(1, 5, n),
        "Tariff Plan":        np.random.randint(1, 3, n),
        "Status":             np.random.randint(1, 3, n),
        "Age":                np.random.randint(18, 65, n),
    })

    # Churn label: higher probability if complaints + low usage
    churn_data["Churn"] = (
        (churn_data["Complaints"] == 1) |
        (churn_data["Frequency of Use"] < 30)
    ).astype(int)

    print(f"\nDataset shape: {churn_data.shape}")
    print(f"Churn rate: {churn_data['Churn'].mean():.2%}")

    X = churn_data.drop("Churn", axis=1)
    y = churn_data["Churn"]

    # Train/test split: 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    print("\nRandom Forest Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances.sort_values(ascending=True).tail(10).plot(
        kind="barh", color="#E84855", figsize=(8, 5)
    )
    plt.title("Figure 4: Top Feature Importances — Churn Prediction")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Saved: feature_importance.png")

    return rf


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset    = task1_data_types_and_encoding()
    reg_model  = task1_regression(dataset)
    clf_model  = task2_classification(dataset)
    kmeans     = task2_clustering(dataset)
    churn_model = task3_churn_prediction()

    print("\nAll tasks complete.")
