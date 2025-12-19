import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("diabetes.csv")

# -----------------------------
# 2. Data Cleaning
# -----------------------------
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# -----------------------------
# 3. Feature & Target Split
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. Logistic Regression Model
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

log_acc = accuracy_score(y_test, log_pred)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", log_acc)
print(classification_report(y_test, log_pred))

# -----------------------------
# 7. Support Vector Machine (SVM)
# -----------------------------
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)

print("\n===== SVM Results =====")
print("Accuracy:", svm_acc)
print(classification_report(y_test, svm_pred))

# -----------------------------
# 8. Model Comparison
# -----------------------------
print("\n===== Model Comparison =====")
print(f"Logistic Regression Accuracy: {log_acc:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}")

# -----------------------------
# 9. Save Best Model
# -----------------------------
if svm_acc > log_acc:
    joblib.dump(svm_model, "diabetes_model.pkl")
    best_model = "SVM"
else:
    joblib.dump(log_model, "diabetes_model.pkl")
    best_model = "Logistic Regression"

joblib.dump(scaler, "scaler.pkl")

print(f"\nBest Model Selected: {best_model}")
print("Model and scaler saved successfully!")
