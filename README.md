# Diabetes Prediction System

An end-to-end Machine Learning project that predicts whether a person is diabetic
based on medical parameters using Logistic Regression and Support Vector Machine (SVM).

---

## 🔍 Problem Statement
Early detection of diabetes helps in preventing severe health complications.
This project uses machine learning to predict diabetes from patient medical data.

---

## 📊 Dataset
- PIMA Indians Diabetes Dataset
- Features include:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- Target variable: **Outcome (0 = Non-diabetic, 1 = Diabetic)**

---

## ⚙️ Workflow
1. Data loading and cleaning
2. Handling missing and invalid values
3. Feature scaling using StandardScaler
4. Model training (Logistic Regression & SVM)
5. Model comparison and selection
6. Model deployment using Streamlit UI

---

## 🧠 Models Used
- Logistic Regression
- Support Vector Machine (SVM)

The best-performing model is selected automatically based on accuracy.

---

## 🖥️ Web Application
A Streamlit-based UI allows users to enter medical details
and receive real-time diabetes predictions.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Git & GitHub

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
python model_training.py
python -m streamlit run app.py
