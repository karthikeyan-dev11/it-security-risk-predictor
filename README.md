# Enterprise IT Security Risk Predictor

## 📌 Project Overview
This project is an end-to-end Machine Learning system designed to predict IT access risk in enterprise environments.

It uses:
- Logistic Regression (from scratch)
- K-Means clustering (from scratch)

to detect risky and anomalous user access behavior.

---

## ⚙️ Features
- Synthetic IT access log generation
- Data preprocessing & feature engineering
- Logistic Regression implemented using NumPy
- K-Means anomaly detection
- Risk scoring system (0–100)
- Final security decision: ALLOW / REVIEW / BLOCK
- Dashboard visualization

---

## 🛠️ Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib

---

## ▶️ How to Run

```bash
python data/generate_data.py
python src/preprocess.py
python src/logistic_regression.py
python src/final_decision.py
python src/dashboard.py
