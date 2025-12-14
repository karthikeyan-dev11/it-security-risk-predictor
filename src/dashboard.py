import matplotlib.pyplot as plt
import numpy as np
from prepare_data import get_features_and_labels
from logistic_regression import LogisticRegressionScratch

# Load data
X, y = get_features_and_labels("data/access_logs.csv")

# Train model
model = LogisticRegressionScratch(learning_rate=0.01, epochs=800)
model.fit(X, y)

# Predict probabilities and risk scores
probs = model.predict_proba(X)
risk_scores = probs * 100

# ---- PLOTS ----

# 1. Risk Score Distribution
plt.figure()
plt.hist(risk_scores, bins=30)
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Number of Users")
plt.show()

# 2. Risk Level Count
low = np.sum(risk_scores < 30)
medium = np.sum((risk_scores >= 30) & (risk_scores < 70))
high = np.sum(risk_scores >= 70)

plt.figure()
plt.bar(["LOW", "MEDIUM", "HIGH"], [low, medium, high])
plt.title("Risk Level Count")
plt.ylabel("Users")
plt.show()
