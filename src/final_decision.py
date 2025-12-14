import numpy as np
from prepare_data import get_features_and_labels
from logistic_regression import LogisticRegressionScratch
from kmeans_anomaly import KMeansScratch

def risk_level(score):
    if score < 30:
        return "LOW"
    elif score < 70:
        return "MEDIUM"
    else:
        return "HIGH"

def final_action(risk, is_anomaly):
    if risk == "HIGH" or is_anomaly:
        return "BLOCK"
    elif risk == "MEDIUM":
        return "REVIEW"
    else:
        return "ALLOW"


if __name__ == "__main__":
    # Load data
    X, y = get_features_and_labels("data/access_logs.csv")

    # ---------- Train Logistic Regression ----------
    lr_model = LogisticRegressionScratch(
        learning_rate=0.01,
        epochs=800
    )
    lr_model.fit(X, y)

    probabilities = lr_model.predict_proba(X)
    risk_scores = (probabilities * 100).astype(int)
    risk_levels = [risk_level(s) for s in risk_scores]

    # ---------- Train K-Means ----------
    kmeans = KMeansScratch(k=3, epochs=100)
    kmeans.fit(X)
    anomaly_scores = kmeans.anomaly_scores(X)

    threshold = np.percentile(anomaly_scores, 95)
    anomalies = anomaly_scores > threshold

    # ---------- Final Decision ----------
    print("\nFINAL SECURITY DECISIONS (Sample):\n")
    for i in range(10):
        action = final_action(risk_levels[i], anomalies[i])
        print(
            f"User {i+1} | "
            f"Risk Score: {risk_scores[i]} ({risk_levels[i]}) | "
            f"Anomaly: {anomalies[i]} | "
            f"Action: {action}"
        )
