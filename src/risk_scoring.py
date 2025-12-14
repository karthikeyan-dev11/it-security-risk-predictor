import numpy as np
from prepare_data import get_features_and_labels
from logistic_regression import LogisticRegressionScratch

def calculate_risk_score(probability):
    """
    Convert probability (0–1) to risk score (0–100)
    """
    return int(probability * 100)

def assign_risk_level(score):
    if score < 30:
        return "LOW"
    elif score < 70:
        return "MEDIUM"
    else:
        return "HIGH"


if __name__ == "__main__":
    # Load data
    X, y = get_features_and_labels("data/access_logs.csv")

    # Train model
    model = LogisticRegressionScratch(
        learning_rate=0.01,
        epochs=800
    )
    model.fit(X, y)

    # Get probabilities
    probabilities = model.predict_proba(X)

    # Convert to risk scores & levels
    risk_scores = [calculate_risk_score(p) for p in probabilities]
    risk_levels = [assign_risk_level(s) for s in risk_scores]

    # Show sample output
    print("\nSample Risk Predictions:\n")
    for i in range(10):
        print(
            f"User {i+1} → "
            f"Risk Score: {risk_scores[i]} | "
            f"Risk Level: {risk_levels[i]}"
        )
