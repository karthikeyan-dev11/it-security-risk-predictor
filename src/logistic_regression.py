import numpy as np
from prepare_data import get_features_and_labels

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights & bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias

            # Apply sigmoid
            y_pred = self.sigmoid(linear_model)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = -np.mean(
                    y * np.log(y_pred + 1e-9) +
                    (1 - y) * np.log(1 - y_pred + 1e-9)
                )
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


if __name__ == "__main__":
    # Load data
    X, y = get_features_and_labels("data/access_logs.csv")

    # Train model
    model = LogisticRegressionScratch(
        learning_rate=0.01,
        epochs=1000
    )

    model.fit(X, y)

    # Test predictions
    predictions = model.predict(X)

    accuracy = np.mean(predictions == y)
    print("\nTraining Accuracy:", accuracy)
