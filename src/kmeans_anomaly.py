import numpy as np
from prepare_data import get_features_and_labels

class KMeansScratch:
    def __init__(self, k=3, epochs=100):
        self.k = k
        self.epochs = epochs
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape

        # Randomly initialize centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.epochs):
            # Assign clusters
            clusters = self._assign_clusters(X)

            # Update centroids
            new_centroids = np.array([
                X[clusters == i].mean(axis=0)
                for i in range(self.k)
            ])

            # Stop if centroids don’t change
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centroids,
            axis=2
        )
        return np.argmin(distances, axis=1)

    def anomaly_scores(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centroids,
            axis=2
        )
        return np.min(distances, axis=1)


if __name__ == "__main__":
    # Load features
    X, _ = get_features_and_labels("data/access_logs.csv")

    # Train K-Means
    kmeans = KMeansScratch(k=3, epochs=100)
    kmeans.fit(X)

    # Compute anomaly scores
    scores = kmeans.anomaly_scores(X)

    # Threshold (top 5% as anomalies)
    threshold = np.percentile(scores, 95)
    anomalies = scores > threshold

    print("Anomaly Detection Results:\n")
    print("Total records:", len(scores))
    print("Anomalies detected:", np.sum(anomalies))

    print("\nSample anomaly scores:")
    for i in range(10):
        print(f"Record {i+1} → Score: {scores[i]:.4f}")
