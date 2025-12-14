import numpy as np
from preprocess import load_and_preprocess

def get_features_and_labels(csv_path):
    df = load_and_preprocess(csv_path)

    # Separate label
    y = df['label'].values

    # Drop label from features
    X = df.drop(columns=['label']).values

    return X, y


if __name__ == "__main__":
    X, y = get_features_and_labels("data/access_logs.csv")

    print("Feature matrix shape (X):", X.shape)
    print("Label vector shape (y):", y.shape)
