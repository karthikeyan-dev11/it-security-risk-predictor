import pandas as pd
import numpy as np

def load_and_preprocess(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['is_work_hour'] = df['hour'].between(9, 18).astype(int)

    # Drop original timestamp (not ML-friendly)
    df.drop(columns=['timestamp', 'user_id'], inplace=True)

    # Encode categorical columns
    categorical_cols = [
        'role', 'department', 'location',
        'device_type', 'request_type'
    ]

    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # Scale numeric columns (simple normalization)
    numeric_cols = [
        'previous_violations_count',
        'failed_login_count_24h',
        'avg_session_duration',
        'account_age_days'
    ]

    for col in numeric_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    return df


# Run only if this file is executed directly
if __name__ == "__main__":
    processed_df = load_and_preprocess("data/access_logs.csv")
    print("Preprocessing completed successfully!")
    print(processed_df.head())
