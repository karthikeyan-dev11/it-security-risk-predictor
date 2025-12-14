# data/generate_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

n = 5000  # number of records

roles = ['engineer', 'admin', 'hr', 'devops', 'qa']
departments = ['platform', 'apps', 'security', 'hr', 'finance']
locations = ['IN', 'US', 'UK', 'DE', 'SG']
devices = ['laptop', 'desktop', 'mobile', 'server']
request_types = ['read', 'write', 'execute', 'admin']

def rand_dates(n):
    start = datetime.now() - timedelta(days=180)
    # generate n random datetimes in last 180 days
    return [
        (start + timedelta(seconds=int(np.random.rand() * 180 * 24 * 3600))).isoformat()
        for _ in range(n)
    ]

df = pd.DataFrame({
    'user_id': [f"user_{i}" for i in range(n)],
    'role': np.random.choice(roles, n),
    'department': np.random.choice(departments, n),
    'location': np.random.choice(locations, n),
    'device_type': np.random.choice(devices, n),
    'timestamp': rand_dates(n),
    'request_type': np.random.choice(request_types, n),
    'previous_violations_count': np.random.poisson(0.2, n),
    'failed_login_count_24h': np.random.poisson(0.5, n),
    'avg_session_duration': np.random.exponential(300, n),  # seconds
    'is_privileged_account': np.random.choice([0,1], n, p=[0.9,0.1]),
    'account_age_days': np.random.randint(1,2000,n)
})

# Create a risk label (1 = risky)
risk = (
    (df['previous_violations_count'] > 0) |
    (df['failed_login_count_24h'] > 3) |
    ((df['is_privileged_account'] == 1) & (df['account_age_days'] < 30))
)

df['label'] = risk.astype(int)

# Save CSV inside data folder (ensure working dir is project root)
df.to_csv("data/access_logs.csv", index=False)

print("Dataset created successfully → data/access_logs.csv")

