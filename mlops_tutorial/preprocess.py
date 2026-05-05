"""
preprocess.py — clean, split, and scale the raw Heart Disease data.
Reads from data/raw, writes to data/processed.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH       = Path('data/raw/heart_disease.csv')
PROCESSED_DIR  = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print('Loading raw data...')
df = pd.read_csv(RAW_PATH)

# Drop rows with missing values
df = df.dropna()
print(f'Rows after dropping NaN: {len(df)}')

# Convert target to binary
df['target'] = (df['target'] > 0).astype(int)
print(f"Target distribution:\n{df['target'].value_counts().to_string()}")

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train/test split — fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features — fit on train only, transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save splits as CSVs
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    PROCESSED_DIR / 'X_train.csv', index=False
)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    PROCESSED_DIR / 'X_test.csv', index=False
)
pd.Series(y_train).to_csv(PROCESSED_DIR / 'y_train.csv', index=False)
pd.Series(y_test).to_csv(PROCESSED_DIR / 'y_test.csv',  index=False)

# Save the fitted scaler so it can be reused at inference time
joblib.dump(scaler, PROCESSED_DIR / 'scaler.pkl')

print(f'Training samples: {len(X_train)}')
print(f'Test samples:     {len(X_test)}')
print('Preprocessing complete.')
