"""
ingest.py — download the Heart Disease dataset and save to data/raw.
This is the first stage of the DVC pipeline.
"""
import pandas as pd
from pathlib import Path

# Output path — DVC expects this file to exist after the script runs
OUTPUT_PATH = Path('data/raw/heart_disease.csv')

# Ensure the output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases'
    '/heart-disease/processed.cleveland.data'
)

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

print('Downloading Heart Disease dataset from UCI...')
df = pd.read_csv(URL, header=None, names=COLUMN_NAMES, na_values='?')
df.to_csv(OUTPUT_PATH, index=False)
print(f'Saved {len(df)} rows to {OUTPUT_PATH}')
