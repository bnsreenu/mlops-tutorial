"""
evaluate.py — evaluate the trained model and log metrics.
Reads model from models/, test data from data/processed.
Writes metrics to reports/metrics.json.
"""
import pandas as pd
import joblib
import mlflow
import json
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

os.chdir(Path(__file__).parent.parent)

PROCESSED_DIR = Path('data/processed')
MODELS_DIR    = Path('models')
REPORTS_DIR   = Path('reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load test data and model
X_test  = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
y_test  = pd.read_csv(PROCESSED_DIR / 'y_test.csv').squeeze()
model   = joblib.load(MODELS_DIR / 'model.pkl')

# Evaluate
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'ROC AUC:  {roc_auc:.4f}')

# Write metrics to JSON — DVC tracks this file
metrics = {'accuracy': round(accuracy, 4), 'roc_auc': round(roc_auc, 4)}
metrics_path = REPORTS_DIR / 'metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'Metrics saved to {metrics_path}')

# Log metrics to the most recent MLflow run
# We find the latest run in the experiment and log to it
mlflow.set_experiment('heart-disease-classification')
runs = mlflow.search_runs(order_by=['start_time DESC'], max_results=1)
run_id = runs.iloc[0]['run_id']

with mlflow.start_run(run_id=run_id):
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('roc_auc',  roc_auc)

print('Evaluation complete.')
