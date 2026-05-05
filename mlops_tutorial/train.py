"""
train.py — train the Gradient Boosting model and log to MLflow.
Reads from data/processed, writes model to models/.
"""
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier

# Fix working directory so MLflow finds the correct mlruns folder
# When DVC runs this script it sets the working directory to the project root
# so this line is a safety net — os.getcwd() should already be the root
os.chdir(Path(__file__).parent.parent)

PROCESSED_DIR = Path('data/processed')
MODELS_DIR    = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load processed training data
X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv').squeeze()

# Hyperparameters
N_ESTIMATORS  = 200
LEARNING_RATE = 0.05
MAX_DEPTH     = 4

mlflow.set_experiment('heart-disease-classification')

with mlflow.start_run(run_name='pipeline-gb-v1'):
    mlflow.log_param('model_type',    'GradientBoosting')
    mlflow.log_param('n_estimators',  N_ESTIMATORS)
    mlflow.log_param('learning_rate', LEARNING_RATE)
    mlflow.log_param('max_depth',     MAX_DEPTH)

    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save model to disk — evaluate.py will load it from here
    model_path = MODELS_DIR / 'model.pkl'
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

    # Also log to MLflow for the registry
    mlflow.sklearn.log_model(model, 'model')
    print(f'MLflow run ID: {mlflow.active_run().info.run_id}')

print('Training complete.')
