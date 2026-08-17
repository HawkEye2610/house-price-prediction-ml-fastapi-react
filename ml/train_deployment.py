from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# Load dataset
print("Loading California Housing dataset...")

data = fetch_california_housing()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target


# Same train/test split used during model development
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train deployment model
print("Training deployment model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="log2",
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)


# Evaluate deployment model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nDeployment model performance:")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")


# Save model
print("\nSaving deployment model...")

joblib.dump(
    model,
    MODEL_DIR / "house_model.joblib",
    compress=3
)

joblib.dump(
    list(X.columns),
    MODEL_DIR / "house_features.joblib"
)


# Save deployment metrics
deployment_metrics = {
    "model": "Random Forest Regressor",
    "mae": mae,
    "rmse": rmse,
    "r2": r2
}

with open(
    MODEL_DIR / "model_metrics.json",
    "w"
) as file:
    json.dump(
        deployment_metrics,
        file,
        indent=4
    )


# Save deployment model configuration
deployment_info = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "log2"
}

with open(
    MODEL_DIR / "deployment_model_info.json",
    "w"
) as file:
    json.dump(
        deployment_info,
        file,
        indent=4
    )


print("\nDeployment model created successfully.")