from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


print("Loading California Housing dataset...")

data = fetch_california_housing()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target


print("Training final deployment model...")

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="log2",
    random_state=42,
    n_jobs=1
)

model.fit(X, y)


print("Saving deployment model...")

joblib.dump(
    model,
    MODEL_DIR / "house_model.joblib",
    compress=3
)

joblib.dump(
    list(X.columns),
    MODEL_DIR / "house_features.joblib"
)

print("Deployment model created successfully.")