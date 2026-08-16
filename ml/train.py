# %%
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor



# %%
# Project paths

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# %%
# Load dataset

print("Loading California Housing dataset...")

data = fetch_california_housing()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

print(f"Total records: {X.shape[0]}")
print(f"Total features: {X.shape[1]}")


# %%
# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training records: {X_train.shape[0]}")
print(f"Testing records: {X_test.shape[0]}")


# %%
# Define models

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# %%
# Train and evaluate models

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"MAE : ${mae * 100000:,.0f}")
    print(f"RMSE: ${rmse * 100000:,.0f}")
    print(f"R²  : {r2:.4f}")


# %%
# Compare models

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE"
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# %%
# Save comparison results

results_df.to_csv(
    REPORTS_DIR / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")
print(REPORTS_DIR / "model_comparison.csv")

# %%
# Feature importance from Random Forest

random_forest = trained_models["Random Forest"]

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_ 
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance.to_string(index=False))

# %%
# Hyperparameter tuning for Random Forest

random_forest = trained_models["Random Forest"]

param_distributions = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, "sqrt", "log2"]
}

random_search = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=param_distributions,
    n_iter=15,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

print("\nStarting hyperparameter tuning...")

random_search.fit(X_train, y_train)

print("\nBest parameters:")
print(random_search.best_params_)

print(
    "\nBest cross-validation RMSE:",
    -random_search.best_score_
)


# %%
# Evaluate tuned Random Forest on the test set

tuned_model = random_search.best_estimator_

y_pred_tuned = tuned_model.predict(X_test)

tuned_mae = mean_absolute_error(y_test, y_pred_tuned)

tuned_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_tuned)
)

tuned_r2 = r2_score(
    y_test,
    y_pred_tuned
)

print("\n========== TUNED RANDOM FOREST ==========")
print(f"MAE : ${tuned_mae * 100000:,.0f}")
print(f"RMSE: ${tuned_rmse * 100000:,.0f}")
print(f"R²  : {tuned_r2:.4f}")


# %%
# Compare baseline and tuned Random Forest

baseline_rf = trained_models["Random Forest"]

y_pred_baseline = baseline_rf.predict(X_test)

baseline_mae = mean_absolute_error(
    y_test,
    y_pred_baseline
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_baseline)
)

baseline_r2 = r2_score(
    y_test,
    y_pred_baseline
)

comparison = pd.DataFrame({
    "Model": [
        "Baseline Random Forest",
        "Tuned Random Forest"
    ],
    "MAE": [
        baseline_mae,
        tuned_mae
    ],
    "RMSE": [
        baseline_rmse,
        tuned_rmse
    ],
    "R2": [
        baseline_r2,
        tuned_r2
    ]
})

print("\n========== BASELINE VS TUNED ==========")
print(comparison.to_string(index=False))

# %%
# Save the final tuned model

final_model = random_search.best_estimator_

joblib.dump(
    final_model,
    MODEL_DIR / "house_model.joblib"
)

joblib.dump(
    list(X.columns),
    MODEL_DIR / "house_features.joblib"
)

print("\nFinal tuned model saved successfully.")
print(f"Model path: {MODEL_DIR / 'house_model.joblib'}")
print(f"Features path: {MODEL_DIR / 'house_features.joblib'}")


# %%
# Save final model metrics

metrics = {
    "model": "Random Forest Regressor",
    "mae": tuned_mae,
    "rmse": tuned_rmse,
    "r2": tuned_r2
}

metrics_path = MODEL_DIR / "model_metrics.json"

import json

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel metrics saved successfully.")
print(f"Metrics path: {metrics_path}")

# %%
