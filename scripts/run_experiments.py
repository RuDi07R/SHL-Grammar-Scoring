from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline_common import OUTPUTS, feature_matrix, load_feature_sets

KFOLD = KFold(n_splits=5, shuffle=True, random_state=42)
RIDGE_ALPHAS = [0.1, 1, 3, 10, 30, 100]
FEATURE_SETS = ["acoustic", "original_linguistic", "v2_linguistic", "original_linguistic+acoustic", "v2_linguistic+acoustic"]


def metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return float(mean_squared_error(y, prediction) ** 0.5), float(pearsonr(y, prediction).statistic)


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])


def tree_pipeline(model) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def evaluate(model, x, y):
    prediction = cross_val_predict(model, x, y, cv=KFOLD, n_jobs=1)
    return metrics(y.to_numpy(), prediction), prediction


def alpha_from_label(value: str) -> float:
    return float(str(value).split("=", 1)[1])


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    if not (OUTPUTS / "linguistic_features_v2.csv").exists():
        from scripts.pipeline_common import build_linguistic_v2
        transcripts = pd.read_csv(OUTPUTS / "train_transcripts.csv")
        build_linguistic_v2(transcripts).to_csv(OUTPUTS / "linguistic_features_v2.csv", index=False)
    frames = load_feature_sets()
    rows = []
    predictions = {}
    for feature_set in FEATURE_SETS:
        x, y = feature_matrix(feature_set, frames)
        for alpha in RIDGE_ALPHAS:
            score, prediction = evaluate(ridge_pipeline(alpha), x, y)
            key = f"ridge|{feature_set}|{alpha}"
            predictions[key] = prediction
            rows.append({"model": "Ridge", "features": feature_set, "alpha_or_params": f"alpha={alpha}", "rmse": score[0], "pearson": score[1]})
        print(f"Finished Ridge feature set: {feature_set}")

    ridge_frame = pd.DataFrame(rows)
    best_feature_sets = (
        ridge_frame.sort_values(["rmse", "pearson"], ascending=[True, False])["features"]
        .drop_duplicates()
        .head(2)
        .tolist()
    )
    nonlinear = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=4, random_state=42, n_jobs=1),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=2, min_samples_leaf=5, random_state=42),
    }
    for feature_set in best_feature_sets:
        x, y = feature_matrix(feature_set, frames)
        for name, estimator in nonlinear.items():
            score, prediction = evaluate(tree_pipeline(estimator), x, y)
            key = f"{name}|{feature_set}|default"
            predictions[key] = prediction
            rows.append({"model": name, "features": feature_set, "alpha_or_params": "conservative_default", "rmse": score[0], "pearson": score[1]})
        print(f"Finished nonlinear models: {feature_set}")

    result = pd.DataFrame(rows)
    best_ridge_row = result[result.model == "Ridge"].sort_values(["rmse", "pearson"], ascending=[True, False]).iloc[0]
    nonlinear_result = result[result.model != "Ridge"]
    best_nonlinear_row = nonlinear_result.sort_values(["rmse", "pearson"], ascending=[True, False]).iloc[0]
    ridge_key = f"ridge|{best_ridge_row.features}|{alpha_from_label(best_ridge_row.alpha_or_params):g}"
    nonlinear_key = f"{best_nonlinear_row.model}|{best_nonlinear_row.features}|default"
    _, ensemble_labels = feature_matrix(best_ridge_row.features, frames)
    ensemble_candidates = []
    for weight in [0.25, 0.5, 0.75]:
        prediction = weight * predictions[ridge_key] + (1 - weight) * predictions[nonlinear_key]
        rmse, pearson = metrics(ensemble_labels.to_numpy(), prediction)
        ensemble_candidates.append((rmse, pearson, weight, prediction))
    best_ensemble = min(ensemble_candidates, key=lambda item: (item[0], -item[1]))
    if best_ensemble[0] < best_ridge_row.rmse and best_ensemble[1] > best_ridge_row.pearson:
        rows.append({"model": "Ensemble", "features": f"{best_ridge_row.features}+{best_nonlinear_row.features}", "alpha_or_params": f"ridge_weight={best_ensemble[2]:.2f}", "rmse": best_ensemble[0], "pearson": best_ensemble[1]})
        selected = {"kind": "ensemble", "ridge": best_ridge_row.to_dict(), "nonlinear": best_nonlinear_row.to_dict(), "ridge_weight": best_ensemble[2]}
    else:
        selected = {"kind": "single", "model": best_ridge_row.to_dict()}
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUTS / "experiment_results.csv", index=False)
    (OUTPUTS / "selected_model.json").write_text(json.dumps(selected, indent=2, default=str), encoding="utf-8")

    if selected["kind"] == "ensemble":
        selected_feature_set = best_ridge_row.features
        selected_alpha = alpha_from_label(best_ridge_row.alpha_or_params)
        ridge_x, y = feature_matrix(selected_feature_set, frames)
        nonlinear_x, _ = feature_matrix(best_nonlinear_row.features, frames)
        final_model = {
            "kind": "ensemble",
            "ridge": ridge_pipeline(selected_alpha).fit(ridge_x, y),
            "nonlinear": tree_pipeline(nonlinear[best_nonlinear_row.model]).fit(nonlinear_x, y),
            "ridge_features": selected_feature_set,
            "nonlinear_features": best_nonlinear_row.features,
            "ridge_columns": list(ridge_x.columns),
            "nonlinear_columns": list(nonlinear_x.columns),
            "ridge_weight": best_ensemble[2],
        }
    else:
        selected_feature_set = best_ridge_row.features
        selected_alpha = alpha_from_label(best_ridge_row.alpha_or_params)
        x, y = feature_matrix(selected_feature_set, frames)
        final_model = {
            "kind": "single",
            "pipeline": ridge_pipeline(selected_alpha).fit(x, y),
            "features": selected_feature_set,
            "columns": list(x.columns),
        }
    joblib.dump(final_model, OUTPUTS / "final_model.joblib")
    print(result.sort_values(["rmse", "pearson"], ascending=[True, False]).head(10).to_string(index=False))
    print(f"Selected final model: {selected}")


if __name__ == "__main__":
    main()