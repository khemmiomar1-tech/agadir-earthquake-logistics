# -*- coding: utf-8 -*-
"""
ML extension
============
Demand prediction for aid delivery points (ADPs), Agadir Ida-Outanane case study.

Context
-------
The Master's thesis (defended June 2026) estimated ADP demand deterministically:
each aid delivery point's demand is the aggregated demand of the beneficiary
points (schools) assigned to it. No predictive ML model was fitted in the thesis.

This extension — a first step towards the proposed PhD work package WP1
(uncertainty-aware prediction for crisis logistics) —
fits Random Forest and Gradient Boosting regressors that predict ADP demand
from spatial and assignment features, so that demand can be *predicted* (with
uncertainty, see conformal_prediction.py) instead of only counted after the fact.

Data
----
- ../03_cvrp_heterogeneous_heuristic/adp_cvrp_demande_positive_clean.csv
  (16 ADPs: demand_final, cluster, projected coordinates)
- ../03_cvrp_heterogeneous_heuristic/distance_matrix_cvrp_16adp.csv
  (road-network distances/times from the depot, DEPOT_18)
- ../data/BPs_Affecte_au_ADPs.csv
  (65 candidate schools and their assignment to the 16 selected points)

Evaluation
----------
n = 16 ADPs is small, so models are evaluated with leave-one-out cross
-validation (LOOCV) and results are reported as a proof of concept, not as
production-grade accuracy. The PhD will scale this to synthetic and
semi-synthetic scenario ensembles.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def build_dataset() -> pd.DataFrame:
    adp = pd.read_csv(ROOT / "03_cvrp_heterogeneous_heuristic" / "adp_cvrp_demande_positive_clean.csv")
    adp["cluster_id"] = adp["cluster_id"].astype(str).str.strip().astype(int)
    adp["bp_id"] = adp["bp_id"].astype(str).str.strip()

    matrix = pd.read_csv(ROOT / "03_cvrp_heterogeneous_heuristic" / "distance_matrix_cvrp_16adp.csv")
    # The QGIS export contains duplicate pairs (append artefact) — the thesis
    # code removes them the same way.
    matrix = matrix.drop_duplicates(subset=["InputID", "TargetID"], keep="first")
    depot = matrix[matrix["InputID"] == "DEPOT_18"].set_index("TargetID")
    adp["dist_depot_km"] = adp["bp_id"].map(depot["Distance"] / 1000.0)
    adp["time_depot_min"] = adp["bp_id"].map(depot["time_min"])

    assign = pd.read_csv(ROOT / "data" / "BPs_Affecte_au_ADPs.csv")
    grp = assign.groupby("bp_id").agg(
        n_assigned=("original_id", "count"),
        mean_assign_dist=("distance", "mean"),
    )
    df = adp.join(grp, on="bp_id")
    df["n_assigned"] = df["n_assigned"].fillna(0)
    df["mean_assign_dist"] = df["mean_assign_dist"].fillna(df["mean_assign_dist"].mean())
    return df


def loocv_metrics(model, X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        m = model.__class__(**model.get_params())
        m.fit(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict(X[test_idx])
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    return preds, mae, rmse


def main():
    df = build_dataset()
    features = ["x", "y", "cluster_id", "dist_depot_km", "time_depot_min",
                "n_assigned", "mean_assign_dist"]
    X = df[features].to_numpy(dtype=float)
    y = df["demand_final"].to_numpy(dtype=float)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    rows = []
    print(f"Dataset: {len(df)} ADPs, demand range {y.min():.0f}–{y.max():.0f}, "
          f"mean {y.mean():.1f}")
    print(f"Baseline (predict the mean): MAE = "
          f"{mean_absolute_error(y, np.full_like(y, y.mean())):.2f}")
    for name, model in models.items():
        preds, mae, rmse = loocv_metrics(model, X, y)
        rows.append({"model": name, "loocv_mae": round(mae, 2), "loocv_rmse": round(rmse, 2)})
        print(f"{name:18s} LOOCV  MAE = {mae:7.2f}   RMSE = {rmse:7.2f}")
        df[f"pred_{name}"] = preds

    pd.DataFrame(rows).to_csv(BASE / "demand_model_results.csv", index=False)
    df[["bp_id", "demand_final"] + [f"pred_{n}" for n in models]].to_csv(
        BASE / "demand_predictions_loocv.csv", index=False)
    print("\nSaved: demand_model_results.csv, demand_predictions_loocv.csv")


if __name__ == "__main__":
    main()
