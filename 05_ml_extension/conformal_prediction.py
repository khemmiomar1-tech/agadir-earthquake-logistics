# -*- coding: utf-8 -*-
"""
ML extension
============
Conformal prediction intervals for ADP demand (jackknife+).

Why this matters
----------------
The proposed PhD (WP1, "Shift-Aware Multivariate Conformal Calibration")
replaces point demand forecasts with *uncertainty sets* that carry a
finite-sample coverage guarantee, and passes those sets to the robust
location-routing optimiser. This prototype is the first step: it wraps the
Random Forest demand model of demand_prediction.py with jackknife+
intervals (Barber, Candès, Ramdas & Tibshirani, 2021), a conformal method
designed for very small samples such as this case study (n = 16).

Output: for each ADP, a 90% prediction interval, its empirical coverage
under leave-one-out evaluation, and the mean interval width — the exact
quantities a robust optimiser consumes as an uncertainty set.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from demand_prediction import build_dataset

BASE = Path(__file__).resolve().parent
ALPHA = 0.10  # target miscoverage -> 90% intervals
FEATURES = ["x", "y", "cluster_id", "dist_depot_km", "time_depot_min",
            "n_assigned", "mean_assign_dist"]


def main():
    df = build_dataset()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["demand_final"].to_numpy(dtype=float)
    n = len(y)

    # Leave-one-out models and predictions.
    loo_pred = np.zeros(n)
    models = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = RandomForestRegressor(n_estimators=300, random_state=42)
        m.fit(X[mask], y[mask])
        models.append(m)
        loo_pred[i] = m.predict(X[i].reshape(1, -1))[0]
    resid = np.abs(y - loo_pred)

    # Jackknife+ interval for each point i: quantiles over j != i of
    # (LOO prediction of model j at x_i) -/+ (LOO residual of j).
    lower = np.zeros(n)
    upper = np.zeros(n)
    for i in range(n):
        lo_vals = np.array([models[j].predict(X[i].reshape(1, -1))[0] - resid[j]
                            for j in range(n) if j != i])
        hi_vals = np.array([models[j].predict(X[i].reshape(1, -1))[0] + resid[j]
                            for j in range(n) if j != i])
        lower[i] = np.quantile(lo_vals, ALPHA, method="lower")
        upper[i] = np.quantile(hi_vals, 1 - ALPHA, method="higher")
    lower = np.maximum(lower, 0)

    covered = (y >= lower) & (y <= upper)
    out = pd.DataFrame({
        "bp_id": df["bp_id"], "demand_actual": y,
        "prediction": np.round(loo_pred, 1),
        "interval_lower": np.round(lower, 1), "interval_upper": np.round(upper, 1),
        "covered_90": covered,
    })
    out.to_csv(BASE / "conformal_intervals.csv", index=False)
    print(f"Jackknife+ 90% intervals over {n} ADPs")
    print(f"Empirical coverage : {covered.mean() * 100:.1f}%  (target 90%)")
    print(f"Mean interval width: {(upper - lower).mean():.1f} demand units")
    print(f"Mean demand        : {y.mean():.1f}")
    print("\nSaved: conformal_intervals.csv")


if __name__ == "__main__":
    main()
