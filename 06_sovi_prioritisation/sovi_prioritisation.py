"""
Stage 06 — SoVI-proxy prioritisation
=====================================
A transparent, reproducible vulnerability-prioritisation score for the 16
aid delivery points (ADPs) of the Agadir Ida-Outanane case study, in the
spirit of the Social Vulnerability Index (SoVI; Cutter, Boruff & Shirley,
2003).

Why a proxy: the full SoVI is built from census socio-economic variables
that are not available at ADP level for the study communes. This stage
therefore computes a **SoVI proxy** from the variables the case study
actually measured — the name Omar's own thesis files used
("ADP_SoVI_T6_proxy"). In the defended thesis, prioritisation under partial
coverage was argued qualitatively in the T6 analysis; this script is the
reproducible formalisation of that argument, built after the defence.

Components (each z-standardised; higher = more vulnerable / higher priority):
  1. demand            — people depending on the ADP (exposure).
  2. remoteness        — road distance from the depot DEPOT_18 (response delay).
  3. isolation         — mean distance from the ADP to the beneficiary points
                         it serves (last-mile difficulty inside its zone).
  4. t6_disruption     — 1 if the ADP became unreachable in the disrupted
                         (T6) network, else 0 (exposure to network failure).

Composite = unweighted mean of the component z-scores. Equal weights are
used deliberately: with proxy variables and n = 16 there is no defensible
empirical basis for differential weights, and equal weighting is the
transparent default a decision-maker can audit. Two sensitivity variants
are computed alongside — (a) without the disruption component, (b) PCA
first-component weights (Cutter et al.'s original construction) — and the
rank agreement (Spearman) with the main score is reported.

Outputs: sovi_proxy_scores.csv — components, z-scores, composite, rank,
T6 reachability and the T6 service-priority order among reachable ADPs.

Run from this folder:  python sovi_prioritisation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd > 0 else 1.0)


def main() -> None:
    dem = pd.read_csv(ROOT / "04_cvrp_heterogeneous_heuristic" /
                      "adp_cvrp_demande_positive_clean.csv")
    mat = pd.read_csv(ROOT / "04_cvrp_heterogeneous_heuristic" /
                      "distance_matrix_cvrp_16adp_clean.csv")
    asg = pd.read_csv(ROOT / "data" / "BPs_Affecte_au_ADPs.csv")
    t6r = pd.read_csv(ROOT / "05_cvrp_t6_ortools" /
                      "cvrp_heterogeneous_T6_final_routes.csv")

    df = dem[["bp_id", "nom_etablissement", "cluster_id", "demand_final"]].copy()
    df = df.rename(columns={"demand_final": "demand"})

    depot = mat[(mat["InputID"] == "DEPOT_18")][["TargetID", "Distance"]]
    df = df.merge(depot, left_on="bp_id", right_on="TargetID", how="left") \
           .drop(columns="TargetID").rename(columns={"Distance": "remoteness_m"})

    iso = asg.groupby("bp_id")["distance"].mean().rename("isolation_deg")
    df = df.merge(iso, left_on="bp_id", right_index=True, how="left")

    served_t6 = set(t6r.loc[t6r["node_id"] != "DEPOT_18", "node_id"])
    df["reachable_T6"] = df["bp_id"].isin(served_t6)
    df["t6_disruption"] = (~df["reachable_T6"]).astype(int)

    comps = ["demand", "remoteness_m", "isolation_deg", "t6_disruption"]
    for c in comps:
        df[f"z_{c}"] = z(df[c].astype(float))
    Z = df[[f"z_{c}" for c in comps]]

    df["sovi_proxy"] = Z.mean(axis=1)
    # sensitivity (a): no disruption component
    df["sovi_nodisr"] = Z[["z_demand", "z_remoteness_m", "z_isolation_deg"]].mean(axis=1)
    # sensitivity (b): PCA first component (sign aligned with the main score)
    pc = PCA(n_components=1).fit_transform(Z.values).ravel()
    if np.corrcoef(pc, df["sovi_proxy"])[0, 1] < 0:
        pc = -pc
    df["sovi_pca"] = pc

    df["rank"] = df["sovi_proxy"].rank(ascending=False, method="min").astype(int)
    df["t6_priority"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    reach = df["reachable_T6"]
    df.loc[reach, "t6_priority"] = df.loc[reach, "sovi_proxy"].rank(
        ascending=False, method="min").astype("Int64")
    df = df.sort_values("rank")

    rho_a = spearmanr(df["sovi_proxy"], df["sovi_nodisr"]).statistic
    rho_b = spearmanr(df["sovi_proxy"], df["sovi_pca"]).statistic

    cols = ["bp_id", "nom_etablissement", "cluster_id", "demand", "remoteness_m",
            "isolation_deg", "reachable_T6", "sovi_proxy", "rank", "t6_priority",
            "sovi_nodisr", "sovi_pca"]
    df[cols].round(4).to_csv(BASE / "sovi_proxy_scores.csv", index=False)

    print("SoVI-proxy ranking (all 16 ADPs):")
    print(df[["rank", "bp_id", "demand", "remoteness_m", "reachable_T6",
              "sovi_proxy"]].to_string(index=False))
    print("\nT6 service priority (reachable ADPs only):")
    t6 = df[df["reachable_T6"]].sort_values("t6_priority")
    print(t6[["t6_priority", "bp_id", "sovi_proxy"]].to_string(index=False))
    print(f"\nSensitivity: Spearman vs no-disruption variant = {rho_a:.3f}; "
          f"vs PCA variant = {rho_b:.3f}")
    print("Saved: sovi_proxy_scores.csv")


if __name__ == "__main__":
    main()
