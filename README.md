# AI-Driven Emergency Response Optimisation for Earthquake Disasters

**A location–routing framework for post-earthquake humanitarian aid distribution — Agadir Ida-Outanane, Morocco**

Master's thesis codebase of **Omar Khemmi**, ENSA Khouribga (Université Sultan
Moulay Slimane), defended **23 June 2026**, graded **19/20**.
Supervisor: Prof. Bouchra Karoum.

This repository contains the code and data behind the thesis, plus the
pipeline's ML prediction layer (Stage 02), implemented after the defence as
the first step towards a proposed PhD project at Liverpool John Moores
University: *Uncertainty-Aware Learning-Augmented Optimisation for
Decision-Making in Crisis Logistics*.

---

## Pipeline

```
OSM / GIS data
  → Stage 01  Clustering (K-Means, K = 6) + p-median selection of 16 ADPs
  → Stage 02  ML prediction layer: demand prediction (RF / GB)
              + conformal uncertainty intervals; road-state classification
  → Stage 03  CVRP baseline — OR-Tools, Guided Local Search
  → Stage 04  Heterogeneous-fleet CVRP — custom insertion + 2-opt heuristic
  → Stage 05  Re-planning on the disrupted network (T6) — OR-Tools
  (+ SoVI prioritisation whenever coverage is partial)
```

Predictions sit **before** the optimisation stages they feed: demand decides
vehicle loads, road states decide which routes exist. The reasoning behind
every method choice — alternatives tested, measured results, literature —
is documented in [METHODS.md](METHODS.md).

## What the system does

After an earthquake, relief must reach affected communities from a central
depot over a damaged road network. The framework:

1. **Clusters** beneficiary points (geocoded schools of Agadir Ida-Outanane)
   with K-Means / DBSCAN and selects aid delivery points (ADPs) with a
   p-median logic — 65 candidate points → **16 ADPs** in 6 clusters.
2. **Predicts** each ADP's demand (Random Forest / Gradient Boosting) with
   conformal uncertainty intervals, and classifies road segments as
   Open / Slow / Blocked (Stage 02 — in the thesis, demand was aggregated
   deterministically and road states classified manually).
3. **Routes** a heterogeneous vehicle fleet from the depot (DEPOT_18) with a
   capacitated vehicle routing model (CVRP).
4. **Re-plans** on a disrupted network (T6 scenario: roads open / slow /
   blocked), where two ADPs become unreachable and coverage is prioritised
   with a Social Vulnerability Index (SoVI).

## Headline results

Every figure below is the output of the code in this repository — each script
was re-run in November 2026 and reproduces the thesis outputs exactly.

| Model | Points served | Fleet | Verified output |
|---|---|---|---|
| Baseline CVRP (OR-Tools) | 17 beneficiary points + depot | 4 vehicles × 20 capacity | Total load 65, total time **479 min** |
| Heterogeneous-fleet CVRP (custom heuristic) | 16 ADPs | 6 vehicles (2 small, 3 medium trucks, 1 light truck), post-quake capacity factors 0.7–0.8 | Total load 2,145, total distance **574.1 km**, vehicle utilisation up to 99.8% |
| Disrupted network T6 (OR-Tools) | 14 of 16 ADPs (2 unreachable after the quake) | Same heterogeneous fleet | Demand served 60, total time **556 min** |
| Demand prediction (Stage 02) | 16 ADPs | RF / GB regressors | LOOCV MAE ≈ 53 (vs 63 mean baseline); jackknife+ 90% intervals |
| Road-state classification (Stage 02) | 679 labelled segments | RF / GB classifiers | Accuracy 0.83 (RF) / 0.90 (GB); Blocked recall 0.86 |

The defended thesis report additionally analyses the T0 (undamaged) and T6
(disrupted) scenarios on the real OpenStreetMap network in QGIS, including
the Social Vulnerability Index prioritisation under which coverage of the
served population reaches 92.2% at T6 and the reinforcement vehicle is
activated for the most vulnerable, hardest-to-reach point (ADP_13).

## Methods — stated exactly as implemented

- **Clustering:** K-Means (K selected by elbow/silhouette analysis over
  K = 2…13) and DBSCAN, scikit-learn, in the Colab notebooks of `01_clustering/`.
- **ML prediction:** Random Forest and Gradient Boosting regressors for ADP
  demand, wrapped with jackknife+ conformal prediction intervals; Random
  Forest and Gradient Boosting classifiers for road state (`02_ml_prediction/`).
- **Baseline routing:** Google **OR-Tools** Routing Solver —
  `PATH_CHEAPEST_ARC` first solution, **Guided Local Search** metaheuristic,
  30 s time limit (`03_cvrp_baseline_ortools/`). Running the script reproduces
  the thesis results exactly.
- **Heterogeneous-fleet model:** a custom constructive heuristic —
  farthest-seed initialisation, cheapest-insertion growth, **2-opt** local
  improvement — implemented in pure Python/pandas
  (`04_cvrp_heterogeneous_heuristic/`).
- **Disrupted-network (T6) final run:** OR-Tools with per-vehicle capacities
  (`05_cvrp_t6_ortools/`).
- **GIS:** QGIS / OpenStreetMap network; CVRP tours converted to real road
  geometries with shortest-path tools; blocked/slow road sets define the T6
  scenario (`qgis/`, `data/`).

Full justifications — alternatives considered, measured comparisons, and
references — are in [METHODS.md](METHODS.md).

## Repository structure

```
01_clustering/                   Stage 01 notebooks: preprocessing, K-Means/DBSCAN, K selection
02_ml_prediction/                Stage 02: demand prediction, conformal prediction,
                                 road-state classification (RF + GB)
03_cvrp_baseline_ortools/        Stage 03: OR-Tools CVRP baseline + inputs + expected output
04_cvrp_heterogeneous_heuristic/ Stage 04: heterogeneous-fleet CVRP (custom heuristic + 2-opt)
05_cvrp_t6_ortools/              Stage 05: disrupted-network (T6) OR-Tools model + inputs/outputs
data/                            Case-study CSVs (assignments, segments, T6 road-state labels)
qgis/                            PyQGIS accessibility-extraction script
METHODS.md                       Method choices and justifications, with references
```

The QGIS project (`.qgz`) and GeoPackage (`.gpkg`, ~530 MB) exceed GitHub's
file limits and are available on request.

## Requirements

Python 3.11+, `pandas`, `numpy`, `scikit-learn`, `ortools` — see
`requirements.txt`. Each folder's scripts run standalone from their own
directory.

## Citation

Khemmi, O. (2026). *Optimisation intelligente de la réponse d'urgence aux
catastrophes sismiques : un cadre de localisation-routage enrichi par
l'apprentissage automatique et l'indice de vulnérabilité sociale (SoVI).*
Master's thesis, ENSA Khouribga, Université Sultan Moulay Slimane, Morocco.
