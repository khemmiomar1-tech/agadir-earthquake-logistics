# AI-Driven Emergency Response Optimisation for Earthquake Disasters

**A location–routing framework for post-earthquake humanitarian aid distribution — Agadir Ida-Outanane, Morocco**

Master's thesis codebase of **Omar Khemmi**, ENSA Khouribga (Université Sultan
Moulay Slimane), defended **23 June 2026**, graded **19/20**.
Supervisor: Prof. Bouchra Karoum.

This repository contains the code and data behind the thesis, plus a clearly
dated post-thesis extension (November 2026) that prototypes the prediction
layer of a proposed PhD project at Liverpool John Moores University:
*Uncertainty-Aware Learning-Augmented Optimisation for Decision-Making in
Crisis Logistics*.

---

## What the system does

After an earthquake, relief must reach affected communities from a central
depot over a damaged road network. The framework:

1. **Clusters** beneficiary points (geocoded schools of Agadir Ida-Outanane)
   with K-Means / DBSCAN and selects aid delivery points (ADPs) with a
   p-median logic — 65 candidate points → **16 ADPs** in 6 clusters.
2. **Assigns** beneficiary points to ADPs and aggregates demand.
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

The defended thesis report additionally analyses the T0 (undamaged) and T6
(disrupted) scenarios on the real OpenStreetMap network in QGIS, including
the Social Vulnerability Index prioritisation under which coverage of the
served population reaches 92.2% at T6 and the reinforcement vehicle is
activated for the most vulnerable, hardest-to-reach point (ADP_13).

## Methods — stated exactly as implemented

- **Clustering:** K-Means (K selected by elbow/silhouette analysis over
  K = 2…13) and DBSCAN, scikit-learn, in the Colab notebooks of `01_clustering/`.
- **Baseline routing:** Google **OR-Tools** Routing Solver —
  `PATH_CHEAPEST_ARC` first solution, **Guided Local Search** metaheuristic,
  30 s time limit (`02_cvrp_baseline_ortools/`). Running the script reproduces
  the thesis results exactly.
- **Heterogeneous-fleet model:** a custom constructive heuristic —
  farthest-seed initialisation, cheapest-insertion growth, **2-opt** local
  improvement — implemented in pure Python/pandas
  (`03_cvrp_heterogeneous_heuristic/`).
- **Disrupted-network (T6) final run:** OR-Tools with per-vehicle capacities
  (`04_cvrp_t6_ortools/`).
- **GIS:** QGIS / OpenStreetMap network; CVRP tours converted to real road
  geometries with shortest-path tools; blocked/slow road sets define the T6
  scenario (`qgis/`, `data/`).

## Repository structure

```
01_clustering/                  Thesis notebooks: preprocessing, K-Means/DBSCAN, K selection
02_cvrp_baseline_ortools/       OR-Tools CVRP baseline + inputs + expected output
03_cvrp_heterogeneous_heuristic/ Heterogeneous-fleet CVRP (custom heuristic + 2-opt)
04_cvrp_t6_ortools/             Final disrupted-network (T6) OR-Tools model + inputs/outputs
05_ml_extension_nov2026/        POST-THESIS extension (Nov 2026): demand prediction,
                                conformal prediction intervals, road-state classification
data/                           Case-study CSVs (assignments, segments, T6 road-state labels)
qgis/                           PyQGIS accessibility-extraction script
```

The QGIS project (`.qgz`) and GeoPackage (`.gpkg`, ~530 MB) exceed GitHub's
file limits and are available on request.

## Post-thesis extension (November 2026)

`05_ml_extension_nov2026/` prototypes the PhD's prediction layer on the
thesis data: Random Forest / Gradient Boosting **demand prediction**,
**jackknife+ conformal prediction intervals** for that demand (the uncertainty
sets of PhD work package WP1), and a **road-state classifier** (Open / Slow /
Blocked) trained on the T6 labels. See that folder's README for results and
scope. This extension is dated and labelled as post-thesis work; nothing in
it is claimed as part of the defended thesis.

## Requirements

Python 3.11+, `pandas`, `numpy`, `scikit-learn`, `ortools` — see
`requirements.txt`. Each folder's scripts run standalone from their own
directory.

## Citation

Khemmi, O. (2026). *Optimisation intelligente de la réponse d'urgence aux
catastrophes sismiques : un cadre de localisation-routage enrichi par
l'apprentissage automatique et l'indice de vulnérabilité sociale (SoVI).*
Master's thesis, ENSA Khouribga, Université Sultan Moulay Slimane, Morocco.
