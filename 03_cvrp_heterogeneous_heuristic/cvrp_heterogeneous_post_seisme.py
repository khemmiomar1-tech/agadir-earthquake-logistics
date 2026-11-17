# -*- coding: utf-8 -*-
"""
CVRP post-séisme avec flotte hétérogène
Projet PFE Omar — Emergency Response Optimization

Entrées attendues dans le même dossier :
- adp_cvrp_demande_positive_clean.csv
- vehicles_heterogeneous_post_seisme.csv
- distance_matrix_cvrp_16adp.csv

Sorties :
- distance_matrix_cvrp_16adp_clean.csv
- cvrp_heterogeneous_post_seisme_routes.csv
- cvrp_heterogeneous_post_seisme_summary.csv

Méthode :
Heuristique constructive + amélioration locale 2-opt.
Cette méthode est adaptée pour produire une solution faisable et interprétable.
"""

import math
from pathlib import Path
import pandas as pd


DEPOT_ID = "DEPOT_18"

ADP_FILE = "adp_cvrp_demande_positive_clean.csv"
VEHICLES_FILE = "vehicles_heterogeneous_post_seisme.csv"
MATRIX_FILE = "distance_matrix_cvrp_16adp.csv"

OUT_MATRIX_CLEAN = "distance_matrix_cvrp_16adp_clean.csv"
OUT_ROUTES = "cvrp_heterogeneous_post_seisme_routes.csv"
OUT_SUMMARY = "cvrp_heterogeneous_post_seisme_summary.csv"


def read_inputs(base_dir: Path):
    adp = pd.read_csv(base_dir / ADP_FILE)
    vehicles = pd.read_csv(base_dir / VEHICLES_FILE)
    matrix = pd.read_csv(base_dir / MATRIX_FILE)

    required_adp = {"bp_id", "demand_final", "x", "y"}
    required_veh = {"vehicle_id", "type", "capacity_useful"}
    required_matrix = {"InputID", "TargetID", "Distance", "time_min"}

    missing_adp = required_adp - set(adp.columns)
    missing_veh = required_veh - set(vehicles.columns)
    missing_matrix = required_matrix - set(matrix.columns)

    if missing_adp:
        raise ValueError(f"Colonnes manquantes dans {ADP_FILE}: {missing_adp}")
    if missing_veh:
        raise ValueError(f"Colonnes manquantes dans {VEHICLES_FILE}: {missing_veh}")
    if missing_matrix:
        raise ValueError(f"Colonnes manquantes dans {MATRIX_FILE}: {missing_matrix}")

    adp["demand_final"] = pd.to_numeric(adp["demand_final"], errors="raise")
    vehicles["capacity_useful"] = pd.to_numeric(vehicles["capacity_useful"], errors="raise")
    matrix["Distance"] = pd.to_numeric(matrix["Distance"], errors="raise")
    matrix["time_min"] = pd.to_numeric(matrix["time_min"], errors="raise")

    # Sécurité : supprimer BP_17 si jamais il existe encore dans la matrice.
    matrix = matrix[
        (matrix["InputID"] != "BP_17") &
        (matrix["TargetID"] != "BP_17")
    ].copy()

    # Sécurité : supprimer les doublons issus d'un export QGIS avec append.
    matrix = matrix.drop_duplicates(subset=["InputID", "TargetID"], keep="first").copy()

    return adp, vehicles, matrix


def validate_inputs(adp, vehicles, matrix):
    customers = sorted(adp["bp_id"].astype(str).tolist(), key=lambda x: int(x.split("_")[1]))
    nodes = [DEPOT_ID] + customers

    demand_total = float(adp["demand_final"].sum())
    capacity_total = float(vehicles["capacity_useful"].sum())

    if capacity_total < demand_total:
        raise ValueError(
            f"Flotte insuffisante : capacité utile totale = {capacity_total}, "
            f"demande totale = {demand_total}"
        )

    expected_pairs = len(nodes) * (len(nodes) - 1)
    if len(matrix) != expected_pairs:
        raise ValueError(
            f"Matrice incohérente : {len(matrix)} lignes trouvées, "
            f"{expected_pairs} attendues pour {len(nodes)} points."
        )

    dist = {}
    tmin = {}
    for _, row in matrix.iterrows():
        i = str(row["InputID"])
        j = str(row["TargetID"])
        dist[(i, j)] = float(row["Distance"])
        tmin[(i, j)] = float(row["time_min"])

    for n in nodes:
        dist[(n, n)] = 0.0
        tmin[(n, n)] = 0.0

    missing = [(i, j) for i in nodes for j in nodes if i != j and (i, j) not in dist]
    if missing:
        raise ValueError(f"Distances manquantes dans la matrice, exemples : {missing[:10]}")

    return customers, nodes, dist, tmin


def route_distance(route, dist):
    seq = [DEPOT_ID] + route + [DEPOT_ID]
    return sum(dist[(seq[i], seq[i + 1])] for i in range(len(seq) - 1))


def route_time(route, tmin):
    seq = [DEPOT_ID] + route + [DEPOT_ID]
    return sum(tmin[(seq[i], seq[i + 1])] for i in range(len(seq) - 1))


def two_opt(route, dist):
    """Amélioration locale 2-opt sur une route."""
    best = route[:]
    improved = True

    while improved:
        improved = False
        n = len(best)
        for i in range(n - 1):
            for k in range(i + 1, n):
                candidate = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if route_distance(candidate, dist) + 1e-9 < route_distance(best, dist):
                    best = candidate
                    improved = True

    return best


def solve_greedy_heterogeneous(adp, vehicles, dist, tmin):
    demands = dict(zip(adp["bp_id"].astype(str), adp["demand_final"].astype(float)))
    unserved = set(demands.keys())

    # On commence par les véhicules avec capacité utile élevée.
    # Pourquoi ? Ils peuvent absorber les ADP éloignés ou fortement demandeurs.
    vehicles_sorted = vehicles.sort_values(
        by=["capacity_useful", "vehicle_id"], ascending=[False, True]
    ).reset_index(drop=True)

    routes = []

    for _, v in vehicles_sorted.iterrows():
        vehicle_id = str(v["vehicle_id"])
        vehicle_type = str(v["type"])
        capacity = float(v["capacity_useful"])

        load = 0.0
        route = []

        feasible = [i for i in unserved if demands[i] <= capacity - load]
        if not feasible:
            routes.append({
                "vehicle_id": vehicle_id,
                "type": vehicle_type,
                "capacity_useful": capacity,
                "route": route,
                "load": load
            })
            continue

        # Point de départ de la tournée : ADP le plus éloigné du dépôt.
        # Cette stratégie évite de laisser les points éloignés aux petits véhicules en fin d'affectation.
        seed = max(feasible, key=lambda i: dist[(DEPOT_ID, i)])
        route.append(seed)
        load += demands[seed]
        unserved.remove(seed)

        # Insertion au moindre coût marginal tant que la capacité le permet.
        while True:
            feasible = [i for i in unserved if demands[i] <= capacity - load]
            if not feasible:
                break

            best = None
            current = [DEPOT_ID] + route + [DEPOT_ID]

            for node in feasible:
                for pos in range(len(current) - 1):
                    a = current[pos]
                    b = current[pos + 1]
                    incremental_cost = dist[(a, node)] + dist[(node, b)] - dist[(a, b)]

                    if best is None or incremental_cost < best[0]:
                        best = (incremental_cost, node, pos)

            _, chosen_node, insert_pos = best
            route.insert(insert_pos, chosen_node)
            load += demands[chosen_node]
            unserved.remove(chosen_node)

        route = two_opt(route, dist)

        routes.append({
            "vehicle_id": vehicle_id,
            "type": vehicle_type,
            "capacity_useful": capacity,
            "route": route,
            "load": load
        })

    if unserved:
        raise ValueError(
            "Solution non faisable avec cette heuristique. "
            f"ADP non servis : {sorted(unserved)}"
        )

    return routes, demands


def build_outputs(routes, demands, dist, tmin, adp):
    name_by_bp = {}
    if "nom_etablissement" in adp.columns:
        name_by_bp = dict(zip(adp["bp_id"].astype(str), adp["nom_etablissement"].astype(str)))

    route_rows = []
    summary_rows = []

    for r in routes:
        vehicle_id = r["vehicle_id"]
        vehicle_type = r["type"]
        capacity = r["capacity_useful"]
        route = r["route"]

        seq = [DEPOT_ID] + route + [DEPOT_ID]
        cumulative_load = 0.0
        cumulative_distance = 0.0
        cumulative_time = 0.0

        for order, node in enumerate(seq):
            demand = 0.0 if node == DEPOT_ID else float(demands[node])
            if node != DEPOT_ID:
                cumulative_load += demand

            if order > 0:
                prev = seq[order - 1]
                cumulative_distance += dist[(prev, node)]
                cumulative_time += tmin[(prev, node)]

            route_rows.append({
                "vehicle_id": vehicle_id,
                "vehicle_type": vehicle_type,
                "route_order": order,
                "node_id": node,
                "nom_etablissement": "DEPOT" if node == DEPOT_ID else name_by_bp.get(node, ""),
                "demand": demand,
                "cumulative_load": cumulative_load,
                "capacity_useful": capacity,
                "cumulative_distance_m": cumulative_distance,
                "cumulative_time_min": cumulative_time
            })

        total_distance = route_distance(route, dist)
        total_time = route_time(route, tmin)
        total_load = sum(demands[n] for n in route)

        summary_rows.append({
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "route": " -> ".join(seq),
            "total_load": total_load,
            "capacity_useful": capacity,
            "utilization_rate_percent": round((total_load / capacity) * 100, 2) if capacity else 0,
            "total_distance_m": total_distance,
            "total_distance_km": total_distance / 1000,
            "total_time_min": total_time,
            "number_of_adp_served": len(route)
        })

    return pd.DataFrame(route_rows), pd.DataFrame(summary_rows)


def main():
    base_dir = Path(__file__).resolve().parent

    adp, vehicles, matrix = read_inputs(base_dir)
    customers, nodes, dist, tmin = validate_inputs(adp, vehicles, matrix)

    # Sauvegarde d'une matrice propre, sans BP_17 et sans doublons.
    matrix.to_csv(base_dir / OUT_MATRIX_CLEAN, index=False, encoding="utf-8-sig")

    routes, demands = solve_greedy_heterogeneous(adp, vehicles, dist, tmin)
    route_df, summary_df = build_outputs(routes, demands, dist, tmin, adp)

    route_df.to_csv(base_dir / OUT_ROUTES, index=False, encoding="utf-8-sig")
    summary_df.to_csv(base_dir / OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("=== CVRP hétérogène post-séisme terminé ===")
    print(f"ADP servis : {len(customers)}")
    print(f"Demande totale : {adp['demand_final'].sum()}")
    print(f"Capacité utile totale : {vehicles['capacity_useful'].sum()}")
    print(f"Matrice propre : {len(matrix)} lignes")
    print()
    print(summary_df[[
        "vehicle_id",
        "vehicle_type",
        "total_load",
        "capacity_useful",
        "utilization_rate_percent",
        "total_distance_km",
        "total_time_min",
        "number_of_adp_served"
    ]].to_string(index=False))
    print()
    print(f"Fichiers générés :")
    print(f"- {OUT_MATRIX_CLEAN}")
    print(f"- {OUT_ROUTES}")
    print(f"- {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
