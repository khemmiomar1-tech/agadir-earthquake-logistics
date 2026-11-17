# -*- coding: utf-8 -*-
"""
CVRP T6 final — PFE Séisme Agadir
Entrées attendues dans le même dossier que ce script :
  1) CVRP_Nodes_T6_final.csv
  2) distance_matrix_cvrp_T6_final.csv
  3) vehicles_heterogeneous_post_seisme.csv
Sorties :
  - cvrp_heterogeneous_T6_final_routes.csv
  - cvrp_heterogeneous_T6_final_summary.csv
  - cvrp_heterogeneous_T6_final_segments.csv
"""

from pathlib import Path
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

BASE_DIR = Path(__file__).resolve().parent

NODES_FILE = BASE_DIR / "CVRP_Nodes_T6_final.csv"
MATRIX_FILE = BASE_DIR / "distance_matrix_cvrp_T6_final.csv"
VEHICLES_FILE = BASE_DIR / "vehicles_heterogeneous_post_seisme.csv"

OUT_ROUTES = BASE_DIR / "cvrp_heterogeneous_T6_final_routes.csv"
OUT_SUMMARY = BASE_DIR / "cvrp_heterogeneous_T6_final_summary.csv"
OUT_SEGMENTS = BASE_DIR / "cvrp_heterogeneous_T6_final_segments.csv"

DEPOT_ID = "DEPOT_18"


def normalize_id(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    # sécurité si certains anciens fichiers utilisent DEPOT18
    if x == "DEPOT18":
        return "DEPOT_18"
    return x


def read_inputs():
    nodes = pd.read_csv(NODES_FILE)
    matrix = pd.read_csv(MATRIX_FILE)
    vehicles = pd.read_csv(VEHICLES_FILE)

    # Nettoyage noms colonnes
    nodes.columns = [c.strip() for c in nodes.columns]
    matrix.columns = [c.strip() for c in matrix.columns]
    vehicles.columns = [c.strip() for c in vehicles.columns]

    if "node_id" not in nodes.columns:
        raise ValueError("Le fichier nodes doit contenir la colonne node_id")

    # Colonne demande
    if "new_demand_T6" in nodes.columns:
        demand_col = "new_demand_T6"
    elif "demand_final" in nodes.columns:
        demand_col = "demand_final"
    else:
        raise ValueError("Le fichier nodes doit contenir new_demand_T6 ou demand_final")

    # Colonne capacité véhicules
    if "capacity_useful" in vehicles.columns:
        cap_col = "capacity_useful"
    elif "capacity_max" in vehicles.columns:
        cap_col = "capacity_max"
    elif "Capacity" in vehicles.columns:
        cap_col = "Capacity"
    else:
        raise ValueError("Le fichier véhicules doit contenir capacity_useful ou capacity_max")

    if "vehicle_id" not in vehicles.columns:
        raise ValueError("Le fichier véhicules doit contenir vehicle_id")

    required_matrix = {"InputID", "TargetID", "Distance", "time_min"}
    missing = required_matrix - set(matrix.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans la matrice : {missing}")

    nodes["node_id"] = nodes["node_id"].apply(normalize_id)
    matrix["InputID"] = matrix["InputID"].apply(normalize_id)
    matrix["TargetID"] = matrix["TargetID"].apply(normalize_id)
    vehicles["vehicle_id"] = vehicles["vehicle_id"].astype(str).str.strip()

    nodes[demand_col] = pd.to_numeric(nodes[demand_col], errors="coerce").fillna(0).astype(int)
    vehicles[cap_col] = pd.to_numeric(vehicles[cap_col], errors="coerce").fillna(0).astype(int)
    matrix["Distance"] = pd.to_numeric(matrix["Distance"], errors="coerce")
    matrix["time_min"] = pd.to_numeric(matrix["time_min"], errors="coerce")

    # Garder uniquement les nœuds actifs + dépôt
    nodes = nodes[nodes["node_id"] != ""].copy()
    nodes = nodes.drop_duplicates(subset=["node_id"], keep="first")

    if DEPOT_ID not in set(nodes["node_id"]):
        raise ValueError("DEPOT_18 est absent de CVRP_Nodes_T6_final.csv")

    # Ordre : dépôt d'abord, puis BP triés numériquement
    customers = [n for n in nodes["node_id"].tolist() if n != DEPOT_ID]
    customers = sorted(customers, key=lambda s: int(s.split("_")[1]) if s.startswith("BP_") else 9999)
    ids = [DEPOT_ID] + customers

    # Dictionnaire demande
    demand_by_id = dict(zip(nodes["node_id"], nodes[demand_col]))
    demand_by_id[DEPOT_ID] = 0
    demands = [int(demand_by_id.get(i, 0)) for i in ids]

    vehicle_ids = vehicles["vehicle_id"].tolist()
    vehicle_types = vehicles["type"].astype(str).tolist() if "type" in vehicles.columns else ["vehicle"] * len(vehicle_ids)
    vehicle_capacities = vehicles[cap_col].astype(int).tolist()

    return nodes, matrix, ids, demands, vehicle_ids, vehicle_types, vehicle_capacities


def build_matrices(matrix, ids):
    n = len(ids)
    id_to_idx = {node_id: i for i, node_id in enumerate(ids)}

    # très grande valeur si arc manquant
    BIG = 10**9
    distance_matrix = [[0 if i == j else BIG for j in range(n)] for i in range(n)]
    time_matrix = [[0 if i == j else BIG for j in range(n)] for i in range(n)]

    matrix = matrix[matrix["InputID"].isin(ids) & matrix["TargetID"].isin(ids)].copy()
    matrix = matrix[matrix["InputID"] != matrix["TargetID"]].copy()
    matrix = matrix.drop_duplicates(subset=["InputID", "TargetID"], keep="first")

    expected = n * (n - 1)
    if len(matrix) != expected:
        print(f"ATTENTION : matrice = {len(matrix)} lignes, attendu = {expected}")

    for _, row in matrix.iterrows():
        i = id_to_idx[row["InputID"]]
        j = id_to_idx[row["TargetID"]]
        distance_matrix[i][j] = int(round(float(row["Distance"])))
        time_matrix[i][j] = int(round(float(row["time_min"])))

    # Vérifier arcs manquants
    missing = []
    for i in range(n):
        for j in range(n):
            if i != j and time_matrix[i][j] >= BIG:
                missing.append((ids[i], ids[j]))
    if missing:
        raise ValueError(f"Arcs manquants dans la matrice, exemples : {missing[:10]}")

    return distance_matrix, time_matrix


def solve_cvrp(ids, demands, vehicle_ids, vehicle_capacities, time_matrix):
    n = len(ids)
    depot_index = ids.index(DEPOT_ID)
    num_vehicles = len(vehicle_ids)

    if sum(vehicle_capacities) < sum(demands):
        raise ValueError(f"Capacité totale insuffisante : {sum(vehicle_capacities)} < demande {sum(demands)}")

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        vehicle_capacities,
        True,
        "Capacity"
    )

    # Dimension distance/temps pour équilibrer les tournées.
    # Important car les capacités sont grandes par rapport à la demande.
    routing.AddDimension(
        transit_callback_index,
        0,
        10**9,
        True,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    time_dimension.SetGlobalSpanCostCoefficient(100)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 60

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("Aucune solution trouvée. Vérifie la matrice, les demandes et les capacités.")

    return manager, routing, solution


def export_results(ids, demands, vehicle_ids, vehicle_types, vehicle_capacities, manager, routing, solution, time_matrix):
    route_rows = []
    summary_rows = []
    segment_rows = []
    total_load = 0
    total_time = 0

    for v in range(len(vehicle_ids)):
        index = routing.Start(v)
        route_nodes = []
        route_load = 0
        route_time = 0
        order = 0

        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            node_id = ids[node_idx]
            node_demand = demands[node_idx]
            route_nodes.append(node_id)
            route_load += node_demand

            next_index = solution.Value(routing.NextVar(index))
            next_node_idx = manager.IndexToNode(next_index)
            next_node_id = ids[next_node_idx]
            arc_time = routing.GetArcCostForVehicle(index, next_index, v)

            route_rows.append({
                "vehicle_id": vehicle_ids[v],
                "vehicle_type": vehicle_types[v],
                "route_order": order,
                "node_id": node_id,
                "demand": node_demand,
                "cumulative_load": route_load,
                "capacity_useful": vehicle_capacities[v]
            })

            if not (node_id == DEPOT_ID and next_node_id == DEPOT_ID):
                segment_rows.append({
                    "vehicle_id": vehicle_ids[v],
                    "vehicle_type": vehicle_types[v],
                    "segment_order": order,
                    "from_node": node_id,
                    "to_node": next_node_id,
                    "time_min": arc_time
                })

            route_time += arc_time
            index = next_index
            order += 1

        # end depot
        node_idx = manager.IndexToNode(index)
        node_id = ids[node_idx]
        route_nodes.append(node_id)
        route_rows.append({
            "vehicle_id": vehicle_ids[v],
            "vehicle_type": vehicle_types[v],
            "route_order": order,
            "node_id": node_id,
            "demand": 0,
            "cumulative_load": route_load,
            "capacity_useful": vehicle_capacities[v]
        })

        # Une route vide = DEPOT_18 -> DEPOT_18
        served_adp = [n for n in route_nodes if n != DEPOT_ID]
        total_load += route_load
        total_time += route_time

        summary_rows.append({
            "vehicle_id": vehicle_ids[v],
            "vehicle_type": vehicle_types[v],
            "route": " -> ".join(route_nodes),
            "total_load": route_load,
            "capacity_useful": vehicle_capacities[v],
            "utilization_rate_percent": round(100 * route_load / vehicle_capacities[v], 2) if vehicle_capacities[v] else 0,
            "total_time_min": route_time,
            "number_of_adp_served": len(served_adp)
        })

    pd.DataFrame(route_rows).to_csv(OUT_ROUTES, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(OUT_SEGMENTS, index=False, encoding="utf-8-sig")

    print("=== CVRP T6 FINAL TERMINÉ ===")
    print("Nombre de nœuds :", len(ids))
    print("Nombre ADP servis :", len(ids) - 1)
    print("Demande totale servie :", total_load)
    print("Temps total :", total_time, "min")
    print("Fichiers générés :")
    print("-", OUT_ROUTES.name)
    print("-", OUT_SUMMARY.name)
    print("-", OUT_SEGMENTS.name)


def main():
    nodes, matrix, ids, demands, vehicle_ids, vehicle_types, vehicle_capacities = read_inputs()

    print("Nœuds :", ids)
    print("Demandes :", dict(zip(ids, demands)))
    print("Demande totale :", sum(demands))
    print("Véhicules :", dict(zip(vehicle_ids, vehicle_capacities)))

    distance_matrix, time_matrix = build_matrices(matrix, ids)
    manager, routing, solution = solve_cvrp(ids, demands, vehicle_ids, vehicle_capacities, time_matrix)
    export_results(ids, demands, vehicle_ids, vehicle_types, vehicle_capacities, manager, routing, solution, time_matrix)


if __name__ == "__main__":
    main()
