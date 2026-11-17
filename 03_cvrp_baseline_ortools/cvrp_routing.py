import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# =========================
# 1. Read CSV files
# =========================

points_file = "point_routing.csv"
matrix_file = "distance_matrix_raw.csv"
demand_file = "demande_BP.csv"

points = pd.read_csv(points_file)
matrix = pd.read_csv(matrix_file)
demands_table = pd.read_csv(demand_file)


# =========================
# 2. Define IDs
# =========================

ids = list(points["routing_id"])

depot_id = [x for x in ids if str(x).startswith("DEPOT")][0]
depot_index = ids.index(depot_id)

print("Depot:", depot_id)
print("Number of points:", len(ids))


# =========================
# 3. Build time matrix
# =========================

n = len(ids)
time_matrix = [[0 for _ in range(n)] for _ in range(n)]

for _, row in matrix.iterrows():
    i_id = row["InputID"]
    j_id = row["TargetID"]

    if i_id in ids and j_id in ids:
        i = ids.index(i_id)
        j = ids.index(j_id)
        time_matrix[i][j] = int(round(row["time_min"]))


# =========================
# 4. Build demands
# =========================

demands = [0 for _ in range(n)]

for _, row in demands_table.iterrows():
    bp_id = row["bp_id"]
    demand = int(row["count"])

    if bp_id in ids:
        idx = ids.index(bp_id)
        demands[idx] = demand

demands[depot_index] = 0

print("Total demand:", sum(demands))
print("Demands:", dict(zip(ids, demands)))


# =========================
# 5. CVRP parameters
# =========================

num_vehicles = 4
vehicle_capacity = 20

vehicle_capacities = [vehicle_capacity] * num_vehicles


# =========================
# 6. Create OR-Tools model
# =========================

manager = pywrapcp.RoutingIndexManager(
    n,
    num_vehicles,
    depot_index
)

routing = pywrapcp.RoutingModel(manager)


def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return time_matrix[from_node][to_node]


transit_callback_index = routing.RegisterTransitCallback(time_callback)

routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


# =========================
# 7. Capacity constraint
# =========================

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


# =========================
# 8. Search parameters
# =========================

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.seconds = 30


# =========================
# 9. Solve
# =========================

solution = routing.SolveWithParameters(search_parameters)


# =========================
# 10. Print and export results
# =========================

if solution:
    results = []

    total_time = 0
    total_load = 0

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route_ids = []
        route_time = 0
        route_load = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_ids.append(ids[node_index])
            route_load += demands[node_index]

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_time += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        node_index = manager.IndexToNode(index)
        route_ids.append(ids[node_index])

        total_time += route_time
        total_load += route_load

        results.append({
            "vehicle": vehicle_id + 1,
            "route": " -> ".join(route_ids),
            "load": route_load,
            "time_min": route_time
        })

        print(f"\nVehicle {vehicle_id + 1}")
        print("Route:", " -> ".join(route_ids))
        print("Load:", route_load)
        print("Time:", route_time, "min")

    print("\n=========================")
    print("Total load:", total_load)
    print("Total time:", total_time, "min")
    print("=========================")

    results_df = pd.DataFrame(results)
    results_df.to_csv("cvrp_results.csv", index=False, encoding="utf-8-sig")

    print("\nResults exported to cvrp_results.csv")

else:
    print("No solution found.")
    print("Try increasing number of vehicles or vehicle capacity.")
