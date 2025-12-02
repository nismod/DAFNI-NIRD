# %%
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta

import geopandas as gpd
from itertools import islice
import networkx as nx
from collections import defaultdict
import json
import warnings


warnings.simplefilter("ignore")

INPUT_PATH_MACC = Path(r"C:\Oxford\Research\MACCHUB\local")

# %%
nodes = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes.gpq"
)
stations = nodes[nodes.station_label == "train_station"].reset_index(drop=True)
stations_set = set(stations.TIPLOC)

edges = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_edges.gpq"  # undirected edges
)
edges.edge_id = edges.edge_id.apply(lambda x: "e_" + str(x).split("_")[-1])
edges["weight"] = edges.geometry.length
graph_df = edges[["from_node", "to_node", "edge_id", "weight"]]
network = nx.from_pandas_edgelist(
    graph_df,
    source="from_node",
    target="to_node",
    edge_attr=True,
)  # default: create_using=nx.Graph()/ create_using=nx.DiGraph()

# %%
timetable = pd.read_csv(
    INPUT_PATH_MACC / "data" / "incoming" / "rails" / "timetable" / "timetable.csv"
)
timetable = timetable[
    (timetable.cargo_type == "ExpP") | (timetable.cargo_type == "OrdP")
].reset_index()  # select the train timetable records
timetable["path"] = timetable["path"].apply(ast.literal_eval)
timetable["stops"] = timetable["stops"].apply(ast.literal_eval)
timetable["times"] = timetable["times"].apply(ast.literal_eval)


# %%
def minutes_diff(t1_str, t2_str):
    fmt = "%H%M"
    t1 = datetime.strptime(t1_str, fmt)
    t2 = datetime.strptime(t2_str, fmt)
    # adjust for crossing midnight
    if t2 < t1:
        t2 += timedelta(days=1)

    return int((t2 - t1).total_seconds())  # seconds


def build_travel_dict(stations, stops, times):
    d = defaultdict(lambda: defaultdict(list))
    prev_stop_idx = None  # index of last station where train stops
    for i, stop in enumerate(stops):
        if stop == "S":
            if prev_stop_idx is not None:
                from_station = stations[prev_stop_idx]
                to_station = stations[i]
                depart_time = times[prev_stop_idx][1]  # departure at from_station
                arrive_time = times[i][0]  # arrival at to_station
                travel_time = minutes_diff(depart_time, arrive_time)
                if travel_time == 0:  # dont append 0 travel time
                    continue
                d[from_station][to_station].append(travel_time)
            prev_stop_idx = i

    return d


# %%
all_travel_dict = defaultdict(lambda: defaultdict(list))
for _, row in timetable.iterrows():
    travel_dict = build_travel_dict(row["path"], row["stops"], row["times"])
    for from_station, to_dict in travel_dict.items():
        for to_station, times_list in to_dict.items():
            if (from_station in stations_set) and (to_station in stations_set):
                all_travel_dict[from_station][to_station].extend(times_list)

# extract unique times and sorted from low (fast) to high (slow)
all_travel_dict_sorted = {
    from_station: {to_station: sorted(list(set(time_list)))}
    for from_station, to_dict in all_travel_dict.items()
    for to_station, time_list in to_dict.items()
}

# %%
# # PADTON: n_2382
# # OXFD: n_1557
# k = 3
# paths = nx.shortest_simple_paths(
#     network, source="n_2382", target="n_1557", weight="weight"
# )
# for i, path in zip(range(k), paths):
#     length = sum(network[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
#     print(f"Path {i+1}: {path}, length={length}")

# %%
# Convert TIPLOC → node_id lookup once
tiploc_node = stations.set_index("TIPLOC")["node_id"].to_dict()

# Precompute a direct (u, v): ignore direction
edge_id_lookup = {}
edge_weight_dict = {}
for u, v, data in network.edges(data=True):
    edge_id_lookup[(u, v)] = data["edge_id"]
    edge_id_lookup[(v, u)] = data["edge_id"]
    edge_weight_dict[(u, v)] = data["weight"]
    edge_weight_dict[(v, u)] = data["weight"]

total_edge_speeds = defaultdict(list)
for from_station, to_dict in all_travel_dict_sorted.items():
    try:
        src = tiploc_node[from_station]
    except KeyError:
        print(f"No source station: {src}")
        continue

    for to_station, times in to_dict.items():
        try:
            tgt = tiploc_node[to_station]
        except KeyError:
            print(f"No target station: {tgt}")
            continue

        k = len(times)
        try:
            paths_gen = nx.shortest_simple_paths(network, src, tgt, weight="weight")
            for path, time in zip(
                islice(paths_gen, k), times
            ):  # time: total travel time between stations (second)
                # - sorted from fast
                # path: sorted from shortest
                edges_data = [
                    (u, v, edge_id_lookup[(u, v)], edge_weight_dict[(u, v)])
                    for u, v in zip(path[:-1], path[1:])
                ]
                total_length = sum(
                    w for _, _, _, w in edges_data
                )  # length of path between stations (meter)
                for _, _, eid, _ in edges_data:
                    total_edge_speeds[eid].append(total_length / time)

        except nx.exception.NetworkXNoPath:
            print(f"No path between {src} and {tgt}")
            continue

edge_speed_dict = dict(total_edge_speeds)  # 7 mins
edge_speed_dict2 = defaultdict(list)
for edge_id, speeds in edge_speed_dict.items():
    for speed in speeds:
        if speed <= 50:
            edge_speed_dict2[edge_id].append(speed)

edge_speed_dict2 = {
    edge_id: sorted(list(set(time_list)))
    for edge_id, time_list in edge_speed_dict2.items()
}

# %%
out_path = (
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "timetable"
    / "edge_speed_dict.json"
)
with open(out_path, "wt") as f:
    json.dump(edge_speed_dict2, f)
# only 30% of the edges were used according to timetable

# %%
edges["track_label"] = "slow"
edges.loc[edges.track_id2.isin(["11", "21", "31"]), "track_label"] = "fast"
edges["speeds_mps"] = edges.edge_id.map(edge_speed_dict2)
edges["speed_mps"] = np.nan
valid_edges = edges[edges.speeds_mps.notnull()].reset_index(drop=True)
valid_edges["speed_mps"] = valid_edges.apply(
    lambda row: (
        max(row["speeds_mps"])  # max -> fast tracks
        if row["track_label"] == "fast"
        else np.median(row["speeds_mps"])  # median -> other tracks
    ),
    axis=1,
)
invalid_edges = edges[edges.speeds_mps.isnull()].reset_index(drop=True)
print(f"valid: {valid_edges.shape}")
print(f"invalid: {invalid_edges.shape}")


# %%
def estimate_speed(valid_edges, invalid_edges):
    from_nodes = valid_edges[["from_node", "speed_mps"]].rename(
        columns={"from_node": "node"}
    )
    to_nodes = valid_edges[["to_node", "speed_mps"]].rename(columns={"to_node": "node"})
    valid_nodes = (
        pd.concat([from_nodes, to_nodes], axis=0, ignore_index=True)
        .groupby("node", as_index=False)
        .agg({"speed_mps": list})
    )
    valid_nodes["speed_mps"] = valid_nodes["speed_mps"].apply(lambda x: np.mean(x))
    node_speed_dict = valid_nodes.set_index("node")["speed_mps"].to_dict()
    invalid_edges["from_node_speed_mps"] = invalid_edges.from_node.map(node_speed_dict)
    invalid_edges["to_node_speed_mps"] = invalid_edges.to_node.map(node_speed_dict)

    cond1 = (
        invalid_edges["from_node_speed_mps"].notna()
        & invalid_edges["to_node_speed_mps"].notna()
    )
    cond2 = (
        invalid_edges["from_node_speed_mps"].isna()
        & invalid_edges["to_node_speed_mps"].notna()
    )
    cond3 = (
        invalid_edges["from_node_speed_mps"].notna()
        & invalid_edges["to_node_speed_mps"].isna()
    )

    invalid_edges["speed_mps"] = np.select(
        [cond1, cond2, cond3],
        [
            (invalid_edges["from_node_speed_mps"] + invalid_edges["to_node_speed_mps"])
            / 2,  # both present
            invalid_edges["to_node_speed_mps"],  # from NaN
            invalid_edges["from_node_speed_mps"],  # to NaN
        ],
        default=invalid_edges["speed_mps"],  # both NaN
    )
    invalid_edges.drop(
        columns=["from_node_speed_mps", "to_node_speed_mps"], inplace=True
    )
    temp = invalid_edges[invalid_edges.speed_mps.notnull()].reset_index(drop=True)
    invalid_edges = invalid_edges[invalid_edges.speed_mps.isnull()].reset_index(
        drop=True
    )
    valid_edges = pd.concat([valid_edges, temp], axis=0, ignore_index=True)
    return valid_edges, invalid_edges


# %%
# loop through to derive edge speeds (until valid number wont change)
prev_shape = None  # to store the last shape
while True:
    # Run your function
    valid_edges, invalid_edges = estimate_speed(valid_edges, invalid_edges)

    # Print status
    print(f"valid: {valid_edges.shape}")
    print(f"invalid: {invalid_edges.shape}")

    # Check if shape stopped changing
    if valid_edges.shape == prev_shape:
        break  # Exit the loop

    # Update previous shape for the next iteration
    prev_shape = valid_edges.shape

# %%
# assign min speeds to unconnected network edges
invalid_edges["speed_mps"] = valid_edges["speed_mps"].min()

# %%
edges = pd.concat([valid_edges, invalid_edges], axis=0, ignore_index=True)
edges = gpd.GeoDataFrame(edges, geometry="geometry", crs="epsg:27700")
edges.drop(
    columns=[
        "weight",
        "track_label",
        "speeds_mps",
    ],
    inplace=True,
)
edges["speed_kmh"] = edges["speed_mps"] * 3.6
edges.to_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_edges_with_speed.gpq"
)
