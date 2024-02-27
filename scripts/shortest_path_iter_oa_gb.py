# %%
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd  # type: ignore

from collections import defaultdict
import igraph  # type: ignore
from tqdm.auto import tqdm

from utils import load_config, get_flow_on_edges

import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])

CONV_METER_TO_MILE = 0.000621371
CONV_MILE_TO_KM = 1.60934
CONV_KM_TO_MILE = 0.621371
PENCE_TO_POUND = 0.01

"""
road_type: M, A, B
form_of_road: collapsed dual/dual, single, roundabout/slip
urban form: urban (maximum speed limit: 30 mph)
"""
flow_breakpoint_dict = {"M": 1200, "A": 1080}  # pcph
flow_cap_dict = {
    "M": 54000,  # 80000
    "A_dual": 54000,  # 80000,
    "A_single": 38400,
    "B": 33600,
}
free_flow_speed_dict = {
    "M": 67.0,
    "A_dual": 45.0,
    "A_single": 37.0,
    "B": 37.0,
}

min_speed_cap = {
    "M": 1.0,
    "A_dual": 0.2,
    "A_single": 0.2,
    "B": 0.4,
}  # while B is constant in this model


# %%
# function
def voc_func(speed: float) -> float:  # speed: mile/hour
    # d = distance * conv_mile_to_km  # km
    s = speed * CONV_MILE_TO_KM  # km/hour
    lpkm = 0.178 - 0.00299 * s + 0.0000205 * (s**2)  # fuel cost (liter/km)
    c = 140 * lpkm * PENCE_TO_POUND  # average petrol cost: 140 pence/liter
    return c  # pound/km


def cost_func(
    time: float, distance: float, voc: float
) -> float:  # time: hour, distance: mile/hour, voc: pound/km
    ave_occ = 1.6
    vot = 20  # value of time: pounds/hour
    d = distance * CONV_MILE_TO_KM  # km
    t = time + d * voc / (ave_occ * vot)
    return t  # hour


# speed functions
def initial_speed_func(road_type: str, form_of_road: str) -> Union[float, None]:
    if road_type == "M":
        return free_flow_speed_dict["M"]
    elif road_type == "A":
        if form_of_road == "Single Carriageway":
            return free_flow_speed_dict["A_single"]
        else:
            return free_flow_speed_dict["A_dual"]
    elif road_type == "B":
        return free_flow_speed_dict["B"]
    else:
        print("Error: initial speed!")
        return None


def speed_flow_func(
    road_type: str, form_of_road: str, isurban: int, vp: float
) -> Union[float, None]:
    vp = vp / 24
    if road_type == "M":
        initial_speed = free_flow_speed_dict["M"]
        if vp > 1200:  # speed starts to decrease
            vt = max((initial_speed - 0.033 * (vp - 1200)), min_speed_cap["M"])
            if isurban:
                return min(47.0, vt)
            else:
                return vt
        else:
            if isurban:
                return min(47.0, initial_speed)
            else:
                return initial_speed
    elif road_type == "A":
        if form_of_road == "Single Carriageway":
            initial_speed = free_flow_speed_dict["A_single"]
            if vp > 1200:
                vt = max(
                    (initial_speed - 0.05 * (vp - 1200)), min_speed_cap["A_single"]
                )
                if isurban:
                    return min(30.0, vt)
                else:
                    return vt
            else:
                if isurban:
                    return min(30.0, initial_speed)
                else:
                    return initial_speed
        else:
            initial_speed = free_flow_speed_dict["A_dual"]
            if vp > 1080:
                vt = max((initial_speed - 0.033 * (vp - 1080)), min_speed_cap["A_dual"])
                if isurban:
                    return min(30.0, vt)
                else:
                    return vt
            else:
                if isurban:
                    return min(30.0, initial_speed)
                else:
                    return initial_speed
    elif road_type == "B":
        initial_speed = free_flow_speed_dict["B"]
        if isurban:
            return min(30.0, initial_speed)
        else:
            return initial_speed
    else:
        print("Please select the road type from [M, A, B]!")
        return None


def filter_less_than_one(arr: np.ndarray) -> np.ndarray:
    return np.where(arr >= 1, arr, 0)


# %%
"""
list of inputs:
    (1) OS open roads.
    (2) Urban mask.
    (2) Population-weighed centroids of admin units.
    (3) O-D matrix (*travel to work by car).
"""

# input1: OS Open Roads
# OS open roads
osoprd_link = gpd.read_parquet(base_path / "networks" / "osoprd_road_links.parquet")
osoprd_node = gpd.read_parquet(base_path / "networks" / "osoprd_road_nodes.parquet")

# links (458827, )
# select major roads
road_link_file = osoprd_link[
    (osoprd_link.road_classification == "A Road")
    | (osoprd_link.road_classification == "B Road")
    | (osoprd_link.road_classification == "Motorway")
]
road_link_file.reset_index(drop=True, inplace=True)

# %%
# input2: urban mask
# apply urban mask
urban_mask = gpd.read_parquet(base_path / "networks" / "urban_mask.parquet")
road_link_file = road_link_file.sjoin(urban_mask, how="left")
road_link_file["urban"] = road_link_file["index_right"].apply(
    lambda x: 0 if pd.isna(x) else 1
)
road_link_file = road_link_file.drop(columns=["index_right", "FID_1"])

road_link_file["e_id"] = road_link_file.id
road_link_file["from_id"] = road_link_file.start_node
road_link_file["to_id"] = road_link_file.end_node

# nodes (423649, )
sel_node_idx = list(
    set(
        list(road_link_file.start_node.tolist())
        + list(road_link_file.end_node.tolist())
    )
)
road_node_file = osoprd_node[osoprd_node.id.isin(sel_node_idx)]
road_node_file.reset_index(drop=True, inplace=True)

road_node_file["nd_id"] = road_node_file.id
road_node_file["lat"] = road_node_file.geometry.y
road_node_file["lon"] = road_node_file.geometry.x

# %%
# input3: centroids of admins
zone_centroids = gpd.read_parquet(base_path / "census_datasets" / "zone_pwc.parquet")

# Find the nearest road node for each zone
nearest_node_dict = {}  # node_idx: zone_idx
for zone_idx, centroid_i in zone_centroids.iterrows():
    closest_road_node = road_node_file.sindex.nearest(
        centroid_i.geometry, return_all=False
    )[1][0]
    # for the first [x]:
    #   [0] represents the index of geometry;
    #   [1] represents the index of gdf
    # the second [x] represents the No. of closest item in the returned list,
    #   which only return one nearest node in this case
    nearest_node_dict[zone_idx] = closest_road_node

zone_to_node = {}
for zone_idx in range(zone_centroids.shape[0]):
    zonei = zone_centroids.loc[zone_idx, "code"]
    node_idx = nearest_node_dict[zone_idx]
    nodei = road_node_file.loc[node_idx, "nd_id"]
    zone_to_node[zonei] = nodei

# %%
# input4: O-D matrix
# original O-D (cross-border trips, by cars, 2011)
# Area of usual residence:
# OA (2011)
# Area of workplace:
# Workplace Zone (ENW, 2011),
# MOSA (ENW, 2011)
# Intermediate Zones (Scotland, 2001)
# OA (Scotland, 2011)
od_df = pd.read_csv(base_path / "census_datasets" / "od_gb_2011.csv")
print(f"total flows: {od_df.car.sum()}")  # 14_203_635 trips/day

from_zone_label = "Area of usual residence"
to_zone_label = "Area of workplace"

# nearest nodes <- O-D information
list_of_origin_node = []
destination_node_dict: dict[str, list[str]] = defaultdict(list)
flow_dict: dict[str, list[float]] = defaultdict(list)

invalid_home = []
invalid_work = []
for idx in tqdm(range(od_df.shape[0]), desc="Processing"):  # 11311607 zones
    from_zone = od_df.loc[idx, from_zone_label]
    to_zone = od_df.loc[idx, to_zone_label]
    count: float = od_df.loc[idx, "car"]  # type: ignore
    try:
        from_node = zone_to_node[from_zone]
    except KeyError:
        invalid_home.append(from_zone)
        print(f"No accessible net nodes to home {from_zone}!")

    try:
        to_node = zone_to_node[to_zone]
    except KeyError:
        invalid_work.append(to_zone)
        print(f"No accessible net nodes to workplace {to_zone}!")

    list_of_origin_node.append(from_node)  # origin
    destination_node_dict[from_node].append(to_node)  # destinations
    flow_dict[from_node].append(count)  # flows

# %%
list_of_origin_node = list(set(list_of_origin_node))  # 124740 nodes
list_of_origin_node.sort()

# %%
# network creation and configuration (igragh)
name_to_index = {name: index for index, name in enumerate(road_node_file.nd_id)}
index_to_name = {value: key for key, value in name_to_index.items()}
# nodes
nodeList = [
    (
        name_to_index[road_node_file.loc[i, "id"]],
        {
            "lon": road_node_file.loc[i, "geometry"].x,
            "lat": road_node_file.loc[i, "geometry"].y,
        },
    )
    for i in range(road_node_file.shape[0])
]
# edges
edge_weight_dict = [
    (
        road_link_file.loc[i, "e_id"],  # edge name
        road_link_file.loc[i, "from_id"],
        road_link_file.loc[i, "to_id"],
        road_link_file.loc[i, "geometry"].length,  # meter
        road_link_file.loc[i, "road_classification"],  # [M, A, B]
        road_link_file.loc[i, "urban"],  # urban/suburban
        road_link_file.loc[i, "form_of_way"],  # [dual, single...]
    )
    for i in range(road_link_file.shape[0])
]

edgeNameList = [
    e_namei
    for e_namei, sourcei, targeti, lengthi, typei, urbani, formi in edge_weight_dict
]
edgeList = [
    (name_to_index[sourcei], name_to_index[targeti])
    for _, sourcei, targeti, _, _, _, _ in edge_weight_dict
]
lengthList = [
    lengthi * CONV_METER_TO_MILE for _, _, _, lengthi, _, _, _ in edge_weight_dict
]  # convert meter to mile
typeList = [typei[0] for _, _, _, _, typei, _, _ in edge_weight_dict]  # M, A, B
formList = [formi for _, _, _, _, _, _, formi in edge_weight_dict]
speedList = np.vectorize(initial_speed_func)(
    typeList, formList
)  # initial speed: free-flow speed
# traveling time
timeList = np.array(lengthList) / np.array(speedList)  # hours
# def voc_func(speed):  # speed: mile/hour
vocList = np.vectorize(voc_func)(speedList)
# def cost_func(time, distance, voc):  # time: hour, distance: mile, voc: pound/km
timeList2 = np.vectorize(cost_func)(timeList, lengthList, vocList)  # hours
# weight: time + f(cost)
weightList = ((timeList + timeList2) * 3600).tolist()  # seconds

# %%
# create the network
test_net_ig = igraph.Graph(directed=False)
test_net_ig.add_vertices(nodeList)
test_net_ig.vs["nd_id"] = road_node_file.id.tolist()
test_net_ig.add_edges(edgeList)
test_net_ig.es["edge_name"] = edgeNameList
test_net_ig.es["weight"] = weightList

# edge_idx -> edge_name
edge_index_to_name = {idx: name for idx, name in enumerate(test_net_ig.es["edge_name"])}

# %%
# initailisation
# road type
road_link_file["road_type_label"] = road_link_file.road_classification.str[0]
road_type_dict = road_link_file.set_index("e_id")[
    "road_type_label"
]  # dict: e_id -> ABM
# form of road dict [dual, single, ...]
form_dict = road_link_file.set_index("e_id")["form_of_way"]
# urban form of road (binary)
isurban_dict = road_link_file.set_index("e_id")["urban"]
# length dict (miles)
length_dict = road_link_file.set_index("e_id")["geometry"].length * CONV_METER_TO_MILE
# M, A_dual, A_single, B
road_link_file["combined_label"] = road_link_file.road_type_label
road_link_file.loc[road_link_file.road_type_label == "A", "combined_label"] = "A_dual"
road_link_file.loc[
    (
        (road_link_file.road_type_label == "A")
        & (road_link_file.form_of_way.str.contains("Single"))
    ),
    "combined_label",
] = "A_single"  # only single carriageways of A roads


# accumulated flow dict (cars/day)
road_link_file["acc_flow"] = 0.0
acc_flow_dict = road_link_file.set_index("e_id")["acc_flow"]
# accumulated capacity dict (cars/day)
road_link_file["acc_capacity"] = road_link_file.combined_label.map(flow_cap_dict)
acc_capacity_dict = road_link_file.set_index("e_id")["acc_capacity"]
# accumulated average flow rate dict (miles/hour)
road_link_file["ave_flow_rate"] = road_link_file.combined_label.map(
    free_flow_speed_dict
)  # initial: free-flow speed
speed_dict = road_link_file.set_index("e_id")["ave_flow_rate"]

# %%
# flow simulation
total_remain = sum(sum(values) for values in flow_dict.values())
number_of_edges = len(list(test_net_ig.es))
# initial info
print(f"The initial number of edges in the network: {number_of_edges}")
print(f"The initial number of origins: {len(list_of_origin_node)}")
number_of_destinations = sum(len(value) for value in destination_node_dict.values())
print(f"The initial number of destinations: {number_of_destinations}")
total_non_allocated_flow = 0

iter_flag = 1
while total_remain > 0:  # iter_flag:
    print(f"No.{iter_flag} iteration starts:")
    list_of_spath = []
    for i in tqdm(range(len(list_of_origin_node)), desc="Processing"):
        name_of_origin_node = list_of_origin_node[i]
        idx_of_origin_node = name_to_index[name_of_origin_node]
        list_of_name_destination_node = destination_node_dict[
            name_of_origin_node
        ]  # a list of destination nodes
        list_of_idx_destination_node = [
            name_to_index[i] for i in list_of_name_destination_node
        ]
        flows = flow_dict[name_of_origin_node]
        paths = test_net_ig.get_shortest_paths(
            v=idx_of_origin_node,
            to=list_of_idx_destination_node,
            weights="weight",
            mode="out",
            output="epath",
        )
        # 7222: [origin, destination list, path list, flow list]
        list_of_spath.append(
            [name_of_origin_node, list_of_name_destination_node, paths, flows]
        )

    # expand the flow matrix
    temp_flow_matrix = pd.DataFrame(
        list_of_spath,
        columns=[
            "origin",
            "destination",
            "path",
            "flow",
        ],
    ).explode(["destination", "path", "flow"])

    # delete origins and destinations, of which "path = []"
    temp_df = temp_flow_matrix[temp_flow_matrix["path"].apply(lambda x: len(x) == 0)]
    non_allocated_flow = temp_df.flow.sum()
    print(f"Non_allocated_flow: {non_allocated_flow}")
    total_non_allocated_flow += non_allocated_flow

    for _, row in temp_df.iterrows():
        origin_temp = row["origin"]
        destination_temp = row["destination"]
        idx_temp = destination_node_dict[origin_temp].index(destination_temp)
        destination_node_dict[origin_temp].remove(destination_temp)
        del flow_dict[origin_temp][idx_temp]
        # ensure the same length of flow_dict and destination_node dict

    # aggregate edge flows
    temp_edge_flow = get_flow_on_edges(temp_flow_matrix, "e_id", "path", "flow")
    # edge_idx -> edge_name
    temp_edge_flow.e_id = temp_edge_flow.e_id.map(edge_index_to_name)

    temp_edge_flow["road_type"] = temp_edge_flow.e_id.map(road_type_dict)
    temp_edge_flow["form_of_way"] = temp_edge_flow.e_id.map(form_dict)
    temp_edge_flow["isurban"] = temp_edge_flow.e_id.map(isurban_dict)
    temp_edge_flow["temp_acc_flow"] = temp_edge_flow.e_id.map(
        acc_flow_dict
    )  # accumulated edge flows of previous step
    temp_edge_flow["temp_acc_capacity"] = temp_edge_flow.e_id.map(
        acc_capacity_dict
    )  # accumulated edge capacities of previous step
    temp_edge_flow["est_overflow"] = (
        temp_edge_flow.flow - temp_edge_flow.temp_acc_capacity
    )  # positive -> has overflow

    # calculate the adjusted flow
    max_overflow = temp_edge_flow.est_overflow.max()
    print(f"The maximum amount of overflow of edges: {max_overflow}")
    if max_overflow <= 0:
        temp_edge_flow["total_flow"] = (
            temp_edge_flow.flow + temp_edge_flow.temp_acc_flow
        )
        temp_edge_flow["speed"] = np.vectorize(speed_flow_func)(
            temp_edge_flow.road_type,
            temp_edge_flow.form_of_way,
            temp_edge_flow.isurban,
            temp_edge_flow.total_flow,
        )
        temp_edge_flow["remaining_capacity"] = (
            temp_edge_flow.temp_acc_capacity - temp_edge_flow.flow
        )
        # accumulated flows that have been assigned (temp: total_flow)
        temp_dict = temp_edge_flow.set_index("e_id")["total_flow"]
        acc_flow_dict.update(
            {key: temp_dict[key] for key in acc_flow_dict.keys() & temp_dict.keys()}
        )
        # average flow rate (temp: speed)
        temp_dict = temp_edge_flow.set_index("e_id")["speed"]
        speed_dict.update(
            {key: temp_dict[key] for key in speed_dict.keys() & temp_dict.keys()}
        )
        # accumulated remaining capacities (temp: remaining_capacity)
        temp_dict = temp_edge_flow.set_index("e_id")["remaining_capacity"]
        acc_capacity_dict.update(
            {key: temp_dict[key] for key in acc_capacity_dict.keys() & temp_dict.keys()}
        )

        # export the last-round of remaining flow matrix
        temp_flow_matrix = temp_flow_matrix[
            temp_flow_matrix["path"].apply(lambda x: len(x) != 0)
        ]
        # temp_flow_matrix.to_csv(base_path / "outputs" / "flow_matrix" / (str(iter_flag) + "_flow_matrix_msoa.csv"))
        print("Iteration stops: no edge overflow!")
        break

    # find the minimum r (ratio of flow assignment)
    temp_edge_flow["r"] = np.where(
        temp_edge_flow["flow"] != 0,
        temp_edge_flow["temp_acc_capacity"] / temp_edge_flow["flow"],
        np.nan,  # set as NaN when flow is zero
    )
    r = temp_edge_flow.r.min()  # r: (0,1)
    if r < 0:
        print("Error: r should not be negative!")
        break
    if r == 0:  # temp_acc_capacity = 0
        print("Error: existing network has zero-capacity links!")
        break
    if r >= 1:
        print("Error: r > 0: there is no edge flow!")
        break
    print(f"r = {r}")  # set as NaN when flow is zero

    # update flow matrix
    temp_flow_matrix = temp_flow_matrix[
        temp_flow_matrix["path"].apply(lambda x: len(x) != 0)
    ]
    temp_flow_matrix.flow = temp_flow_matrix.flow * r
    # temp_flow_matrix.to_csv(base_path / "outputs" / "flow_matrix" / (str(iter_flag) + "_flow_matrix_msoa.csv"))

    temp_edge_flow["adjusted_flow"] = temp_edge_flow.flow * r

    # update: [acc_flow, average_flow_rate (speed), acc_capacity]
    # calculate the total flow of each edge
    temp_edge_flow["total_flow"] = (
        temp_edge_flow.temp_acc_flow + temp_edge_flow.adjusted_flow
    )  # total flow = accumulated flow of previous steps + adjusted flow of current step

    # calculate the average flow rate
    temp_edge_flow["speed"] = np.vectorize(speed_flow_func)(
        temp_edge_flow.road_type,
        temp_edge_flow.form_of_way,
        temp_edge_flow.isurban,
        temp_edge_flow.total_flow,
    )
    # calculate the remaining capacity of each edge
    temp_edge_flow["remaining_capacity"] = (
        temp_edge_flow.temp_acc_capacity - temp_edge_flow.adjusted_flow
    )
    temp_edge_flow.loc[
        temp_edge_flow.remaining_capacity < 0, "remaining_capacity"
    ] = 0.0  # capacity is non-negative

    # update three dicts for next iteration
    # accumulated flows that have been assigned (temp: total_flow)
    temp_dict = temp_edge_flow.set_index("e_id")["total_flow"]
    acc_flow_dict.update(
        {key: temp_dict[key] for key in acc_flow_dict.keys() & temp_dict.keys()}
    )
    # average flow rate (temp: speed)
    temp_dict = temp_edge_flow.set_index("e_id")["speed"]
    speed_dict.update(
        {key: temp_dict[key] for key in speed_dict.keys() & temp_dict.keys()}
    )
    # accumulated remaining capacities (temp: remaining_capacity)
    temp_dict = temp_edge_flow.set_index("e_id")["remaining_capacity"]
    acc_capacity_dict.update(
        {key: temp_dict[key] for key in acc_capacity_dict.keys() & temp_dict.keys()}
    )

    # od matrix: remaining flow of each origin
    # flow_dict[origin_i] = [flow1, flow2,..., flowj, ... flowJ]
    flow_dict = {
        k: filter_less_than_one(np.array(v) * (1 - r)).tolist()
        for k, v in flow_dict.items()
    }

    total_remain = sum(sum(values) for values in flow_dict.values())
    print(f"The total remaining flow: {total_remain}")

    # update network structure
    zero_capacity_egdes = set(
        temp_edge_flow.loc[temp_edge_flow.remaining_capacity < 1, "e_id"].tolist()
    )
    net_edges = test_net_ig.es["edge_name"]
    idx_to_remove = [
        index
        for index, element in enumerate(net_edges)
        if element in zero_capacity_egdes
    ]

    # delete links reaching their full capacities
    test_net_ig.delete_edges(idx_to_remove)
    number_of_edges = len(list(test_net_ig.es))
    print(f"The remaining number of edges in the network: {number_of_edges}")

    # update weights (time: mph) for the remaining links
    remaining_edges = test_net_ig.es["edge_name"]
    lengthList = list(
        map(length_dict.get, filter(length_dict.__contains__, remaining_edges))
    )
    speedList = list(
        map(speed_dict.get, filter(speed_dict.__contains__, remaining_edges))
    )
    timeList = np.where(
        np.array(speedList) != 0, np.array(lengthList) / np.array(speedList), np.nan
    )  # hours

    if np.isnan(timeList).any():
        print("Error: have congested links!")
        # lanes where the average flow speed decreased to zero
        break
    else:
        vocList = np.vectorize(voc_func)(speedList)
        timeList2 = np.vectorize(cost_func)(timeList, lengthList, vocList)  # hours
        weightList = ((timeList + timeList2) * 3600).tolist()  # seconds
        test_net_ig.es["weight"] = weightList
        # update idx_to_name dict
        edge_index_to_name = {
            idx: name for idx, name in enumerate(test_net_ig.es["edge_name"])
        }

    # update the size of OD matrix for next iteration
    # delete zero-supply nodes from "list_of_origin_node"
    # delete zero-demand nodes from "destination_node_dict"
    new_flow_dict = {}
    new_destination_node_dict: dict[str, list[str]] = {}
    new_list_of_origin_node = []
    for origin, list_of_counts in flow_dict.items():
        tt_flow_from_origin = sum(list_of_counts)
        if tt_flow_from_origin > 0:
            # identify the valid origins
            new_list_of_origin_node.append(origin)
            # flow
            new_counts = [od_flow for od_flow in list_of_counts if od_flow != 0]
            new_flow_dict[origin] = new_counts
            # destination
            new_destination_node_dict[origin] = [
                dest
                for idx, dest in enumerate(destination_node_dict[origin])
                if list_of_counts[idx] != 0
            ]

    # replace the original dictionaries and list with the updated ones
    flow_dict = new_flow_dict
    destination_node_dict = new_destination_node_dict
    list_of_origin_node = new_list_of_origin_node

    number_of_destinations = sum(len(value) for value in destination_node_dict.values())
    print(f"The remaining number of origins: {len(list_of_origin_node)}")
    print(f"The remaining number of destinations: {number_of_destinations}")

    iter_flag += 1

print("The flow simulation is completed!")
print(f"The total non-allocated flow is {total_non_allocated_flow}")

# %%
# append estimations about: speed, flows, and remaining capacities
road_link_file.ave_flow_rate = road_link_file.e_id.map(speed_dict)
road_link_file.acc_flow = road_link_file.e_id.map(acc_flow_dict)
road_link_file.acc_capacity = road_link_file.e_id.map(acc_capacity_dict)

# change field types
road_link_file.acc_flow = road_link_file.acc_flow.astype(int)
road_link_file.acc_capacity = road_link_file.acc_capacity.astype(int)

# %%
# export file
road_link_file.to_file(
    base_path / "outputs" / "GB_flows_updated.gpkg",
    driver="GPKG",
)
