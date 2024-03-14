# %%
from typing import Union, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd  # type: ignore

import igraph  # type: ignore

from collections import defaultdict
import constants as cons

from tqdm.auto import tqdm
import warnings

warnings.simplefilter("ignore")


# %%
# functions
# extract major roads
def select_partial_roads(
    road_links: gpd.GeoDataFrame,
    road_nodes: gpd.GeoDataFrame,
    col_name: str,
    list_of_values: list,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    selected_links = []
    # road links selection
    for ci in list_of_values:
        selected_links.append(road_links[road_links[col_name] == ci])

    selected_links = pd.concat(selected_links, ignore_index=True)
    selected_links = gpd.GeoDataFrame(selected_links, geometry="geometry")

    selected_links["e_id"] = selected_links.id
    selected_links["from_id"] = selected_links.start_node
    selected_links["to_id"] = selected_links.end_node

    # road nodes selection
    sel_node_idx = list(
        set(selected_links.start_node.tolist() + selected_links.end_node.tolist())
    )

    selected_nodes = road_nodes[road_nodes.id.isin(sel_node_idx)]
    selected_nodes.reset_index(drop=True, inplace=True)
    selected_nodes["nd_id"] = selected_nodes.id
    selected_nodes["lat"] = selected_nodes.geometry.y
    selected_nodes["lon"] = selected_nodes.geometry.x

    return selected_links, selected_nodes


# urban road classification
def create_urban_mask(etisplus_urban_roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buf_geom = etisplus_urban_roads.geometry.buffer(
        500
    )  # create a buffer of 500 meters
    uni_geom = buf_geom.unary_union  # feature union within the same layer
    temp = gpd.GeoDataFrame(geometry=[uni_geom])
    new_geom = (
        temp.explode()
    )  # explode multi-polygons into separate individual polygons
    cvx_geom = (
        new_geom.convex_hull
    )  # generate convex polygon for each individual polygon

    urban_mask = gpd.GeoDataFrame(
        geometry=cvx_geom[0], crs=etisplus_urban_roads.crs
    ).to_crs("27700")
    return urban_mask


def label_urban_roads(
    road_links: gpd.GeoDataFrame, urban_mask: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    road_links = road_links.sjoin(urban_mask, how="left")
    road_links["urban"] = road_links["index_right"].apply(
        lambda x: 0 if pd.isna(x) else 1
    )
    road_links = road_links.drop(columns=["index_right"])
    return road_links


def voc_func(speed: float) -> float:  # speed: mile/hour
    # d = distance * conv_mile_to_km  # km
    s = speed * cons.CONV_MILE_TO_KM  # km/hour
    lpkm = 0.178 - 0.00299 * s + 0.0000205 * (s**2)  # fuel cost (liter/km)
    c = 140 * lpkm * cons.PENCE_TO_POUND  # average petrol cost: 140 pence/liter
    return c  # pound/km


def cost_func(
    time: float, distance: float, voc: float
) -> float:  # time: hour, distance: mile/hour, voc: pound/km
    ave_occ = 1.6
    vot = 20  # value of time: pounds/hour
    d = distance * cons.CONV_MILE_TO_KM  # km
    t = time + d * voc / (ave_occ * vot)
    return t  # hour


# speed functions
def initial_speed_func(
    road_type: str, form_of_road: str, free_flow_speed_dict: dict
) -> Union[float, None]:
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
    road_type: str,
    form_of_road: str,
    isurban: int,
    vp: float,
    free_flow_speed_dict: dict,
    flow_breakpoint_dict: dict,
    min_speed_cap: dict,
    urban_speed_cap: dict,
) -> Union[float, None]:
    vp = vp / 24
    if road_type == "M":
        initial_speed = free_flow_speed_dict["M"]
        if vp > flow_breakpoint_dict["M"]:  # speed starts to decrease
            vt = max(
                (initial_speed - 0.033 * (vp - flow_breakpoint_dict["M"])),
                min_speed_cap["M"],
            )
            if isurban:
                return min(urban_speed_cap["M"], vt)
            else:
                return vt
        else:
            if isurban:
                return min(urban_speed_cap["M"], initial_speed)
            else:
                return initial_speed
    elif road_type == "A":
        if form_of_road == "Single Carriageway":
            initial_speed = free_flow_speed_dict["A_single"]
            if vp > flow_breakpoint_dict["A_single"]:
                vt = max(
                    (initial_speed - 0.05 * (vp - flow_breakpoint_dict["A_single"])),
                    min_speed_cap["A_single"],
                )
                if isurban:
                    return min(urban_speed_cap["A_single"], vt)
                else:
                    return vt
            else:
                if isurban:
                    return min(urban_speed_cap["A_single"], initial_speed)
                else:
                    return initial_speed
        else:
            initial_speed = free_flow_speed_dict["A_dual"]
            if vp > flow_breakpoint_dict["A_dual"]:
                vt = max(
                    (initial_speed - 0.033 * (vp - flow_breakpoint_dict["A_dual"])),
                    min_speed_cap["A_dual"],
                )
                if isurban:
                    return min(urban_speed_cap["A_dual"], vt)
                else:
                    return vt
            else:
                if isurban:
                    return min(urban_speed_cap["A_dual"], initial_speed)
                else:
                    return initial_speed
    elif road_type == "B":
        initial_speed = free_flow_speed_dict["B"]
        if isurban:
            return min(urban_speed_cap["B"], initial_speed)
        else:
            return initial_speed
    else:
        print("Please select the road type from [M, A, B]!")
        return None


def filter_less_than_one(arr: np.ndarray) -> np.ndarray:
    return np.where(arr >= 1, arr, 0)


# find nearest network node for each admin centroid
def find_nearest_node(zones: gpd.GeoDataFrame, road_nodes: gpd.GeoDataFrame) -> dict:
    nearest_node_dict = {}  # node_idx: zone_idx
    for zidx, z in zones.iterrows():
        closest_road_node = road_nodes.sindex.nearest(z.geometry, return_all=False)[1][
            0
        ]
        # for the first [x]:
        #   [0] represents the index of geometry;
        #   [1] represents the index of gdf
        # the second [x] represents the No. of closest item in the returned list,
        #   which only return one nearest node in this case
        nearest_node_dict[zidx] = closest_road_node

    zone_to_node = {}
    for zidx in range(zones.shape[0]):
        z = zones.loc[zidx, "code"]
        nidx = nearest_node_dict[zidx]
        n = road_nodes.loc[nidx, "nd_id"]
        zone_to_node[z] = n

    return zone_to_node


# interpret od matrix
# {list of origins,
# list of destinations attached to each origin,
# list of supplies from each origin}
def od_interpret(
    od_matrix: pd.DataFrame,
    zone_to_node: dict,
    col_origin: str,
    col_destination: str,
    col_count: str,
) -> Tuple[list, dict, dict]:

    list_of_origins = []
    destination_dict: dict[str, list[str]] = defaultdict(list)
    supply_dict: dict[str, list[float]] = defaultdict(list)

    for idx in tqdm(range(od_matrix.shape[0]), desc="Processing"):
        from_zone = od_matrix.loc[idx, col_origin]
        to_zone = od_matrix.loc[idx, col_destination]
        count: float = od_matrix.loc[idx, col_count]  # type: ignore
        try:
            from_node = zone_to_node[from_zone]
        except KeyError:
            print(f"No accessible network node to attached to home/origin {from_zone}!")
        try:
            to_node = zone_to_node[to_zone]
        except KeyError:
            print(
                f"No accessible network node attached to workplace/destination {to_zone}!"
            )

        list_of_origins.append(from_node)  # origin
        destination_dict[from_node].append(to_node)  # origin -> destinations
        supply_dict[from_node].append(count)  # origin -> supply

    return list_of_origins, destination_dict, supply_dict


# network creation
def create_igraph_network(
    name_to_index: dict, road_links: gpd.GeoDataFrame, road_nodes: gpd.GeoDataFrame
) -> igraph.Graph:
    nodeList = [
        (
            name_to_index[node.id],
            {
                "lon": node.geometry.x,
                "lat": node.geometry.y,
            },
        )
        for _, node in road_nodes.iterrows()
    ]  # (id, {x, y})

    edgeNameList = []
    edgeList = []
    edgeLengthList = []
    edgeTypeList = []
    edgeFormList = []
    for _, link in road_links.iterrows():
        edge_name = link.e_id
        edge_from = link.from_id
        edge_to = link.to_id
        edge_length = link.geometry.length * cons.CONV_METER_TO_MILE  # miles
        edge_type = link.road_classification
        edge_form = link.form_of_way
        edgeNameList.append(edge_name)
        edgeList.append((name_to_index[edge_from], name_to_index[edge_to]))
        edgeLengthList.append(edge_length)
        edgeTypeList.append(edge_type)
        edgeFormList.append(edge_form)

    edgeSpeedList = np.vectorize(initial_speed_func)(
        edgeTypeList, edgeFormList
    )  # miles/hour

    # travel time
    timeList = np.array(edgeLengthList) / np.array(edgeSpeedList)  # hour

    # total travel cost (time-equivalent)
    vocList = np.vectorize(voc_func)(edgeSpeedList)  # £/km
    costList = np.vectorize(cost_func)(timeList, edgeLengthList, vocList)  # hour
    weightList = (costList * 3600).tolist()  # seconds

    test_net = igraph.Graph(directed=False)
    test_net.add_vertices(nodeList)
    test_net.vs["nd_id"] = road_nodes.id.tolist()
    test_net.add_edges(edgeList)
    test_net.es["edge_name"] = edgeNameList
    test_net.es["weight"] = weightList

    return test_net


# network initialization
# road type
def initialise_igraph_network(
    road_links: gpd.GeoDataFrame,
    initial_capacity_dict: dict,
    initial_speed_dict: dict,
    col_id: str,
    col_rdclass: str,
    col_form: str,
    col_urban: str,
) -> Tuple[gpd.GeoDataFrame, dict, dict, dict, dict, dict, dict, dict]:
    # road_types: M, A, B
    road_links["road_type_label"] = road_links[col_rdclass].str[0]
    edge_type_dict = road_links.set_index(col_id)["road_type_label"]
    edge_form_dict = road_links.set_index(col_id)[col_form]
    edge_isUrban_dict = road_links.set_index(col_id)[col_urban]
    edge_length_dict = (
        road_links.set_index(col_id)["geometry"].length * cons.CONV_METER_TO_MILE
    )

    # road_types and road_forms: M, A_dual, A_single, B
    road_links["combined_label"] = road_links.road_type_label
    road_links.loc[road_links.road_type_label == "A", "combined_label"] = "A_dual"
    road_links.loc[
        (
            (road_links.road_type_label == "A")
            & (road_links.form_of_way.str.contains("Single"))
        ),
        "combined_label",
    ] = "A_single"  # only single carriageways of A roads

    # accumulated edge flows (cars/day)
    road_links["acc_flow"] = 0.0
    acc_flow_dict = road_links.set_index(col_id)["acc_flow"]

    # remaining edge capacities (cars/day)
    road_links["acc_capacity"] = road_links.combined_label.map(initial_capacity_dict)
    acc_capacity_dict = road_links.set_index(col_id)["acc_capacity"]

    # average edge flow rates (miles/hour)
    road_links["ave_flow_rate"] = road_links["combined_label"].map(initial_speed_dict)
    acc_speed_dict = road_links.set_index(col_id)["ave_flow_rate"]

    return (
        road_links,
        edge_type_dict,
        edge_form_dict,
        edge_isUrban_dict,
        edge_length_dict,
        acc_flow_dict,
        acc_capacity_dict,
        acc_speed_dict,
    )
