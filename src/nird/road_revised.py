""" Road Network Flow Model - Functions
"""

from typing import Tuple, List, Dict
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import geopandas as gpd  # type: ignore
import igraph  # type: ignore

import nird.constants as cons
from nird.utils import get_flow_on_edges

from multiprocessing import Pool
import warnings
import pickle
import time

warnings.simplefilter("ignore")


def select_partial_roads(
    road_links: gpd.GeoDataFrame,
    road_nodes: gpd.GeoDataFrame,
    col_name: str,
    list_of_values: List[str],
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Extract partial road network based on road types.

    Parameters
    ----------
    road_links: gpd.GeoDataFrame
        Links of the road network.
    road_nodes: gpd.GeoDataFrame
        Nodes of the road network.
    col_name: str
        The road type column.
    list_of_values:
        The road types to be extracted from the network.

    Returns
    -------
    selected_links: gpd.GeoDataFrame
        Partial road links.
    selected_nodes: gpd.GeoDataFrame
        Partial road nodes.
    """
    # road links selection
    selected_links = road_links[road_links[col_name].isin(list_of_values)].reset_index(
        drop=True
    )
    selected_links.rename(
        columns={"id": "e_id", "start_node": "from_id", "end_node": "to_id"},
        inplace=True,
    )
    # road nodes selection
    sel_node_idx = list(
        set(selected_links.from_id.tolist() + selected_links.to_id.tolist())
    )
    selected_nodes = road_nodes[road_nodes.id.isin(sel_node_idx)]
    selected_nodes.reset_index(drop=True, inplace=True)
    selected_nodes.rename(columns={"id": "nd_id"}, inplace=True)
    selected_nodes["lat"] = selected_nodes["geometry"].y
    selected_nodes["lon"] = selected_nodes["geometry"].x

    return selected_links, selected_nodes


def create_urban_mask(
    etisplus_urban_roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """To extract urban areas across Great Britain (GB) based on ETISPLUS datasets.

    Parameters
    ----------
    etisplus_urban_roads: gpd.GeoDataFrame
        D6 ETISplus Roads (2010)

    Returns
    -------
    urban_mask: gpd.GeoDataFrame
        A binary file that spatially identify the urban areas across GB
        with values == 1.
    """
    etisplus_urban_roads = etisplus_urban_roads[
        etisplus_urban_roads["Urban"] == 1
    ].reset_index(drop=True)
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
    """Classify road links into urban/suburban roads.

    Parameters
    ----------
    road_links: gpd.GeoDataFrame
        Links of the road network.
    urban_mask: gpd.GeoDataFrame
        A binary file that spatially identify the urban areas across GB.

    Returns
    -------
    road_links: gpd.GeoDataFrame
        Create a column "urban" to classify road links into urban/suburban links.
    """
    temp_file = road_links.sjoin(urban_mask, how="left")
    temp_file["urban"] = temp_file["index_right"].apply(
        lambda x: 0 if pd.isna(x) else 1
    )
    max_values = temp_file.groupby("e_id")["urban"].max()
    road_links = road_links.merge(max_values, on="e_id", how="left")
    return road_links


def find_nearest_node(
    zones: gpd.GeoDataFrame,
    road_nodes: gpd.GeoDataFrame,
    zone_id_column: str,
    node_id_column: str,
) -> Dict[str, str]:
    """Find the nearest road node for each admin unit.

    Parameters
    ----------
    zones: gpd.GeoDataFrame
        Admin units.
    road nodes: gpd.GeoDataFrame
        Nodes of the road network.

    Returns
    -------
    nearest_node_dict: dict
        A dictionary to convert from admin units to their attached road nodes.
    """
    nearest_nodes = gpd.sjoin_nearest(zones, road_nodes)
    nearest_nodes = nearest_nodes.drop_duplicates(subset=[zone_id_column], keep="first")
    nearest_node_dict = dict(
        zip(nearest_nodes[zone_id_column], nearest_nodes[node_id_column])
    )
    return nearest_node_dict  # zone_idx: node_idx


def voc_func(
    speed: float,
) -> float:
    """Calculate the Vehicle Operating Cost (VOC).

    Parameters
    ----------
    speed: float
        The average flow speed: mph

    Returns
    -------
    float
        The unit vehicle operating cost: £/km
    """
    s = speed * cons.CONV_MILE_TO_KM  # km/hour
    lpkm = 0.178 - 0.00299 * s + 0.0000205 * (s**2)  # fuel consumption (liter/km)
    voc_per_km = (
        140 * lpkm * cons.PENCE_TO_POUND
    )  # average petrol cost: 140 pence/liter
    return voc_per_km  # £/km


def cost_func(
    time: float,
    distance: float,
    voc_per_km: float,
    toll: float,
) -> Tuple[float, float, float]:  # time: hour, distance: mph, voc: £/km
    """Calculate the total travel cost.

    Parameters
    ----------
    time: float
        Travel time: hour
    distance: float
        Travel distance: mile
    voc:
        Vehicle operating cost: £/km

    Returns
    -------
    cost: float
        The total travel costs: £
    c_time: float
        The time-equivalent costs: £
    c_operate: float
        The vehicle operating costs/fuel costs: £
    """
    ave_occ = 1.06  # average car occupancy = 1.6
    vot = 17.69  # value of time (VOT): 17.69 £/hour
    d = distance * cons.CONV_MILE_TO_KM
    c_time = time * ave_occ * vot
    c_operate = d * voc_per_km
    cost = time * ave_occ * vot + d * voc_per_km + toll
    return cost, c_time, c_operate


def edge_reclassification_func(
    road_links: pd.DataFrame,
) -> pd.DataFrame:
    """Reclassify network edges to "M, A_dual, A_single, B"."""
    road_links["combined_label"] = "A_dual"
    road_links.loc[road_links.road_classification == "Motorway", "combined_label"] = "M"
    road_links.loc[road_links.road_classification == "B Road", "combined_label"] = "B"
    road_links.loc[
        (road_links.road_classification == "A Road")
        & (road_links.form_of_way == "Single Carriageway"),
        "combined_label",
    ] = "A_single"
    return road_links


def edge_initial_speed_func(
    road_links: pd.DataFrame,
    free_flow_speed_dict: Dict[str, float],
    urban_flow_speed_dict: Dict[str, float],
    min_flow_speed_dict: Dict[str, float],
    max_flow_speed_dict: Dict[str, float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Calculate the initial vehicle speed for network edges.

    Parameters
    ----------
    road_links: pd.DataFrame
        network link features
    free_flow_speed_dict: Dict
        free-flow vehicle operating speed on roads: M, A(single/dual), B
    urban_flow_speed_dict: Dict
        set the maximum vehicle speed restrictions in urban areas
    min_flow_speed_dict: Dict
        minimum vehicle operating speeds
    max_flow_speed_dict: Dict
        maximum safe vehicel operatin speeds on flooded roads

    Returns
    -------
    road_links: pd.DataFrame
        with speed information
    initial_speed_dict: Dict
        initial vehicle operating speed of existing road links
    """

    assert "combined_label" in road_links.columns, "combined_label column not exists!"
    assert "urban" in road_links.columns, "urban column not exists!"

    if (
        "free_flow_speeds" not in road_links.columns
    ):  # add free-flow speeds if not exist
        road_links["free_flow_speeds"] = road_links.combined_label.map(
            free_flow_speed_dict
        )

    road_links.loc[road_links["urban"] == 1, "free_flow_speeds"] = road_links[
        "combined_label"
    ].map(
        urban_flow_speed_dict
    )  # urban speed restriction
    road_links["min_flow_speeds"] = road_links.combined_label.map(min_flow_speed_dict)
    road_links["initial_flow_speeds"] = road_links["free_flow_speeds"]
    if max_flow_speed_dict is not None:
        road_links["max_speeds"] = road_links["e_id"].map(max_flow_speed_dict)
        # if max < min: close the roads
        road_links.loc[
            road_links.max_speeds < road_links.min_flow_speeds, "initial_flow_speeds"
        ] = 0.0
        # if min < max < free
        road_links.loc[
            (road_links.max_speeds >= road_links.min_flow_speeds)
            & (road_links.max_speeds < road_links.free_flow_speeds),
            "initial_flow_speeds",
        ] = road_links.max_speeds
        # if max > free: free flow speeds (by default)
        # remove the closed/damaged roads
        road_links = road_links[road_links["initial_flow_speeds"] > 0]
        road_links.reset_index(drop=True, inplace=True)

    return road_links


def edge_init(
    road_links: gpd.GeoDataFrame,
    capacity_plph_dict: Dict[str, float],
    free_flow_speed_dict: Dict[str, float],
    urban_flow_speed_dict: Dict[str, float],
    min_flow_speed_dict: Dict[str, float],
    max_flow_speed_dict: Dict[str, float],
) -> gpd.GeoDataFrame:
    """Network edges initialisation.

    Parameters
    ----------
    road_links: gpd.GeoDataFrame
    capacity_plph_dict: Dict
        The designed edge capacity (per lane per hour).
    free_flow_speed_dict: Dict
        The free-flow edge speed of different types of road links.
    urban_flow_speed_dict: Dict
        The maximum vehicle operating speed restriction in urban areas.
    min_flow_speed_dict: Dict
        The minimum vehicle operating speeds.
    max_flow_speed_dict: Dict
        The maximum flow speed of the flooded road links.

    Returns
    -------
    road_links: gpd.GeoDataFrame
        Road links with added attributes.
    """
    # reclassification
    road_links = edge_reclassification_func(road_links)
    road_links = edge_initial_speed_func(
        road_links,
        free_flow_speed_dict,
        urban_flow_speed_dict,
        min_flow_speed_dict,
        max_flow_speed_dict,
    )
    assert "combined_label" in road_links.columns, "combined_label column not exists!"
    assert (
        "initial_flow_speeds" in road_links.columns
    ), "initial_flow_speeds column not exists!"

    # initialise key variables
    road_links["acc_flow"] = 0.0
    road_links["acc_capacity"] = (
        road_links["combined_label"].map(capacity_plph_dict) * road_links["lanes"] * 24
    )
    road_links["acc_speed"] = road_links["initial_flow_speeds"]

    # remove invalid road links
    road_links = road_links[road_links.acc_capacity > 0.5].reset_index(drop=True)

    return road_links


def speed_flow_func(
    combined_label: str,
    total_flow: float,
    initial_flow_speed: float,
    min_flow_speed: float,
    flow_breakpoint_dict: Dict[str, float],
) -> float:
    """Create the speed-flow curves for different types of roads.

    Parameters
    ----------
    combined_label: str
        The type or road links.
    total_flow: float
        The number of vehicles on road links.
    initial_flow_speed: float
        The initial traffic speeds of road links.
    min_flow_speed: float
        The minimum traffic speeds of road links.
    flow_breakpoint_dict: dict
        The breakpoint flows after which traffic flows starts to decrease.

    Returns
    -------
    speed: float
        The "real-time" traffic speeds on road links.
    """

    vp = total_flow / 24
    if combined_label == "M" and vp > flow_breakpoint_dict["M"]:
        speed = initial_flow_speed - 0.033 * (vp - flow_breakpoint_dict["M"])
    elif combined_label == "A_single" and vp > flow_breakpoint_dict["A_single"]:
        speed = initial_flow_speed - 0.05 * (vp - flow_breakpoint_dict["A_single"])
    elif combined_label == "A_dual" and vp > flow_breakpoint_dict["A_dual"]:
        speed = initial_flow_speed - 0.033 * (vp - flow_breakpoint_dict["A_dual"])
    elif combined_label == "B" and vp > flow_breakpoint_dict["B"]:
        speed = initial_flow_speed - 0.05 * (vp - flow_breakpoint_dict["B"])
    else:
        speed = initial_flow_speed
    speed = max(speed, min_flow_speed)
    return speed


def create_igraph_network(
    road_links: gpd.GeoDataFrame,
) -> igraph.Graph:
    """Create an undirected igraph network."""
    road_links["edge_length_mile"] = (
        road_links.geometry.length * cons.CONV_METER_TO_MILE
    )
    road_links["time_hr"] = (
        1.0 * road_links.edge_length_mile / road_links.acc_speed
    )  # return inf -> acc_speed is 0?
    road_links["voc_per_km"] = np.vectorize(voc_func, otypes=None)(road_links.acc_speed)

    (
        road_links["weight"],
        road_links["time_cost"],
        road_links["operating_cost"],
    ) = np.vectorize(cost_func, otypes=None)(
        road_links["time_hr"],
        road_links["edge_length_mile"],
        road_links["voc_per_km"],
        road_links["average_toll_cost"],
    )
    graph_df = road_links[
        [
            "from_id",
            "to_id",
            "e_id",
            "weight",
            "time_cost",
            "operating_cost",
            "average_toll_cost",
        ]
    ]

    network = igraph.Graph.TupleList(
        graph_df.itertuples(index=False),
        edge_attrs=list(graph_df.columns)[2:],
        directed=False,
    )
    return network


def update_network_structure(
    network: igraph.Graph,
    temp_edge_flow: pd.DataFrame,
    road_links: gpd.GeoDataFrame,
) -> igraph.Graph:
    """Drop fully utilised edges and Update edge weights.

    Parameters
    ----------
    network: igraph network
    temp_edge_flow: the remaining edge capacity at the current iteration
    road_links: road links

    Returns
    -------
    The updated igraph network
    """
    # drop fully utilised edges from the network
    zero_capacity_edges = set(
        temp_edge_flow.loc[temp_edge_flow["acc_capacity"] < 0.5, "e_id"].tolist()
    )
    edges_to_remove = [e.index for e in network.es if e["e_id"] in zero_capacity_edges]
    network.delete_edges(edges_to_remove)
    print(f"The remaining number of edges in the network: {len(list(network.es))}")

    # update edges' weights
    remaining_edges = network.es["e_id"]
    graph_df = road_links[road_links.e_id.isin(remaining_edges)][
        [
            "from_id",
            "to_id",
            "e_id",
            "weight",
            "time_cost",
            "operating_cost",
            "average_toll_cost",
        ]
    ]
    network = igraph.Graph.TupleList(
        graph_df.itertuples(index=False),
        edge_attrs=list(graph_df.columns)[2:],
        directed=False,
    )
    return network


def extract_od_pairs(
    od: pd.DataFrame,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[int]]]:
    """Prepare the OD matrix.

    Parameters
    ----------
    od: pd.DataFrame
        Table of origin-destination passenger flows.

    Returns
    -------
    list_of_origin_nodes: list
        A list of origin nodes.
    dict_of_destination_nodes: dict[str, list[str]]
        A dictionary recording a list of destination nodes for each origin node.
    dict_of_origin_supplies: dict[str, list[int]]
        A dictionary recording a list of flows for each origin-destination pair.
    """
    list_of_origin_nodes = []
    dict_of_destination_nodes: Dict[str, List[str]] = defaultdict(list)
    dict_of_origin_supplies: Dict[str, List[float]] = defaultdict(list)
    for row in od.itertuples():
        from_node = row.origin_node
        to_node = row.destination_node
        Count: float = row.Car21
        list_of_origin_nodes.append(from_node)  # [nd_id...]
        dict_of_destination_nodes[from_node].append(to_node)  # {nd_id: [nd_id...]}
        dict_of_origin_supplies[from_node].append(Count)  # {nd_id: [car21...]}

    # Extract identical origin nodes
    list_of_origin_nodes = list(set(list_of_origin_nodes))
    list_of_origin_nodes.sort()

    return (
        list_of_origin_nodes,
        dict_of_destination_nodes,
        dict_of_origin_supplies,
    )


def update_od_matrix(
    temp_flow_matrix: pd.DataFrame,
    remain_od: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide the OD matrix into allocated and unallocated sections.

    Parameters
    ----------
    temp_flow_matrix: origin, destination, path, flow, operating_cost_per_flow,
        time_cost_per_flow, toll_cost_per_flow
    remain_od: origin, destination, flow

    Returns
    -------
    temp_flow_matrix: flow matrix with valid path
    isolated_flow_matrix: flow matrix with no path
    remain_od: the remaining od matrix
    """
    mask = temp_flow_matrix["path"].apply(lambda x: len(x) == 0)
    isolated_flow_matrix = temp_flow_matrix.loc[
        mask, ["origin", "destination", "flow"]
    ]  # drop the cost columns
    print(f"Non_allocated_flow: {isolated_flow_matrix.flow.sum()}")
    temp_flow_matrix = temp_flow_matrix[~mask]
    remain_origins = temp_flow_matrix.origin.unique().tolist()
    remain_destinations = temp_flow_matrix.destination.unique().tolist()
    remain_od = remain_od[
        (
            remain_od.origin_node.isin(remain_origins)
            & (remain_od.destination_node.isin(remain_destinations))
        )
    ].reset_index(drop=True)

    return temp_flow_matrix, remain_od, isolated_flow_matrix


def find_least_cost_path(
    params: Tuple,
) -> Tuple[int, List[str], List[int], List[float]]:
    """Find the least-cost path for each OD trip.

    Parameters:
    -----------
    params: Tuple
        The first element: the origin node (index).
        The second element: a list of destination nodes (indexes).
        The third element: a list of outbound trips from origin to its connected destinations.

    Returns:
    --------
        idx_of_origin_node: int
            Same as input.
        list_of_idx_destination_nodes: list
            Same as input.
        paths: list
            The least-cost paths.
        flows: list
            Same as input.
    """
    origin_node, destination_nodes, flows = params
    paths = shared_network.get_shortest_paths(
        v=origin_node,
        to=destination_nodes,
        weights="weight",
        mode="out",
        output="epath",
    )  # paths: o - d(s)
    edge_paths = []
    operating_costs = []
    time_costs = []
    toll_costs = []
    for path in paths:  # path: o - d
        edge_path = []
        operating_cost = []
        time_cost = []
        toll_cost = []
        for p in path:  # p: each line segment
            edge_path.append(shared_network.es[p]["e_id"])
            operating_cost.append(shared_network.es[p]["operating_cost"])
            time_cost.append(shared_network.es[p]["time_cost"])
            toll_cost.append(shared_network.es[p]["average_toll_cost"])
        edge_paths.append(edge_path)  # a list of lists
        operating_costs.append(sum(operating_cost))  # a list of values
        time_costs.append(sum(time_cost))  # a list of values
        toll_costs.append(sum(toll_cost))  # a list of values

    return (
        origin_node,
        destination_nodes,
        edge_paths,
        flows,
        operating_costs,
        time_costs,
        toll_costs,
    )


def compute_edge_costs(
    # edge_weight_df,
    path: List[int],
) -> Tuple[float, float, float]:
    """Calculate the total travel cost for the path

    Parameters
    ----------
    path: List
        A list of edge indexes that define the path.

    Returns
    -------
    od_voc: float
        Vehicle operating costs of each trip.
    od_vot: float
        Value of time of each trip.
    od_toll: float
        Toll costs of each trip.
    """
    od_voc = edge_weight_df.loc[edge_weight_df["e_id"].isin(path), "voc"].sum()
    od_vot = edge_weight_df.loc[edge_weight_df["e_id"].isin(path), "time_cost"].sum()
    od_toll = edge_weight_df.loc[
        edge_weight_df["e_id"].isin(path), "average_toll_cost"
    ].sum()
    return (od_voc, od_vot, od_toll)


def worker_init_path(
    shared_network_pkl: bytes,
) -> None:
    """Worker initialisation in multiprocesses to create a shared network
    that could be used across different workers.

    Parameters
    ----------
    shared_network_pkl: bytes
        The pickled file of the igraph network.

    Returns
    -------
    None.
    """
    global shared_network
    shared_network = pickle.loads(shared_network_pkl)
    return None


def worker_init_edge(
    shared_network_pkl: bytes,
    shared_weight_pkl: bytes,
) -> None:
    """Worker initialisation in multiprocesses to create a shared network
    that could be used across different workers.

    Parameters
    ----------
    shared_network_pkl: bytes
        The pickled file of the igraph network.
    shared_weight_pkl: bytes
        The pickled flle of a dictionary of network edge weights.

    Returns
    -------
    None.
    """
    # print(os.getpid())
    global shared_network
    shared_network = pickle.loads(shared_network_pkl)
    global edge_weight_df
    edge_weight_df = pickle.loads(shared_weight_pkl)
    return None


def network_flow_model(
    road_links: gpd.GeoDataFrame,
    network: igraph.Graph,
    remain_od: pd.DataFrame,
    flow_breakpoint_dict: Dict[str, float],
    num_of_cpu,
) -> Tuple[gpd.GeoDataFrame, List, List]:
    """Process-based Network Flow Simulation.

    Parameters
    ----------
    road_links: road links
    network: igraph network
    remain_od: od matrix
    flow_breakpoint_dict: flow breakpoints of different types of road links

    Returns
    -------
    road_links: with added attributes (acc_flow, acc_capacity, acc_speed)
    isolation: non-allocated od matrix
    odpfc: allocated od matrix
    """

    partial_speed_flow_func = partial(
        speed_flow_func,
        flow_breakpoint_dict=flow_breakpoint_dict,
    )
    total_remain = remain_od.Car21.sum()
    print(f"The initial supply is {total_remain}")
    number_of_edges = len(list(network.es))
    print(f"The initial number of edges in the network: {number_of_edges}")
    number_of_origins = remain_od.origin_node.unique().shape[0]
    print(f"The initial number of origins: {number_of_origins}")
    number_of_destinations = remain_od.destination_node.unique().shape[0]
    print(f"The initial number of destinations: {number_of_destinations}")

    # starts
    total_cost = 0
    time_equiv_cost = 0
    operating_cost = 0
    toll_cost = 0
    odpfc = []
    isolation = []

    iter_flag = 1
    while total_remain > 0:
        print(f"No.{iter_flag} iteration starts:")
        # check isolations
        mask = remain_od.origin_node.isin(network.vs["name"]) & (
            remain_od.destination_node.isin(network.vs["name"])
        )
        isolation.extend(
            remain_od.loc[
                ~mask, ["origin_node", "destination_node", "Car21"]
            ].values.tolist()
        )
        remain_od = remain_od[mask].reset_index(drop=True)

        # dump the network and edge weight for shared use in multiprocessing
        shared_network_pkl = pickle.dumps(network)

        # find the least-cost path for each OD trip
        list_of_spath = []
        args = []
        list_of_origin_nodes = list(set(remain_od["origin_node"].tolist()))
        for origin_node in list_of_origin_nodes:
            destination_nodes = remain_od.loc[
                remain_od["origin_node"] == origin_node, "destination_node"
            ].tolist()
            flows = remain_od.loc[
                remain_od["origin_node"] == origin_node, "Car21"
            ].tolist()
            args.append((origin_node, destination_nodes, flows))

        st = time.time()
        with Pool(
            processes=num_of_cpu,
            initializer=worker_init_path,
            initargs=(shared_network_pkl,),
        ) as pool:
            list_of_spath = pool.map(find_least_cost_path, args)
            # [origin(name), destinations(name), path(idx), flow(int)]
        print(f"The least-cost path flow allocation time: {time.time() - st}.")

        temp_flow_matrix = pd.DataFrame(
            list_of_spath,
            columns=[
                "origin",
                "destination",
                "path",
                "flow",
                "operating_cost_per_flow",
                "time_cost_per_flow",
                "toll_cost_per_flow",
            ],
        ).explode(
            [
                "destination",
                "path",
                "flow",
                "operating_cost_per_flow",
                "time_cost_per_flow",
                "toll_cost_per_flow",
            ]
        )

        # calculate the non-allocated flows and remaining flows
        (
            temp_flow_matrix,
            remain_od,
            isolated_flow_matrix,
        ) = update_od_matrix(temp_flow_matrix, remain_od)

        # update the origin-destination-path-cost matrix
        odpfc.extend(temp_flow_matrix.to_numpy().tolist())

        # update the isolated flows
        isolation.extend(isolated_flow_matrix.to_numpy().tolist())

        # calculate the remaining flows
        number_of_origins = remain_od.origin_node.unique().shape[0]
        number_of_destinations = remain_od.destination_node.unique().shape[0]
        print(f"The remaining number of origins: {number_of_origins}")
        print(f"The remaining number of destinations: {number_of_destinations}")
        total_remain = remain_od.Car21.sum()

        if total_remain == 0:
            print("Iteration stops: there is no remaining flows!")
            break

        # calculate edge flows: e_id, flow
        temp_edge_flow = get_flow_on_edges(
            temp_flow_matrix,
            "e_id",
            "path",
            "flow",
        )

        # add/update edge attributes
        temp_edge_flow = temp_edge_flow.merge(
            road_links[
                [
                    "e_id",
                    "combined_label",
                    "min_flow_speeds",
                    "initial_flow_speeds",
                    "acc_flow",
                    "acc_capacity",
                    "acc_speed",
                ]
            ],
            on="e_id",
            how="left",
        )

        temp_edge_flow["est_overflow"] = (
            temp_edge_flow["flow"] - temp_edge_flow["acc_capacity"]
        )  # estimated overflow: positive -> has overflow
        max_overflow = temp_edge_flow["est_overflow"].max()
        print(f"The maximum amount of edge overflow: {max_overflow}")

        if max_overflow <= 0:
            # add/update edge key variables: flow/speed/capacity
            temp_edge_flow["acc_flow"] = (
                temp_edge_flow["flow"] + temp_edge_flow["acc_flow"]
            )
            temp_edge_flow["acc_capacity"] = (
                temp_edge_flow["acc_capacity"] - temp_edge_flow["flow"]
            )
            temp_edge_flow["acc_speed"] = np.vectorize(partial_speed_flow_func)(
                temp_edge_flow["combined_label"],
                temp_edge_flow["acc_flow"],
                temp_edge_flow["initial_flow_speeds"],
                temp_edge_flow["min_flow_speeds"],
            )

            # update road_links (flows, capacities, and speeds) based on temp_edge_flow
            road_links = road_links.set_index("e_id")
            road_links.update(
                temp_edge_flow.set_index("e_id")[
                    ["acc_flow", "acc_capacity", "acc_speed"]
                ]
            )
            road_links = road_links.reset_index()

            # update travel costs: based on temp_flow_matrix
            # operating costs
            temp_cost = (
                temp_flow_matrix.operating_cost_per_flow * temp_flow_matrix.flow
            ).sum()
            operating_cost += temp_cost
            # time costs
            temp_cost = (
                temp_flow_matrix.time_cost_per_flow * temp_flow_matrix.flow
            ).sum()
            time_equiv_cost += temp_cost
            # toll costs
            temp_cost = (
                temp_flow_matrix.toll_cost_per_flow * temp_flow_matrix.flow
            ).sum()
            toll_cost += temp_cost
            # total cost
            total_cost += time_equiv_cost + operating_cost + toll_cost

            print("Iteration stops: there is no edge overflow!")
            break

        # calculate the ratio of flow adjustment (0 < r < 1)
        temp_edge_flow["r"] = np.where(
            temp_edge_flow["flow"] != 0,
            temp_edge_flow["acc_capacity"] / temp_edge_flow["flow"],
            np.nan,
        )
        r = temp_edge_flow.r.min()
        if r < 0:
            print("Error: r < 0")
            break
        if r == 0:
            print("Error: existing network has zero-capacity links!")
            break
        if r >= 1:
            print("Error: r >= 1")
            break
        print(f"r = {r}")

        # add/update edge key variables: flows/speeds/capacities
        temp_edge_flow["adjusted_flow"] = temp_edge_flow["flow"] * r
        temp_edge_flow["acc_flow"] = (
            temp_edge_flow.acc_flow + temp_edge_flow.adjusted_flow
        )
        temp_edge_flow["acc_capacity"] = (
            temp_edge_flow.acc_capacity - temp_edge_flow.adjusted_flow
        )
        temp_edge_flow["acc_speed"] = np.vectorize(partial_speed_flow_func)(
            temp_edge_flow["combined_label"],
            temp_edge_flow["acc_flow"],
            temp_edge_flow["initial_flow_speeds"],
            temp_edge_flow["min_flow_speeds"],
        )

        # update road links (flows, capacities, speeds) based on temp_edge_flow
        road_links = road_links.set_index("e_id")
        road_links.update(
            temp_edge_flow.set_index("e_id")[["acc_flow", "acc_capacity", "acc_speed"]]
        )
        road_links = road_links.reset_index()

        # update travel costs based on temp_flow_matrix
        # operating costs
        temp_cost = (
            temp_flow_matrix.operating_cost_per_flow * temp_flow_matrix.flow * r
        ).sum()
        operating_cost += temp_cost
        # time costs
        temp_cost = (
            temp_flow_matrix.time_cost_per_flow * temp_flow_matrix.flow * r
        ).sum()
        time_equiv_cost += temp_cost
        # toll costs
        temp_cost = (
            temp_flow_matrix.toll_cost_per_flow * temp_flow_matrix.flow * r
        ).sum()
        toll_cost += temp_cost
        # total cost
        total_cost += time_equiv_cost + operating_cost + toll_cost

        # if remaining supply < 0.5 -> 0
        remain_od.loc[remain_od.Car21 < 0.5, "Car21"] = 0
        total_remain = remain_od.Car21.sum()
        print(f"The total remaining supply (after flow adjustment) is: {total_remain}")

        # update network structure (nodes and edges)
        network = update_network_structure(network, temp_edge_flow, road_links)

        iter_flag += 1

    # update the road links attributes
    road_links.acc_flow = road_links.acc_flow.astype(int)
    road_links.acc_capacity = road_links.acc_capacity.astype(int)
    road_links = road_links.iloc[:, :-6]  # drop cost-related columns

    print("The flow simulation is completed!")
    print(f"total travel cost is (£): {total_cost}")
    print(f"total time-equiv cost is (£): {time_equiv_cost}")
    print(f"total operating cost is (£): {operating_cost}")
    print(f"total toll cost is (£): {toll_cost}")

    return road_links, isolation, odpfc
