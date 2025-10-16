"""Road Network Flow Model - Functions"""

# Standard library
import logging
import pickle
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Tuple

# Third-party
import geopandas as gpd  # type: ignore
import igraph  # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local
import nird.constants as cons

# from nird.utils import get_flow_on_edges

warnings.simplefilter("ignore")
tqdm.pandas()


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
    flow_breakpoint_dict: Dict[str, float],
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
    flow_breakpoint_dict: Dict
        the breakpoint flow beyond which vehicle speed start to decrease
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
    road_links["breakpoint_flows"] = road_links.combined_label.map(flow_breakpoint_dict)
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
    flow_breakpoint_dict: Dict[str, float],
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
        flow_breakpoint_dict,
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


def update_edge_speed(
    combined_label: str,
    total_flow: float,
    initial_flow_speed: float,
    min_flow_speed: float,
    breakpoint_flow: float,
    # flow_breakpoint_dict: Dict[str, float],
) -> float:
    """Create the speed-flow curves for different types of roads.

    Parameters
    ----------
    combined_label: str
        The type or road links.
    total_flow: float
        The number of vehicles on road links (passneger-cars).
    initial_flow_speed: float
        The initial traffic speeds of road links (mph).
    min_flow_speed: float
        The minimum traffic speeds of road links (mph).
    breakpoint_flow: float
        The breakpoint flow after which speed start to decrease. (passenger-cars/hour/lane)

    Returns
    -------
    speed: float
        The "real-time" traffic speeds on road links. (mph)
    """

    vp = total_flow / 24  # trips/hour
    if combined_label == "M" and vp > breakpoint_flow:
        speed = initial_flow_speed - 0.033 * (vp - breakpoint_flow)
    elif combined_label == "A_single" and vp > breakpoint_flow:
        speed = initial_flow_speed - 0.05 * (vp - breakpoint_flow)
    elif combined_label == "A_dual" and vp > breakpoint_flow:
        speed = initial_flow_speed - 0.033 * (vp - breakpoint_flow)
    elif combined_label == "B" and vp > breakpoint_flow:
        speed = initial_flow_speed - 0.05 * (vp - breakpoint_flow)
    else:
        speed = initial_flow_speed
    speed = max(speed, min_flow_speed)

    return speed


def create_igraph_network(
    road_links: gpd.GeoDataFrame,
) -> igraph.Graph:
    """Create an undirected igraph network."""
    # cols = road_links.columns.tolist()
    road_links["edge_length_mile"] = (
        road_links.geometry.length * cons.CONV_METER_TO_MILE
    )
    road_links["time_hr"] = 1.0 * road_links.edge_length_mile / road_links.acc_speed
    road_links["voc_per_km"] = np.vectorize(voc_func, otypes=None)(road_links.acc_speed)
    road_links[["weight", "time_cost", "operating_cost"]] = road_links.apply(
        lambda row: pd.Series(
            cost_func(
                row["time_hr"],
                row["edge_length_mile"],
                row["voc_per_km"],
                row["average_toll_cost"],
            )
        ),
        axis=1,
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
    index_map = {eid: idx for idx, eid in enumerate(network.es["e_id"])}
    road_links["e_idx"] = road_links["e_id"].map(index_map)
    if len(road_links[road_links.e_idx.isnull()]) > 0:
        logging.info("Error: cannot find e_id in the network!")
        sys.exit()

    return network, road_links


def update_network_structure(
    num_of_edges: int,
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
    # update the remaining capacity
    road_links_valid = road_links.dropna(subset=["e_idx"]).set_index("e_idx")
    print(f"#road_links: {len(road_links)}, #valid_links: {len(road_links_valid)}")

    temp_edge_flow = temp_edge_flow.set_index("e_idx")
    temp_edge_flow["acc_capacity"].update(road_links_valid["acc_capacity"])
    temp_edge_flow.reset_index(inplace=True)

    # drop fully utilised edges from the network
    zero_capacity_edges = set(
        temp_edge_flow.loc[temp_edge_flow["acc_capacity"] < 1, "e_idx"].tolist()
    )
    network.delete_edges(list(zero_capacity_edges))
    num_of_edges_update = len(list(network.es))
    if num_of_edges_update == num_of_edges:
        logging.info("The network structure does not change!")
        return network, road_links
    logging.info(f"The remaining number of edges in the network: {num_of_edges_update}")

    # update edges' weights
    # remaining_edges = network.es["e_id"]
    # graph_df = road_links[road_links.e_id.isin(remaining_edges)][
    #     [
    #         "from_id",
    #         "to_id",
    #         "e_id",
    #         "weight",
    #         "time_cost",
    #         "operating_cost",
    #         "average_toll_cost",
    #     ]
    # ].reset_index(drop=True)

    # network = igraph.Graph.TupleList(
    #     graph_df.itertuples(index=False),
    #     edge_attrs=list(graph_df.columns)[2:],
    #     directed=False,
    # )
    # convert edge_id to edge_idx as per network edges
    index_map = {eid: idx for idx, eid in enumerate(network.es["e_id"])}
    road_links["e_idx"] = road_links["e_id"].map(index_map)  # return nan if empty

    return network, road_links


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
        from_node = row["origin_node"]
        to_node = row["destination_node"]
        Count: float = row["Car21"]
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
    # remain_od: pd.DataFrame,
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
    # isolated od
    isolated_flow_matrix = temp_flow_matrix.loc[mask].reset_index(drop=True)
    isolated_flow_matrix.drop(columns="path", inplace=True)
    temp_isolation = isolated_flow_matrix.flow.sum()  # 666
    # allocated od (before adjustment)
    temp_flow_matrix = temp_flow_matrix[~mask].reset_index(drop=True)  # 1032

    return (
        temp_flow_matrix,
        isolated_flow_matrix,
        temp_isolation,
    )


def find_least_cost_path(
    params: Tuple,
) -> Tuple[int, List[str], List[int], List[float]]:
    """Find the least-cost path for each OD trip.

    Parameters:
    -----------
    params: Tuple
        The first element: the origin node (index).
        The second element: a list of destination nodes (indexes).
        The third element: a list of outbound trips from origin to its connected
            destinations.

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

    return (
        origin_node,
        destination_nodes,
        paths,
        flows,
    )


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


def itter_path(
    network,
    temp_flow_matrix: pd.DataFrame,
    chunk_size: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Iterate through all the paths to calculate edge flows and travel costs."""
    if chunk_size is None:
        chunk_size = len(temp_flow_matrix)  # default = process all at once
    edges = network.es
    edges_df = pd.DataFrame(
        {
            "path": range(len(edges)),
            "edge_id": edges["e_id"],
            "time_cost": edges["time_cost"],
            "operating_cost": edges["operating_cost"],
            "average_toll_cost": edges["average_toll_cost"],
        }
    ).set_index("path")

    edge_flows = []
    od_results = []
    for start in tqdm(
        range(0, len(temp_flow_matrix), chunk_size),
        desc="Processing chunks:",
        unit="chunk",
    ):
        chunk = temp_flow_matrix.iloc[start : start + chunk_size].explode("path")
        chunk = chunk.join(edges_df, on="path").rename(
            columns={
                "time_cost": "time_cost_per_flow",
                "operating_cost": "operating_cost_per_flow",
                "average_toll_cost": "toll_cost_per_flow",
            }
        )
        # aggregate OD results
        od_results.append(
            chunk.groupby(["origin", "destination"], as_index=False).agg(
                {
                    "path": list,
                    "edge_id": list,
                    "flow": "first",
                    "operating_cost_per_flow": "sum",
                    "time_cost_per_flow": "sum",
                    "toll_cost_per_flow": "sum",
                }
            )
        )
        # edge flows
        edge_flows.append(
            chunk.groupby("path")["flow"]
            .sum()
            .reset_index()
            .rename(columns={"path": "e_idx"})
        )

    temp_flow_matrix = pd.concat(od_results, ignore_index=True)
    temp_edge_flow = (
        pd.concat(edge_flows, ignore_index=True)
        .groupby("e_idx")["flow"]
        .sum()
        .reset_index()
    )

    return (temp_edge_flow, temp_flow_matrix)


def retrieve_path_attributes(path: List[int]) -> Dict:
    """Retrieve attributes for a given path."""
    if not path:
        return None
    edges = shared_network.es
    edges_df = pd.DataFrame(
        {
            "path": range(len(edges)),
            "edge_id": edges["e_id"],
            "time_cost": edges["time_cost"],
            "operating_cost": edges["operating_cost"],
            "average_toll_cost": edges["average_toll_cost"],
        }
    ).set_index("path")
    # (edge_id, time_cost, operating_cost, average_toll_cost)

    selected = edges_df.loc[
        path, ["edge_id", "time_cost", "operating_cost", "average_toll_cost"]
    ]

    # directly compute aggregated values, no Series
    edge_id_list = selected["edge_id"].tolist()
    time_cost_sum = selected["time_cost"].sum()
    operating_cost_sum = selected["operating_cost"].sum()
    toll_cost_sum = selected["average_toll_cost"].sum()

    return edge_id_list, time_cost_sum, operating_cost_sum, toll_cost_sum


def calculate_path(tfm_chunk: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate path attributes for a chunk of the flow matrix."""

    tfm_chunk[
        [
            "edge_id",
            "time_cost_per_flow",
            "operating_cost_per_flow",
            "toll_cost_per_flow",
        ]
    ] = tfm_chunk["path"].apply(lambda p: pd.Series(retrieve_path_attributes(p)))

    return tfm_chunk


def chunk_df(df, n_chunks: int):
    """Split dataframe into n roughly equal chunks."""
    chunk_size = int(np.ceil(len(df) / n_chunks))
    return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


def network_flow_model(
    road_links: gpd.GeoDataFrame,
    network: igraph.Graph,
    remain_od: pd.DataFrame,
    flow_breakpoint_dict: Dict[str, float],
    num_of_chunk: int,
    num_of_cpu: int,
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
    road_links_columns = road_links.columns.tolist()

    total_remain = remain_od["Car21"].sum()
    logging.info(f"The initial supply is {total_remain}")
    number_of_edges = len(list(network.es))
    logging.info(f"The initial number of edges in the network: {number_of_edges}")
    number_of_origins = remain_od["origin_node"].unique().shape[0]
    logging.info(f"The initial number of origins: {number_of_origins}")
    number_of_destinations = remain_od["destination_node"].unique().shape[0]
    logging.info(f"The initial number of destinations: {number_of_destinations}")

    # starts
    total_cost = cost_time = cost_fuel = cost_toll = 0
    odpfc, isolation = [], []
    initial_sumod = remain_od["Car21"].sum()
    assigned_sumod = 0
    iter_flag = 1
    while total_remain > 0:
        logging.info(f"No.{iter_flag} iteration starts:")
        # check isolations: [origin_node, destination_node, flow]
        mask = remain_od["origin_node"].isin(network.vs["name"]) & (
            remain_od["destination_node"].isin(network.vs["name"])
        )
        isolated_flow_matrix = remain_od.loc[
            ~mask, ["origin_node", "destination_node", "Car21"]
        ]  # 38
        isolation.extend(isolated_flow_matrix.values.tolist())
        remain_od = remain_od[mask].reset_index(drop=True)  # 1698
        temp_isolation = isolated_flow_matrix.Car21.sum()
        logging.info(f"Initial isolated flows: {temp_isolation}")
        # initial_sumod -= temp_isolation
        # dump the network and edge weight for shared use in multiprocessing
        shared_network_pkl = pickle.dumps(network)

        # find the least-cost path for each OD trip
        args = []
        for origin_node, subset in tqdm(
            remain_od.groupby("origin_node"), desc="Creating argument list: "
        ):
            args.append(
                (
                    origin_node,
                    subset.destination_node.tolist(),
                    subset.Car21.tolist(),
                )
            )

        # batch-processing
        st = time.time()
        list_of_spath = []
        if num_of_cpu > 1:
            with Pool(
                processes=num_of_cpu,
                initializer=worker_init_path,
                initargs=(shared_network_pkl,),
            ) as pool:
                for i, shortest_path in enumerate(
                    pool.imap_unordered(find_least_cost_path, args)
                ):
                    list_of_spath.append(shortest_path)
                    if i % 10_000 == 0:
                        logging.info(
                            f"Completed {i} of {len(args)}, {100 * i / len(args):.2f}%"
                        )
        else:
            global shared_network
            shared_network = network
            list_of_spath = [find_least_cost_path(arg) for arg in args]
            # -> [origin(name), destinations(name), path(idx), flow(int)]
        logging.info(f"The least-cost path flow allocation time: {time.time() - st}.")
        temp_flow_matrix = (
            pd.DataFrame(
                list_of_spath,
                columns=[
                    "origin",
                    "destination",
                    "path",
                    "flow",
                ],
            )
            .explode(
                [
                    "destination",
                    "path",
                    "flow",
                ]
            )
            .reset_index(drop=True)
        )
        # calculate the non-allocated flows and remaining flows
        (
            temp_flow_matrix,  # to-be-allocated: origin-destination-path-flow
            isolated_flow_matrix,  # isolated: origin-destination-flow
            temp_isolation,
        ) = update_od_matrix(temp_flow_matrix)
        # update isolation: [origin, destination, flow]
        logging.info(f"Non_allocated_flow: {temp_isolation}")
        # initial_sumod -= temp_isolation
        isolation.extend(isolated_flow_matrix.to_numpy().tolist())

        # %%
        # calculate edge flows -> [e_idx, flow]
        # and attach cost matrix (fuel, time, toll) to temp_flow_matrix
        logging.info("Calculating edge flows...")
        # number_of_chunks = 100 # user-defined parameter: memory vs speed
        (
            temp_edge_flow,
            temp_flow_matrix,
        ) = itter_path(
            network,
            temp_flow_matrix,
            chunk_size=int(len(temp_flow_matrix)) // num_of_chunk,
        )

        # # check point2: revise to parallel processing
        # chunks = chunk_df(temp_flow_matrix, n_chunks=num_of_cpu)
        # logging.info(
        #     f"Created {len(chunks)} chunks from {len(temp_flow_matrix)} rows "
        #     f"for parallel processing with {num_of_cpu} CPUs."
        # )
        # st = time.time()
        # logging.info("Calculating o-d-path costs...")
        # temp_flow_matrix_list = []
        # with Pool(
        #     processes=num_of_cpu,
        #     initializer=worker_init_path,
        #     initargs=(shared_network_pkl,),
        # ) as pool:
        #     for i, flow_matrices in enumerate(
        #         pool.imap_unordered(calculate_path, chunks)
        #     ):
        #         temp_flow_matrix_list.append(flow_matrices)
        #         if i % 10_000 == 0:
        #             logging.info(
        #                 f"Completed {i} of {len(chunks)}, {100 * i / len(chunks):.2f}%"
        #             )
        # logging.info(f"Path attributes retrieval time: {time.time() - st}.")
        # temp_flow_matrix = pd.concat(temp_flow_matrix_list, ignore_index=True)

        # logging.info("Calculating edge flows...")
        # temp_edge_flow = get_flow_on_edges(temp_flow_matrix, "e_idx", "path", "flow")

        # should first check overflows
        # compare the remaining capacity and to-be-allocated flows
        temp_edge_flow = temp_edge_flow.merge(
            road_links[["e_idx", "acc_capacity"]], on="e_idx", how="left"
        )
        r = (
            temp_edge_flow["acc_capacity"] / temp_edge_flow["flow"].replace(0, np.nan)
        ).min(skipna=True)
        logging.info(f"The minimum r value is: {r}.")

        # %%
        # use this r to adjust temp_flow_matrix and the remain_od matrix
        if iter_flag <= 5:
            temp_flow_matrix["flow"] *= min(r, 1.0)
        temp_flow_matrix["flow"] = temp_flow_matrix.flow.apply(np.ceil)

        assigned_sumod += temp_flow_matrix["flow"].sum()
        percentage_sumod = assigned_sumod / initial_sumod

        temp_flow_matrix_indexed = temp_flow_matrix.set_index(["origin", "destination"])
        remain_od[["origin", "destination"]] = remain_od[
            ["origin_node", "destination_node"]
        ]
        remain_od_indexed = remain_od.set_index(["origin", "destination"])
        merged = temp_flow_matrix_indexed.join(remain_od_indexed, how="inner")
        merged["Car21"] = (merged["Car21"] - merged["flow"]).clip(
            lower=0
        )  # non-negative
        remain_od = merged.reset_index()[["origin_node", "destination_node", "Car21"]]
        # remain_od.Car21 = remain_od.Car21.apply(np.floor)
        remain_od = remain_od[remain_od.Car21 > 0].reset_index(drop=True)

        total_remain = remain_od["Car21"].sum()
        logging.info(f"The total remain flow (after adjustment) is: {total_remain}.")

        # %%
        # update road link attributes (acc_flow, acc_capacity, acc_speed)
        if iter_flag <= 5:
            temp_edge_flow["flow"] = temp_edge_flow["flow"] * min(r, 1.0)
        temp_edge_flow["flow"] = temp_edge_flow["flow"].apply(np.ceil)

        road_links = road_links.merge(
            temp_edge_flow[["e_idx", "flow"]], on="e_idx", how="left"
        )
        road_links["flow"] = road_links["flow"].fillna(0.0)
        road_links["acc_flow"] += road_links["flow"]
        road_links["acc_capacity"] = (
            road_links["acc_capacity"] - road_links["flow"]
        ).clip(lower=0)
        logging.info("Updating edge speeds: ")
        road_links["acc_speed"] = road_links.progress_apply(
            lambda x: update_edge_speed(
                x["combined_label"],
                x["acc_flow"],
                x["initial_flow_speeds"],
                x["min_flow_speeds"],
                x["breakpoint_flows"],
            ),
            axis=1,
        )
        road_links.drop(columns=["flow"], inplace=True)
        # %%
        # estimate total travel costs:
        # [origin_node_id, destination_node_id,
        # path(edge_idx), path(edge_id), flow, fuel, time, toll]
        odpfc.extend(temp_flow_matrix.to_numpy().tolist())
        cost_fuel += (
            temp_flow_matrix["flow"] * temp_flow_matrix["operating_cost_per_flow"]
        ).sum()
        cost_time += (
            temp_flow_matrix["flow"] * temp_flow_matrix["time_cost_per_flow"]
        ).sum()
        cost_toll += (
            temp_flow_matrix["flow"] * temp_flow_matrix["toll_cost_per_flow"]
        ).sum()
        total_cost = cost_fuel + cost_time + cost_toll

        # %%
        # check point for next iteration
        if r >= 1:
            logging.info(
                f"Stop: {percentage_sumod*100}% of flows have been allocated"
                "and there is no edge overflow!"
            )
            break

        if percentage_sumod >= 0.99:
            # origin_node, destination_node, Car21
            temp_isolation = remain_od.Car21.sum()
            isolation.extend(remain_od.to_numpy().tolist())
            logging.info(
                f"Stop: {percentage_sumod*100}% of flows have been allocated with "
                f"{temp_isolation} extra isolated flows."
            )
            break

        if iter_flag > 5:
            # temp_isolation = remain_od.Car21.sum()
            # isolation.extend(remain_od.to_numpy().tolist())
            # logging.info(
            #     "Stop: Maximum iterations reached (5) with "
            #     f"{temp_isolation} extra isolated flows. "
            # )
            logging.info("Stop: Maximum iterations reached (5)!")
            break

        # %%
        # update network structure (nodes and edges) for next iteration
        network, road_links = update_network_structure(
            number_of_edges, network, temp_edge_flow, road_links
        )
        iter_flag += 1

    cList = [cost_time, cost_fuel, cost_toll, total_cost]
    road_links = road_links[road_links_columns]

    logging.info("The flow simulation is completed!")
    logging.info(f"total travel cost is (£): {total_cost}")
    logging.info(f"total time-equiv cost is (£): {cost_time}")
    logging.info(f"total operating cost is (£): {cost_fuel}")
    logging.info(f"total toll cost is (£): {cost_toll}")

    return road_links, isolation, odpfc, cList
