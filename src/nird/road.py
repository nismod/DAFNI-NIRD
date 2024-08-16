""" Road Network Flow Model - Functions
"""

from typing import Union, Tuple, List, Dict
from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
import geopandas as gpd  # type: ignore
import igraph  # type: ignore
import nird.constants as cons
from nird.utils import get_flow_on_edges

# from tqdm.auto import tqdm
from multiprocessing import Pool
import warnings
import pickle
import time

warnings.simplefilter("ignore")


def select_partial_roads(
    road_links: gpd.GeoDataFrame,
    road_nodes: gpd.GeoDataFrame,
    col_name: str,
    list_of_values: List,
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


def create_urban_mask(etisplus_urban_roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
    zones: gpd.GeoDataFrame, road_nodes: gpd.GeoDataFrame
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
    nearest_node_dict = {}
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

    return nearest_node_dict  # zone_idx: node_idx


def extract_od_pairs(
    od: pd.DataFrame,
) -> Tuple[List, Dict[str, List[str]], Dict[str, List[int]]]:
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
    dict_of_destination_nodes: dict[str, list[str]] = defaultdict(list)
    dict_of_origin_supplies: dict[str, list[float]] = defaultdict(list)
    for _, row in od.iterrows():
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


def voc_func(speed: float) -> float:
    """Calculate the Vehicle Operating Cost (VOC).

    Parameters
    ----------
    speed: float
        The average flow speed: mile/hour

    Returns
    -------
    float
        The unit vehicle operating cost: £/km
    """
    s = speed * cons.CONV_MILE_TO_KM  # km/hour
    lpkm = 0.178 - 0.00299 * s + 0.0000205 * (s**2)  # fuel consumption (liter/km)
    uvoc = 140 * lpkm * cons.PENCE_TO_POUND  # average petrol cost: 140 pence/liter
    return uvoc


def cost_func(
    time: float,
    distance: float,
    voc: float,
    toll: float,
) -> Tuple[float, float, float]:  # time: hour, distance: mile/hour, voc: pound/km
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
    ave_occ = 1.6  # average car occupancy
    vot = 17.69  # value of time (VOT): 17.69 pounds/hour
    d = distance * cons.CONV_MILE_TO_KM
    c_time = time * ave_occ * vot
    c_operate = d * voc
    cost = time * ave_occ * vot + d * voc + toll
    return cost, c_time, c_operate


def initial_speed_func(
    road_type: str,
    form_of_way: str,
    free_flow_speed_dict: Dict[str, float],
) -> Union[float, None]:
    """Append free-flow speed to each road segment.

    Parameters
    ----------
    road_type: str
        The road type in ['M','A','B']
    form_of_road: str
        The form of way in ['Single Carriageway', 'Roundabout', 'Dual Carriageway',
       'Collapsed Dual Carriageway', 'Slip Road'].
    free_flow_speed_dict: dict
        Free-flow speed of different combined road types, miles/hour

    Returns
    -------
    float or None
        The initial speed, miles/hour
    """
    if road_type == "M":
        return free_flow_speed_dict["M"]
    elif road_type == "A":
        if form_of_way == "Single Carriageway":
            return free_flow_speed_dict["A_single"]
        else:
            return free_flow_speed_dict["A_dual"]
    elif road_type == "B":
        return free_flow_speed_dict["B"]
    else:
        print("Error: initial speed!")
        return None


# update speed (mile/hour) according to edge flow (car/day)
def speed_flow_func(
    road_type: str,
    isurban: int,
    vp: float,  # edge flow
    free_flow_speed_dict: Dict[str, float],
    flow_breakpoint_dict: Dict[str, float],
    min_speed_cap: Dict[str, float],
    urban_speed_cap: Dict[str, float],
) -> Union[float, None]:
    """Modelling the reduction in average flow speed in response to increased traffic
    on various road types.

    Parameters
    ----------
    road_type: str
        The column of road type.
    isUrban: int
        1: urban, 0: suburban
    vp: float
        The average daily flow, cars/day.
    free_flow_speed_dict: dict
        The free-flow speeds of different combined road types, mile/hour.
    min_speed_cap: dict
        The restriction on the lowest flow rate, mile/hour.
    urban_speed_cap: dict
        The restriction on the maximum flow rate in urban areas, mile/hour.

    Returns
    -------
    float OR None
        The average flow rate, mile/hour.
    """
    vp = vp / 24
    if road_type == "M":
        initial_speed = free_flow_speed_dict["M"]
        if vp > flow_breakpoint_dict["M"]:
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
    elif road_type == "A_single" or road_type == "B":
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
    elif road_type == "A_dual":
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
    else:
        print("Please select the road type from [M, A, B]!")
        return None


def filter_less_than_one(arr: np.ndarray) -> np.ndarray:
    """Convert values less than one to zero"""
    return np.where(arr >= 1, arr, 0)


def create_igraph_network(
    road_links: gpd.GeoDataFrame,
    road_nodes: gpd.GeoDataFrame,
    initialSpeeds: Dict[str, float],
) -> Tuple[igraph.Graph, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Create an undirected igraph network.

    Parameters
    ----------
    road_links: gpd.GeoDataFrame
    road_nodes: gpd.GeoDataFrame
    initialSpeeds: dict
        The free-flow speed of different types of road links.

    Returns
    -------
    igraph.Graph
        The road network.
    edge_cost_dict,
        Total travel cost of each edge.
    edge_timecost_dict,
        The time-equivalent cost of each edge.
    edge_operatecost_dict,
        The fuel cost of each edge.
    """
    nodeList = [(node.id) for _, node in road_nodes.iterrows()]
    edgeIndexList = []
    edgeNameList = []
    edgeList = []
    edgeLengthList = []
    edgeTypeList = []
    edgeFormList = []
    edgeTollList = []
    for idx, link in road_links.iterrows():
        edge_name = link.e_id
        edge_from = link.from_id
        edge_to = link.to_id
        edge_length = link.geometry.length * cons.CONV_METER_TO_MILE  # miles
        edge_type = link.road_classification[0]
        edge_form = link.form_of_way
        edge_toll = link.average_toll_cost
        edgeIndexList.append(idx)
        edgeNameList.append(edge_name)
        edgeList.append((edge_from, edge_to))
        edgeLengthList.append(edge_length)
        edgeTypeList.append(edge_type)
        edgeFormList.append(edge_form)
        edgeTollList.append(edge_toll)

    edgeSpeedList = np.vectorize(initial_speed_func, otypes=None)(
        edgeTypeList,
        edgeFormList,
        initialSpeeds,
    )  # miles/hour

    # travel time
    timeList = np.array(edgeLengthList) / np.array(edgeSpeedList)  # hour

    # total travel cost (£)
    vocList = np.vectorize(voc_func, otypes=None)(edgeSpeedList)  # £/km
    costList, timeCostList, operateCostList = np.vectorize(cost_func, otypes=None)(
        timeList, edgeLengthList, vocList, edgeTollList
    )
    weightList = costList.tolist()

    # Node/Egde-seq objects: indices and attributes
    test_net = igraph.Graph(directed=False)
    test_net.add_vertices(nodeList)
    test_net.add_edges(edgeList)
    test_net.es["edge_name"] = edgeNameList
    test_net.es["weight"] = weightList

    # Cmponent costs (£)
    edge_cost_dict = dict(zip(edgeNameList, weightList))
    edge_timecost_dict = dict(zip(edgeNameList, timeCostList))
    edge_operatecost_dict = dict(zip(edgeNameList, operateCostList))
    edge_toll_dict = dict(zip(edgeNameList, edgeTollList))

    edge_compc_df = pd.DataFrame(
        {
            "edge_idx": edgeIndexList,
            "edge_voc": operateCostList,
            "edge_vot": timeCostList,
            "edge_toll": edgeTollList,
        }
    )
    return (
        test_net,
        edge_cost_dict,
        edge_timecost_dict,
        edge_operatecost_dict,
        edge_toll_dict,
        edge_compc_df,
    )


def network_init(
    road_links: gpd.GeoDataFrame,
    initial_capacity_dict: Dict[str, float],
    initial_speed_dict: Dict[str, float],
    col_road_classification=str,
) -> gpd.GeoDataFrame:
    """Initialise network key parameters: edge flows, capacities,
    and average flow speeds.

    Parameters
    ----------
    road_links: gpd.GeoDataFrame
    initial_capacity_dict: dict
        The capacity of different types of road links.
    initial_speed_dict: dict
        The free-flow speed of different types of road links.
    col_road_classification: str
        The road type column.

    Returns
    -------
    gpd.GeoDataFrame
        Road links with updated attributes,
        (1) road_type_label: M, A, and B.
        (2) combined_label: M, A_dual, A_single, and B.
        and key parameters,
        (1) acc_flow: edge flows (cars).
        (2) acc_capacities: edge capacities (cars/day).
        (3) ave_flow_rate: average flow speeds (miles/hour).
    """
    # road_types: M, A, B
    road_links["road_type_label"] = road_links[col_road_classification].str[0]
    # road_forms: M, A_dual, A_single, B
    road_links["combined_label"] = road_links["road_type_label"]
    # A_single: all the other types of A roads except (collapsed) dual carriageway
    road_links.loc[road_links.road_type_label == "A", "combined_label"] = "A_single"
    # A_dual: (collapsed) dual carriageway
    road_links.loc[
        (
            (road_links.road_type_label == "A")
            & (
                road_links.form_of_way.isin(
                    ["Dual Carriageway", "Collapsed Dual Carriageway"]
                )
            )
        ),
        "combined_label",
    ] = "A_dual"

    # accumulated edge flows (cars/day)
    road_links["acc_flow"] = 0.0
    # remaining edge capacities (cars/day)
    road_links["acc_capacity"] = road_links["combined_label"].map(initial_capacity_dict)
    # average edge flow rates (miles/hour)
    road_links["ave_flow_rate"] = road_links["combined_label"].map(initial_speed_dict)

    # remove edges with zero capacities
    road_links = road_links[road_links.acc_capacity > 0].reset_index(drop=True)

    return road_links


def update_od_matrix(
    temp_flow_matrix: pd.DataFrame,
    supply_dict: Dict[str, List[float]],
    destination_dict: Dict[str, List[str]],
    isolated_flow_dict: Dict[Tuple[str], float],
) -> Tuple[pd.DataFrame, List, Dict[str, List[float]], Dict[str, List[str]]]:
    """Update the OD matrix by removing unreachable desitinations from each origin;
    and origins with zero supplies.

    Parameters
    ----------
    temp_flow_matrix: pd.DataFrame
        A temporary flow matrix: [origins, destinations, paths, flows]
    supply_dict: dict
        The number of outbound trips from each origin.
    destination_dict: dict
        A list of destinations attached to each origin.
    isolated_flow_dict: dict
        The number of isolated trips between each OD pair.

    Returns
    -------
    temp_flow_matrix: pd.DataFrame
    new_list_of_origins: list
    new_supply_dict: dict
    new_destination: dict
    """
    # drop the OD trips with no accessible route ("path = []")
    # drop destinations with no accessible route from each origin
    temp_df = temp_flow_matrix[temp_flow_matrix["path"].apply(lambda x: len(x) == 0)]
    print(f"Non_allocated_flow: {temp_df.flow.sum()}")
    for _, row in temp_df.iterrows():
        origin_temp = row["origin"]
        destination_temp = row["destination"]
        flow_temp = row["flow"]
        isolated_flow_dict[(origin_temp, destination_temp)] += flow_temp
        idx_temp = destination_dict[origin_temp].index(destination_temp)
        destination_dict[origin_temp].remove(destination_temp)
        del supply_dict[origin_temp][idx_temp]

    # drop origins of which all trips have been sent to the network
    new_supply_dict = {}
    new_destination_dict = {}
    new_list_of_origins = []
    for origin, list_of_counts in supply_dict.items():
        tt_supply = sum(list_of_counts)
        if tt_supply > 0:
            new_list_of_origins.append(origin)
            new_counts = [od_flow for od_flow in list_of_counts if od_flow != 0]
            new_supply_dict[origin] = new_counts
            new_destination_dict[origin] = [
                dest
                for idx, dest in enumerate(destination_dict[origin])
                if list_of_counts[idx] != 0
            ]
    temp_flow_matrix = temp_flow_matrix[
        temp_flow_matrix["path"].apply(lambda x: len(x) != 0)
    ]
    return (
        temp_flow_matrix,
        new_list_of_origins,
        new_supply_dict,
        new_destination_dict,
    )


def update_network_structure(
    network: igraph.Graph,
    length_dict: Dict[str, float],
    speed_dict: Dict[str, float],
    toll_dict: Dict[str, float],
    temp_edge_flow: pd.DataFrame,
) -> Tuple[igraph.Graph, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Update the road network structure by removing the fully utilised edges,
    and updating the weights of the remaining edges.

    Parameters
    ----------
    network: igraph.Graph
        A road network.
    length_dict: dict
        The length of road links.
    speed_dict:
        The average flow speed of road links.
    temp_edge_flow: pd.DataFrame
        A table that records edge flows.

    Returns
    -------
    network: igraph.Graph
        Th updated road network.
    edge_cost_dict: dict
        The updated travel cost of edges.
    edge_timecost_dict: dict
        The updated time-equivalent cost of edges.
    edge_operatecost_dict: dict
        The updated vehicle operating cost of edges.
    """
    zero_capacity_edges = set(
        temp_edge_flow.loc[temp_edge_flow["remaining_capacity"] < 1, "e_id"].tolist()
    )  # edge names
    net_edges = network.es["edge_name"]
    idx_to_remove = [
        idx for idx, element in enumerate(net_edges) if element in zero_capacity_edges
    ]
    # drop fully utilised edges
    network.delete_edges(idx_to_remove)
    number_of_edges = len(list(network.es))
    print(f"The remaining number of edges in the network: {number_of_edges}")

    # update edge weights
    remaining_edges = network.es["edge_name"]
    lengthList = list(
        map(length_dict.get, filter(length_dict.__contains__, remaining_edges))
    )
    speedList = list(
        map(speed_dict.get, filter(speed_dict.__contains__, remaining_edges))
    )
    timeList = np.where(
        np.array(speedList) != 0, np.array(lengthList) / np.array(speedList), np.nan
    )  # hours
    tollList = list(map(toll_dict.get, filter(toll_dict.__contains__, remaining_edges)))

    if np.isnan(timeList).any():
        idx_first_nan = np.where(np.isnan(timeList))[0][0]
        length_nan = lengthList[idx_first_nan]
        speed_nan = speedList[idx_first_nan]
        print("ERROR: Network contains congested edges.")
        print(f"The first nan item - length: {length_nan}")
        print(f"The first nan item - speed: {speed_nan}")
        exit()
    else:
        vocList = np.vectorize(voc_func, otypes=None)(speedList)
        costList, timeCostList, operateCostList = np.vectorize(cost_func, otypes=None)(
            timeList, lengthList, vocList, tollList
        )  # hours
        weightList = costList.tolist()  # pounds
        network.es["weight"] = weightList
        # estimate edge traveling cost (£)
        edge_cost_dict = dict(
            zip(
                network.es["edge_name"],
                weightList,
            )
        )
        edge_timecost_dict = dict(zip(network.es["edge_name"], timeCostList))
        edge_operatecost_dict = dict(zip(network.es["edge_name"], operateCostList))
        edge_tollcost_dict = dict(zip(network.es["edge_name"], tollList))

    edge_compc_df = pd.DataFrame(
        {
            "edge_idx": [edge.index for edge in network.es],
            "edge_voc": operateCostList,
            "edge_vot": timeCostList,
            "edge_toll": tollList,
        }
    )
    return (
        network,
        edge_cost_dict,
        edge_timecost_dict,
        edge_operatecost_dict,
        edge_tollcost_dict,
        edge_compc_df,
    )


def find_least_cost_path(params: Tuple) -> Tuple[int, List, List, List]:
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
    idx_of_origin_node, list_of_idx_destination_nodes, flows = params
    paths = shared_network.get_shortest_paths(
        v=idx_of_origin_node,
        to=list_of_idx_destination_nodes,
        weights="weight",
        mode="out",
        output="epath",
    )
    return (
        idx_of_origin_node,
        list_of_idx_destination_nodes,
        paths,
        flows,
    )


def compute_edge_costs(
    path: List,
) -> Tuple[Dict[Tuple[str], float], Dict[Tuple[str], float], Dict[Tuple[str]]]:
    """Calculate the total travel cost for the path

    Parameters
    ----------
    path: List
        A list of edge indexes that define the path.

    Returns
    -------
    od_voc: Dict
        Vehicle operating costs of each trip.
    od_vot: Dict
        Value of time of each trip.
    od_toll: Dict
        Toll costs of each trip.
    """
    od_voc = edge_weight_df.loc[edge_weight_df["edge_idx"].isin(path), "edge_voc"].sum()
    od_vot = edge_weight_df.loc[edge_weight_df["edge_idx"].isin(path), "edge_vot"].sum()
    od_toll = edge_weight_df.loc[
        edge_weight_df["edge_idx"].isin(path), "edge_toll"
    ].sum()
    return (od_voc, od_vot, od_toll)


def worker_init_path(shared_network_pkl: bytes) -> None:
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


def worker_init_edge(shared_network_pkl: bytes, shared_weight_pkl: bytes) -> None:
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
    road_nodes: gpd.GeoDataFrame,
    list_of_origins: List,
    supply_dict: Dict[str, List[float]],
    destination_dict: Dict[str, List[str]],
    free_flow_speed_dict: Dict[str, float],
    flow_breakpoint_dict: Dict[str, float],
    flow_capacity_dict: Dict[str, float],
    min_speed_cap: Dict[str, float],
    urban_speed_cap: Dict[str, float],
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[Tuple[str], float],
    Dict[Tuple[str], float],
    Dict[Tuple[str], float],
    Dict[Tuple[str], float],
]:
    """Model the passenger flows on the road network.

    Parameters
    ----------
    road_links: gpd.GeoDataFrame
    road_nodes: gpd.GeoDataFrame
    list_of_origins: list
        A list of unique origin nodes of OD flows.
    supply_dict: dict
        The number of outbound trips of individual origins.
    destination_dict: dict
        A list of destinations attached to individual origins.
    free_flow_speed_dict: dict
        A dictionary recording the free-flow speed on different types of road, mile/hour.
    flow_breakpoint_dict: dict
        A dictionary recording the breakpoint flow of different types of roads,
        beyond which the flow speeds starts to decrease according to the remaining road capacities.
    flow_capacity_dict: dict
        A dictionary recording the capacity of different types of road.
    min_speed_cap: dict
        The restriction on the lowest flow rate, mile/hour.
    urban_speed_cap: dict
        The restriction on the maximum flow rate in urban areas, mile/hour.

    Returns
    -------
    acc_speed_dict: dict
        The updated average flow rate on each road link.
    acc_flow_dict: dict
        The accumulated flow amount on each road link.
    acc_capacity_dict: dict
        The remaining capacity of each road link.
    od_voc_dict: dict
        The vehicle operating costs of each trip.
    od_vot_dict: dict
        The value of time of each trip.
    od_toll_dict: dict
        The toll costs of each trip.
    isolated_flow_dict: dict
        The isolated trips between each OD pair.
    """
    # network creation (igraph)
    (
        network,
        edge_cost_dict,
        edge_timeC_dict,
        edge_operateC_dict,
        edge_toll_dict,
        edge_compc_df,
    ) = create_igraph_network(
        road_links, road_nodes, free_flow_speed_dict
    )  # this returns a network and edge weights dict(edge_name, edge_weight)

    # network initialisation
    road_links = network_init(
        road_links,
        flow_capacity_dict,
        free_flow_speed_dict,
        col_road_classification="road_classification",
    )

    # record total cost of travelling: weight * flow
    total_cost = 0
    time_equiv_cost = 0
    operating_cost = 0
    toll_cost = 0
    partial_speed_flow_func = partial(
        speed_flow_func,
        free_flow_speed_dict=free_flow_speed_dict,
        flow_breakpoint_dict=flow_breakpoint_dict,
        min_speed_cap=min_speed_cap,
        urban_speed_cap=urban_speed_cap,
    )

    total_remain = sum(sum(values) for values in supply_dict.values())
    print(f"The initial total supply is {total_remain}")
    number_of_edges = len(list(network.es))
    print(f"The initial number of edges in the network: {number_of_edges}")
    print(f"The initial number of origins: {len(list_of_origins)}")
    number_of_destinations = sum(len(value) for value in destination_dict.values())
    print(f"The initial number of destinations: {number_of_destinations}")

    # road link properties
    edge_cbtype_dict = road_links.set_index("e_id")["combined_label"].to_dict()
    edge_isUrban_dict = road_links.set_index("e_id")["urban"].to_dict()
    edge_length_dict = (
        road_links.set_index("e_id")["geometry"].length * cons.CONV_METER_TO_MILE
    ).to_dict()
    # edge_toll_dict = road_links.set_index("e_id")["average_toll_cost"].to_dict()
    acc_flow_dict = road_links.set_index("e_id")["acc_flow"].to_dict()
    acc_capacity_dict = road_links.set_index("e_id")["acc_capacity"].to_dict()
    acc_speed_dict = road_links.set_index("e_id")["ave_flow_rate"].to_dict()

    # starts
    iter_flag = 1
    isolated_flow_dict = defaultdict(float)
    # od_cost_dict = defaultdict(float)
    od_voc_dict = defaultdict(float)
    od_vot_dict = defaultdict(float)
    od_toll_dict = defaultdict(float)
    while total_remain > 0:
        print(f"No.{iter_flag} iteration starts:")
        # dump the network and edge weight for shared use in multiprocessing
        shared_network_pkl = pickle.dumps(network)
        shared_weight_pkl = pickle.dumps(edge_compc_df)

        # find the least-cost for each OD trip
        list_of_spath = []
        args = []
        for i in range(len(list_of_origins)):
            name_of_origin_node = list_of_origins[i]
            list_of_name_destination_node = destination_dict[
                name_of_origin_node
            ]  # a list of destination nodes
            list_of_flows = supply_dict[name_of_origin_node]
            args.append(
                (
                    name_of_origin_node,
                    list_of_name_destination_node,
                    list_of_flows,
                )
            )
        st = time.time()
        with Pool(
            processes=20,
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
            ],
        ).explode(["destination", "path", "flow"])

        # compute the total travel cost for each OD trip
        st = time.time()
        args = []
        args = [(row["path"],) for _, row in temp_flow_matrix.iterrows()]
        with Pool(
            processes=20,
            initializer=worker_init_edge,
            initargs=(shared_network_pkl, shared_weight_pkl),
        ) as pool:
            # temp_flow_matrix["unit_od_cost"] = pool.starmap(compute_edge_costs, args)
            temp_flow_matrix[["unit_od_voc", "unit_od_vot", "unit_od_toll"]] = (
                pool.starmap(compute_edge_costs, args)
            )
        print(f"The computational time for OD costs: {time.time() - st}.")

        # calculate the non-allocated flows and remaining flows
        (
            temp_flow_matrix,
            list_of_origins,
            supply_dict,
            destination_dict,
            # non_allocated_flow,
        ) = update_od_matrix(
            temp_flow_matrix, supply_dict, destination_dict, isolated_flow_dict
        )
        # total_non_allocated_flow += non_allocated_flow
        number_of_destinations = sum(len(value) for value in destination_dict.values())
        print(f"The remaining number of origins: {len(list_of_origins)}")
        print(f"The remaining number of destinations: {number_of_destinations}")

        # calculate edge flows
        # [edge_name, flow]
        temp_edge_flow = get_flow_on_edges(temp_flow_matrix, "e_idx", "path", "flow")

        # update edge attributes after updating the network structure
        edge_index_to_name = {
            idx: network.es[idx]["edge_name"] for idx in range(len(network.es))
        }
        temp_edge_flow["e_id"] = temp_edge_flow.e_idx.astype(int).map(
            edge_index_to_name
        )
        temp_edge_flow["combined_label"] = temp_edge_flow["e_id"].map(
            edge_cbtype_dict
        )  # combined road type
        temp_edge_flow["isUrban"] = temp_edge_flow["e_id"].map(
            edge_isUrban_dict
        )  # urban/suburban
        temp_edge_flow["temp_acc_flow"] = temp_edge_flow["e_id"].map(
            acc_flow_dict
        )  # flow
        temp_edge_flow["temp_acc_capacity"] = temp_edge_flow["e_id"].map(
            acc_capacity_dict
        )  # capacity
        temp_edge_flow["est_overflow"] = (
            temp_edge_flow["flow"] - temp_edge_flow["temp_acc_capacity"]
        )  # estimated overflow: positive -> has overflow
        max_overflow = temp_edge_flow["est_overflow"].max()
        print(f"The maximum amount of edge overflow: {max_overflow}")

        # break
        if max_overflow <= 0:
            temp_edge_flow["total_flow"] = (
                temp_edge_flow["flow"] + temp_edge_flow["temp_acc_flow"]
            )
            temp_edge_flow["speed"] = np.vectorize(
                partial_speed_flow_func, otypes=None
            )(
                temp_edge_flow["combined_label"],
                temp_edge_flow["isUrban"],
                temp_edge_flow["total_flow"],
            )
            temp_edge_flow["remaining_capacity"] = (
                temp_edge_flow["temp_acc_capacity"] - temp_edge_flow["flow"]
            )
            # update dicts
            # accumulated edge flows
            temp_dict = temp_edge_flow.set_index("e_id")["total_flow"].to_dict()
            acc_flow_dict.update(
                {key: temp_dict[key] for key in acc_flow_dict.keys() & temp_dict.keys()}
            )
            # average flow speeds
            temp_dict = temp_edge_flow.set_index("e_id")["speed"].to_dict()
            acc_speed_dict.update(
                {
                    key: temp_dict[key]
                    for key in acc_speed_dict.keys() & temp_dict.keys()
                }
            )
            # remaining edge capacities
            temp_dict = temp_edge_flow.set_index("e_id")["remaining_capacity"].to_dict()
            acc_capacity_dict.update(
                {
                    key: temp_dict[key]
                    for key in acc_capacity_dict.keys() & temp_dict.keys()
                }
            )
            # update edge travel costs
            temp_cost = (
                temp_edge_flow["e_id"].map(edge_cost_dict) * temp_edge_flow["flow"]
            )
            total_cost += temp_cost.sum()
            temp_cost = (
                temp_edge_flow["e_id"].map(edge_timeC_dict) * temp_edge_flow["flow"]
            )
            time_equiv_cost += temp_cost.sum()
            temp_cost = (
                temp_edge_flow["e_id"].map(edge_operateC_dict) * temp_edge_flow["flow"]
            )
            operating_cost += temp_cost.sum()
            temp_cost = (
                temp_edge_flow["e_id"].map(edge_toll_dict) * temp_edge_flow["flow"]
            )
            toll_cost += temp_cost.sum()

            # update od trave costs: unit_od_cost * adjusted_flows
            for row in temp_flow_matrix.itertuples(index=False):
                key = (row.origin, row.destination)
                od_voc_dict[key] += row.unit_od_voc * row.flow
                od_vot_dict[key] += row.unit_od_vot * row.flow
                od_toll_dict[key] += row.unit_od_toll * row.flow

            print("Iteration stops: there is no edge overflow!")
            break

        # calculate the ratio of flow adjustment (0 < r < 1)
        temp_edge_flow["r"] = np.where(
            temp_edge_flow["flow"] != 0,
            temp_edge_flow["temp_acc_capacity"] / temp_edge_flow["flow"],
            np.nan,
        )
        r = temp_edge_flow.r.min()
        if r < 0:
            print("Error: negative r!")
            break
        if r == 0:
            print("Error: (r==0) existing network has zero-capacity links!")
            break
        if r >= 1:
            print("Error: (r>=1) there is no edge overflow!")
            break
        print(f"r = {r}")  # set as NaN when flow is zero

        # update edge flows/speeds/capacities
        temp_edge_flow["adjusted_flow"] = temp_edge_flow["flow"] * r
        temp_edge_flow["total_flow"] = (
            temp_edge_flow.temp_acc_flow + temp_edge_flow.adjusted_flow
        )

        temp_edge_flow["speed"] = np.vectorize(partial_speed_flow_func, otypes=None)(
            temp_edge_flow.combined_label,
            temp_edge_flow.isUrban,
            temp_edge_flow.total_flow,
        )
        temp_edge_flow["remaining_capacity"] = (
            temp_edge_flow.temp_acc_capacity - temp_edge_flow.adjusted_flow
        )
        temp_edge_flow.loc[
            temp_edge_flow.remaining_capacity < 0, "remaining_capacity"
        ] = 0.0  # capacity is non-negative

        # update dicts
        # accumulated flows
        temp_dict = temp_edge_flow.set_index("e_id")["total_flow"].to_dict()
        acc_flow_dict.update(
            {key: temp_dict[key] for key in acc_flow_dict.keys() & temp_dict.keys()}
        )
        # average flow rate
        temp_dict = temp_edge_flow.set_index("e_id")["speed"].to_dict()
        acc_speed_dict.update(
            {key: temp_dict[key] for key in acc_speed_dict.keys() & temp_dict.keys()}
        )
        # accumulated remaining capacities
        temp_dict = temp_edge_flow.set_index("e_id")["remaining_capacity"].to_dict()
        acc_capacity_dict.update(
            {key: temp_dict[key] for key in acc_capacity_dict.keys() & temp_dict.keys()}
        )
        # if remaining supply < 1 -> 0
        supply_dict = {
            k: filter_less_than_one(np.array(v) * (1 - r)).tolist()
            for k, v in supply_dict.items()
        }
        total_remain = sum(sum(values) for values in supply_dict.values())
        print(f"The total remaining supply is: {total_remain}")

        # update edge travel costs
        temp_cost = temp_edge_flow["e_id"].map(edge_cost_dict) * temp_edge_flow["flow"]
        total_cost += temp_cost.sum()
        temp_cost = temp_edge_flow["e_id"].map(edge_timeC_dict) * temp_edge_flow["flow"]
        time_equiv_cost += temp_cost.sum()
        temp_cost = (
            temp_edge_flow["e_id"].map(edge_operateC_dict) * temp_edge_flow["flow"]
        )
        operating_cost += temp_cost.sum()
        temp_cost = temp_edge_flow["e_id"].map(edge_toll_dict) * temp_edge_flow["flow"]
        toll_cost += temp_cost.sum()

        # update OD travel costs
        for row in temp_flow_matrix.itertuples(index=False):
            key = (row.origin, row.destination)
            od_voc_dict[key] += row.unit_od_voc * row.flow
            od_vot_dict[key] += row.unit_od_vot * row.flow
            od_toll_dict[key] += row.unit_od_toll * row.flow

        # update network structure (nodes and edges)
        (
            network,
            edge_cost_dict,
            edge_timeC_dict,
            edge_operateC_dict,
            edge_toll_dict,
            edge_compc_df,
        ) = update_network_structure(
            network,
            edge_length_dict,
            acc_speed_dict,
            temp_edge_flow,
        )

        iter_flag += 1

    print("The flow simulation is completed!")
    print(f"total travel cost is (£): {total_cost}")
    print(f"total time-equiv cost is (£): {time_equiv_cost}")
    print(f"total operating cost is (£): {operating_cost}")
    print(f"total toll cost is (£): {toll_cost}")

    return (
        acc_speed_dict,
        acc_flow_dict,
        acc_capacity_dict,
        od_voc_dict,
        od_vot_dict,
        od_toll_dict,
        isolated_flow_dict,
    )
