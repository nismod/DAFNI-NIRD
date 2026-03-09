"""Road Network Flow Model - Functions"""

# Standard library
import os
import logging
import pickle
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional

# Third-party
import geopandas as gpd  # type: ignore
import igraph  # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# Local
import nird.constants as cons
import duckdb

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


# %%
def compute_costs_for_links(
    road_links: pd.DataFrame,
    vehicle_type: str = "car",
    cols_out=("time_cost", "operating_cost"),
    chunksize: int | None = None,
    inplace: bool = True,
    eps: float = 1e-6,
):
    """Compute costs for each road link based on the vehicle type.

    Parameters
    ----------
    road_links: pd.DataFrame
        DataFrame containing road link attributes, including 'time_hr' and 'length_mile'.
    vehicle_type: str
        Type of vehicle for cost calculation (e.g., 'car', 'lgv', 'ogv', 'psv', 'rail').
    cols_out: tuple of str
        Names of the output columns for time cost and operating cost.
    chunksize: int or None
        Number of rows to process in each chunk; if None, process all at once.
    inplace: bool
    eps: float
            Small constant to prevent division by zero in speed calculations.

    Returns
    -------
    pd.DataFrame or None
    """

    def _process_block(df_block: pd.DataFrame):
        # ensure floats for numeric ops
        time_hr = df_block["time_hr"].to_numpy(dtype=float)
        distance_km = (
            df_block["length_mile"].to_numpy(dtype=float) * cons.CONV_MILE_TO_KM
        )
        # toll = df_block["average_toll_cost"].to_numpy(dtype=float)

        # safe speed
        v_kmph = distance_km / np.maximum(time_hr, eps)

        # operate_cost per km vectorised (use values from cons)
        a, b, c, d = tuple(cons.FUEL_LITRE_PER_KM[vehicle_type].values())
        L = a / np.maximum(v_kmph, eps) + b + c * v_kmph + d * v_kmph**2
        FC = L * 1.4  # GBP per km

        a1, b1 = tuple(cons.NON_FUEL_PENCE_PER_KM[vehicle_type].values())
        NFC = a1 + b1 / np.maximum(v_kmph, eps)  # pence per km
        NFC = NFC / 100.0  # GBP per km

        operate_cost_per_km = FC + NFC
        operate_cost = operate_cost_per_km * distance_km  # GBP (vector)

        # value of time: allow dict or function
        if hasattr(cons, "VOT_POUND_PER_HOUR"):
            vot = cons.VOT_POUND_PER_HOUR.get(vehicle_type, None)
        else:
            vot = None

        if vot is None:
            try:
                vot = cons.VOT_POUND_PER_HOUR(vehicle_type)
            except Exception as e:
                raise RuntimeError(
                    "Could not obtain VOT from cons (VOT_POUND_PER_HOUR)"
                ) from e

        # compute according to vehicle_type
        if vehicle_type == "car":
            ave_occ = 1.06
            c_time = time_hr * ave_occ * vot
            # total_cost = c_time + operate_cost + toll
            out = np.vstack([c_time, operate_cost]).T

        elif vehicle_type in ("lgv", "ogv"):
            c_time = time_hr * vot
            # total_cost = c_time + operate_cost + toll
            out = np.vstack([c_time, operate_cost]).T

        elif vehicle_type == "psv":
            # waiting time and fare should be counted only once per od flow
            c_time = time_hr * vot
            # c_fare = np.minimum(2 + 0.15 * distance_km, 4.5)
            # total_cost = c_time + c_fare
            out = np.vstack([c_time, np.zeros_like(c_time)]).T

        elif vehicle_type == "rail":
            # waiting time and fare should be counted only once per od flow
            c_time = time_hr * vot
            # c_fare = np.minimum(3 + 0.2 * distance_km, 250)
            # total_cost = c_time + c_fare
            out = np.vstack([c_time, np.zeros_like(c_time)]).T

        else:
            raise ValueError(f"Unknown vehicle_type: {vehicle_type}")

        return pd.DataFrame(out, index=df_block.index, columns=cols_out)

    # choose chunking strategy
    n = len(road_links)
    if chunksize is None or chunksize >= n:
        results = _process_block(road_links)
        if inplace:
            road_links.loc[:, cols_out] = results
            return None
        else:
            return results

    # chunked processing
    if inplace:
        # create columns to avoid reallocation during assignment
        for c in cols_out:
            if c not in road_links.columns:
                road_links[c] = np.nan

        for start in range(0, n, chunksize):
            end = min(start + chunksize, n)
            block_idx = slice(start, end)
            block = road_links.iloc[block_idx]
            res_block = _process_block(block)
            road_links.loc[block.index, cols_out] = res_block.values
        return None
    else:
        pieces = []
        for start in range(0, n, chunksize):
            end = min(start + chunksize, n)
            block = road_links.iloc[start:end]
            pieces.append(_process_block(block))
        return pd.concat(pieces, axis=0).sort_index()


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
        Links of the road network.
    flow_breakpoint_dict: Dict
        The breakpoint flow beyond which vehicle speed start to decrease.
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
    # Baseline capacity used for utilisation ratio checks.
    if "current_capacity" not in road_links.columns:
        road_links["current_capacity"] = road_links["acc_capacity"]
    road_links["acc_speed"] = road_links["initial_flow_speeds"]

    # remove invalid road links
    road_links = road_links[road_links.acc_capacity > 0.5].reset_index(drop=True)

    return road_links


def update_edge_speed(
    road_links: pd.DataFrame, inplace: bool = True
) -> pd.DataFrame | None:
    """
    Update the edge speeds based on accumulated flows.

    Parameters
    ----------
    road_links: pd.DataFrame
        DataFrame containing road link attributes.
    inplace: bool
        If True, update the DataFrame in place; otherwise, return a new DataFrame.

    Returns
    -------
    pd.DataFrame or None
        Updated road links DataFrame if inplace=False, otherwise None.
    """
    acc_flow = road_links["acc_flow"].to_numpy(dtype=float)  # vehicles per day
    vp = acc_flow / 24.0  # vehicles per hour
    initial_speed = road_links["initial_flow_speeds"].to_numpy(dtype=float)
    min_speed = road_links["min_flow_speeds"].to_numpy(dtype=float)
    breakpoint_flow = road_links["breakpoint_flows"].to_numpy(dtype=float)

    # label reduced speed factors
    factor_map = {
        "M": 0.033,
        "A_dual": 0.033,
        "A_single": 0.05,
        "B": 0.05,
        "B_dual": 0.05,
        "B_single": 0.05,
    }
    # map to factors, default 0.0 for labels not in map
    labels = road_links["combined_label"].astype(
        object
    )  # ensure dtype suitable for map
    factor_series = labels.map(factor_map).fillna(0.0)
    factor = factor_series.to_numpy(dtype=float)

    # compute reduction only where vp > breakpoint_flow
    excess = vp - breakpoint_flow
    excess = np.where(excess > 0.0, excess, 0.0)
    reduction = factor * excess
    speed = initial_speed - reduction

    # enforce minimum
    speed = np.maximum(speed, min_speed)

    # write result
    if inplace:
        road_links["acc_speed"] = speed
        return None
    else:
        return road_links.assign(acc_speed=speed)


def create_igraph_network(
    road_links: gpd.GeoDataFrame,
    vehicle_type: str = "car",
) -> igraph.Graph:
    """Create an undirected igraph network."""
    # cols = road_links.columns.tolist()
    road_links["length_mile"] = road_links.geometry.length * cons.CONV_METER_TO_MILE
    road_links["time_hr"] = 1.0 * road_links.length_mile / road_links.acc_speed
    compute_costs_for_links(
        road_links=road_links,
        vehicle_type=vehicle_type,
        chunksize=np.maximum(road_links.shape[0] // 10, 100_000),
        inplace=True,
    )  # time_cost, operating_cost
    road_links["weight"] = (
        road_links.time_cost + road_links.operating_cost + road_links.average_toll_cost
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
            "length_mile",
        ]
    ]

    network = igraph.Graph.TupleList(
        graph_df.itertuples(index=False),
        edge_attrs=list(graph_df.columns)[2:],
        directed=False,
    )
    eids = list(network.es["e_id"])
    index_map = dict(zip(eids, range(len(eids))))
    road_links["e_idx"] = road_links["e_id"].map(index_map)
    if len(road_links[road_links.e_idx.isnull()]) > 0:
        logging.info("Error: cannot find e_id in the network!")
        sys.exit()

    return network, road_links


def update_network_structure(
    num_of_edges: int,  # initial network
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
    # update remaining edge capacities
    road_links_valid = road_links.dropna(subset=["e_idx"])
    logging.info(
        f"#road_links: {len(road_links)}, #valid_links: {len(road_links_valid)}"
    )
    temp_edge_flow = temp_edge_flow.merge(
        road_links_valid[["e_id", "e_idx", "acc_capacity", "current_capacity"]],
        on="e_id",
        how="left",
    )
    temp_edge_flow["e_idx"] = temp_edge_flow["e_idx"].astype(int)
    # create mask
    ratio = temp_edge_flow["acc_capacity"] / temp_edge_flow["current_capacity"].replace(
        0, np.nan
    )
    # considered fully utilised if remaining capacity < 1 vehicle
    # or < 1% of original capacity
    mask = (temp_edge_flow["acc_capacity"] < 1) | (ratio < 0.01)

    # drop fully utilised edges from the network
    zero_capacity_edges = set(
        temp_edge_flow.loc[
            mask,
            "e_idx",
        ].tolist()
    )
    network.delete_edges(list(zero_capacity_edges))
    num_of_edges_update = len(list(network.es))
    if num_of_edges_update == num_of_edges:
        logging.info("The network structure does not change!")
        return network, road_links
    logging.info(f"The remaining number of edges in the network: {num_of_edges_update}")

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
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Split OD allocations into routable rows and isolated leftovers.

    Parameters
    ----------
    temp_flow_matrix : pd.DataFrame
        DataFrame with columns ``origin``, ``destination``, ``path`` (list of edge ids),
        and ``flow`` representing per-OD assignments prior to capacity checks.

    Returns
    -------
    pd.DataFrame
        Filtered copy containing only rows with a non-empty path.
    pd.DataFrame
        Rows whose ``path`` lists are empty (no feasible route found).
    float
        Total isolated flow (sum of the ``flow`` values with empty paths).
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
    """Solve shortest paths for all destinations of a single origin.

    Parameters
    ----------
    params : Tuple
        Tuple of ``(origin_node, destination_nodes, flows)`` where the first entry is
        the origin vertex id (string), the second is a list of destination ids, and the
        third is the corresponding list of OD flows.

    Returns
    -------
    Tuple[int, List[str], List[List[int]], List[float]]
        The origin id, destination id list, list of edge-id paths (one per destination),
        and the list of flows matching the inputs.
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
    """Load the shared igraph network inside each worker process.

    Parameters
    ----------
    shared_network_pkl : bytes
        Pickled bytes of the igraph ``Graph`` to be shared across worker processes.

    Returns
    -------
    None
        The function sets the module-level ``shared_network`` variable in-place.
    """
    global shared_network
    shared_network = pickle.loads(shared_network_pkl)
    return None


def itter_path(
    network,
    road_links,
    temp_flow_matrix: Optional[pd.DataFrame] = None,
    num_of_chunk: int = None,
    num_of_cpu: Optional[int] = None,
    apply_od_adjustment: bool = True,
    db_path: str = "results.duckdb",
    conn=None,
    temp_flow_table: Optional[str] = None,
) -> None:
    """Iteratively update the OD matrix and edge flows based on the current network structure."""

    if temp_flow_table is None and temp_flow_matrix is None:
        raise ValueError("Either temp_flow_matrix or temp_flow_table must be provided.")

    if conn is None:
        conn = duckdb.connect(db_path)

    # Set DuckDB to use multiple threads for query execution
    # Note, SQL operators already parallelised internally
    if num_of_cpu is not None and num_of_cpu > 0:
        conn.execute(f"PRAGMA threads={int(num_of_cpu)}")

    if temp_flow_table is not None:
        total_rows = (
            conn.execute(f"SELECT COUNT(*) FROM {temp_flow_table}").fetchone()[0] or 0
        )
    else:
        total_rows = len(temp_flow_matrix)

    if total_rows == 0:
        logging.info("No rows available for itter_path; skipping.")
        return

    edges = network.es
    edges_df = pd.DataFrame(
        {
            "path": range(len(edges)),
            "e_id": edges["e_id"],
            "time": edges["time_cost"],
            "fuel": edges["operating_cost"],
            "toll": edges["average_toll_cost"],
            "length_mile": edges["length_mile"],
        }
    )
    road_caps_df = road_links[["e_id", "acc_capacity"]]
    conn.execute("DROP TABLE IF EXISTS od_results_iter")  # reset table
    conn.execute("DROP TABLE IF EXISTS edge_flows")  # reset table
    conn.register("edges_tmp", edges_df)  # edge attributes
    conn.register("road_caps_tmp", road_caps_df)  # road capacities
    conn.execute("CREATE OR REPLACE TEMP TABLE edges_attr AS SELECT * FROM edges_tmp")
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE road_caps AS SELECT * FROM road_caps_tmp"
    )
    conn.unregister("edges_tmp")
    conn.unregister("road_caps_tmp")

    source_table = temp_flow_table
    if source_table is None:
        conn.execute("DROP TABLE IF EXISTS temp_flow_matrix_input_mem")
        conn.register("temp_flow_matrix_input_mem_tmp", temp_flow_matrix)
        conn.execute(
            """
            CREATE TABLE temp_flow_matrix_input_mem AS
            SELECT *
            FROM temp_flow_matrix_input_mem_tmp;
            """
        )
        conn.unregister("temp_flow_matrix_input_mem_tmp")
        source_table = "temp_flow_matrix_input_mem"

    # Start the path explosion and aggregation in DuckDB with chunked process
    chunk_count = int(num_of_chunk) if num_of_chunk and num_of_chunk > 0 else 1
    chunk_count = max(1, min(chunk_count, total_rows))
    chunk_size = (total_rows + chunk_count - 1) // chunk_count
    created_results = False

    for chunk_idx in range(chunk_count):
        offset = chunk_idx * chunk_size
        if offset >= total_rows:
            break
        limit = min(chunk_size, total_rows - offset)
        conn.execute("DROP TABLE IF EXISTS exploded_paths_chunk")
        conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE exploded_paths_chunk AS
            SELECT
                t.origin,
                t.destination,
                t.flow,
                list_extract(t.path, u.ord) AS path_idx,
                u.ord AS path_ord
            FROM (
                SELECT *
                FROM {source_table}
                LIMIT {limit} OFFSET {offset}
            ) t
            CROSS JOIN LATERAL (
                SELECT UNNEST(range(1, array_length(t.path) + 1)) AS ord
            ) AS u;
            """
        )

        """
        Example:
        origin | destination | flow | path_idx |
        -------|-------------|------|----------|
        A      | B           | 100  | [1, 5, 9]|

        -> exploded_paths:
        origin | destination | flow | path_idx | path_ord |
        -------|-------------|------|----------|---------|
        A      | B           | 100  | 1        | 1       |
        A      | B           | 100  | 5        | 2       |
        A      | B           | 100  | 9        | 3       |
        """

        od_sql = """
            SELECT
                x.origin,
                x.destination,
                LIST(e.e_id ORDER BY x.path_ord) AS e_id,
                MAX(x.flow) AS flow,
                SUM(e.fuel) AS fuel,
                SUM(e.time) AS time,
                SUM(e.toll) AS toll,
                SUM(e.length_mile) AS length_mile
            FROM exploded_paths_chunk x
            JOIN edges_attr e
                ON x.path_idx = e.path
            GROUP BY x.origin, x.destination
        """

        """
        Example:
        origin | destination | flow | path_idx | path_ord | fuel | time | toll | length_mile
        -------|-------------|------|----------|---------|------|------|------|-------------
        A      | B           | 100  | 1        | 1       | 10   | 0.5  | 2    | 5
        A      | B           | 100  | 5        | 2       | 20   | 1.0  | 0    | 3
        A      | B           | 100  | 9        | 3       | 15   | 0.8  | 1    | 4

        -> od_results:
        origin | destination | flow | e_id       | fuel | time | toll | length_mile
        -------|-------------|------|------------|------|------|------|-------------
        A      | B           | 100  | [1, 5, 9]  | 45   | 2.3  | 3    | 12
        """

        edge_sql = """
            SELECT
                e.e_id,
                x.origin,
                x.destination,
                MAX(c.acc_capacity) AS acc_capacity,
                SUM(x.flow) AS flow
            FROM exploded_paths_chunk x
            JOIN edges_attr e
                ON x.path_idx = e.path
            LEFT JOIN road_caps c
                ON e.e_id = c.e_id
            GROUP BY e.e_id, x.origin, x.destination
        """

        """
        Example:
        origin | destination | flow | path_idx | path_ord
        -------|-------------|------|----------|---------
        A      | B           | 100  | 1        | 1
        A      | B           | 100  | 5        | 2
        A      | B           | 100  | 9        | 3

        -> join with edge attributes:
        origin | destination | flow | path_idx | path_ord | e_id
        -------|-------------|------|----------|---------|------
        A      | B           | 100  | 1        | 1       | 1
        A      | B           | 100  | 5        | 2       | 5
        A      | B           | 100  | 9        | 3       | 9

        -> join with road_caps:
        origin | destination | flow | path_idx | path_ord | e_id | acc_capacity
        -------|-------------|------|----------|---------|------|-------------
        A      | B           | 100  | 1        | 1       | 1    | 500
        A      | B           | 100  | 5        | 2       | 5    | 300
        A      | B           | 100  | 9        | 3       | 9    | 200

        -> edge_flows (group by e_id, origin, destination):
        e_id | origin | destination | acc_capacity | flow
        -----|--------|-------------|--------------|------
        1    | A      | B           | 500          | 100
        5    | A      | B           | 300          | 100
        9    | A      | B           | 200          | 100
        """

        if not created_results:
            conn.execute(f"CREATE TABLE od_results_iter AS {od_sql};")
            conn.execute(f"CREATE TABLE edge_flows AS {edge_sql};")
            created_results = True
        else:
            conn.execute(f"INSERT INTO od_results_iter {od_sql};")
            conn.execute(f"INSERT INTO edge_flows {edge_sql};")

        conn.execute("DROP TABLE IF EXISTS exploded_paths_chunk")

    if not created_results:
        conn.execute(
            """
            CREATE TABLE od_results_iter (
                origin VARCHAR,
                destination VARCHAR,
                e_id VARCHAR[],
                flow DOUBLE,
                fuel DOUBLE,
                time DOUBLE,
                toll DOUBLE,
                length_mile DOUBLE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE edge_flows (
                e_id VARCHAR,
                origin VARCHAR,
                destination VARCHAR,
                acc_capacity DOUBLE,
                flow DOUBLE
            );
            """
        )
    if source_table == "temp_flow_matrix_input_mem":
        conn.execute("DROP TABLE IF EXISTS temp_flow_matrix_input_mem")

    logging.info(
        f"Completed SQL path explosion and aggregation in DuckDB across {chunk_count} chunk(s)."
    )

    if apply_od_adjustment:
        # od results
        # origin, destination, path, flow, cost
        # 1) create base
        conn.execute(
            """
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT
            e_id,
            origin,
            destination,
            MAX(acc_capacity) AS acc_capacity,
            SUM(flow) AS flow
        FROM edge_flows
        GROUP BY e_id, origin, destination;
        """
        )

        # 2) create total (per e_id)
        conn.execute(
            """
        CREATE OR REPLACE TEMP TABLE total AS
        SELECT e_id, SUM(flow) AS total_flow
        FROM base
        GROUP BY e_id;
        """
        )

        # 3) create od_adjustment
        conn.execute(
            """
        CREATE OR REPLACE TEMP TABLE od_adjustment AS
        SELECT
            b.origin,
            b.destination,
            MIN(LEAST(b.acc_capacity / NULLIF(t.total_flow, 0), 1.0)) AS adjust_r
        FROM base b
        JOIN total t USING (e_id)
        GROUP BY b.origin, b.destination;
        """
        )

        """
        Example:
        e_id | origin | destination | acc_capacity | flow | adsjust_r
        -----|--------|-------------|--------------|------|---------
        1    | A      | B           | 500          | 100  | min(500/100, 1) = 1.0
        5    | A      | B           | 300          | 100  | min(300/100, 1) = 1.0
        9    | A      | B           | 200          | 100  | min(200/100, 1) = 1.0

        od_adjustment:
        origin | destination | adjust_r
        -------|-------------|---------
        A      | B           | 1.0

        """
        # 4) final result with od adjustment
        conn.execute(
            """
        CREATE OR REPLACE TEMP TABLE temp_flow_matrix AS
        SELECT
            o.origin,
            o.destination,
            o.e_id,
            o.flow * COALESCE(a.adjust_r, 1.0) AS flow,
            o.fuel,
            o.time,
            o.toll,
            o.length_mile
        FROM od_results_iter o
        LEFT JOIN od_adjustment a USING (origin, destination);
        """
        )
    else:
        # Final-iteration capacity mode: keep all found-path flow (no OD scaling).
        conn.execute(
            """
        CREATE OR REPLACE TEMP TABLE temp_flow_matrix AS
        SELECT
            o.origin,
            o.destination,
            o.e_id,
            o.flow AS flow,
            o.fuel,
            o.time,
            o.toll,
            o.length_mile
        FROM od_results_iter o;
        """
        )

    logging.info("Complete creating temp_flow_matrix table in Duckdb!")

    if temp_flow_table is not None:
        conn.execute(f"DROP TABLE IF EXISTS {temp_flow_table}")

    return


def network_flow_model(
    road_links: gpd.GeoDataFrame,
    network: igraph.Graph,
    remain_od: pd.DataFrame,
    flow_breakpoint_dict: Dict[str, float],
    num_of_chunk: int,
    num_of_cpu: int,
    db_path: str = "results.duckdb",
    iso_out_path: str = None,
    odpfc_out_path: str = None,
    vehicle_type: str = "car",
    capacity_mode: bool = False,
) -> Tuple[gpd.GeoDataFrame, List[float]]:
    """Iteratively assign OD demand, update road attributes, and export results.

    Parameters
    ----------
    road_links : gpd.GeoDataFrame
        Road network edges with geometry and accumulated metrics.
    network : igraph.Graph
        Directed network used for least-cost path searches.
    remain_od : pd.DataFrame
        Input OD matrix with columns ``origin_node``, ``destination_node``, ``Car21``.
    flow_breakpoint_dict : Dict[str, float]
        Dictionary mapping combined labels to breakpoint flows for speed updates.
    num_of_chunk : int
        Number of chunks to split per-iteration path results when exploding paths.
    num_of_cpu : int
        Number of worker processes used for path finding (>=1).
    db_path : str, default ``"results.duckdb"``
        Location of the DuckDB database for temporaries and final tables.
    iso_out_path : str
        File path where the final isolated OD Parquet file will be written.
    odpfc_out_path : str
        File path where the per-path flow/cost Parquet file will be written.
    vehicle_type : str, default ``"car"``
        Vehicle type for cost calculations; one of ``["car", "lgv", "ogv", "psv", "rail"]``.
    capacity_mode : bool, default ``False``
        If ``True``, residual OD demand at stop conditions is assigned as overflow on
        already-found paths (allowing over-capacity edges) where possible, instead of
        being fully moved to isolation.

    Returns
    -------
    gpd.GeoDataFrame
        Road links with updated accumulated flow, capacity, and speed attributes.
    List[float]
        Aggregate costs in the order ``[cost_time, cost_fuel, cost_toll, total_cost]``.
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
    total_cost = cost_time = cost_fuel = cost_toll = cost_fare = 0
    initial_sumod = remain_od["Car21"].sum()
    assigned_sumod = 0
    iter_flag = 1
    max_iterations = 5  # user-defined

    # create db (remove the pre-exist one)
    if os.path.exists(db_path):
        os.remove(db_path)
    # create isolated_od table
    conn = duckdb.connect(db_path)
    conn.execute(
        """
        CREATE OR REPLACE TABLE isolated_od (
            origin_node VARCHAR,
            destination_node VARCHAR,
            flow DOUBLE
        );
    """
    )
    # create odpfc table
    conn.execute(
        """
        CREATE OR REPLACE TABLE odpfc (
            origin VARCHAR,
            destination VARCHAR,
            path VARCHAR,
            flow DOUBLE,
            fuel DOUBLE,
            time DOUBLE,
            toll DOUBLE,
            fare DOUBLE,
        )
        """
    )
    # create remain_od table
    conn.execute("DROP TABLE IF EXISTS remain_od")
    conn.register(
        "remain_od_tmp",
        remain_od[["origin_node", "destination_node", "Car21"]],
    )
    conn.execute(
        """
        CREATE TABLE remain_od AS
        SELECT
            origin_node,
            destination_node,
            Car21
        FROM remain_od_tmp;
        """
    )
    conn.unregister("remain_od_tmp")
    total_remain = (
        conn.execute("SELECT COALESCE(SUM(Car21), 0.0) FROM remain_od").fetchone()[0]
        or 0.0
    )
    del remain_od
    gc.collect()

    while total_remain > 0:
        logging.info(f"No.{iter_flag} iteration starts:")
        # remove OD pairs whose nodes are not present in the current network
        conn.register("current_valid_nodes", pd.DataFrame({"node": network.vs["name"]}))
        conn.execute("DROP TABLE IF EXISTS isolated_tmp")
        conn.execute(
            """
        CREATE TEMP TABLE isolated_tmp AS
        SELECT
            origin_node,
            destination_node,
            Car21 AS flow
        FROM remain_od
        WHERE origin_node NOT IN (SELECT node FROM current_valid_nodes)
           OR destination_node NOT IN (SELECT node FROM current_valid_nodes);
        """
        )
        temp_isolation = (
            conn.execute(
                "SELECT COALESCE(SUM(flow), 0.0) FROM isolated_tmp"
            ).fetchone()[0]
            or 0.0
        )
        if temp_isolation > 0:
            conn.execute("INSERT INTO isolated_od SELECT * FROM isolated_tmp")
            conn.execute(
                """
            DELETE FROM remain_od
            WHERE origin_node NOT IN (SELECT node FROM current_valid_nodes)
               OR destination_node NOT IN (SELECT node FROM current_valid_nodes);
            """
            )
        conn.unregister("current_valid_nodes")
        conn.execute("DROP TABLE IF EXISTS isolated_tmp")
        logging.info(f"Initial isolated flows: {temp_isolation}")

        # dump the network and edge weight for shared use in multiprocessing
        shared_network_pkl = pickle.dumps(network)

        # find the least-cost path for each OD trip
        args_df = conn.execute(
            """
        SELECT
            origin_node,
            LIST(destination_node ORDER BY destination_node) AS destinations,
            LIST(Car21 ORDER BY destination_node) AS flows
        FROM remain_od
        GROUP BY origin_node
        """
        ).fetchdf()
        total_args = len(args_df)

        def iter_args():
            for row in args_df.itertuples(index=False):
                yield (
                    row.origin_node,
                    list(row.destinations) if row.destinations is not None else [],
                    list(row.flows) if row.flows is not None else [],
                )

        conn.execute("DROP TABLE IF EXISTS temp_flow_matrix_input")
        conn.execute(
            """
            CREATE TABLE temp_flow_matrix_input (
                origin VARCHAR,
                destination VARCHAR,
                path INT[],
                flow DOUBLE
            );
            """
        )
        conn.execute("DROP TABLE IF EXISTS temp_isolated_flow_matrix")
        conn.execute(
            """
            CREATE TEMP TABLE temp_isolated_flow_matrix (
                origin VARCHAR,
                destination VARCHAR,
                flow DOUBLE
            );
            """
        )

        flow_batch: List[Tuple[str, str, List[int], float]] = []
        isolated_batch: List[Tuple[str, str, float]] = []
        batch_size = 100_000

        def flush_flow_batch() -> None:
            nonlocal flow_batch
            if not flow_batch:
                return
            batch_df = pd.DataFrame(
                flow_batch, columns=["origin", "destination", "path", "flow"]
            )
            conn.register("temp_flow_batch", batch_df)
            conn.execute(
                "INSERT INTO temp_flow_matrix_input SELECT * FROM temp_flow_batch"
            )
            conn.unregister("temp_flow_batch")
            flow_batch = []

        def flush_isolated_batch() -> None:
            nonlocal isolated_batch
            if not isolated_batch:
                return
            iso_df = pd.DataFrame(
                isolated_batch, columns=["origin", "destination", "flow"]
            )
            conn.register("temp_isolated_batch", iso_df)
            conn.execute(
                "INSERT INTO temp_isolated_flow_matrix SELECT * FROM temp_isolated_batch"
            )
            conn.unregister("temp_isolated_batch")
            isolated_batch = []

        def handle_shortest_path(
            shortest_path: Tuple[str, List[str], List[List[int]], List[float]],
        ) -> None:
            origin_node, destinations, paths, flows = shortest_path
            for dest, path, flow in zip(destinations, paths, flows):
                flow_val = float(flow) if flow is not None else 0.0
                if not path:  # if no path between OD -> isolation
                    isolated_batch.append((origin_node, dest, flow_val))
                else:
                    flow_batch.append((origin_node, dest, path, flow_val))
                if len(flow_batch) >= batch_size:
                    flush_flow_batch()
                if len(isolated_batch) >= batch_size:
                    flush_isolated_batch()

        # batch-processing
        st = time.time()
        if num_of_cpu > 1:
            with Pool(
                processes=num_of_cpu,
                initializer=worker_init_path,
                initargs=(shared_network_pkl,),
            ) as pool:
                for i, shortest_path in enumerate(
                    pool.imap_unordered(find_least_cost_path, iter_args()), start=1
                ):
                    handle_shortest_path(shortest_path)
                    if i % 10_000 == 0:
                        logging.info(
                            f"Completed {i} of {total_args}, "
                            f"{100 * i / total_args:.2f}%"
                        )
        else:
            global shared_network
            shared_network = network
            for i, shortest_path in enumerate(
                (find_least_cost_path(arg) for arg in iter_args()), start=1
            ):
                handle_shortest_path(shortest_path)
                if i % 10_000 == 0:
                    logging.info(
                        f"Completed {i} of {total_args}, "
                        f"{100 * i / total_args:.2f}%"
                    )

        del args_df
        gc.collect()

        flush_flow_batch()
        flush_isolated_batch()
        logging.info(f"The least-cost path flow allocation time: {time.time() - st}.")

        # isolated flow
        temp_isolation = (
            conn.execute(
                "SELECT COALESCE(SUM(flow), 0.0) FROM temp_isolated_flow_matrix"
            ).fetchone()[0]
            or 0.0
        )
        logging.info(f"Non_allocated_flow: {temp_isolation}")
        if temp_isolation > 0:
            conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE temp_isolated_flow_matrix_agg AS
                SELECT
                    origin,
                    destination,
                    SUM(flow) AS flow
                FROM temp_isolated_flow_matrix
                GROUP BY origin, destination;
                """
            )
            conn.execute(
                "INSERT INTO isolated_od SELECT * FROM temp_isolated_flow_matrix_agg"
            )
            conn.execute("DROP TABLE IF EXISTS temp_isolated_flow_matrix_agg")
        conn.execute("DROP TABLE IF EXISTS temp_isolated_flow_matrix")

        # remaining flow
        temp_flow_count = (
            conn.execute("SELECT COUNT(*) FROM temp_flow_matrix_input").fetchone()[0]
            or 0
        )
        if temp_flow_count == 0:
            logging.info("Stop: no remaining flows!")
            conn.execute("DROP TABLE IF EXISTS temp_flow_matrix_input")
            break

        # %%
        logging.info("Create temp_flow_matrix table in duckdb...")
        # origin (name), destination(name), path(idx), flow(int)
        itter_path(
            network,
            road_links,
            temp_flow_matrix=None,
            num_of_chunk=num_of_chunk,
            num_of_cpu=num_of_cpu,
            apply_od_adjustment=not (capacity_mode and iter_flag >= max_iterations),
            db_path=db_path,
            conn=conn,
            temp_flow_table="temp_flow_matrix_input",
        )  # -> xxx, fuel, time, toll

        assigned_iter_sum = (
            conn.execute(
                "SELECT COALESCE(SUM(flow), 0.0) FROM temp_flow_matrix"
            ).fetchone()[0]
            or 0.0
        )

        # assign flows (scalar) for this iteration
        assigned_sumod += assigned_iter_sum
        percentage_sumod = assigned_sumod / initial_sumod if initial_sumod > 0 else 1.0

        # append the adjusted per-OD rows (e_id) into odpfc (-> path)
        if vehicle_type == "psv":
            time_expr = f"time + 0.25 * {cons.VOT_POUND_PER_HOUR[vehicle_type]}"
            fare_expr = f"LEAST(2.0 + 0.15 * length_mile * {cons.CONV_MILE_TO_KM}, 4.5)"
        elif vehicle_type == "rail":
            time_expr = f"time + 0.15 * {cons.VOT_POUND_PER_HOUR[vehicle_type]}"
            fare_expr = (
                f"LEAST(3.0 + 0.2  * length_mile * {cons.CONV_MILE_TO_KM}, 250.0)"
            )
        else:
            time_expr = "time"
            fare_expr = "0.0"

        conn.execute(
            f"""
            INSERT INTO odpfc (
                origin,
                destination,
                path,
                flow,
                fuel,
                time,
                toll,
                fare
            )
            SELECT
                origin,
                destination,
                e_id AS path,
                flow,
                fuel,
                {time_expr} AS time,
                toll,
                {fare_expr} AS fare
            FROM temp_flow_matrix
            """
        )

        # compute costs
        iter_cost_fuel = (
            conn.execute(
                "SELECT COALESCE(SUM(flow * fuel), 0.0) FROM temp_flow_matrix"
            ).fetchone()[0]
            or 0.0
        )
        iter_cost_time = (
            conn.execute(
                f"SELECT COALESCE(SUM(flow * {time_expr}), 0.0) FROM temp_flow_matrix"
            ).fetchone()[0]
            or 0.0
        )
        iter_cost_toll = (
            conn.execute(
                "SELECT COALESCE(SUM(flow * toll), 0.0) FROM temp_flow_matrix"
            ).fetchone()[0]
            or 0.0
        )
        iter_cost_fare = (
            conn.execute(
                f"SELECT COALESCE(SUM(flow * {fare_expr}), 0.0) FROM temp_flow_matrix"
            ).fetchone()[0]
            or 0.0
        )

        cost_fuel += iter_cost_fuel
        cost_time += iter_cost_time
        cost_toll += iter_cost_toll
        cost_fare += iter_cost_fare
        total_cost = cost_fuel + cost_time + cost_toll + cost_fare

        # aggregate edge flows from edge_flows table
        temp_edge_flow = conn.execute(
            """
            SELECT
                e AS e_id,
                SUM(flow) AS flow
            FROM (
                SELECT
                    flow,
                    UNNEST(e_id) AS e
                FROM temp_flow_matrix
            ) AS t
            GROUP BY e
        """
        ).fetchdf()

        # merge edge flows into road_links and update accumulators
        road_links = road_links.merge(
            temp_edge_flow[["e_id", "flow"]], on="e_id", how="left"
        )
        road_links["flow"] = road_links["flow"].fillna(0.0)
        road_links["acc_flow"] += road_links["flow"]
        road_links["acc_capacity"] = road_links["acc_capacity"] - road_links["flow"]

        # Recalculate edge speeds for edges that changed (vectorized if possible)
        logging.info("Updating edge speeds: ")
        update_edge_speed(road_links, inplace=True)
        road_links.drop(columns=["flow"], inplace=True)

        # update remain od using DuckDB
        conn.execute(
            """
        CREATE OR REPLACE TEMP TABLE remain_od_updated AS
        SELECT
            r.origin_node,
            r.destination_node,
            GREATEST((r.Car21 - COALESCE(a.flow_assigned, 0.0)), 0.0) AS Car21
        FROM remain_od r
        LEFT JOIN (
            SELECT
                origin,
                destination,
                SUM(flow) AS flow_assigned
            FROM temp_flow_matrix
            GROUP BY origin, destination
        ) a
        ON r.origin_node = a.origin AND r.destination_node = a.destination;
        """
        )

        conn.execute("DELETE FROM remain_od")
        conn.execute(
            """
        INSERT INTO remain_od
        SELECT origin_node, destination_node, Car21
        FROM remain_od_updated
        WHERE Car21 > 0;
        """
        )

        total_remain = (
            conn.execute("SELECT COALESCE(SUM(Car21), 0.0) FROM remain_od").fetchone()[
                0
            ]
            or 0.0
        )
        logging.info(f"The total remain flow (after adjustment) is: {total_remain}.")
        gc.collect()

        # %%
        # check point for next iteration
        stop_reason = None
        if percentage_sumod >= 0.99:
            stop_reason = "target_allocated"
        elif iter_flag >= max_iterations:
            stop_reason = "max_iterations"

        if stop_reason is not None:
            temp_isolation = (
                conn.execute(
                    "SELECT COALESCE(SUM(Car21), 0.0) FROM remain_od"
                ).fetchone()[0]
                or 0.0
            )
            if temp_isolation > 0:
                conn.execute(
                    """
                    INSERT INTO isolated_od
                    SELECT
                        origin_node,
                        destination_node,
                        Car21 AS flow
                    FROM remain_od;
                    """
                )

            if stop_reason == "target_allocated":
                logging.info(
                    f"Stop: {percentage_sumod*100}% of flows have been allocated with "
                    f"{temp_isolation} extra isolated flows."
                )
            else:
                logging.info(
                    f"Stop: Maximum iterations reached ({max_iterations}) with "
                    f"{temp_isolation} extra isolated flows. "
                )
                if capacity_mode:
                    logging.info(
                        "Capacity mode: skipped OD adjustment on the final iteration."
                    )
            break

        # %%
        # update network structure (nodes and edges) for next iteration
        network, road_links = update_network_structure(
            number_of_edges, network, temp_edge_flow, road_links
        )

        del temp_edge_flow
        gc.collect()

        iter_flag += 1

    cList = [cost_time, cost_fuel, cost_toll, total_cost]
    road_links = road_links[road_links_columns]

    # create isolation and odpfc from db
    if iso_out_path is not None:
        conn.execute(
            f"""
            COPY (
                SELECT
                    origin_node,
                    destination_node,
                    SUM(flow) AS flow
                FROM isolated_od
                GROUP BY origin_node, destination_node
            ) TO '{iso_out_path}' (FORMAT PARQUET);
        """
        )
    if odpfc_out_path is not None:
        conn.execute(
            f"""
            COPY (
                SELECT
                    origin AS origin_node,
                    destination AS destination_node,
                    path,
                    SUM(flow) AS flow,
                    MIN(fuel) AS operating_cost_per_flow,
                    MIN(time) AS time_cost_per_flow,
                    MIN(toll) AS toll_cost_per_flow,
                    MIN(fare) AS fare_cost_per_flow
                FROM odpfc
                GROUP BY origin, destination, path
            ) TO '{odpfc_out_path}' (FORMAT PARQUET);
        """
        )
    conn.close()

    logging.info("The flow simulation is completed!")
    logging.info(f"total travel cost is (£): {total_cost}")
    logging.info(f"total time-equiv cost is (£): {cost_time}")
    logging.info(f"total operating cost is (£): {cost_fuel}")
    logging.info(f"total toll cost is (£): {cost_toll}")

    return road_links, cList
