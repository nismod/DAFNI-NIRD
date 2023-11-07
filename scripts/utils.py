"""Functions for preprocessing road data
    WILL MODIFY LATER
"""
import os
import json
from collections import defaultdict
from math import sin, cos, atan2, sqrt, pi
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import igraph
import networkx
import numpy as np
import pandas as pd
import snkit
from shapely.geometry import LineString, Polygon
from scipy.spatial import cKDTree


def components(
    edges: pd.DataFrame, nodes: pd.DataFrame, node_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    G = networkx.Graph()
    G.add_nodes_from(
        (getattr(n, node_id_col), {"geometry": n.geometry}) for n in nodes.itertuples()
    )
    G.add_edges_from(
        (e.from_node, e.to_node, {"edge_id": e.edge_id, "geometry": e.geometry})
        for e in edges.itertuples()
    )
    components = networkx.connected_components(G)
    for num, c in enumerate(components):
        print(f"Component {num} has {len(c)} nodes")
        edges.loc[(edges.from_node.isin(c) | edges.to_node.isin(c)), "component"] = num
        nodes.loc[nodes[node_id_col].isin(c), "component"] = num

    return edges, nodes


def add_lines(
    x: pd.DataFrame,
    from_nodes_df: pd.DataFrame,
    to_nodes_df: pd.DataFrame,
    from_nodes_id: str,
    to_nodes_id: str,
) -> LineString:
    from_point = from_nodes_df[from_nodes_df[from_nodes_id] == x[from_nodes_id]]
    to_point = to_nodes_df[to_nodes_df[to_nodes_id] == x[to_nodes_id]]

    return LineString(from_point.geometry.values[0], to_point.geometry.values[0])


def ckdnearest(gdA: gpd.GeoDataFrame, gdB: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Taken from https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas"""
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB_nearest, pd.Series(dist, name="dist")], axis=1
    )

    return gdf


def gdf_geom_clip(gdf_in: gpd.GeoDataFrame, clip_geom: Polygon) -> gpd.GeoDataFrame:
    """Filter a dataframe to contain only features within a clipping geometry

    Parameters
    ---------
    gdf_in
        geopandas dataframe to be clipped in
    province_geom
        shapely geometry of province for what we do the calculation

    Returns
    -------
    filtered dataframe
    """
    return gdf_in.loc[
        gdf_in["geometry"].apply(lambda x: x.within(clip_geom))
    ].reset_index(drop=True)


def get_nearest_values(
    x: gpd.GeoSeries, input_gdf: gpd.GeoDataFrame, column_name: str
) -> gpd.GeoDataFrame:
    polygon_index = input_gdf.distance(x.geometry).sort_values().index[0]
    return input_gdf.loc[polygon_index, column_name]


def extract_gdf_values_containing_nodes(
    x: gpd.GeoSeries, input_gdf: gpd.GeoDataFrame, column_name: str
) -> gpd.GeoSeries:
    a = input_gdf.loc[list(input_gdf.geometry.contains(x.geometry))]
    if len(a.index) > 0:
        return a[column_name].values[0]
    else:
        polygon_index = input_gdf.distance(x.geometry).sort_values().index[0]
        return input_gdf.loc[polygon_index, column_name]


def load_config() -> Dict[str, Dict[str, str]]:
    """Read config.json"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    with open(config_path, "r") as config_fh:
        config: Dict[str, Dict[str, str]] = json.load(config_fh)
    return config


def create_network_from_nodes_and_edges(
    nodes: Optional[gpd.GeoDataFrame],
    edges: gpd.GeoDataFrame,
    node_edge_prefix: str,
    snap_distance: Optional[float] = None,
    by: Optional[List[str]] = None,
) -> snkit.Network:
    edges.columns = map(str.lower, edges.columns)
    if "id" in edges.columns.values.tolist():  # type: ignore
        edges.rename(columns={"id": "e_id"}, inplace=True)

    # Deal with empty edges (drop)
    empty_idx = edges.geometry.apply(lambda e: e is None or e.is_empty)
    if empty_idx.sum():
        empty_edges = edges[empty_idx]
        print(f"Found {len(empty_edges)} empty edges.")
        print(empty_edges)
        edges = edges[~empty_idx].copy()

    network = snkit.Network(nodes, edges)
    print("* Done with network creation")

    network = snkit.network.split_multilinestrings(network)
    print("* Done with splitting multilines")

    if nodes is not None:
        if snap_distance is not None:
            network = snkit.network.link_nodes_to_edges_within(
                network, snap_distance, tolerance=1e-09
            )
            print("* Done with joining nodes to edges")
        else:
            network = snkit.network.snap_nodes(network)
            print("* Done with snapping nodes to edges")
        # network.nodes = snkit.network.drop_duplicate_geometries(network.nodes)
        # print ('* Done with dropping same geometries')

        # network = snkit.network.split_edges_at_nodes(network,tolerance=9e-10)
        # print ('* Done with splitting edges at nodes')

    network = snkit.network.add_endpoints(network)
    print("* Done with adding endpoints")

    network.nodes = snkit.network.drop_duplicate_geometries(network.nodes)
    print("* Done with dropping same geometries")

    network = snkit.network.split_edges_at_nodes(network, tolerance=1e-11)
    print("* Done with splitting edges at nodes")

    network = snkit.network.add_ids(
        network, edge_prefix=f"{node_edge_prefix}e", node_prefix=f"{node_edge_prefix}n"
    )
    network = snkit.network.add_topology(network, id_col="id")
    print("* Done with network topology")

    if by is not None:
        network = snkit.network.merge_edges(network, by=by)
        print("* Done with merging network")

    network.edges.rename(
        columns={"from_id": "from_node", "to_id": "to_node", "id": "edge_id"},
        inplace=True,
    )
    network.nodes.rename(columns={"id": "node_id"}, inplace=True)

    return network


NodeEdgeID = Union[str, float, int]


def network_od_path_estimations(
    graph: igraph.Graph,
    source: NodeEdgeID,
    target: NodeEdgeID,
    cost_criteria: str,
) -> Tuple[List[List[NodeEdgeID]], List[float]]:
    """Estimate the paths, distances, times, and costs for given OD pair

    Parameters
    ---------
    graph
        igraph network structure
    source
        name of Origin node ID
    target
        name of Destination node ID
    cost_criteria
        name of generalised cost criteria to be used: min_gcost or max_gcost

    Returns
    -------
    edge_path_list
        nested lists of edge ID's in routes
    path_gcost_list
        estimated generalised costs of routes

    """
    paths = graph.get_shortest_paths(
        source, target, weights=cost_criteria, output="epath"
    )

    edge_path_list = []
    path_gcost_list = []
    # for p in range(len(paths)):
    for path in paths:
        edge_path = []
        path_gcost = 0.0
        if path:
            for n in path:
                edge_id: NodeEdgeID = graph.es[n]["edge_id"]
                edge_path.append(edge_id)
                path_gcost += graph.es[n][cost_criteria]

        edge_path_list.append(edge_path)
        path_gcost_list.append(path_gcost)

    return edge_path_list, path_gcost_list


# At the equator / on another great circle???
nauticalMilePerLat = 60.00721
nauticalMilePerLongitude = 60.10793
rad = pi / 180.0
milesPerNauticalMile = 1.15078
kmsPerNauticalMile = 1.85200
degreeInMiles = milesPerNauticalMile * 60
degreeInKms = kmsPerNauticalMile * 60
# earth's mean radius = 6,371km
earthradius = 6371.0

Number = Union[float, int]


def getDistance(loc1: Tuple[Number, Number], loc2: Tuple[Number, Number]) -> float:
    """Haversine formula - give coordinates as (lat_decimal,lon_decimal) tuples"""
    lat1, lon1 = loc1
    lat2, lon2 = loc2
    # convert to radians
    lon1 = lon1 * pi / 180.0
    lon2 = lon2 * pi / 180.0
    lat1 = lat1 * pi / 180.0
    lat2 = lat2 * pi / 180.0
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2)) ** 2 + cos(lat1) * cos(lat2) * (sin(dlon / 2.0)) ** 2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
    km = earthradius * c
    return km


def get_flow_on_edges(
    save_paths_df: pd.DataFrame,
    edge_id_column: str,
    edge_path_column: str,
    flow_column: str,
) -> pd.DataFrame:
    """Get flows from paths onto edges
    Parameters
    ---------
    save_paths_df
        Pandas DataFrame of OD flow paths and their flow values
    edge_id_column
        String name of ID column of edge dataset
    edge_path_column
        String name of column which contains edge paths
    flow_column
        String name of column which contains flow values
    Result
    -------
    DataFrame with edge_ids and their total flows
    """
    """Example:
        save_path_df:
            origin_id | destination_id | edge_path          |  flux (or traffic)
            node_1      node_2          ['edge_1','edge_2']     10
            node_1      node_3          ['edge_1','edge_3']     20
        edge_id_column = "edge_id"
        edge_path_column = "edge_path"
        flow_column = "traffic"
        Result
            edge_id | flux (or traffic)
            edge_1      30
            edge_2      10
            edge_3      20
    """
    edge_flows: Dict[str, float] = defaultdict(float)
    for row in save_paths_df.itertuples():
        for item in getattr(row, edge_path_column):
            edge_flows[item] += getattr(row, flow_column)

    return pd.DataFrame(
        [(k, v) for k, v in edge_flows.items()], columns=[edge_id_column, flow_column]
    )
