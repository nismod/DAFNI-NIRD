# %%
"""
This script is for flow modelling at MSOA level in England and Wales.
"""
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

import geopandas as gpd
import igraph

from skmob.models.radiation import Radiation
from utils import load_config, get_flow_on_edges

radiationFunc = Radiation()

base_path = Path(load_config()["paths"]["base_path"])

# %%
# Data Preparation
# open roads links and nodes
# node file
road_node_file = gpd.read_file(
    base_path / "inputs" / "processed_data" / "road" / "MSOA.gpkg",
    layer="ABM_Road_Node_ENW",
    engine="pyogrio",
)
road_node_file["nd_id"] = road_node_file.id
road_node_file["lat"] = road_node_file.geometry.y
road_node_file["lon"] = road_node_file.geometry.x
# link file
road_link_file = gpd.read_file(
    base_path / "inputs" / "processed_data" / "road" / "MSOA.gpkg",
    layer="ABM_Road_Link_ENW",
    engine="pyogrio",
)
road_link_file["e_id"] = road_link_file.id
road_link_file["from_id"] = road_link_file.start_node
road_link_file["to_id"] = road_link_file.end_node

# %%
# Network Configuration (Population Census)
# LAD
MSOA_centroids = gpd.read_file(
    base_path / "inputs" / "processed_data" / "census" / "LAD.gpkg",
    layer="lad_inner_traveler_excluded_enw",
    engine="pyogrio",
)

MSOA_centroids["geometry"] = MSOA_centroids.geometry.centroid
MSOA_centroids = MSOA_centroids.rename(
    columns={
        "LAD_CODE": "MSOA_CODE",
        "commuters by roads": "road_commuters",
        "Inflows": "Inflow_2021",
        "Outflows": "Outflow_2021",
    }
)

# real MSOA
"""
MSOA_centroids = gpd.read_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\processed_data\census\MSOA.gpkg",
    layer="MSOA_centroids_attr",
    engine="pyogrio",
)
"""

MSOA_attr_dict = defaultdict(list)
for idx, centroid_i in MSOA_centroids.iterrows():
    # msoa_code = centroid_i.MSOA_CODE
    inflow = centroid_i.Inflow_2021
    outflow = centroid_i.Outflow_2021
    tt_comm = centroid_i.total_commuters
    rd_comm = centroid_i.road_commuters
    wk_pop = centroid_i.WKPOP_2021
    xi = centroid_i.geometry.x
    yi = centroid_i.geometry.y
    MSOA_attr_dict[idx].append([inflow, outflow, tt_comm, rd_comm, wk_pop, xi, yi])

# attach population of each MSOA to its nearest node of the network
nearest_node_list = {}  # node_idx: centroid_idx
for idx, centroid_i in MSOA_centroids.iterrows():
    closest_road_node = road_node_file.sindex.nearest(
        centroid_i.geometry, return_all=False
    )[1][0]
    # for the first [x]: [0] represents the index of geometry;
    # [1] represents the index of gdf
    # the second [x] represents the No. of closest item in the returned list
    nearest_node_list[closest_road_node] = idx

road_node_file["inflow"] = 0.0
road_node_file["outflow"] = 0.0
road_node_file["tt_commuter"] = 0.0
road_node_file["rd_commuter"] = 0.0
road_node_file["wk_pop"] = 0.0
for idx in range(road_node_file.shape[0]):
    if idx in nearest_node_list.keys():
        road_node_file.loc[idx, "inflow"] = MSOA_attr_dict[nearest_node_list[idx]][0][0]
        road_node_file.loc[idx, "outflow"] = MSOA_attr_dict[nearest_node_list[idx]][0][
            1
        ]
        road_node_file.loc[idx, "tt_commuter"] = MSOA_attr_dict[nearest_node_list[idx]][
            0
        ][2]
        road_node_file.loc[idx, "rd_commuter"] = MSOA_attr_dict[nearest_node_list[idx]][
            0
        ][3]
        road_node_file.loc[idx, "wk_pop"] = MSOA_attr_dict[nearest_node_list[idx]][0][4]

road_node_file.reset_index(drop=True, inplace=True)

# %%
# Network Creation
name_to_index = {name: index for index, name in enumerate(road_node_file.nd_id)}
index_to_name = {value: key for key, value in name_to_index.items()}

# create igraph network
nodeList = [
    (name_to_index[road_node_file.loc[i, "id"]], {"nd_id": road_node_file.loc[i, "id"]})
    for i in range(road_node_file.shape[0])
]

edge_and_weight = [
    (
        road_link_file.loc[i, "e_id"],
        road_link_file.loc[i, "from_id"],
        road_link_file.loc[i, "to_id"],
        road_link_file.loc[i, "geometry"].length,
    )
    for i in range(road_link_file.shape[0])
]

edgeList = [
    (name_to_index[source], name_to_index[target])
    for _, source, target, _ in edge_and_weight
]
weightList = [weight for _, _, _, weight in edge_and_weight]
edgeNameList = [e_name for e_name, _, _, _ in edge_and_weight]

test_net = igraph.Graph(directed=False)
test_net.add_vertices(nodeList)
test_net.add_edges(edgeList)
test_net.es["weight"] = weightList
test_net.es["edge_name"] = edgeNameList

# edge_length_table
df_edge_length = pd.DataFrame(
    [(idx, len) for idx, len in enumerate(test_net.es["weight"])],
    columns=["e_id", "e_len"],
)

# %%
# Network Analysis
# drop nodes with zero attraction
road_node_file = road_node_file[
    (road_node_file.inflow != 0) & (road_node_file.outflow != 0)
]
road_node_file.reset_index(drop=True, inplace=True)

# estimate road_outflows
road_node_file["outflow_commuters_vehicles"] = (
    road_node_file.rd_commuter / road_node_file.tt_commuter * road_node_file.outflow
)

# %%
# flow modelling
test_flow = radiationFunc.generate(
    road_node_file,
    name_to_index,
    index_to_name,
    test_net,
    df_edge_length,
    alpha=0.1,  # calibration: 2, 5, 8
    tile_id_column="nd_id",
    tot_outflows_column="outflow_commuters_vehicles",
    employment_column="inflow",
    relevance_column="wk_pop",
    out_format="flows_sample",  # ["flows", "flows_sample", "probabilities"]
    # rm_ver=1,  # [1: spatial calibrated model, 2: normalisation, 3: original]
)
# %%
# OD-matrix based on nodes
test_flow.to_csv(base_path / "outputs" / "od_matrix_nodes_enw.csv", index=False)

# %%
# OD-matrix based on LAD
from_node_to_zone = {}
for i in range(road_node_file.shape[0]):
    if i in nearest_node_list.keys():
        node_name = road_node_file.loc[i, "nd_id"]
        centroid_idx = nearest_node_list.get(i)
        centroid_name = MSOA_centroids.loc[centroid_idx, "MSOA_CODE"]
        from_node_to_zone[node_name] = centroid_name

test_flow["from_lad"] = test_flow["origin_id"].map(from_node_to_zone)
test_flow["to_lad"] = test_flow["destination_id"].map(from_node_to_zone)

# export file
test_flow.to_csv(base_path / "outputs" / "od_matrix_zones_enw.csv", index=False)

# %%
# Edge flows on road links
edge_flows = get_flow_on_edges(test_flow, "e_id", "edge_path", "flux")

# convert edge list from idx to edge name
edge_id_to_name = {id: name for id, name in enumerate(test_net.es["edge_name"])}
edge_flows["e_id"] = edge_flows.e_id.map(edge_id_to_name)

# export csv file
edge_flows.to_csv(base_path / "outputs" / "edge_flows.csv", index=False)

# export shapefile
road_link_file["flux"] = np.nan
flux_dict = edge_flows.set_index("e_id")["flux"].to_dict()
road_link_file.flux = road_link_file.e_id.map(flux_dict)

road_link_file.to_file(
    base_path / "inputs" / "processed_data" / "road" / "LAD.gpkg",
    layer="road_link_flux",
    driver="GPKG",
)
# Observation data: C:\Oxford\Research\DAFNI\local\inputs\processed_data\
# road\major_roads_counts_2021.gpkg
