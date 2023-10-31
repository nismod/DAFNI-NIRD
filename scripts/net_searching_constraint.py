# %%
#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import snkit
from skmob.models.radiation import Radiation

from utils import getDistance, load_config

network = snkit.Network()
radiationFun = Radiation()

base_path = Path(load_config()["paths"]["base_path"])

# %%
# Data Preparation
# node file
oprd_node = gpd.read_file(
    base_path / "inputs" / "processed_data" / "road" / "cliped_file.gpkg",
    layer="oproad_nodes_cliped",
)
oprd_node["nd_id"] = oprd_node.id
oprd_node["lat"] = oprd_node.geometry.y
oprd_node["lon"] = oprd_node.geometry.x
# edge file
oprd_edge = gpd.read_file(
    base_path / "inputs" / "processed_data" / "road" / "cliped_file.gpkg",
    layer="oproad_cliped",
)
oprd_edge["e_id"] = oprd_edge.id
oprd_edge["from_id"] = oprd_edge.start_node
oprd_edge["to_id"] = oprd_edge.end_node

# %%
# Network Configuration
# attributes file
oa_attr = gpd.read_file(
    base_path / "inputs" / "processed_data" / "census" / "london.gpkg",
    layer="OA",
)
oa_attr["centroid"] = oa_attr.centroid  # point geometry
oa_centroid = oa_attr[
    ["OA_code", "tt_pop_oa", "employment_oa", "outflow_oa", "centroid"]
].rename(columns={"OA_code": "oa_id", "centroid": "geometry"})

# nodes: x-y
nodes_dict = defaultdict(list)
for i in range(oprd_node.shape[0]):
    nodes_dict[str(oprd_node.loc[i, "nd_id"])].append(
        [oprd_node.loc[i, "lon"], oprd_node.loc[i, "lat"]]
    )
# OA: x-y, tt_pop, wk_pop
oa_dict = defaultdict(list)
for i in range(oa_attr.shape[0]):
    oa_id = oa_centroid.loc[i, "oa_id"]
    xi = oa_centroid.loc[i, "geometry"].x
    yi = oa_centroid.loc[i, "geometry"].y
    pi = oa_centroid.loc[i, "tt_pop_oa"]
    ei = oa_centroid.loc[i, "employment_oa"]
    ti = oa_centroid.loc[i, "outflow_oa"]
    oa_dict[str(oa_id)].append([pi, ei, ti, xi, yi])

# inner: use the intersection index; retain the left geometry (which is oa in this case)
oa_nd_sj = oa_attr.sjoin(oprd_node, how="inner", predicate="intersects")

# geometry: polygon
oa_nd_sj.reset_index(drop=True, inplace=True)
oa_nd_sj["dist"] = np.nan
for i in range(oa_nd_sj.shape[0]):
    oa_id = str(oa_nd_sj.loc[i, "OA_code"])
    nd_id = str(oa_nd_sj.loc[i, "nd_id"])
    if str(oa_id) in oa_dict.keys() and str(nd_id) in nodes_dict.keys():
        dist_oa_nd = getDistance(
            (oa_dict[str(oa_id)][0][-1], oa_dict[str(oa_id)][0][-2]),
            (nodes_dict[str(nd_id)][0][1], nodes_dict[str(nd_id)][0][0]),
        )
        # the distance between the centroid of OA and each node located within the OA
        oa_nd_sj.loc[i, "dist"] = dist_oa_nd

# identify the closest node in the network
oa_nd_sj_gp = oa_nd_sj.groupby(by=["OA_code"], as_index=False).agg(
    {"nd_id": list, "dist": list}
)
oa_nd_sj_gp = oa_nd_sj_gp.rename(columns={"OA_code": "oa_id"})
oa_nd_dict = {}
for i in range(oa_nd_sj_gp.shape[0]):
    idx = oa_nd_sj_gp.loc[i, "dist"].index(min(oa_nd_sj_gp.loc[i, "dist"]))
    sel_nd_id = oa_nd_sj_gp.loc[i, "nd_id"][idx]
    oa_nd_dict[str(sel_nd_id)] = oa_nd_sj_gp.loc[i, "oa_id"]

#!!! assumption: attach the centroid of each OA to the nearest node point of the network
# for analysis
oprd_node["population"] = 0
oprd_node["employment"] = 0
oprd_node["outflow"] = 0
for i in range(oprd_node.shape[0]):
    nd_id = oprd_node.loc[i, "nd_id"]
    if nd_id in oa_nd_dict.keys():
        oa_id = oa_nd_dict[nd_id]
        try:
            pi = oa_dict[oa_id][0][0]
            ei = oa_dict[oa_id][0][1]
            ti = oa_dict[oa_id][0][2]
            oprd_node.loc[i, "population"] = pi
            oprd_node.loc[i, "employment"] = ei
            oprd_node.loc[i, "outflow"] = ti
        except Exception as e:
            print(e)

# %%
# Network Creation
# delete the edges of which any end node is not in the nodes list (dangling edges)
nodeset = set(oprd_node.nd_id.unique())  # 167,903
idxList = []
for i in range(oprd_edge.shape[0]):  # 210,611
    from_id = oprd_edge.loc[i, "from_id"]
    to_id = oprd_edge.loc[i, "to_id"]
    if (from_id in nodeset) and (to_id in nodeset):
        idxList.append(i)

oprd_edge = oprd_edge[oprd_edge.index.isin(idxList)]
oprd_edge.reset_index(drop=True, inplace=True)  # 210,247

# create an igraph network
# to use get_shortest_paths(): one-to-multiple
"""
Structure of Elements:
nodes = [(1, {"latitude": x, "longitude": x}),
(2...)]
egdes = [(1,2, {"length": x}), ()...]
"""
name_to_index = {
    name: index for index, name in enumerate(oprd_node.nd_id)
}  #!!! this is important: to convert the nd_id (str) into index (int)
index_to_name = {value: key for key, value in name_to_index.items()}
nodeList = [name_to_index[oprd_node.loc[i, "id"]] for i in range(oprd_node.shape[0])]
edge_and_weight = [
    (
        oprd_edge.loc[i, "from_id"],
        oprd_edge.loc[i, "to_id"],
        oprd_edge.loc[i, "geometry"].length,
    )
    for i in range(oprd_edge.shape[0])
]
"""
edgeList = [
    (name_to_index[source], name_to_index[target])
    for source, target, weight in edge_and_weight
]
weightList = [weight for source, target, weight in edge_and_weight]
test_net = igraph.Graph(directed=False)
test_net.add_vertices(nodeList)
test_net.add_edges(edgeList)
test_net.es["weight"] = weightList
"""
# create a networkx network
# to use single_source_dijkstra(): cut-off parameter
test_net2 = nx.Graph()
test_net2.add_nodes_from(nodeList)
edgeList2 = [
    (name_to_index[source], name_to_index[target], {"weight": weight})
    for source, target, weight in edge_and_weight
]
# This is a complete network comprising all the original nodes and edges
# (open road dataset)
test_net2.add_edges_from(edgeList2)

# %%
# Network Analysis
# drop nodes with zero population or zero working population before modelling flows
oprd_node.population = oprd_node.population.astype(int)
oprd_node.employment = oprd_node.employment.astype(int)
oprd_node.outflow = oprd_node.outflow.astype(int)

oprd_node = oprd_node[
    (oprd_node.population != 0) & (oprd_node.employment != 0) & (oprd_node.outflow != 0)
]
oprd_node.reset_index(drop=True, inplace=True)  # 19,716 none zero nodes

# %%
# (1) Add a searching constraint
test_flow = radiationFun.generate(
    oprd_node,
    name_to_index,
    index_to_name,
    test_net2,
    employment_column="employment",
    cut_off=1000,  # distance/time, depending on "weight" in the network
    tile_id_column="nd_id",
    tot_outflows_column="outflow",
    relevance_column="population",
    out_format="flows_sample",
    ver="method1",
)  # method1: nc-paper; method2: michael batty's paper

# test_flow.to_csv(
# r"outputs\test_flow_1000.csv", index = False)
