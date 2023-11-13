# %%
from pathlib import Path
from collections import defaultdict

import geopandas as gpd

# import networkx as nx
import pandas as pd
import numpy as np
import igraph

from skmob.models.radiation import Radiation
from utils import load_config, getDistance, get_flow_on_edges

# network = snkit.Network()
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
    layer="oproad_edges_major_roads",  # only select the major roads
)
oprd_edge["e_id"] = oprd_edge.id
oprd_edge["from_id"] = oprd_edge.start_node
oprd_edge["to_id"] = oprd_edge.end_node

# %%
# import o-d matrix for calibration
od_file = pd.read_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "MSOA_collection"
    / "ODWP01EW_MSOA.csv"
)
od_file = od_file[od_file["Place of work indicator (4 categories) code"] == 3]
od_file.reset_index(drop=True, inplace=True)
od_dict = defaultdict(lambda: defaultdict(list))
for i in range(od_file.shape[0]):
    from_lad = od_file.loc[i, "Middle layer Super Output Areas code"]
    to_lad = od_file.loc[i, "MSOA of workplace code"]
    count = od_file.loc[i, "Count"]
    od_dict[from_lad][to_lad] = count

# %%
# Network configuration (MSOA level)
lad_attr = gpd.read_file(
    base_path / "inputs" / "processed_data" / "census" / "MSOA.gpkg",
    layer="MSOA_attr_london_cliped",  #!!! update
)  # outflow (by roads) and inflow (opportunities)

lad_attr["centroid"] = lad_attr.centroid
lad_centroid = lad_attr[
    [
        "MSOA_CODE",
        "Inflow_2021",
        "Outflow_2021",
        "total_commuters",
        "road_commuters",
        "WKPOP_2021",
        "centroid",
    ]
].rename(columns={"centroid": "geometry"})

lad_dict = defaultdict(list)
for i in range(lad_centroid.shape[0]):
    lad_id = lad_centroid.loc[i, "MSOA_CODE"]
    inflow = lad_centroid.loc[i, "Inflow_2021"]
    outflow = lad_centroid.loc[i, "Outflow_2021"]
    tt_comm = lad_centroid.loc[i, "total_commuters"]
    rd_comm = lad_centroid.loc[i, "road_commuters"]
    wk_pop = (lad_centroid.loc[i, "WKPOP_2021"],)
    xi = lad_centroid.loc[i, "geometry"].x
    yi = lad_centroid.loc[i, "geometry"].y
    lad_dict[lad_id].append([inflow, outflow, tt_comm, rd_comm, wk_pop, xi, yi])

# %%
# nodes of the road network: x-y
nodes_dict = defaultdict(list)
for i in range(oprd_node.shape[0]):
    nodes_dict[str(oprd_node.loc[i, "nd_id"])].append(
        [oprd_node.loc[i, "lon"], oprd_node.loc[i, "lat"]]
    )

# inner: use the intersection index; retain the left geometry (which is oa in this case)
lad_nd_sj = lad_attr.sjoin(oprd_node, how="inner", predicate="intersects")

# geometry: polygon
lad_nd_sj.reset_index(drop=True, inplace=True)
lad_nd_sj["dist"] = np.nan
for i in range(lad_nd_sj.shape[0]):
    lad_id = str(lad_nd_sj.loc[i, "MSOA_CODE"])
    nd_id = str(lad_nd_sj.loc[i, "nd_id"])
    if str(lad_id) in lad_dict.keys() and str(nd_id) in nodes_dict.keys():
        dist_lad_nd = getDistance(
            (lad_dict[str(lad_id)][0][-1], lad_dict[str(lad_id)][0][-2]),
            (nodes_dict[str(nd_id)][0][1], nodes_dict[str(nd_id)][0][0]),
        )
        # the distance between the centroid of OA and each node located within the OA
        lad_nd_sj.loc[i, "dist"] = dist_lad_nd

# identify the closest node in the network
lad_nd_sj_gp = lad_nd_sj.groupby(by=["MSOA_CODE"], as_index=False).agg(
    {"nd_id": list, "dist": list}
)
# lad_nd_sj_gp = lad_nd_sj_gp.rename(columns={"MSOA_CODE": "lad_id"})
lad_nd_dict = {}
for i in range(lad_nd_sj_gp.shape[0]):
    idx = lad_nd_sj_gp.loc[i, "dist"].index(min(lad_nd_sj_gp.loc[i, "dist"]))
    sel_nd_id = lad_nd_sj_gp.loc[i, "nd_id"][idx]
    lad_nd_dict[str(sel_nd_id)] = lad_nd_sj_gp.loc[i, "MSOA_CODE"]

# assumption: attach the centroid of each LAD to the nearest node point of the network
# for analysis (33 LAD cliped for London)
oprd_node["inflow"] = 0.0
oprd_node["outflow"] = 0.0
oprd_node["tt_commuter"] = 0.0
oprd_node["rd_commuter"] = 0.0
oprd_node["wk_pop"] = 0.0
for i in range(oprd_node.shape[0]):
    nd_id = oprd_node.loc[i, "nd_id"]
    if nd_id in lad_nd_dict.keys():
        lad_id = lad_nd_dict[nd_id]
        try:
            inflow = lad_dict[lad_id][0][0]
            outflow = lad_dict[lad_id][0][1]
            tt_comm = lad_dict[lad_id][0][2]
            rd_comm = lad_dict[lad_id][0][3]
            wk_pop = lad_dict[lad_id][0][4]
            oprd_node.loc[i, "inflow"] = inflow
            oprd_node.loc[i, "outflow"] = outflow
            oprd_node.loc[i, "tt_commuter"] = tt_comm
            oprd_node.loc[i, "rd_commuter"] = rd_comm
            oprd_node.loc[i, "wk_pop"] = wk_pop
        except Exception as e:
            print(e)

# %%
# Network Creation
# delete the dangling edges
nodeset = set(oprd_node.nd_id.unique())
idxList = []
for i in range(oprd_edge.shape[0]):  # 210,611
    from_id = oprd_edge.loc[i, "from_id"]
    to_id = oprd_edge.loc[i, "to_id"]
    if (from_id in nodeset) and (to_id in nodeset):
        idxList.append(i)

oprd_edge = oprd_edge[oprd_edge.index.isin(idxList)]  # 210,611
oprd_edge.reset_index(drop=True, inplace=True)  # 210,247

# delete the hanging nodes
nodeset = set(oprd_edge.from_id.tolist() + oprd_edge.to_id.tolist())
idxList = []
for i in range(oprd_node.shape[0]):
    node_id = oprd_node.loc[i, "nd_id"]
    if node_id in nodeset:
        idxList.append(i)

oprd_node = oprd_node[oprd_node.index.isin(idxList)]  # 167,903
oprd_node.reset_index(drop=True, inplace=True)  # 167,860

# relationship between node indexes and names
name_to_index = {
    name: index for index, name in enumerate(oprd_node.nd_id)
}  #!!! this is important: to convert the nd_id (str) into index (int)
index_to_name = {value: key for key, value in name_to_index.items()}

# relationship between edges and nodes
nodes_to_edge = {}
for i in range(oprd_edge.shape[0]):
    e_id = oprd_edge.loc[i, "e_id"]
    from_id = oprd_edge.loc[i, "from_id"]
    to_id = oprd_edge.loc[i, "to_id"]
    node_set = frozenset([from_id, to_id])  # !!! frozenset
    nodes_to_edge[node_set] = e_id

# create a networkx network
# to use single_source_dijkstra(): cut-off parameter
nodeList = [
    (name_to_index[oprd_node.loc[i, "id"]], {"nd_id": oprd_node.loc[i, "id"]})
    for i in range(oprd_node.shape[0])
]

edge_and_weight = [
    (
        oprd_edge.loc[i, "e_id"],
        oprd_edge.loc[i, "from_id"],
        oprd_edge.loc[i, "to_id"],
        oprd_edge.loc[i, "geometry"].length,
    )
    for i in range(oprd_edge.shape[0])
]

"""
edgeList = [
    (name_to_index[source], name_to_index[target], {"weight": weight})
    for source, target, weight in edge_and_weight
]

test_net2 = nx.Graph()
test_net2.add_nodes_from(nodeList)
test_net2.add_edges_from(edgeList)
"""

# create a igraph-based network
edgeList_ig = [
    (name_to_index[source], name_to_index[target])
    for _, source, target, _ in edge_and_weight
]

weightList_ig = [weight for _, _, _, weight in edge_and_weight]
edgeNameList_id = [e_name for e_name, _, _, _ in edge_and_weight]

test_net = igraph.Graph(directed=False)
test_net.add_vertices(nodeList)
test_net.add_edges(edgeList_ig)
test_net.es["weight"] = weightList_ig
test_net.es["edge_name"] = edgeNameList_id

# %%
# Network Analysis
# drop nodes with zero population or zero working population before modelling flows
oprd_node = oprd_node[(oprd_node.inflow != 0) & (oprd_node.outflow != 0)]
oprd_node.reset_index(drop=True, inplace=True)

# to estimate the outflow by roads
oprd_node["outflow_by_roads"] = (
    oprd_node.rd_commuter / oprd_node.tt_commuter * oprd_node.outflow
)

# %%
# Start the calibration
alpha_list = []
error_list = []

alpha = 0.09
# [0, 0.1], step = 0.01
# [0.09, 0.1], step = 0.001
while 0 <= alpha < 0.1:  # city level: 0 < alpha < 0.1
    print(alpha)
    test_flow = radiationFun.generate(
        oprd_node,
        name_to_index,
        index_to_name,
        nodes_to_edge,  # add: relationship between edges and nodes
        test_net,  # test_net: igraph; test_net2: networkx
        cut_off=None,  # distance/time, depending on "weight" in the network
        alpha=alpha,  # spatial calibration factor
        tile_id_column="nd_id",
        tot_outflows_column="outflow_by_roads",  # commuters between different zones
        employment_column="inflow",  # employment opportunities
        relevance_column="wk_pop",  # total population
        out_format="flows_sample",
    )

    test_flow["from_lad"] = test_flow["origin_id"].map(lad_nd_dict)
    test_flow["to_lad"] = test_flow["destination_id"].map(lad_nd_dict)

    # calculate the sum of the errors
    test_flow["Counts"] = np.nan
    for i in range(test_flow.shape[0]):
        from_lad = test_flow.loc[i, "from_lad"]
        to_lad = test_flow.loc[i, "to_lad"]
        if from_lad in od_dict.keys():
            if to_lad in od_dict[from_lad].keys():
                count = od_dict[from_lad][to_lad]
                test_flow.loc[i, "Counts"] = count
    test_flow["Errors"] = test_flow.flux - test_flow.Counts
    test_flow.Errors = test_flow.Errors.fillna(0.0)
    sum_error = np.abs(test_flow.Errors).sum()

    alpha_list.append(alpha)
    error_list.append(sum_error)

    alpha += 0.01

# %%
# city-level simulation
# best alpha is 0.1
test_flow = radiationFun.generate(
    oprd_node,
    name_to_index,
    index_to_name,
    nodes_to_edge,  # add: relationship between edges and nodes
    test_net,
    cut_off=None,  # distance/time, depending on "weight" in the network
    alpha=0.1,  # spatial calibration factor
    tile_id_column="nd_id",
    tot_outflows_column="outflow_by_roads",  # commuters between different zones
    employment_column="inflow",  # employment opportunities
    relevance_column="inflow",  # total population
    out_format="flows_sample",
)

# %%
# to export OD matrix
test_flow["from_lad"] = test_flow["origin_id"].map(lad_nd_dict)
test_flow["to_lad"] = test_flow["destination_id"].map(lad_nd_dict)
test_flow.to_csv(base_path / "outputs" / "od_msoa_1108.csv", index=False)

# to export Edge Flux
# convert from flux to traffic (Oab -> Tij)
edge_flows = get_flow_on_edges(test_flow, "e_id", "edge_path", "flux")
# post process of edge flows
edge_id_to_name = {id: name for id, name in enumerate(test_net.es["edge_name"])}
edge_flows["e_id"] = edge_flows.e_id.map(edge_id_to_name)
edge_flows.to_csv(base_path / "outputs" / "edge_flow_msoa_1108.csv", index=False)

oprd_edge["flux"] = np.nan
flux_dict = {}
for i in range(edge_flows.shape[0]):
    e_id = edge_flows.loc[i, "e_id"]
    flux_i = edge_flows.loc[i, "flux"]
    flux_dict[e_id] = flux_i

for i in range(oprd_edge.shape[0]):
    e_id = oprd_edge.loc[i, "e_id"]
    if e_id in flux_dict.keys():
        oprd_edge.loc[i, "flux"] = flux_dict[e_id]

#!!! to export the edge flow results
oprd_edge.to_file(
    base_path / "inputs" / "processed_data" / "road" / "cliped_file.gpkg",
    layer="oprd_cliped_flux",
    driver="GPKG",
)

# %%
# combine major roads attr to shapefile
major_road_file = gpd.read_file(
    base_path / "inputs" / "processed_data" / "road" / "cliped_file.gpkg",
    layer="major_road_cliped",
)
major_road_file.CP_Number = major_road_file.CP_Number.astype(int)
major_road_file = major_road_file[["CP_Number", "RoadNumber", "geometry"]]

mr_attr_2021 = pd.read_csv(
    r"C:\Oxford\Research\DAFNI\local\inputs\incoming_data\road\mrdb-traffic counts\dft_traffic_counts_aadf_2021.csv"
)
mr_attr_2021 = mr_attr_2021.rename(columns={"Count_point_id": "CP_Number"})
mr_attr_2021_dict = mr_attr_2021.set_index("CP_Number")["commuters_vehicles"]
major_road_file = major_road_file.merge(
    mr_attr_2021[["CP_Number", "commuters_vehicles"]], how="left", on="CP_Number"
)

#!!! observations
major_road_file.to_file(
    base_path / "inputs" / "processed_data" / "road" / "cliped_file.gpkg",
    driver="GPKG",
    layer="major_road_cliped",
)
