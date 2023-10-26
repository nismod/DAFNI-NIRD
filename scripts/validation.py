# %%
from pathlib import Path
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import snkit
from skmob.models.radiation import Radiation

from utils import create_network_from_nodes_and_edges, load_config

network = snkit.Network()
radiationFun = Radiation()

base_path = Path(load_config()["paths"]["base_path"])

# %%
# data preparation
edgeFile = gpd.read_file(
    base_path
    / "inputs"
    / "incoming_data"
    / "lca_radiation_model_data"
    / "lca_radiation_model_data"
    / "lca_roads.gpkg",
    layer="edges",
)

edgeFile["time"] = 60 * edgeFile.length_m / (1000 * edgeFile.max_speed)  # minutes
edgeFile.rename(
    columns={"edge_id": "edge_id0", "from_node": "from_node0", "to_node": "to_node0"},
    inplace=True,
)
net = create_network_from_nodes_and_edges(
    nodes=None, edges=edgeFile, node_edge_prefix="rd"
)
edges = net.edges  # "edge_id", "from_node", "to_node"
edges.crs = edgeFile.crs
nodes = net.nodes  # "node_id"
nodes.crs = edgeFile.crs

# %%
"""
edges.to_file(
    "inputs"
    / "incoming_data"
    / "lca_radiation_model_data"
    / "lca_radiation_model_data"
    / "processed"
    / "road_network.gpkg",
    layer="edges",
    driver="GPKG",
)

nodes.to_file(
    "inputs"
    / "incoming_data"
    / "lca_radiation_model_data"
    / "lca_radiation_model_data"
    / "processed"
    / "road_network.gpkg",
    layer="nodes",
    driver="GPKG",
)
"""

# %%
# create a network using networkX
name_to_index = {
    name: index for index, name in enumerate(nodes.node_id)
}  #!!! this is important: to convert the nd_id (str) into index (int)
index_to_name = {value: key for key, value in name_to_index.items()}
nodeList = [name_to_index[nodes.loc[i, "node_id"]] for i in range(nodes.shape[0])]
# using time as the cutoff constraint
edge_and_weight = [
    (edges.loc[i, "from_node"], edges.loc[i, "to_node"], edges.loc[i, "time"])
    for i in range(edges.shape[0])
]
edgeList = [
    (name_to_index[source], name_to_index[target], {"weight": weight})
    for source, target, weight in edge_and_weight
]
net2 = nx.Graph()
net2.add_nodes_from(nodeList)
net2.add_edges_from(edgeList)

# %%
# network configuration
voroniFile = gpd.read_file(
    base_path
    / "inputs"
    / "incoming_data"
    / "lca_radiation_model_data"
    / "lca_radiation_model_data"
    / "LCA_roads_voronoi.gpkg",
    layer="areas",
)
voroni_node_sj = voroniFile.sjoin(
    nodes, how="inner", predicate="intersects"
)  # geometry: polygon
voroni_node_sj_gp = voroni_node_sj.groupby(by=["node_id_left"], as_index=False).agg(
    {"node_id_right": list}
)
voroni_pop_dict = {}
for i in range(voroniFile.shape[0]):
    voroni_id = voroniFile.loc[i, "node_id"]
    popi = voroniFile.loc[i, "pop_2020"]
    voroni_pop_dict[voroni_id] = popi

node_to_voroni_dict = {}  # one-to-one
for i in range(voroni_node_sj_gp.shape[0]):
    node_id = voroni_node_sj_gp.iloc[i, -1][0]
    voroni_id = voroni_node_sj_gp.iloc[i, -2]
    node_to_voroni_dict[node_id] = voroni_id

# attach population information to nodes file
nodes["population"] = 0
for i in range(nodes.shape[0]):
    node_id = nodes.loc[i, "node_id"]
    if node_id in node_to_voroni_dict.keys():
        voroni_id = node_to_voroni_dict[node_id]
        if voroni_id in voroni_pop_dict.keys():
            popi = voroni_pop_dict[voroni_id]
            nodes.loc[i, "population"] = popi

# %%
# drop areas with zero population
nodes.population = nodes.population.astype(int)
nodes = nodes[(nodes.population != 0)]  # 8275 -> 7741
nodes.reset_index(drop=True, inplace=True)

# %%
# network analysis (60mins-cutoff: 35mins; 30mins-cutoff: 20mins)
test_flow = radiationFun.generate(
    nodes,
    name_to_index,
    index_to_name,
    net2,
    ver="method1",
    employment_column="population",
    cut_off=30,
    tile_id_column="node_id",
    tot_outflows_column="population",
    relevance_column="population",
    out_format="flows_sample",
)

# %%
"""
test_flow.to_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "lca_radiation_model_data"
    / "lca_radiation_model_data"
    / "processed"
    / "test_flow_20231025.csv",
    index=False,
)
"""

# %%
# for visualization
test_flow_dict = defaultdict(lambda: defaultdict(list))
for i in range(test_flow.shape[0]):
    from_id = test_flow.loc[i, "origin"]
    to_id = test_flow.loc[i, "destination"]
    flow = test_flow.loc[i, "flow"]
    test_flow_dict[from_id][to_id].append(flow)

edges["flow"] = 0
for i in range(edges.shape[0]):
    from_id = edges.loc[i, "from_node"]
    to_id = edges.loc[i, "to_node"]
    if from_id in test_flow_dict.keys():
        if to_id in test_flow_dict[from_id].keys():
            edges.loc[i, "flow"] = test_flow_dict[from_id][to_id][0]

# %%
"""
edges.to_file(
    base_path
    / "inputs"
    / "incoming_data"
    / "lca_radiation_model_data"
    / "lca_radiation_model_data"
    / "processed"
    / "road_network.gpkg",
    layer="edges_30_20211025",
    driver="GPKG",
)
"""
