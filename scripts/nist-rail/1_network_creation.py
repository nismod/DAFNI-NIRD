# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd
import snkit.network as snx
from collections import defaultdict

import warnings
from utils import create_network_from_nodes_and_edges

warnings.simplefilter("ignore")

INPUT_PATH_MACC = Path(r"C:\Oxford\Research\MACCHUB\local")


# %%
national_edges = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "NWR_TrackCentreLines_revised.gpq"  # attached with track id and direction info
)  # 40,845

stations_snapped = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "train_stations_calibrated.gpq"  # calibrated based on OSM data
)  # 2,852

# %%
# split edges at nodes
split_edges = snx._split_edges_at_nodes(
    edges=national_edges, nodes=stations_snapped, tolerance=1e-3
)
split_edges = pd.concat(split_edges, axis=0).reset_index().drop("index", axis=1)
print("Done with split edges at nodes")

# %%
# create network based on edges and nodes
network = create_network_from_nodes_and_edges(
    nodes=stations_snapped, edges=split_edges, node_edge_prefix=""
)
nodes = network.nodes  # including all network nodes
edges = network.edges
edges.crs = "epsg:27700"  # set CRS for edges
print(f"Number of edges: {len(edges)}")  # 51,996
print(f"Number of nodes: {len(nodes)}")  # 40,912

# %%
# export nodes and edges
nodes.to_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes.gpq"
)
edges.to_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_edges.gpq"
)

# %%
# match ELR values to nodes
elr = gpd.read_file(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "NWR_ELRs.shp"
)

elr_dict = defaultdict(lambda: defaultdict(float))
for row in elr.itertuples():
    elr_dict[row.ASSET_ID]["start"] = row.START
    elr_dict[row.ASSET_ID]["end"] = row.END

matched_elr_values = []
for _, node in nodes.iterrows():
    # Find nearest ELR line to this node
    nearest_idx = elr.distance(node.geometry).idxmin()
    line_geom = elr.loc[nearest_idx, "geometry"]
    line_id = elr.loc[nearest_idx, "ASSET_ID"]

    # Get start/end ELR values for this line
    start_val = elr_dict[line_id]["start"]
    end_val = elr_dict[line_id]["end"]

    # Distance along the line where the node projects
    dist_along = line_geom.project(node.geometry)
    total_len = line_geom.length

    # Interpolate ELR value
    elr_val = start_val + (end_val - start_val) * (dist_along / total_len)

    matched_elr_values.append(elr_val)

# Add to nodes GeoDataFrame
nodes["elr_value"] = matched_elr_values
node_elr_dict = nodes.set_index("node_id")["elr_value"].to_dict()


# %%
# correct direction for up/down links
def check_up_down_direction(from_node, to_node, direction):
    assert direction in ["1", "2"], "direction should be 1 or 2!"
    from_elr_value = node_elr_dict[from_node]
    end_elr_value = node_elr_dict[to_node]
    if direction == 1:  # up, away from major cities, start < end
        if from_elr_value <= end_elr_value:
            return from_node, to_node
        else:
            return to_node, from_node
    else:  # down, towards major cities, start > end
        if from_elr_value <= end_elr_value:
            return to_node, from_node
        else:
            return from_node, to_node


# %%
non_bi_edges = edges[(edges.direction == "1") | (edges.direction == "2")].reset_index(
    drop=True
)
non_bi_edges[["from_node", "to_node"]] = non_bi_edges.apply(
    lambda row: check_up_down_direction(
        row["from_node"], row["to_node"], row["direction"]
    ),
    axis=1,
    result_type="expand",
)

# %%
# bidirectional links
bi_edges = edges[edges.direction == "3"].reset_index(drop=True)
bi_edges_copy = bi_edges.copy()
bi_edges_copy[["from_node", "to_node"]] = bi_edges[["to_node", "from_node"]]
i = len(edges)
for idx, row in bi_edges_copy.iterrows():
    edge_id = "e_" + str(idx + i)
    bi_edges_copy.loc[idx, "edge_id"] = edge_id

directed_edges = pd.concat(
    [non_bi_edges, bi_edges, bi_edges_copy], axis=0, ignore_index=True
)
directed_edges = gpd.GeoDataFrame(directed_edges, geometry="geometry", crs="epsg:27700")
directed_edges.drop(columns=["edge_id", "from_node", "to_node"], inplace=True)

# %%
# rebuild network and export nodes and edges
network = create_network_from_nodes_and_edges(
    nodes=stations_snapped, edges=directed_edges, node_edge_prefix=""
)

nodes = network.nodes  # including all network nodes
edges = network.edges
edges.crs = "epsg:27700"  # set CRS for edges
print(f"Number of edges: {len(edges)}")  # 73,321
print(f"Number of nodes: {len(nodes)}")  # 40,912

# %%
# export nodes and edges
nodes.to_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes_directed.gpq"
)
edges.to_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_edges_directed.gpq"
)
