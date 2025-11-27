# %%
import geopandas as gpd
import pandas as pd
import warnings

warnings.simplefilter("ignore")

# %%
od = pd.read_csv(
    r"C:\Oxford\Research\DAFNI\local\processed_data\census_datasets\od_matrix\od_gb_oa_2021_node_with_bridges.csv"
)
# %%
# origin_strs = od.origin_node.to_numpy()
# destination_strs = od.destination_node.to_numpy()

# origin_map = {}  # dict to map generated uuid5 -> original string
# destination_map = {}


# def safe_uuid_bytes(s, map_dict):
#     try:
#         # Try to interpret as a real UUID
#         return uuid.UUID(s).bytes
#     except Exception:
#         # Not a valid UUID — generate deterministic uuid5
#         u5 = uuid.uuid5(uuid.NAMESPACE_DNS, s)
#         map_dict[u5] = s  # record mapping for lookup
#         return u5.bytes


# # Convert all to bytes
# origin_bt = np.array([safe_uuid_bytes(s, origin_map) for s in origin_strs], dtype="V16")
# destination_bt = np.array(
#     [safe_uuid_bytes(s, destination_map) for s in destination_strs], dtype="V16"
# )


# %%
nodes = gpd.read_parquet(
    r"C:\Oxford\Research\DAFNI\local\processed_data\networks\road\GB_road_nodes_with_bridges.gpq"
)
edges = gpd.read_parquet(
    r"C:\Oxford\Research\DAFNI\local\processed_data\networks\road\GB_road_links_with_bridges.gpq"
)

# %%
ids = [f"roadn_{i}" for i in range(len(nodes))]
nd_map = {name: f"roadn_{i}" for i, name in enumerate(nodes.id)}

# %%
nodes.rename(columns={"id": "id_copy"}, inplace=True)
nodes["id"] = nodes.id_copy.map(nd_map)

# %%
edges.rename(columns={"from_id": "from_id_copy", "to_id": "to_id_copy"}, inplace=True)
edges["from_id"] = edges["from_id_copy"].map(nd_map)
edges["to_id"] = edges["to_id_copy"].map(nd_map)

# %%
od.rename(
    columns={
        "origin_node": "origin_node_copy",
        "destination_node": "destination_node_copy",
    },
    inplace=True,
)
# %%
od["origin_node"] = od.origin_node_copy.map(nd_map)
od["destination_node"] = od.destination_node_copy.map(nd_map)
