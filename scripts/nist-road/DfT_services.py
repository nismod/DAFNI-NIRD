# %%
from typing import Dict
from pathlib import Path
import pandas as pd
import json
import geopandas as gpd
import networkx as nx
from tqdm.auto import tqdm

base_dir = Path(r"C:\Oxford\Research\NIST\DfT Model\processed_data")
nist_dir = Path(r"C:\Oxford\Research\NIST\local\data\processed")


# %%
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


def filter_by_attraction(row):
    filtered = [
        (target, time, attr)
        for target, time, attr in zip(
            row["target_nodes"], row["travel_times_sec"], row["target_populations"]
        )
        if pd.notna(attr)
    ]

    return pd.Series(
        {
            "target_nodes": [x[0] for x in filtered],
            "travel_times_sec": [x[1] for x in filtered],
            "target_populations": [x[2] for x in filtered],
        }
    )


# %%
# source | production | attraction | list of destinations | list of destination attractions
# load network components
# To check network connectivity (components) - good!
nodes = gpd.read_parquet(nist_dir / "England_road_nodes_with_bridges.gpq")
nodes.rename(columns={"id": "node_id"}, inplace=True)
edges = gpd.read_parquet(nist_dir / "England_links_with_time.gpq")  # 0.4 million edges

# %%
lut = pd.read_csv(base_dir / "simulation" / "lut.csv")
msoa11cd_to_msoa21cd = lut.set_index("MSOA11CD")["MSOA21CD"].to_dict()
oa21cd_to_msoa21cd = lut.set_index("OA21CD")["MSOA21CD"].to_dict()
oa21cd_to_lsoa11cd = lut.set_index("OA21CD")["LSOA11CD"].to_dict()
with open(base_dir / "simulation" / "zonename_to_msoa11cd.json", "r") as f:
    zonename_to_msoa11cd = json.load(f)
with open(base_dir / "simulation" / "msoa21_to_oa21_pop.json", "r") as f:
    msoa21_to_oa21_ratio = json.load(f)
# with open(base_dir / "simulation" / "msoa11_to_msoa21.json", "r") as f:
#     msoa11cd_to_msoa21cd = json.load(f)
# with open(base_dir / "simulation" / "oa21cd_to_lsoa11cd.json", "r") as f:
#     oa21cd_to_lsoa11cd = json.load(f)

# %%
# To generate OD for other services (downscale from MSOA to OA level)
# load PA at msoa level
purpose = "Town"  # Town, Employment, Education
df_pa = pd.read_csv(base_dir / "simulation" / "PA_MSOA11CD_CarEducation_2021.csv")
df_pa.loc[
    df_pa["ZoneName"].str.contains("King`s Lynn and West Norfolk", na=False),
    "ZoneName",
] = df_pa.loc[
    df_pa["ZoneName"].str.contains("King`s Lynn and West Norfolk", na=False),
    "ZoneName",
].str.replace(
    "`", "'", regex=False
)
df_pa["MSOA11CD"] = df_pa["ZoneName"].map(zonename_to_msoa11cd)
df_pa = df_pa[df_pa.MSOA11CD.notnull()].reset_index(drop=True)
df_pa["MSOA21CD"] = df_pa["MSOA11CD"].map(msoa11cd_to_msoa21cd)

# %%
# downscale from MSOA to OA level
# Columns that should NOT be scaled
id_cols = ["ZoneID", "ZoneName", "Authority", "MSOA11CD", "MSOA21CD"]

# Everything else will be distributed
value_cols = [c for c in df_pa.columns if c not in id_cols]
rows = []
for _, row in df_pa.iterrows():
    msoa = row["MSOA21CD"]
    if msoa not in msoa21_to_oa21_ratio:
        continue

    for oa, ratio in msoa21_to_oa21_ratio[msoa].items():
        new_row = row.copy()
        new_row["OA21CD"] = oa
        for col in value_cols:
            new_row[col] = row[col] * ratio
        rows.append(new_row)

df_pa_oa = pd.DataFrame(rows)
df_pa_oa.reset_index(drop=True, inplace=True)
df_pa_oa = df_pa_oa[
    [
        "OA21CD",
        "Total2021_Production_AllCoreDBs",
        "Total2021_Attraction_AllCoreDBs",
        "Population_2021_AllCoreDBs",
        "Workers_2021_AllCoreDBs",
        "Households_2021_AllCoreDBs",
        "Jobs_2021_AllCoreDBs",
    ]
]
# export disaggreagted PA at OA level
df_pa_oa.to_parquet(base_dir / "simulation" / f"PA_{purpose}_OA_2021.pq")

# %%
# attach maximum travel time for service within each OA
df_pa_oa["LSOA11CD"] = df_pa_oa["OA21CD"].map(oa21cd_to_lsoa11cd)
jts = pd.read_csv(base_dir / "simulation" / "jts_2019_lsoa_msoa.csv")

df_pa_oa = df_pa_oa.merge(
    jts[["LSOA11CD", f"{purpose}Cart"]], on="LSOA11CD", how="left"
)
median_cutoff = df_pa_oa[f"{purpose}Cart"].median()
df_pa_oa.loc[df_pa_oa[f"{purpose}Cart"].isnull(), f"{purpose}Cart"] = median_cutoff

# %%
oa_shp = gpd.read_parquet(base_dir / "simulation" / "oa21_shp.gpq")
oa_shp = oa_shp[oa_shp.OA21CD.str.startswith("E")].reset_index(drop=True)
zone_to_node_dict = find_nearest_node(oa_shp, nodes, "OA21CD", "node_id")

# %%
# source | population | production | [destinations] | [populations] | [times]
df_pa_oa["source"] = df_pa_oa["OA21CD"].map(zone_to_node_dict)  # 177_357
df_pa_oa_node = (
    df_pa_oa.groupby("source")
    .agg(
        {
            "Population_2021_AllCoreDBs": "sum",
            "Total2021_Production_AllCoreDBs": "sum",
            "Total2021_Attraction_AllCoreDBs": "sum",
            f"{purpose}Cart": "max",
        }
    )
    .reset_index()
)
df_pa_oa_node = df_pa_oa_node[
    df_pa_oa_node.Total2021_Production_AllCoreDBs > 0
].reset_index(
    drop=True
)  # 109_606

# %%
G = nx.from_pandas_edgelist(
    edges,
    source="from_id",
    target="to_id",
    edge_attr=["e_id", "time_sec"],
    create_using=nx.Graph(),
)

# %%
sources = []
target_lists = []
time_lists = []

for _, row in tqdm(df_pa_oa_node.iterrows(), desc="Processing sources"):
    source = row["source"]
    lengths = nx.single_source_dijkstra_path_length(
        G, source=source, weight="time_sec", cutoff=row[f"{purpose}Cart"]
    )

    # remove the source node itself if present
    filtered = [(t, tt) for t, tt in lengths.items() if t != source]
    sources.append(source)
    target_lists.append([t for t, _ in filtered])
    time_lists.append([tt for _, tt in filtered])

targets_and_times = pd.DataFrame(
    {"source": sources, "target_nodes": target_lists, "travel_times_sec": time_lists}
)
# %%
attraction_dict = (
    df_pa_oa_node[df_pa_oa_node.Total2021_Attraction_AllCoreDBs > 0]
    .set_index("source")["Population_2021_AllCoreDBs"]
    .to_dict()
)
targets_and_times["target_populations"] = targets_and_times["target_nodes"].apply(
    lambda nodes: [attraction_dict.get(node) for node in nodes]
)  # select only nodes with "attractions" as destinations (not any junction nodes)
targets_and_times[["target_nodes", "travel_times_sec", "target_populations"]] = (
    targets_and_times.apply(filter_by_attraction, axis=1)
)
# %%
df_pa_oa_node = df_pa_oa_node.merge(targets_and_times, on="source", how="left")
df_pa_oa_node["target_counts"] = df_pa_oa_node.target_nodes.apply(len)
df_pa_oa_node = df_pa_oa_node[df_pa_oa_node["target_counts"] > 0].reset_index(drop=True)
# around 30% of source nodes do not have reachable destinations nodes

# %%
df_pa_oa_node.to_parquet(base_dir / "simulation" / f"radiation_inputs_{purpose}.pq")
