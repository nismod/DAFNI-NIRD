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
dafni_dir = Path(r"C:\Oxford\Research\DAFNI\local\processed_data")


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
# nodes = gpd.read_parquet(nist_dir / "England_road_nodes_with_bridges.gpq")
# edges = gpd.read_parquet(nist_dir / "England_links_with_time.gpq")  # 0.4 million edges
nodes = gpd.read_parquet(base_dir / "networks" / "road_nodes_gb.gpq")
nodes.rename(columns={"id": "node_id"}, inplace=True)
edges = gpd.read_parquet(base_dir / "networks" / "road_edges_gb.gpq")

# %%
lut = pd.read_parquet(base_dir / "simulation" / "gb_lut.pq")
msoa11cd_to_msoa21cd = lut.set_index("MSOA11CD")["MSOA21CD"].to_dict()
oa21cd_to_msoa21cd = lut.set_index("OA21CD")["MSOA21CD"].to_dict()
oa21cd_to_lsoa11cd = lut.set_index("OA21CD")["LSOA11CD"].to_dict()
with open(base_dir / "simulation" / "zonename_to_msoa11cd.json", "r") as f:
    zonename_to_msoa11cd = json.load(
        f
    )  # convert from DFT zones to MSOA11 for England and Wales
scot_df = pd.read_csv(base_dir / "simulation" / "scot_zone_msoa21_popshare.csv")
scotzone_to_msoa21_popshare_df = (
    scot_df.groupby("ZoneName")
    .agg({"MSOA21CD": list, "MSOA_POP_SHARE": list})
    .reset_index()
)  # downscaling factor from ZoneName to MSOA21CD (Scotland)

gb_df = pd.read_csv(base_dir / "simulation" / "msoa21_oa21_popshare.csv")
msoa21_to_oa21_popshare_df = (
    gb_df.groupby("MSOA21CD").agg({"OA21CD": list, "OA_POP_SHARE": list}).reset_index()
)  # downscaling factor from MOSA21CD to OA21CD (GB)

# %%
# To generate OD for other services (downscale from MSOA to OA level)
# load PA at msoa level
purpose = "Comm"  # Town, Employment, Education
year = 2050
df_pa = pd.read_csv(
    base_dir / "simulation" / f"PA_MSOA11CD_Car{purpose}_{year}.csv"
)  # baseline scenario: 2021, future scenario: 2050
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

# for scotland
df_pa_scot = df_pa[df_pa.MSOA21CD.isnull()]  # 499
df_pa_scot.drop(columns=["MSOA21CD"], inplace=True)
df_pa_scot = df_pa_scot.merge(scotzone_to_msoa21_popshare_df, on="ZoneName", how="left")
df_pa_scot = df_pa_scot.explode(["MSOA21CD", "MSOA_POP_SHARE"], ignore_index=True)

# for england and wales
df_pa_ew = df_pa[df_pa.MSOA21CD.notnull()]  # 7201
df_pa_ew["MSOA_POP_SHARE"] = 1

# combined df at MSOA level
df_pa = pd.concat([df_pa_ew, df_pa_scot], axis=0).reset_index(drop=True)  # 11_308

# downscale from MSOA21 to OA21
df_pa = df_pa.merge(msoa21_to_oa21_popshare_df, on="MSOA21CD", how="left")
df_pa = df_pa.explode(["OA21CD", "OA_POP_SHARE"], ignore_index=True)  # 337_637

# downscale from MSOA to OA level
# for England and Wales
# Columns that should NOT be scaled
df_pa_oa = df_pa[["OA21CD"]]
df_cols = [
    f"Total{year}_Production_AllCoreDBs",
    f"Total{year}_Attraction_AllCoreDBs",
    f"Population_{year}_AllCoreDBs",
    # f"Workers_{year}_AllCoreDBs",
    # f"Households_{year}_AllCoreDBs",
    # f"Jobs_{year}_AllCoreDBs",
]
for col in df_cols:
    df_pa_oa[col] = df_pa[col] * df_pa["MSOA_POP_SHARE"] * df_pa["OA_POP_SHARE"]

# %%
# export disaggreagted PA at OA level
df_pa_oa.to_parquet(base_dir / "simulation" / f"PA_{purpose}_OA_{year}_GB.pq")

# %%
# attach maximum travel time for service within each OA
df_pa_oa["LSOA11CD"] = df_pa_oa["OA21CD"].map(oa21cd_to_lsoa11cd)
jts = pd.read_csv(
    base_dir / "simulation" / "jts_2019_lsoa_msoa.csv"
)  # searching cutoff

df_pa_oa = df_pa_oa.merge(
    jts[["LSOA11CD", f"{purpose}Cart"]], on="LSOA11CD", how="left"
)
median_cutoff = df_pa_oa[f"{purpose}Cart"].median()
df_pa_oa.loc[df_pa_oa[f"{purpose}Cart"].isnull(), f"{purpose}Cart"] = median_cutoff

# %%
oa_shp = gpd.read_parquet(
    dafni_dir
    / "census_datasets"
    / "admin_census_boundary_stats"
    / "gb_oa_2021_estimates.geoparquet"
)
# oa_shp = gpd.read_parquet(base_dir / "simulation" / "oa21_shp.gpq")  # England and Wales
# oa_shp = oa_shp[oa_shp.OA21CD.str.startswith("E")].reset_index(drop=True)
zone_to_node_dict = find_nearest_node(oa_shp, nodes, "OA21CD", "node_id")

# %%
# source | population | production | [destinations] | [populations] | [times]
df_pa_oa["source"] = df_pa_oa["OA21CD"].map(zone_to_node_dict)  # 177_357
df_pa_oa_node = (
    df_pa_oa.groupby("source")
    .agg(
        {
            f"Population_{year}_AllCoreDBs": "sum",
            f"Total{year}_Production_AllCoreDBs": "sum",
            f"Total{year}_Attraction_AllCoreDBs": "sum",
            f"{purpose}Cart": "max",
        }
    )
    .reset_index()
)
df_pa_oa_node = df_pa_oa_node[
    df_pa_oa_node[f"Total{year}_Production_AllCoreDBs"] > 0
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
    df_pa_oa_node[df_pa_oa_node[f"Total{year}_Attraction_AllCoreDBs"] > 0]
    .set_index("source")[f"Population_{year}_AllCoreDBs"]
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
df_pa_oa_node.to_parquet(
    base_dir / "simulation" / f"radiation_inputs_{purpose}_{year}_GB.pq"
)
