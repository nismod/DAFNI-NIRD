# %%
# import sys
from typing import Dict
from pathlib import Path
import pandas as pd
import geopandas as gpd
import warnings
from nird.utils import load_config

warnings.simplefilter("ignore")
dafni_path = Path(load_config()["paths"]["base_path"])
macc_path = Path(load_config()["paths"]["macc_path"])
nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"


# %%
def find_nearest_node(
    zones: gpd.GeoDataFrame, road_nodes: gpd.GeoDataFrame
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
    nearest_node_dict = {}
    for zidx, z in zones.iterrows():
        closest_road_node = road_nodes.sindex.nearest(z.geometry, return_all=False)[1][
            0
        ]
        # for the first [x]:
        #   [0] represents the index of geometry;
        #   [1] represents the index of gdf
        # the second [x] represents the No. of closest item in the returned list,
        #   which only return one nearest node in this case
        nearest_node_dict[zidx] = closest_road_node

    return nearest_node_dict  # zone_idx: node_idx


# %%
# futrue population estimates at the LAD level (calibrated with subnational estimates)
# For the whole GB
oa_shp = gpd.read_parquet(
    dafni_path
    / "census_datasets"
    / "admin_census_boundary_stats"
    / "gb_oa_2021_estimates.geoparquet"
)
rail_nodes = gpd.read_parquet(
    macc_path
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes.gpq"
)  # 40_912

stations = rail_nodes[
    rail_nodes.station_label == "train_station"
].reset_index()  # 2_847

# To ensure trips to be commenced between train stations instead of any junction nodes
nearest_node_dict = find_nearest_node(oa_shp, stations)
nearest_node_dict2 = {}
for oa_idx, node_idx in nearest_node_dict.items():
    oa_id = oa_shp.loc[oa_idx, "OA21CD"]
    node_id = stations.loc[node_idx, "node_id"]
    nearest_node_dict2[oa_id] = node_id

# %%
# England only
oa_ppp = pd.read_parquet(nist_path / "processed" / "oa21_ppp_estimates.pq")
oa_hhh = pd.read_parquet(nist_path / "processed" / "oa21_hhh_estimates.pq")

# %%
df = pd.DataFrame(list(nearest_node_dict2.items()), columns=["OA21CD", "node_id"])
df_ppp = df.merge(oa_ppp, on="OA21CD", how="right")
group_ppp = df_ppp.groupby(by=["node_id"], as_index=False).sum(numeric_only=True)
df_hhh = df.merge(oa_hhh, on="OA21CD", how="right")
group_hhh = df_hhh.groupby(by=["node_id"], as_index=False).sum(numeric_only=True)
group_ppp["origin_node_id"] = group_ppp["node_id"]
group_ppp["destination_node_id"] = group_ppp["node_id"]
group_hhh["origin_node_id"] = group_hhh["node_id"]
group_hhh["destination_node_id"] = group_hhh["node_id"]

# %%
# load od flow data
od_node_2025 = pd.read_csv(
    macc_path / "incoming" / "rails" / "od_adjusted_0519.csv"
)  # 20250519 - Monday
od_node_2025["flows_25"] = od_node_2025["flows_adjusted"]

# %%
# future od projectipn
od_ppp = (
    od_node_2025.merge(
        group_ppp[
            ["origin_node_id", "OA21_PPP_2025", "OA21_PPP_2030", "OA21_PPP_2050"]
        ],
        on="origin_node_id",
        how="left",
    )
    .rename(
        columns={
            "OA21_PPP_2025": "origin_ppp_2025",
            "OA21_PPP_2030": "origin_ppp_2030",
            "OA21_PPP_2050": "origin_ppp_2050",
        }
    )
    .merge(
        group_ppp[
            ["destination_node_id", "OA21_PPP_2025", "OA21_PPP_2030", "OA21_PPP_2050"]
        ],
        on="destination_node_id",
        how="left",
    )
    .rename(
        columns={
            "OA21_PPP_2025": "destination_ppp_2025",
            "OA21_PPP_2030": "destination_ppp_2030",
            "OA21_PPP_2050": "destination_ppp_2050",
        }
    )
)

od_hhh = (
    od_node_2025.merge(
        group_hhh[
            ["origin_node_id", "OA21_HHH_2025", "OA21_HHH_2030", "OA21_HHH_2050"]
        ],
        on="origin_node_id",
        how="left",
    )
    .rename(
        columns={
            "OA21_HHH_2025": "origin_hhh_2025",
            "OA21_HHH_2030": "origin_hhh_2030",
            "OA21_HHH_2050": "origin_hhh_2050",
        }
    )
    .merge(
        group_hhh[
            ["destination_node_id", "OA21_HHH_2025", "OA21_HHH_2030", "OA21_HHH_2050"]
        ],
        on="destination_node_id",
        how="left",
    )
    .rename(
        columns={
            "OA21_HHH_2025": "destination_hhh_2025",
            "OA21_HHH_2030": "destination_hhh_2030",
            "OA21_HHH_2050": "destination_hhh_2050",
        }
    )
)

# %%
# projection
od_ppp["flows_30"] = (
    od_ppp["flows_25"]
    * (od_ppp.origin_ppp_2030 / od_ppp.origin_ppp_2025)
    * (od_ppp.destination_ppp_2030 / od_ppp.destination_ppp_2025)
)

od_ppp["flows_50"] = (
    od_ppp["flows_25"]
    * (od_ppp.origin_ppp_2050 / od_ppp.origin_ppp_2025)
    * (od_ppp.destination_ppp_2050 / od_ppp.destination_ppp_2025)
)
od_hhh["flows_30"] = (
    od_hhh["flows_25"]
    * (od_hhh.origin_hhh_2030 / od_hhh.origin_hhh_2025)
    * (od_hhh.destination_hhh_2030 / od_hhh.destination_hhh_2025)
)
od_hhh["flows_50"] = (
    od_hhh["flows_25"]
    * (od_hhh.origin_hhh_2050 / od_hhh.origin_hhh_2025)
    * (od_hhh.destination_hhh_2050 / od_hhh.destination_hhh_2025)
)
od_ppp["flows_30"] = od_ppp["flows_30"].fillna(od_ppp["flows_25"])
od_ppp["flows_50"] = od_ppp["flows_50"].fillna(od_ppp["flows_25"])
od_hhh["flows_30"] = od_hhh["flows_30"].fillna(od_hhh["flows_25"])
od_hhh["flows_50"] = od_hhh["flows_50"].fillna(od_hhh["flows_25"])

# %%
# export results
od_ppp["flows_30"] = od_ppp["flows_30"].round().astype(int)
od_ppp["flows_50"] = od_ppp["flows_50"].round().astype(int)
od_hhh["flows_30"] = od_hhh["flows_30"].round().astype(int)
od_hhh["flows_50"] = od_hhh["flows_50"].round().astype(int)

# %%
od_ppp.to_parquet(
    nist_path / "outputs" / "rails" / "od_oa_ppp_estimates.pq",
    index=False,
)
od_hhh.to_parquet(
    nist_path / "outputs" / "rails" / "od_oa_hhh_estimates.pq",
    index=False,
)
