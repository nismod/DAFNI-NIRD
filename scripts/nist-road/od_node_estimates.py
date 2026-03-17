# %%
from typing import Dict
from pathlib import Path
import geopandas as gpd
import pandas as pd
import warnings
from nird.utils import load_config

warnings.simplefilter("ignore")
dafni_path = Path(load_config()["paths"]["base_path"])
nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"


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
od_ppp = pd.read_parquet(nist_path / "processed" / "od_oa_ppp_estimates.pq")
# ppp: 14 million (2021), 20 million (2030), 24 million (2050)
od_hhh = pd.read_parquet(nist_path / "processed" / "od_oa_hhh_estimates.pq")
# hhh: 14 million (2021), 21 million (2030), 29 million (2050)
# %%
road_nodes = gpd.read_parquet(
    dafni_path / "networks" / "road" / "GB_road_nodes_with_bridges.gpq"
)
oa21_shp = gpd.read_parquet(
    dafni_path
    / "census_datasets"
    / "admin_census_boundary_stats"
    / "gb_oa_2021_estimates.geoparquet"
)
nearest_node_dict = find_nearest_node(oa21_shp, road_nodes)
nearest_node_dict2 = {}
for oa_idx, node_idx in nearest_node_dict.items():
    oa_id = oa21_shp.loc[oa_idx, "OA21CD"]
    node_id = road_nodes.loc[node_idx, "id"]
    nearest_node_dict2[oa_id] = node_id
# %%
od_ppp_temp = od_ppp[["origin", "destination", "Car21", "Car30", "Car50"]]
od_hhh_temp = od_hhh[["origin", "destination", "Car21", "Car30", "Car50"]]
od = od_ppp_temp.merge(
    od_hhh_temp, on=["origin", "destination"], suffixes=("_ppp", "_hhh"), how="left"
)

# %%
od["origin_node"] = od["origin"].map(nearest_node_dict2)
od["destination_node"] = od["destination"].map(nearest_node_dict2)
od_node_ppp = od[
    ["origin_node", "destination_node", "Car21_ppp", "Car30_ppp", "Car50_ppp"]
].rename(columns={"Car21_ppp": "Car21", "Car30_ppp": "Car30", "Car50_ppp": "Car50"})
od_node_hhh = od[
    ["origin_node", "destination_node", "Car21_hhh", "Car30_hhh", "Car50_hhh"]
].rename(columns={"Car21_hhh": "Car21", "Car30_hhh": "Car30", "Car50_hhh": "Car50"})

# rows_with_null = df[df.isnull().any(axis=1)]
# %%
# export results
od_node_hhh.to_parquet(
    nist_path / "outputs" / "roads" / "od_node_hhh_estimates.pq", index=False
)
od_node_ppp.to_parquet(
    nist_path / "outputs" / "roads" / "od_node_ppp_estimates.pq", index=False
)
