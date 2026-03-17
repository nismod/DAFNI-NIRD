# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd
from nird.utils import load_config
import json

import warnings

pd.set_option("display.max_columns", None)
warnings.simplefilter("ignore")

nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"

macc_path = Path(load_config()["paths"]["macc_path"])

# %%
nodes = gpd.read_parquet(
    macc_path
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes.gpq"
)
stations = nodes[nodes.station_label == "train_station"].reset_index()
station_node_to_name = stations.set_index("node_id")["station_name"].to_dict()
station_node_to_tiploc = stations.set_index("node_id")["TIPLOC"].to_dict()

with open(
    nist_path / "processed_data" / "node_idx_to_id_dict.json",
    "r",
) as f:
    node_idx_to_id = json.load(f)

# %%
od_2021 = pd.read_csv(nist_path / "outputs" / "rails" / "od_adjusted_0519.csv")
od_ppp = pd.read_parquet(nist_path / "outputs" / "rails" / "od_oa_ppp_estimates.pq")
od_hhh = pd.read_parquet(
    nist_path / "outputs" / "rails" / "processed_data" / "od_oa_hhh_estimates.pq"
)

# generate future od
od_ppp_30 = od_ppp[["origin_node_idx", "destination_node_idx", "flows_30"]]
od_ppp_50 = od_ppp[["origin_node_idx", "destination_node_idx", "flows_50"]]
od_hhh_30 = od_hhh[["origin_node_idx", "destination_node_idx", "flows_30"]]
od_hhh_50 = od_hhh[["origin_node_idx", "destination_node_idx", "flows_50"]]


# %%
def name_stations(od, node_idx_to_id, node_id_to_name):
    od["origin_node"] = od["origin_node_idx"].astype(str).map(node_idx_to_id)
    od["destination_node"] = od["destination_node_idx"].astype(str).map(node_idx_to_id)
    od["origin_TIPLOC"] = od["origin_node"].map(node_id_to_name)
    od["destination_TIPLOC"] = od["destination_node"].map(node_id_to_name)

    return od


od_ppp_30 = name_stations(od_ppp_30, node_idx_to_id, station_node_to_tiploc)
od_hhh_30 = name_stations(od_hhh_30, node_idx_to_id, station_node_to_tiploc)
od_ppp_50 = name_stations(od_ppp_50, node_idx_to_id, station_node_to_tiploc)
od_hhh_50 = name_stations(od_hhh_50, node_idx_to_id, station_node_to_tiploc)

# %%
edges = gpd.read_parquet(nist_path / "outputs" / "rails" / "future_flows.gpq")
edges["vc_ppp_30"] = edges["flow_ppp_30"] / edges["capacity"]
edges["vc_hhh_30"] = edges["flow_hhh_30"] / edges["capacity"]
edges["vc_ppp_50"] = edges["flow_ppp_50"] / edges["capacity"]
edges["vc_hhh_50"] = edges["flow_hhh_50"] / edges["capacity"]

# %%
lad = gpd.read_parquet(nist_path / "processed_data" / "lad24_shp.gpq")
edge_split = gpd.overlay(edges, lad[["LAD24CD", "geometry"]], how="intersection")
edge_split["length_m"] = edge_split.geometry.length
edge_split["length_km"] = edge_split["length_m"] / 1000.0
edge_split = edge_split[edge_split["length_km"] > 0].copy()
agg = (
    edge_split.groupby("LAD24CD")
    .apply(
        lambda g: pd.Series(
            {
                "vc_baseline_lw": (
                    g["baseline_flow"] / g["capacity"] * g["length_km"]
                ).sum()
                / g["length_km"].sum(),
                "vc_30_ppp_lw": (g["vc_ppp_30"] * g["length_km"]).sum()
                / g["length_km"].sum(),
                "vc_30_hhh_lw": (g["vc_hhh_30"] * g["length_km"]).sum()
                / g["length_km"].sum(),
                "vc_50_ppp_lw": (g["vc_ppp_50"] * g["length_km"]).sum()
                / g["length_km"].sum(),
                "vc_50_hhh_lw": (g["vc_hhh_50"] * g["length_km"]).sum()
                / g["length_km"].sum(),
            }
        )
    )
    .reset_index()
)
lad_with_stats = lad.merge(agg, on="LAD24CD", how="left")
lad_with_stats.to_parquet(nist_path / "outputs" / "rails" / "agg_vc.gpq")
