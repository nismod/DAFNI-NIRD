"""
To estimate the bridge width for OS Open Roads
"""

# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd  # type: ignore
import numpy as np

from nird.utils import load_config

import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])

# %%
# inputs
os_links = gpd.read_parquet(base_path / "networks" / "road" / "GB_road_link_file.pq")
masteros_links = gpd.read_parquet(
    base_path / "networks" / "road" / "GB_RoadLinks.pq"
)  # averageWidth

lut_bridge = pd.read_csv(
    base_path / "tables" / "OSM_OSMasterMap_OSOpenRoadLookUpTable_bridges.csv"
)

# %%
# os link: a list of masteros links with bridge
masteros_links["Roadlink_Id"] = masteros_links["gml_id"]
lut_bridge = lut_bridge.merge(
    masteros_links[["Roadlink_Id", "averageWidth"]], on="Roadlink_Id", how="left"
)

masteros_bridges_gp = lut_bridge.groupby(
    by="OSOpenRoads_RoadLinkIdentifier", as_index=False
).agg(
    {
        "averageWidth": list,
    }
)

# %%
# select partial os links with bridge
os_links_bridge = os_links.merge(
    masteros_bridges_gp, how="right", on="OSOpenRoads_RoadLinkIdentifier"
)
# sum or average?
# collapsed dual carriageways (sum of two random bridge widths)
# others (the average of all bridge widths)
os_links_bridge["est_averageWidth"] = os_links_bridge.apply(
    lambda row: (
        np.sum(sorted(row["averageWidth"], reverse=True)[:2])
        if row["form_of_way"] == "Collapsed Dual Carriageway"
        else np.mean(row["averageWidth"])
    ),
    axis=1,
)

os_links = os_links.merge(
    os_links_bridge[
        ["OSOpenRoads_RoadLinkIdentifier", "averageWidth", "est_averageWidth"]
    ],
    how="left",
    on="OSOpenRoads_RoadLinkIdentifier",
)

os_links.drop(columns="averageWidth", inplace=True)
os_links.rename(
    columns={
        "est_averageWidth": "aveBridgeWidth",
    },
    inplace=True,
)

os_links.aveBridgeWidth = os_links.aveBridgeWidth.fillna(0)
# os_links.to_parquet(base_path / "networks" / "road" / "GB_road_link_file.pq")
