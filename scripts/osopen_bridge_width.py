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
os_links = gpd.read_parquet(
    base_path / "networks" / "road" / "GB_road_link_file.geoparquet"
)
masteros_links = gpd.read_parquet(
    base_path / "networks" / "road" / "GB_RoadLinks.pq"
)  # averageWidth

lut_bridge = pd.read_csv(
    base_path / "tables" / "OSM_OSMasterMap_OSOpenRoadLookUpTable_bridges.csv"
)

# %%
# os link: a list of masteros links with bridge
masteros_2_os = lut_bridge.set_index("Roadlink_Id")[
    "OSOpenRoads_RoadLinkIdentifier"
].to_dict()
masteros_links["OSOpenRoads_RoadLinkIdentifier"] = masteros_links.gml_id.map(
    masteros_2_os
)
masteros_bridges = masteros_links[
    masteros_links.OSOpenRoads_RoadLinkIdentifier.notnull()
][["gml_id", "OSOpenRoads_RoadLinkIdentifier", "averageWidth"]]

masteros_bridges_gp = masteros_bridges.groupby(
    by=["OSOpenRoads_RoadLinkIdentifier"], as_index=False
).agg(
    {
        # "gml_id": list,
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
        np.sum(sorted(row["averageWidth_y"], reverse=True)[:2])
        if row["form_of_way"] == "Collapsed Dual Carriageway"
        else np.mean(row["averageWidth_y"])
    ),
    axis=1,
)

os_links = os_links.merge(
    os_links_bridge[
        ["OSOpenRoads_RoadLinkIdentifier", "averageWidth_y", "est_averageWidth"]
    ],
    how="left",
    on="OSOpenRoads_RoadLinkIdentifier",
)
os_links.drop(
    columns=["averageWidth", "minimumWidth", "max_z", "min_z", "mean_z"], inplace=True
)
os_links.rename(
    columns={
        "averageWidth_y": "list_of_bridge_width",
        "est_averageWidth": "aveBridgeWidth",
    },
    inplace=True,
)
os_links.to_parquet(base_path / "networks" / "road" / "GB_road_link_file_bridge.pq")
