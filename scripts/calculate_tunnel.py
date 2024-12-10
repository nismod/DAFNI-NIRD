# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config

import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])

# %%
lut = pd.read_csv(base_path / "tables" / "OSOpenRoadLookUpTable_major_roads.csv")
openos_links = gpd.read_parquet(
    base_path / "networks" / "road" / "GB_road_link_file.geoparquet"
)
masteros_links = gpd.read_parquet(base_path / "networks" / "road" / "GB_RoadLinks.pq")

# %%
open_to_master = lut.groupby(by="OSOpenRoads_RoadLinkIdentifier", as_index=False).agg(
    {"Roadlink_Id": list}
)
master_to_tunnel = masteros_links[["gml_id", "roadStructure"]]
master_to_tunnel["hasTunnel"] = 0
master_to_tunnel.loc[master_to_tunnel.roadStructure.notnull(), "hasTunnel"] = 1
master_has_tunnel = master_to_tunnel.set_index("gml_id")["hasTunnel"].to_dict()
open_to_master["hasTunnel"] = open_to_master.apply(
    lambda row: (
        1
        if any(master_has_tunnel.get(value, 0) == 1 for value in row["Roadlink_Id"])
        else 0
    ),
    axis=1,
)

# %%
openos_links = openos_links.merge(
    open_to_master[["OSOpenRoads_RoadLinkIdentifier", "hasTunnel"]],
    how="left",
    on="OSOpenRoads_RoadLinkIdentifier",
)

openos_links.drop(
    columns=["averageWidth", "minimumWidth", "max_z", "min_z", "mean_z"], inplace=True
)

openos_links["hasTunnel"] = openos_links["hasTunnel"].fillna(0).astype(int)
# openos_links.to_parquet(base_path / "networks" / "road" / "GB_road_link_file.pq")
# 06CC7473-D5CE-4BAF-B723-4EFF5E3C7ED4: os open: no, masteros: yes
