# %%
from pathlib import Path

import pandas as pd
import geopandas as gpd

from utils import load_config

base_path = Path(load_config()["paths"]["base_path"])

# %%
# shapefile of MSOA
msoa_shp = gpd.read_file(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "MSOA_collection"
    / "England_msoa_2021"
    / "england_msoa_2021.shp"
)

msoa_shp = msoa_shp.rename(columns={"msoa21cd": "MSOA_CODE"})
# %%
# shapefile of the population-weighted centroids of MSOA
msoa_shp = gpd.read_file(
    base_path / "inputs" / "processed_data" / "census" / "MSOA.gpkg",
    layer="pop_weighted_centroids",
)
msoa_shp = msoa_shp.rename(columns={"MSOA21CD": "MSOA_CODE"})

# %%
# working population
wkpop = pd.read_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "MSOA_collection"
    / "WP001_msoa.csv"
)
wkpop = wkpop.rename(
    columns={"Middle layer Super Output Areas Code": "MSOA_CODE", "Count": "WKPOP_2021"}
)
msoa_shp = msoa_shp.merge(
    wkpop[["MSOA_CODE", "WKPOP_2021"]], how="left", on="MSOA_CODE"
)

# %%
# population traveling to work (by methods)
travel_by_method_file = pd.read_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "MSOA_collection"
    / "TS061_msoa.csv"
)

# total travelers in each MSOA
travel_by_method = travel_by_method_file.loc[
    travel_by_method_file.iloc[:, 2].isin(list(range(2, 12)))
]

travel_by_method = travel_by_method.groupby(
    by=["Middle layer Super Output Areas Code"], as_index=False
).agg({"Observation": sum})

travel_by_method = travel_by_method.rename(
    columns={
        "Middle layer Super Output Areas Code": "MSOA_CODE",
        "Observation": "total_commuters",
    }
)

# travelers by roads in each MSOA
travel_by_road = travel_by_method_file.loc[
    travel_by_method_file.iloc[:, 2].isin([2, 4, 5, 6, 7, 8, 9, 11])
]

travel_by_road = travel_by_road.groupby(
    by=["Middle layer Super Output Areas Code"], as_index=False
).agg({"Observation": sum})

travel_by_road = travel_by_road.rename(
    columns={
        "Middle layer Super Output Areas Code": "MSOA_CODE",
        "Observation": "road_commuters",
    }
)

msoa_shp = msoa_shp.merge(
    travel_by_method[["MSOA_CODE", "total_commuters"]],
    how="left",
    on="MSOA_CODE",
)
msoa_shp = msoa_shp.merge(
    travel_by_road[["MSOA_CODE", "road_commuters"]], how="left", on="MSOA_CODE"
)

# %%
# inflow and outflow
od_file = pd.read_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "MSOA_collection"
    / "ODWP01EW_MSOA.csv"
)

# select population working in the UK but not at/from home
od_file = od_file[od_file["Place of work indicator (4 categories) code"] == 3]
# delete population traveling within the origins
od_file = od_file[
    od_file["Middle layer Super Output Areas code"] != od_file["MSOA of workplace code"]
]

outflow = od_file.groupby(
    by=["Middle layer Super Output Areas code"], as_index=False
).agg({"Count": sum})
outflow = outflow.rename(
    columns={
        "Middle layer Super Output Areas code": "MSOA_CODE",
        "Count": "Outflow_2021",
    }
)

inflow = od_file.groupby(by=["MSOA of workplace code"], as_index=False).agg(
    {"Count": sum}
)
inflow = inflow.rename(
    columns={"MSOA of workplace code": "MSOA_CODE", "Count": "Inflow_2021"}
)

msoa_shp = msoa_shp.merge(
    outflow[["MSOA_CODE", "Outflow_2021"]], how="left", on="MSOA_CODE"
)
msoa_shp = msoa_shp.merge(
    inflow[["MSOA_CODE", "Inflow_2021"]], how="left", on="MSOA_CODE"
)

# %%
# export msoa_shp with attributes attached
msoa_shp.to_file(
    base_path / "inputs" / "processed_data" / "census" / "MSOA.gpkg",
    driver="GPKG",
    layer="MSOA_centroids_attr",
)
