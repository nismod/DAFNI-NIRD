# %%
from pathlib import Path

import pandas as pd
import geopandas as gpd

from utils import load_config

base_path = Path(load_config()["paths"]["base_path"])

# %%
# LOWER TIER LOCAL AUTHORITY
lad_shp = gpd.read_file(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "Local_Authority_Districts_May_2022_UK"
    / "LAD_MAY_2022_UK_BFE_V3.shp"
)

lad_shp = lad_shp.rename(columns={"LAD22CD": "LAD_CODE"})

# %%
# working population of each LAD
wkpop = pd.read_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "wp001-workplace population"
    / "WP001_ltla.csv"
)
wkpop = wkpop.rename(
    columns={"Lower tier local authorities Code": "LAD_CODE", "Count": "WKPOP_2021"}
)
lad_shp = lad_shp.merge(wkpop, how="left", on="LAD_CODE")
# %%
# population traveling to work (by methods)
lad_commuter = pd.read_csv(
    base_path / "inputs" / "incoming_data" / "census" / "TS061-2021-4.csv"
)

lad_tt_commuter = lad_commuter.loc[
    lad_commuter.iloc[:, 2].isin([2, 3, 4, 5, 6, 7, 8, 11])
]
lad_tt_commuter = lad_tt_commuter.groupby(
    by=["Lower tier local authorities Code"], as_index=False
).agg({"Observation": sum})
lad_tt_commuter = lad_tt_commuter.rename(
    columns={
        "Lower tier local authorities Code": "LAD_CODE",
        "Observation": "total_commuters",
    }
)

lad_rd_commuter = lad_commuter.loc[lad_commuter.iloc[:, 2].isin([4, 5, 6, 7, 8, 11])]
lad_rd_commuter = lad_rd_commuter.groupby(
    by=["Lower tier local authorities Code"], as_index=False
).agg({"Observation": "sum"})
lad_rd_commuter = lad_rd_commuter.rename(
    columns={
        "Lower tier local authorities Code": "LAD_CODE",
        "Observation": "commuters by roads",
    }
)

lad_shp = lad_shp.merge(lad_tt_commuter, how="left", on="LAD_CODE")
lad_shp = lad_shp.merge(lad_rd_commuter, how="left", on="LAD_CODE")

# %%
# O-D matrix (INFLOW AND OUTFLOW)
od_file = pd.read_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "odwp01ew-location of usual residence and place of work"
    / "ODWP01EW_LTLA.csv"
)

od_file = od_file[od_file["Place of work indicator (4 categories) code"] == 3]

#!!! To delete population traveling within the origins
od_file = od_file[
    od_file["Lower tier local authorities code"] != od_file["LTLA of workplace code"]
]

lad_outflow = od_file.groupby(
    by=["Lower tier local authorities code"], as_index=False
).agg({"Count": sum})

lad_outflow = lad_outflow.rename(
    columns={"Lower tier local authorities code": "LAD_CODE", "Count": "Outflow_2021"}
)

lad_inflow = od_file.groupby(by=["LTLA of workplace code"], as_index=False).agg(
    {"Count": "sum"}
)
lad_inflow = lad_inflow.rename(
    columns={"LTLA of workplace code": "LAD_CODE", "Count": "Inflow_2021"}
)

lad_outflow_dict = lad_outflow.set_index("LAD_CODE")["Outflow_2021"].to_dict()
lad_inflow_dict = lad_inflow.set_index("LAD_CODE")["Inflow_2021"].to_dict()

lad_shp["Inflows"] = 0
lad_shp["Outflows"] = 0
lad_shp.Inflows = lad_shp["LAD_CODE"].map(lad_inflow_dict)
lad_shp.Outflows = lad_shp["LAD_CODE"].map(lad_outflow_dict)

# %%
lad_shp.to_file(
    base_path / "inputs" / "processed_data" / "census" / "LAD.gpkg",
    driver="GPKG",
    layer="LAD_inner_traveler_included",
)
