from pathlib import Path

import pandas as pd
import geopandas as gpd

from utils import load_config

# network attributes
# total population (P)
# traveller (T)
# employment (E)

base_path = Path(load_config()["paths"]["base_path"])

# LAD (local authority districts)
local_authority_file = gpd.read_file(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "Local_Authority_Districts_May_2022_UK_BFE_V3_2022_3331011932393166417"
    / "LAD_MAY_2022_UK_BFE_V3.shp"
)
local_authority_file["area_m2"] = local_authority_file.geometry.area
la_population = pd.read_excel(
    base_path / "inputs" / "incoming_data" / "census" / "population_density.xlsx"
)
la_commuter = pd.read_excel(
    base_path / "inputs" / "incoming_data" / "census" / "commuter_england_wales.xlsx"
)
la_employment = pd.read_excel(
    base_path / "inputs" / "incoming_data" / "census" / "employment_england_wales.xlsx"
)
df_attr = local_authority_file[["LAD22CD", "area_m2", "geometry"]]
df_attr = df_attr.rename(columns={"LAD22CD": "Area code"})
df_attr = df_attr.merge(la_population, on="Area code")
df_attr = df_attr.merge(la_commuter, on="Area code")
df_attr = df_attr.merge(la_employment, on="Area code")
df_attr = df_attr[
    [
        "Area code",
        "Area name_x",
        "geometry",
        "area_m2",
        "Population_2021",
        "Commuters",
        "Employment_2021",
    ]
].rename(
    columns={
        "Area code": "LAD_code",
        "Area name_x": "LAD_name",
        "area_m2": "LAD_area_m2",
        "Commuters": "Commuters_2021",
        "Population_2021": "Population_Density_2021",
    }
)
df_attr["Population_2021"] = (
    df_attr.LAD_area_m2 * df_attr.Population_Density_2021 * 1e-6
)
df_attr2 = df_attr.drop(columns=["geometry", "Population_Density_2021"])  # dataframe

"""
df_attr.to_file(
    base_path / "inputs" / "processed_data" / "census" / "admin_eng_wales.gpkg",
    driver="GPKG",
    layer="LAD",
)
df_attr2.to_excel(
    base_path / "inputs" / "processed_data" / "census" / "lad_attr.xlsx", index=False
)
"""

# OA (output area, smallest administration level)
# look-up-table: from oa to lad
lut_file = pd.read_excel(
    base_path / "inputs" / "processed_data" / "census" / "lut.xlsx"
)
oa_to_lad = {}
for i in range(lut_file.shape[0]):
    oa = lut_file.loc[i, "OA11CD"]
    lad = lut_file.loc[i, "LAD21CD"]
    oa_to_lad[str(oa)] = str(lad)

oa_london = gpd.read_file(
    base_path / "inputs" / "processed_data" / "road" / "cliped_file.gpkg",
    layer="oa_pop_cliped",
)
oa_london["tt_pop_oa"] = (
    oa_london.geometry.area * oa_london.pop_den * 1e-6
)  # area: m2; pop_den: person/km2
oa_london["area_m2_oa"] = oa_london.geometry.area
oa_london["OA_code"] = oa_london.geo_code
oa_london["LAD_code"] = oa_london.OA_code.apply(lambda x: oa_to_lad.get(x))
df_attr_oa = oa_london[["OA_code", "geometry", "tt_pop_oa", "area_m2_oa", "LAD_code"]]
df_attr_oa = df_attr_oa.merge(df_attr2, on="LAD_code")
df_attr_oa2 = df_attr_oa.drop(columns=["geometry"])  # dataframe

"""
df_attr_oa.to_file(
    base_path / "inputs" / "processed_data" / "census" / "london.gpkg",
    driver="GPKG",
    layer="OA",
)
df_attr_oa2.to_excel(
    base_path / "inputs" / "processed_data" / "census" / "oa_attr.xlsx", index=False
)
"""

# %%
# estimate employment and travellers at the OA level
df_attr_oa["tt_pop_oa_hat"] = (
    df_attr_oa.area_m2_oa / df_attr_oa.LAD_area_m2 * df_attr_oa.Population_2021
)
df_attr_oa["delta"] = df_attr_oa.tt_pop_oa - df_attr_oa.tt_pop_oa_hat
df_attr_oa["employment_oa"] = (
    df_attr_oa.area_m2_oa / df_attr_oa.LAD_area_m2 * df_attr_oa.Employment_2021
    + df_attr_oa.delta
)
df_attr_oa["outflow_oa"] = (
    df_attr_oa.area_m2_oa / df_attr_oa.LAD_area_m2 * df_attr_oa.Commuters_2021
    + df_attr_oa.delta
)
df_attr_oa.loc[df_attr_oa[df_attr_oa.employment_oa < 0].index, "employment_oa"] = 0
df_attr_oa.loc[df_attr_oa[df_attr_oa.outflow_oa < 0].index, "outflow_oa"] = 0

"""
df_attr_oa.to_file(
    base_path / "inputs" / "processed_data" / "census" / "london.gpkg",
    driver="GPKG",
    layer="OA",
)
"""
