# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd

pd.set_option("display.max_columns", None)

base_path = Path(r"C:\Oxford\Research\DAFNI\local\outputs\disruption_analysis\revision")
for depth_key in [15, 30, 60]:
    # intersections
    intersect_path = base_path / f"{depth_key}" / "intersections"
    for file in intersect_path.iterdir():
        if file.is_file():
            print("Reading:", file)
            intersection = pd.read_parquet(file)
            if {"flood_depth_river", "damage_level_river"}.issubset(
                intersection.columns
            ):
                print("Find RIVER flood-related damages...")
                intersection.loc[
                    intersection.flood_depth_river < 0, "damage_level_river"
                ] = "no"
            if {"flood_depth_surface", "damage_level_surface"}.issubset(
                intersection.columns
            ):
                print("Find SURFACE flood-related damages...")
                intersection.loc[
                    intersection.flood_depth_surface < 0, "damage_level_surface"
                ] = "no"
            intersection.to_parquet(file)

    # road links
    road_path = base_path / f"{depth_key}" / "links"
    for file in road_path.iterdir():
        print("Reading:", file)
        road = gpd.read_parquet(file)
        road.loc[road.flood_depth_max < 0, "damage_level_max"] = "no"
        road.to_parquet(file)
