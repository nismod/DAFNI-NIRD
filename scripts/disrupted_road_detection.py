# %%
from pathlib import Path
from nird.utils import load_config
import geopandas as gpd
from snail import intersection, io
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["SRE_base_path"])

month = "May"  # update month: ["May", "June", "July"]
scenario = "FLRF"  # update scenario: ["FLSW", "FLRF"]

# %%
floodPath = (
    base_path
    / "inputs"
    / "incoming_data"
    / "12-14_2007 Summer_UK Floods"
    / f"{month}"
    / f"UK_2007_{month}_{scenario}_RD_5m_4326.tif"
)

roads = gpd.read_file(
    base_path / "inputs" / "processed_data" / f"{month}" / f"{scenario}_links.shp",
    engine="pyogrio",
)
# project roads to 4326
roads_prj = roads.to_crs("epsg:4326")

# read the raster data
flood_data = io.read_raster_band_data(floodPath)  # RD: depth (meter)

# run the intersection analysis
grid, bands = io.read_raster_metadata(floodPath)
prepared = intersection.prepare_linestrings(roads_prj)
flood_intersections = intersection.split_linestrings(prepared, grid)
flood_intersections = intersection.apply_indices(flood_intersections, grid)

flood_intersections["level_of_flood"] = intersection.get_raster_values_for_splits(
    flood_intersections, flood_data
)
# attach the maximum flood depth to each road link
# in case one road link intersects with multiple rasters
roads_flooddepth = flood_intersections.groupby(by=["id"], as_index=False).agg(
    {"level_of_flood": "max"}
)

# %%
roads_flooddepth.to_csv(
    base_path
    / "inputs"
    / "processed_data"
    / f"{month}"
    / f"{scenario}_flooded_links.csv",
    index=False,
)
