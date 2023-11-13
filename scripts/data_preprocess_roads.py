# pip install pyogrio
# %%
import geopandas as gpd

"""
The data input is OS roads for the whole GB area;
This script is used for only extracting "major and minor" road links and nodes.

Major and Minor Roads include: A Road, B Road, and Motorway
"""
# edge processing
road_links = gpd.read_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\incoming_data\road\oproad_gpkg_gb\Data\oproad_gb.gpkg",
    layer="road_link",
    engine="pyogrio",
)
major_minor_road_links = road_links[
    road_links.road_classification.isin(["A Road", "B Road", "Motorway"])
]
major_minor_road_links.reset_index(drop=True, inplace=True)

# %%
major_minor_road_links.to_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\processed_data\road\MSOA.gpkg",
    driver="GPKG",
    layer="ABM_Road_Link",
)

# %%
# nodes processing
# delete the hanging nodes based on the filtered road links
road_nodes = gpd.read_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\incoming_data\road\oproad_gpkg_gb\Data\oproad_gb.gpkg",
    layer="road_node",
    engine="pyogrio",
)

nodeset = set(
    major_minor_road_links.start_node.tolist()
    + major_minor_road_links.end_node.tolist()
)
idxList = []
for i in range(road_nodes.shape[0]):
    node_id = road_nodes.loc[i, "id"]
    if node_id in nodeset:
        idxList.append(i)

major_minor_road_nodes = road_nodes[road_nodes.index.isin(idxList)]  # 167,903
major_minor_road_nodes.reset_index(drop=True, inplace=True)  # 167,860

# %%
major_minor_road_nodes.to_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\processed_data\road\MSOA.gpkg",
    driver="GPKG",
    layer="ABM_Road_Node",
)

# %%
# Clip out major and minor roads (links and nodes) located within England and Wales
england_polygon = gpd.read_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\incoming_data\census\bdline_essh_gb\Data\GB\english_region_region.shp"
)
wales_polygon = gpd.read_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\incoming_data\census\bdline_essh_gb\Data\GB\wales_region.shp"
)

polygons = gpd.GeoDataFrame(
    geometry=[
        [england_polygon.geometry.unary_union, wales_polygon.geometry.unary_union]
    ]
)
polygons = polygons.geometry.unary_union

# %%
# clip the nodes
road_nodes_enw = gpd.clip(major_minor_road_nodes, polygons)
road_nodes_enw = road_nodes_enw.reset_index(drop=True)

# deleted the hanging links
nodeset = set(road_nodes_enw.id.unique())
idxList = []
for i in range(major_minor_road_links.shape[0]):
    from_id = major_minor_road_links.loc[i, "start_node"]
    to_id = major_minor_road_links.loc[i, "end_node"]
    if (from_id in nodeset) and (to_id in nodeset):
        idxList.append(i)

roads_links_enw = major_minor_road_links[major_minor_road_links.index.isin(idxList)]
roads_links_enw.reset_index(drop=True, inplace=True)

# %%
# export road_nodes_enw and major_minor_roads_enw
road_nodes_enw.to_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\processed_data\road\MSOA.gpkg",
    driver="GPKG",
    layer="ABM_Road_Node_ENW",
)
roads_links_enw.to_file(
    r"C:\Oxford\Research\DAFNI\local\inputs\processed_data\road\MSOA.gpkg",
    driver="GPKG",
    layer="ABM_Road_Link_ENW",
)
