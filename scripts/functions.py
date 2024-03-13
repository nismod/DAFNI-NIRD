# %%
from typing import Union

import numpy as np
import pandas as pd
import geopandas as gpd  # type: ignore

import constants as cons

import warnings

warnings.simplefilter("ignore")


# %%
# functions
# extract major roads
def select_partial_roads(
    road_links: gpd.GeoDataFrame,
    road_nodes: gpd.GeoDataFrame,
    col_name: str,
    list_of_values: list,
) -> gpd.GeoDataFrame:

    # road links selection
    for ci in list_of_values:
        road_links = road_links[road_links[col_name] == ci]
    selected_links = road_links.reset_index(drop=True)
    selected_links["e_id"] = road_links.id
    selected_links["from_id"] = road_links.start_node
    selected_links["to_id"] = road_links.end_node

    # road nodes selection
    sel_node_idx = list(
        set(list(road_links.start_node.to_list())) + list(road_links.end_node.tolist())
    )
    selected_nodes = road_nodes[road_nodes.id.isin(sel_node_idx)]
    selected_nodes.reset_index(drop=True, inplace=True)
    selected_nodes["nd_id"] = selected_nodes.id
    selected_nodes["lat"] = selected_nodes.geometry.y
    selected_nodes["lon"] = selected_nodes.geometry.x

    return selected_links, selected_nodes


# urban road classification
def create_urban_mask(etisplus_urban_roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buf_geom = etisplus_urban_roads.geometry.buffer(
        500
    )  # create a buffer of 500 meters
    uni_geom = buf_geom.unary_union  # feature union within the same layer
    temp = gpd.GeoDataFrame(geometry=[uni_geom])
    new_geom = (
        temp.explode()
    )  # explode multi-polygons into separate individual polygons
    cvx_geom = (
        new_geom.convex_hull
    )  # generate convex polygon for each individual polygon

    urban_mask = gpd.GeoDataFrame(
        geometry=cvx_geom[0], crs=etisplus_urban_roads.crs
    ).to_crs("27700")
    return urban_mask


def label_urban_roads(
    road_links: gpd.GeoDataFrame, urban_mask: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    road_links = road_links.sjoin(urban_mask, how="left")
    road_links["urban"] = road_links["index_right"].apply(
        lambda x: 0 if pd.isna(x) else 1
    )
    road_links = road_links.drop(columns=["index_right", "FID_1"])
    return road_links


def voc_func(speed: float) -> float:  # speed: mile/hour
    # d = distance * conv_mile_to_km  # km
    s = speed * cons.CONV_MILE_TO_KM  # km/hour
    lpkm = 0.178 - 0.00299 * s + 0.0000205 * (s**2)  # fuel cost (liter/km)
    c = 140 * lpkm * cons.PENCE_TO_POUND  # average petrol cost: 140 pence/liter
    return c  # pound/km


def cost_func(
    time: float, distance: float, voc: float
) -> float:  # time: hour, distance: mile/hour, voc: pound/km
    ave_occ = 1.6
    vot = 20  # value of time: pounds/hour
    d = distance * cons.CONV_MILE_TO_KM  # km
    t = time + d * voc / (ave_occ * vot)
    return t  # hour


# speed functions
def initial_speed_func(
    road_type: str, form_of_road: str, free_flow_speed_dict: dict
) -> Union[float, None]:
    if road_type == "M":
        return free_flow_speed_dict["M"]
    elif road_type == "A":
        if form_of_road == "Single Carriageway":
            return free_flow_speed_dict["A_single"]
        else:
            return free_flow_speed_dict["A_dual"]
    elif road_type == "B":
        return free_flow_speed_dict["B"]
    else:
        print("Error: initial speed!")
        return None


def speed_flow_func(
    road_type: str,
    form_of_road: str,
    isurban: int,
    vp: float,
    free_flow_speed_dict: dict,
    flow_breakpoint_dict: dict,
    min_speed_cap: dict,
) -> Union[float, None]:
    vp = vp / 24
    if road_type == "M":
        initial_speed = free_flow_speed_dict["M"]
        if vp > flow_breakpoint_dict["M"]:  # speed starts to decrease
            vt = max(
                (initial_speed - 0.033 * (vp - flow_breakpoint_dict["M"])),
                min_speed_cap["M"],
            )
            if isurban:
                return min(47.0, vt)
            else:
                return vt
        else:
            if isurban:
                return min(47.0, initial_speed)
            else:
                return initial_speed
    elif road_type == "A":
        if form_of_road == "Single Carriageway":
            initial_speed = free_flow_speed_dict["A_single"]
            if vp > flow_breakpoint_dict["A_single"]:
                vt = max(
                    (initial_speed - 0.05 * (vp - flow_breakpoint_dict["A_single"])),
                    min_speed_cap["A_single"],
                )
                if isurban:
                    return min(30.0, vt)
                else:
                    return vt
            else:
                if isurban:
                    return min(30.0, initial_speed)
                else:
                    return initial_speed
        else:
            initial_speed = free_flow_speed_dict["A_dual"]
            if vp > flow_breakpoint_dict["A_dual"]:
                vt = max(
                    (initial_speed - 0.033 * (vp - flow_breakpoint_dict["A_dual"])),
                    min_speed_cap["A_dual"],
                )
                if isurban:
                    return min(30.0, vt)
                else:
                    return vt
            else:
                if isurban:
                    return min(30.0, initial_speed)
                else:
                    return initial_speed
    elif road_type == "B":
        initial_speed = free_flow_speed_dict["B"]
        if isurban:
            return min(30.0, initial_speed)
        else:
            return initial_speed
    else:
        print("Please select the road type from [M, A, B]!")
        return None


def filter_less_than_one(arr: np.ndarray) -> np.ndarray:
    return np.where(arr >= 1, arr, 0)
