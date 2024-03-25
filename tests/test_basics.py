"""Initial test file
"""

from geopandas import GeoDataFrame
from typing import List
from pandas.testing import assert_frame_equal
from shapely.geometry import Point, LineString

from nird.road import select_partial_roads


def test_select_partial_roads():
    # Set up inputs
    a = Point((0, 0))
    b = Point((0, 2))
    c = Point((0, 0))
    d = Point((0, 2))
    ab = LineString([a, b])
    cd = LineString([c, d])
    nodes = GeoDataFrame(geometry=[a, b, c, d], data={"id": ["a", "b", "c", "d"]})
    links = GeoDataFrame(
        geometry=[ab, cd],
        data={
            "id": ["ab", "cd"],
            "road_class": ["A", "B"],
            "start_node": ["a", "c"],
            "end_node": ["b", "d"],
        },
    )
    col_name: str = "road_class"
    list_of_values: List[str] = ["A"]

    # Run function to test
    actual_links, actual_nodes = select_partial_roads(
        links, nodes, col_name, list_of_values
    )

    # Check against expected outputs
    expected_nodes = GeoDataFrame(
        data={
            "id": ["a", "b"],
            "nd_id": ["a", "b"],
            "lat": [0.0, 2.0],
            "lon": [0.0, 0.0],
        },
        geometry=[a, b],
    )[["id", "geometry", "nd_id", "lat", "lon"]]

    expected_links = GeoDataFrame(
        geometry=[ab],
        data={
            "id": ["ab"],
            "e_id": ["ab"],
            "road_class": ["A"],
            "start_node": ["a"],
            "from_id": ["a"],
            "end_node": ["b"],
            "to_id": ["b"],
        },
    )[
        [
            "id",
            "road_class",
            "start_node",
            "end_node",
            "geometry",
            "e_id",
            "from_id",
            "to_id",
        ]
    ]

    assert_frame_equal(actual_nodes, expected_nodes)
    assert_frame_equal(actual_links, expected_links)
