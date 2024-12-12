import logging
import sys
from pathlib import Path

import geopandas
from nird.utils import load_config
from tqdm.auto import tqdm

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = None
    base_path = Path(load_config(config_path)["paths"]["base_path"])
    all_splits = geopandas.read_file(
        base_path / "processed_data" / "networks" / "GB_road_link_file_100k.gpkg"
    )
    for (index_i, index_j), df in tqdm(all_splits.groupby(["index_i", "index_j"])):
        df.to_parquet(
            base_path
            / "processed_data"
            / "networks"
            / "GB_road_link_file_100k.parquet"
            / f"{index_i}_{index_j}.parquet"
        )
