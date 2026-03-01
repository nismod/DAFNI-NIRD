# %%
import warnings
import sys

from pathlib import Path
import logging
from typing import List, Set, Tuple, Optional

import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from nird.utils import load_config

warnings.simplefilter("ignore")
macc_path = Path(load_config()["paths"]["MACCHUB"])
tqdm.pandas()


# %%
def find_flood_links(path: List, disrupted_set: Set) -> Optional[Tuple]:
    """Return disrupted edge ids along a path, or None if intact."""
    common = disrupted_set.intersection(path)
    if not common:
        return None
    return tuple(common)


# %%
base_od = pd.read_parquet(macc_path / "damages" / "od" / "odpfc_2050_ssp5.pq")
out_path = macc_path.parent / "outputs"
out_path.mkdir(parents=True, exist_ok=True)


# %%
def main(event_key):
    road_links = gpd.read_parquet(
        macc_path / "damages" / "links" / f"road_links_{event_key}_future.gpq"
    )  # england, wales, and scotland
    logging.info(f"Extracting od for event {event_key}...")
    flood_links = road_links.loc[road_links.flood_depth_max > 0, "e_id"]
    flood_links_set = set(flood_links)
    base_od["flood_links"] = base_od.path.progress_apply(
        lambda path: find_flood_links(path, flood_links_set)
    )
    disrupted_mask = base_od["flood_links"].notna()
    disrupted_candidates = base_od.loc[disrupted_mask].reset_index(drop=True)
    disrupted_candidates.to_parquet(out_path / f"odpfc_{event_key}.pq")

    logging.info("Completed!")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        event_key = sys.argv[1]
        main(event_key)
    except (IndexError, ValueError):
        logging.info("Please provide event_key!")
        sys.exit(1)
