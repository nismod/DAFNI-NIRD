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
base_path = Path(load_config()["paths"]["soge_clusters"])
tqdm.pandas()


# %%
def find_flood_links(path: List, disrupted_set: Set) -> Optional[Tuple]:
    """Return disrupted edge ids along a path, or None if intact."""
    common = disrupted_set.intersection(path)
    if not common:
        return None
    return tuple(common)


# %%
base_od = pd.read_parquet(
    base_path.parent / "results" / "base_scenario" / "revision" / "odpfc.pq"
)
out_path = base_path.parent / "results" / "disruption_analysis" / "revision" / "od"
out_path.mkdir(parents=True, exist_ok=True)


# %%
def main(depth_key):
    for event_key in range(18, 20):  # for events 18-19
        # if event_key not in [18, 19]:
        #     continue
        road_links = gpd.read_parquet(
            base_path.parent
            / "results"
            / "disruption_analysis"
            / "revision"
            / str(depth_key)
            / "links"
            / f"road_links_{event_key}.gpq"
        )
        logging.info(f"Extracting od for depth {depth_key} and event {event_key}...")
        flood_links = road_links.loc[road_links.flood_depth_max > 0, "e_id"]
        flood_links_set = set(flood_links)
        base_od["flood_links"] = base_od.path.progress_apply(
            lambda path: find_flood_links(path, flood_links_set)
        )
        disrupted_mask = base_od["flood_links"].notna()
        disrupted_candidates = base_od.loc[disrupted_mask].reset_index(drop=True)
        disrupted_candidates.to_parquet(out_path / f"odpfc_{depth_key}_{event_key}.pq")

    logging.info("Completed!")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        depth_key = sys.argv[1]
        main(depth_key)
    except (IndexError, ValueError):
        logging.info("provide depth_key!")
        sys.exit(1)
