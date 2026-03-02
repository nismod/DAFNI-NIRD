# %%
import warnings
import sys
import logging
from pathlib import Path
from typing import List, Set, Tuple, Optional

import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.dataset as ds

from nird.utils import load_config

warnings.simplefilter("ignore")
tqdm.pandas()

macc_path = Path(load_config()["paths"]["MACCHUB"])
out_path = macc_path.parent / "outputs"
out_path.mkdir(parents=True, exist_ok=True)

# Default chunk size (number of rows per pyarrow record batch).
# Tune this based on your RAM. 50k-200k are common sensible defaults.
CHUNK_SIZE = 100_000


# %%
def find_flood_links(path: List, disrupted_set: Set) -> Optional[Tuple]:
    """Return disrupted edge ids along a path, or None if intact."""
    if not isinstance(path, (list, tuple, pd.Series)):
        return None
    common = disrupted_set.intersection(path)
    if not common:
        return None
    return tuple(common)


# %%
def process_chunk(df_chunk: pd.DataFrame, flood_links_set: Set) -> pd.DataFrame:
    """Take a dataframe chunk, convert path -> tokens, compute flood_links,
    and return only disrupted rows (with flood_links column)."""
    # Defensive: ensure path is string or list-like
    # Convert bracketed strings like "[roade_1, roade_2]" into token lists
    # Prefer regex extraction (fast and robust)
    df = df_chunk.copy()

    # if path column already lists for some rows, leave them untouched
    # apply regex only to strings
    mask_str = df["path"].apply(lambda x: isinstance(x, str))
    if mask_str.any():
        # use findall to extract tokens like roade_123
        df.loc[mask_str, "path"] = (
            df.loc[mask_str, "path"].str.findall(r"[\w_]+")
        )

    # now compute flood_links column
    df["flood_links"] = df["path"].apply(lambda p: find_flood_links(p, flood_links_set))

    # return only rows with disruption
    disrupted = df[df["flood_links"].notna()].reset_index(drop=True)
    return disrupted


# %%
def main(event_key: str, chunk_size: int = CHUNK_SIZE, consolidate: bool = False):
    """
    event_key: e.g. 'england'
    chunk_size: number of rows per pyarrow scanner record_batch
    consolidate: if True, read all part files and write single combined parquet at end
                 (may use more memory / disk I/O; off by default)
    """
    logging.info(f"Loading road links for event {event_key}...")
    road_links = gpd.read_parquet(
        macc_path / "damages" / "links" / f"road_links_{event_key}_future.gpq"
    )  # england, wales, and scotland
    flood_links = road_links.loc[road_links.flood_depth_max > 0, "e_id"]
    flood_links_set = set(flood_links.tolist())
    logging.info(f"Found {len(flood_links_set)} flooded links.")

    # Use pyarrow dataset to stream base_od in batches
    base_od_path = str(macc_path / "damages" / "od" / "odpfc_2050_ssp5.pq")
    dataset = ds.dataset(base_od_path, format="parquet")

    scanner = dataset.scanner(batch_size=chunk_size)
    batches = scanner.to_batches()  # generator

    part_files = []
    logging.info(f"Starting chunked processing with chunk_size={chunk_size} ...")
    for i, record_batch in enumerate(tqdm(batches, desc="parquet-batches")):
        # Convert to pandas DataFrame
        df_chunk = record_batch.to_pandas()
        # Process chunk
        disrupted = process_chunk(df_chunk, flood_links_set)

        if not disrupted.empty:
            part_file = out_path / f"odpfc_{event_key}_part{i:04d}.pq"
            # Use pandas to_parquet; parquet engine default is pyarrow
            disrupted.to_parquet(part_file, index=False)
            part_files.append(part_file)
            logging.info(f"Wrote {len(disrupted)} disrupted rows to {part_file.name}")
        else:
            logging.debug(f"No disrupted rows in chunk {i}")

    logging.info("Chunk processing complete.")

    if consolidate and part_files:
        logging.info("Consolidating part files into single parquet (this may use more memory)...")
        # Read each part and append to a list of dataframes (avoid reading all at once if huge)
        # We'll stream read and write into a single resultant parquet by concatenating in chunks
        combined_file = out_path / f"odpfc_{event_key}.pq"
        # Simple approach: read parts one by one and append to a single file by collecting and writing in one concat
        # If dataset is too large for memory, consider using pyarrow.parquet writer to append batches.
        df_list = []
        for pf in part_files:
            df_list.append(pd.read_parquet(pf))
        combined = pd.concat(df_list, ignore_index=True)
        combined.to_parquet(combined_file, index=False)
        logging.info(f"Wrote consolidated file {combined_file.name} ({len(combined)} rows).")
        # Optionally remove parts
        for pf in part_files:
            try:
                pf.unlink()
            except Exception:
                logging.warning(f"Could not remove temporary part file {pf}")
    elif part_files:
        logging.info(
            f"Finished. {len(part_files)} part files created in {out_path}. You can consolidate later if needed."
        )
    else:
        logging.info("No disrupted rows found; no output files produced.")


# %%
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        event_key = sys.argv[1]
        # Optionally accept chunk size and consolidate flag via CLI args
        # e.g. python script.py england 50000 True
        try:
            chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else CHUNK_SIZE
        except Exception:
            chunk_size = CHUNK_SIZE
        consolidate_flag = False
        if len(sys.argv) > 3:
            consolidate_flag = sys.argv[3].lower() in ("1", "true", "yes", "y")
        main(event_key, chunk_size=chunk_size, consolidate=consolidate_flag)
    except (IndexError, ValueError):
        logging.info("Usage: python script.py <event_key> [chunk_size] [consolidate]")
        logging.info("Example: python script.py england 100000 False")
        sys.exit(1)
