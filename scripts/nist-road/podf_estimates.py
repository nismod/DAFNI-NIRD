# %%
import sys
import pandas as pd

from pathlib import Path
from nird.utils import load_config
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import logging
import warnings
import shutil
import zlib
from typing import Optional

warnings.simplefilter("ignore")
pd.set_option("display.max_columns", None)
try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


def get_paths():
    cfg = load_config()
    nist_path = Path(cfg["paths"]["NIST"])  # path to NIST folder
    out_path = nist_path.parent / "outputs"
    return nist_path, out_path


def load_odpfc(od: str):
    _, out_path = get_paths()
    odpfc_out_path = out_path / f"odpfc_{od}.pq"  # od_agg / od_agg_2050
    if not odpfc_out_path.exists():
        raise FileNotFoundError(f"{odpfc_out_path} does not exist.")
    return ds.dataset(odpfc_out_path, format="parquet")


def main(od: str, batch_size: int = 100_000, n_shards: int = 1024):
    dataset = load_odpfc(od)
    scanner = dataset.scanner(
        columns=["origin_node", "destination_node", "path", "flow"],
        batch_size=batch_size,
    )

    from collections import defaultdict

    # Prepare temporary shard directory
    _, out_path = get_paths()
    shard_dir = out_path / f"odpfc_{od}_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # During scanning: write per-batch aggregated rows into shard CSVs to avoid holding everything in memory
    for batch in tqdm(scanner.to_batches(), desc="Scanning batches", unit="batch"):
        df = batch.to_pandas()
        # Drop rows without a path
        df = df.dropna(subset=["path"]).copy()
        if df.empty:
            continue
        logging.info("processing batch, first path: %s", df.loc[0, "path"][0])
        # Convert string to list
        df["path"] = (
            df["path"]
            .str.strip("[]")
            .str.split(",")
            .apply(lambda x: [s.strip() for s in x])
        )
        # Explode the path column so each edge becomes one row and aggregate within the batch
        out_df = (
            df.explode("path", ignore_index=True)
            .groupby(["path", "origin_node", "destination_node"], as_index=False)[
                "flow"
            ]
            .sum()
        )

        if out_df.empty:
            continue

        # assign each row to a shard using crc32 on the key string
        keys = (
            out_df["path"].astype(str)
            + "|"
            + out_df["origin_node"].astype(str)
            + "|"
            + out_df["destination_node"].astype(str)
        )
        shards = keys.apply(lambda s: zlib.crc32(s.encode("utf-8")) % n_shards)
        out_df["_shard"] = shards

        # write each shard group to its CSV file (append)
        for sid, group in tqdm(
            out_df.groupby("_shard"), desc="Writing shards", leave=False
        ):
            file_path = shard_dir / f"shard_{sid}.csv"
            write_header = not file_path.exists()
            group.loc[:, ["path", "origin_node", "destination_node", "flow"]].to_csv(
                file_path, mode="a", header=write_header, index=False
            )

    # Now aggregate each shard independently and write to a single parquet file incrementally
    writer: Optional[pq.ParquetWriter] = None
    out_file = out_path / f"odpfc_{od}_expanded.parquet"

    shard_files = sorted(
        [p for p in shard_dir.iterdir() if p.name.startswith("shard_")]
    )
    if not shard_files:
        logging.info("No data to write for od=%s", od)
        return

    for shard_file in tqdm(shard_files, desc="Processing shards", unit="shard"):
        acc_shard = defaultdict(float)
        # read shard in chunks to avoid memory spikes
        for chunk in tqdm(
            pd.read_csv(shard_file, chunksize=100_000),
            desc=f"Reading {shard_file.name}",
            unit="chunk",
            leave=False,
        ):
            grp = chunk.groupby(
                ["path", "origin_node", "destination_node"], as_index=False
            )["flow"].sum()
            for r in grp.itertuples(index=False):
                acc_shard[(r.path, r.origin_node, r.destination_node)] += r.flow

        if not acc_shard:
            continue

        shard_df = pd.DataFrame(
            [(k[0], k[1], k[2], v) for k, v in acc_shard.items()],
            columns=["path", "origin_node", "destination_node", "flow"],
        )
        shard_df = shard_df.sort_values(
            by=["path", "origin_node", "destination_node"]
        ).reset_index(drop=True)

        table = pa.Table.from_pandas(shard_df, preserve_index=False)
        if writer is None:
            out_path.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(out_file, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    # cleanup shards
    try:
        shutil.rmtree(shard_dir)
    except Exception:
        logging.warning("Could not remove temporary shard directory %s", shard_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        od = sys.argv[1]
        main(od)
    except (IndexError, NameError):
        logging.info("Provide input parameters!")
