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

warnings.simplefilter("ignore")
pd.set_option("display.max_columns", None)


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


def main(od: str, batch_size: int = 100_000):
    dataset = load_odpfc(od)
    scanner = dataset.scanner(
        columns=["origin_node", "destination_node", "path", "flow"],
        batch_size=batch_size,
    )
    writer = None

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        # Drop rows without a path
        df = df.dropna(subset=["path"]).copy()
        logging.info("od", df.loc[0, "path"][0])
        # Convert string to list
        df["path"] = (
            df["path"]
            .str.strip("[]")
            .str.split(",")
            .apply(lambda x: [s.strip() for s in x])
        )
        logging.info("od", df.loc[0, "path"][0])
        # Explode the path column so each edge becomes one row
        out_df = (
            df.explode("path", ignore_index=True)
            .groupby(["path", "origin_node", "destination_node"], as_index=False)[
                "flow"
            ]
            .sum()
            .reset_index(drop=True)
        )
        logging.info("gp", out_df.head(5))
        table = pa.Table.from_pandas(out_df, preserve_index=False)

        if writer is None:
            # ensure output directory exists
            _, out_path = get_paths()
            out_path.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(
                out_path / f"odpfc_{od}_expanded.parquet", table.schema
            )

        writer.write_table(table)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        od = sys.argv[1]
        main(od)
    except (IndexError, NameError):
        logging.info("Provide input parameters!")
