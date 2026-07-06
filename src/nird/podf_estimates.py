# %%
from pathlib import Path
from nird.utils import load_config
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import logging
import warnings

warnings.simplefilter("ignore")
nist_path = Path(load_config()["paths"]["NIST"])
out_path = nist_path.parent / "outputs"


# %%
def load_odpfc(od: str):
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

        # Explode the path column so each edge becomes one row
        df = df.explode("path", ignore_index=True)

        def parse_edge(edge):
            if isinstance(edge, (tuple, list)) and len(edge) == 2:
                return edge[0], edge[1]
            if isinstance(edge, str) and "->" in edge:
                u, v = edge.split("->", 1)
                return u.strip(), v.strip()
            raise ValueError(f"Unsupported edge format: {edge!r}")

        edge_nodes = df["path"].apply(parse_edge)
        df["edge_origin"] = edge_nodes.apply(lambda x: x[0])
        df["edge_destination"] = edge_nodes.apply(lambda x: x[1])

        out_df = df[
            [
                "origin_node",
                "destination_node",
                "edge_origin",
                "edge_destination",
                "flow",
            ]
        ].copy()

        table = pa.Table.from_pandas(out_df, preserve_index=False)

        if writer is None:
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
