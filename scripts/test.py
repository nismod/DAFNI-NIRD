from pathlib import Path
import pandas as pd
import geopandas as gpd

from nird.utils import load_config, get_flow_on_edges

base_path = Path(load_config()["paths"]["soge_clusters"])

od = pd.read_parquet(
    base_path.parent / "results" / "base_scenario" / "revision" / "odpfc.pq"
)
edge_flows = gpd.read_parquet(
    base_path.parent / "results" / "base_scenario" / "revision" / "edge_flows.gpq"
)

validation = get_flow_on_edges(od, "e_id", "path", "flow")
edge_flows = edge_flows.merge(validation, on="e_id", how="left")
edge_flows.to_parquet(
    base_path.parent / "results" / "base_scenario" / "revision" / "validation.gpq"
)
