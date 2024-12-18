# %%
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
from nird.utils import load_config


base_path = Path(load_config()["paths"]["base_path"])
tqdm.pandas()

od_df = pd.read_csv(
    os.path.join(base_path, "census_datasets", "od_matrix", "od_gb_oa_2021_node.csv")
)
od_df = od_df[od_df["Car21"] > 0]

od_df = od_df.sort_values(by=["Car21"], ascending=False)
od_df["car_total"] = od_df.groupby(["origin_node"])["Car21"].transform("sum")
od_df["car_cumsum"] = od_df.groupby(["origin_node"])["Car21"].transform("cumsum")
od_df["fraction"] = od_df["car_cumsum"] / od_df["car_total"]
od_df["count"] = od_df.groupby(["origin_node"])["origin_node"].transform("count")

before = od_df["Car21"].sum()
print("Before trunction...")
print(f"The total number of trips: {before} ")
print(f"The number of origins: {od_df.origin_node.unique().shape[0]}")
print(f"The number of destinations: {od_df.shape[0]}")

# od_df = od_df[(od_df["fraction"] < 0.7) | (od_df["count"] == 1)]

od_df = od_df[
    (od_df["Car21"] > 1)  # 32.6% (~1.79 millon)
    | (od_df["count"] == 1)  # 32.7% (~1.81 million)
    # | ((od_df["Car21"] == 1) & (od_df["fraction"] < 0.7))  # 69.2% (~7.74 million)
]

after = od_df["Car21"].sum()
print("After trunction...")
print(f"The total number of trips: {after} ")
print(f"The number of origins: {od_df.origin_node.unique().shape[0]}")
print(f"The number of destinations: {od_df.shape[0]}")

print(f"The number of trip reduction: {before - after}")
print(f"The percentage of trips retained: {1.0 * after / before}")
