# %%
from pathlib import Path

import numpy as np
import pandas as pd

from collections import defaultdict

from utils import load_config

base_path = Path(load_config()["paths"]["base_path"])

# %%
test_flow = pd.read_csv(r"C:\Oxford\Research\DAFNI\local\outputs\od_lad_1104.csv")
od_file = pd.read_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "odwp01ew-location of usual residence and place of work"
    / "ODWP01EW_LTLA.csv"
)

od_file = od_file[od_file["Place of work indicator (4 categories) code"] == 3]
od_file.reset_index(drop=True, inplace=True)
# %%
od_dict = defaultdict(lambda: defaultdict(list))
for i in range(od_file.shape[0]):
    from_lad = od_file.loc[i, "Lower tier local authorities code"]
    to_lad = od_file.loc[i, "LTLA of workplace code"]
    count = od_file.loc[i, "Count"]
    od_dict[from_lad][to_lad] = count

test_flow["Counts"] = np.nan
for i in range(test_flow.shape[0]):
    from_lad = test_flow.loc[i, "from_lad"]
    to_lad = test_flow.loc[i, "to_lad"]
    if from_lad in od_dict.keys():
        if to_lad in od_dict[from_lad].keys():
            count = od_dict[from_lad][to_lad]
            test_flow.loc[i, "Counts"] = count

test_flow["error"] = test_flow.flux - test_flow.Counts

# %%
# interpretation
wkpop = pd.read_csv(
    base_path
    / "inputs"
    / "incoming_data"
    / "census"
    / "wp001-workplace population"
    / "WP001_ltla.csv"
)
wkpop = wkpop.rename(
    columns={"Lower tier local authorities Code": "LAD_CODE", "Count": "WKPOP_2021"}
)

wkpop_dict = wkpop.set_index("LAD_CODE")["WKPOP_2021"].to_dict()
test_flow["from_lad_wkpop"] = test_flow["from_lad"].map(wkpop_dict)
test_flow["to_lad_wkpop"] = test_flow["to_lad"].map(wkpop_dict)
