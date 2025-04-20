from pathlib import Path
import pandas as pd
from collections import defaultdict
from nird.utils import load_config

base_path = Path(load_config()["paths"]["base_path"])

# %%
# pre-event indirect costs
base = pd.read_excel(
    base_path.parent
    / "outputs"
    / "rerouting_analysis"
    / "20250302"
    / "cost_matrix"
    / "indirect_rerouting_cost.xlsx",
    sheet_name="base",
)
base_dict = base.set_index("id")["total"].to_dict()

# %%
indirect_dict = defaultdict(lambda: defaultdict(float))
indirect_15 = []
indirect_30 = []
indirect_60 = []
input_path = Path(
    base_path.parent / "outputs" / "rerouting_analysis" / "20250302" / "cost_matrix"
)
for scenario in [15, 30, 60]:
    for event in range(1, 20):
        if event in [2, 17]:
            continue
        path = input_path / str(scenario) / f"cost_{event}.csv"
        df = pd.read_csv(path)
        diff = df.total - base_dict[event]
        diff[diff < 0] = 0
        sum_diff = diff.sum()
        indirect_dict[scenario][event] = sum_diff
        if scenario == 15:
            indirect_15.append(sum_diff / 1e6)
        elif scenario == 30:
            indirect_30.append(sum_diff / 1e6)
        else:
            indirect_60.append(sum_diff / 1e6)

# %%
temp = pd.DataFrame(
    {
        "id": base.id,
        "event": base["Event Name"],
        "15": indirect_15,
        "30": indirect_30,
        "60": indirect_60,
    }
)
