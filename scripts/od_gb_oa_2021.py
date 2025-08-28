"""OD data preparation
- Disaggregate OD matrix from multi-admin-layers to OA level
- Predict OD matrix from 2011 to 2021

Input OD matrix: Number of travel-to-work trips, UK, 2011, Multi-admins
Output OD matrix: Numnber of travel-to-work trips, GB, 2021, OA level.
"""

from pathlib import Path
import os
import pandas as pd
from collections import defaultdict
from nird.utils import load_config
import random
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])
base_path = base_path / "census_datasets" / "od_matrix"

# %%
# Scottish population (2011) at LSOA and OA levels
pop11_Scot = pd.read_csv(os.path.join(base_path, "pop11_scot.csv"))
pop11_Scot_lsoa = pop11_Scot.groupby("LSOA11CD", as_index=False).agg({"Popcount": sum})
pop11_Scot_oa = pop11_Scot[["OA11CD", "Popcount"]]

# Scottish population (2021) at LSOA level
pop21_Scot_lsoa = pd.read_excel(
    os.path.join(base_path, "sape-2021.xlsx"), sheet_name="Sheet1"
)
pop21_Scot_lsoa.rename(
    columns={"Data zone code": "LSOA21CD", "Total population": "Popcount"}, inplace=True
)

# (Population disaggregation) to estimate Scottish population (2021) at OA level
temp_pop = pd.merge(pop11_Scot, pop11_Scot_lsoa, on="LSOA11CD", how="left")
temp_pop["oa/lsoa"] = temp_pop.Popcount_x / temp_pop.Popcount_y
dict_ratio = defaultdict(lambda: defaultdict(list))
for idx, row in temp_pop.iterrows():
    oa = row["OA11CD"]
    lsoa = row["LSOA11CD"]
    rt = row["oa/lsoa"]
    dict_ratio[lsoa][oa] = rt

oaList = []
popList = []
for idx, row in pop21_Scot_lsoa.iterrows():
    lsoa = row["LSOA21CD"]
    pop = row["Popcount"]
    if lsoa in dict_ratio.keys():
        for oa, ratio in dict_ratio[lsoa].items():
            oaList.append(oa)
            popList.append(pop * ratio)

pop21_Scot_oa = pd.DataFrame({"OA21CD": oaList, "Popcount": popList})

# %%
# Read OD data at multiple admin levels
od11_gb = pd.read_csv(
    os.path.join(base_path, "od_gb_2011_multiAdminLevel.csv")
)  # source: WU03BUK_oa_wz_v4 (2011)

# Extract od origining from Scotland (2011)
od11_Scot_oa = od11_gb[od11_gb.from_region == "oa_scot"]
od11_Scot_oa.reset_index(drop=True, inplace=True)
od11_Scot_oa.rename(columns={"car": "travel11"}, inplace=True)

# Project OD from 2011 to 2021: travel21
pop11_scot_dict = pop11_Scot_oa.set_index("OA11CD")["Popcount"].to_dict()
pop21_scot_dict = pop21_Scot_oa.set_index("OA21CD")["Popcount"].to_dict()
od11_Scot_oa["Pop11"] = od11_Scot_oa["Area of usual residence"].map(pop11_scot_dict)
od11_Scot_oa["Pop21"] = od11_Scot_oa["Area of usual residence"].map(pop21_scot_dict)
od11_Scot_oa["travel21"] = (
    od11_Scot_oa.travel11 * od11_Scot_oa.Pop21 / od11_Scot_oa.Pop11
)
od11_Scot_oa["travel21"] = od11_Scot_oa["travel21"].round()  # round the number of trips
# -> pop11, pop21, travel11, travel21

# %%
# OD disaggregation
od11_Scot_oa = od11_Scot_oa[
    od11_Scot_oa["Area of usual residence"] != od11_Scot_oa["Area of workplace"]
]
od11_Scot_oa.reset_index(drop=True, inplace=True)

# Trip Classification
# within Scotland (oa-oa) -> keep
od11_scotoa_scotoa = od11_Scot_oa[od11_Scot_oa.to_region != "msoa_enw"]
od11_scotoa_scotoa.reset_index(drop=True, inplace=True)

# %%
# Extract OD from Scotland to ENW (oa-msoa) -> disaggregate
od11_scotoa_enwmsoa = od11_Scot_oa[od11_Scot_oa.to_region == "msoa_enw"]
od11_scotoa_enwmsoa.reset_index(drop=True, inplace=True)

# ENW 2011/21 population at OA level
pop11_enw_oa = pd.read_csv(os.path.join(base_path, "pop11_enw_oa.csv"))
pop11_enw_oa_dict = pop11_enw_oa.set_index("OA11CD")["POP11"].to_dict()
pop21_enw_oa = pd.read_csv(os.path.join(base_path, "gb_population_2021_estimates.csv"))
pop21_enw_oa_dict = pop21_enw_oa.set_index("OA21CD")["OA_POP_2021"].to_dict()

# from MSOA11 to OA11: MSOA11 contains multiple OA11
lut_enw_msoa11_oa11 = pd.read_csv(
    os.path.join(base_path, "lut\\_lut_enw_msoa11_oa11.csv")
)
msoa11_to_oa11_dict = (
    lut_enw_msoa11_oa11.groupby(by=["MSOA11CD"], as_index=False)
    .agg({"OA11CD": list})
    .set_index("MSOA11CD")["OA11CD"]
    .to_dict()
)
msoa11_oa11_pop11 = defaultdict(lambda: defaultdict(list))
for msoa11 in msoa11_to_oa11_dict.keys():
    for oa11 in msoa11_to_oa11_dict[msoa11]:
        pop11 = pop11_enw_oa_dict[oa11]
        msoa11_oa11_pop11[msoa11][oa11] = pop11  # 2011 OA population

# from OA11 to OA21: OA11 may contain multiple OA21
lut_enw_oa11_oa21 = pd.read_csv(os.path.join(base_path, "lut\\_lut_enw_oa11_oa21.csv"))
oa11_to_oa21_dict = (
    lut_enw_oa11_oa21.groupby(by=["OA11CD"]).agg({"OA21CD": list}).to_dict()["OA21CD"]
)
oa11_oa21_pop21 = defaultdict(lambda: defaultdict(list))
for oa11 in oa11_to_oa21_dict.keys():
    for oa21 in oa11_to_oa21_dict[oa11]:
        pop21 = pop21_enw_oa_dict[oa21]
        oa11_oa21_pop21[oa11][oa21] = pop21  # 2021 OA population

# %%
# Estimate msoa11_oa21_pop21
msoa11_oa21_pop21 = defaultdict(lambda: defaultdict(list))
for msoa11, list_of_oa11 in msoa11_oa11_pop11.items():
    for oa11 in list_of_oa11.keys():
        if oa11 in oa11_oa21_pop21.keys():
            for oa21, pop21 in oa11_oa21_pop21[oa11].items():
                msoa11_oa21_pop21[msoa11][oa21] = pop21
        else:
            print(f"We cannot find oa21 for {oa11}")

msoa11_oa21_ratio21 = defaultdict(lambda: defaultdict(list))
for msoa11, list_of_oa21 in msoa11_oa21_pop21.items():
    ttpop = sum(msoa11_oa21_pop21[msoa11].values())
    for oa21, pop21 in msoa11_oa21_pop21[msoa11].items():
        percentage = pop21 / ttpop
        msoa11_oa21_ratio21[msoa11][oa21] = percentage

msoa11_oa21_ratio21_norm = defaultdict(lambda: defaultdict(list))
for msoa11, list_of_oa21 in msoa11_oa21_ratio21.items():
    sum_ratio = sum(msoa11_oa21_ratio21[msoa11].values())
    for oa21, ratio in list_of_oa21.items():
        normalised_ratio = ratio / sum_ratio
        msoa11_oa21_ratio21_norm[msoa11][oa21] = normalised_ratio

# Disaggregate OD 2021 to OA level
assignment = {}  # scot[oa21] = travel21
for idx, row in od11_scotoa_enwmsoa.iterrows():  # each: 0 ~ 16; total: 8631
    scot11 = row["Area of usual residence"]
    msoa11 = row["Area of workplace"]
    list_of_oa21 = list(msoa11_oa21_ratio21_norm[msoa11].keys())
    # initialisation
    if scot11 not in assignment:
        assignment[scot11] = {oa21: 0 for oa21 in list_of_oa21}
    else:
        for oa21 in list_of_oa21:
            if oa21 not in assignment[scot11]:
                assignment[scot11][oa21] = 0
    # update
    travel21 = row["travel21"]
    list_of_prob21 = list(msoa11_oa21_ratio21_norm[msoa11].values())
    for _ in range(int(travel21)):
        selected_oa21 = random.choices(list_of_oa21, list_of_prob21)[0]
        assignment[scot11][selected_oa21] += 1

scot11List = []
oa21List = []
travel21List = []
for scot11, v in assignment.items():
    for oa21, travel21 in v.items():
        if travel21 > 0:
            scot11List.append(scot11)
            oa21List.append(oa21)
            travel21List.append(travel21)

temp_df = pd.DataFrame(
    {
        "Area of usual residence": scot11List,
        "Area of workplace": oa21List,
        "travel21": travel21List,
    }
)

temp_df = temp_df.groupby(
    by=["Area of usual residence", "Area of workplace"],
    as_index=False,
).agg({"travel21": sum})

# %%
# combine dataframes
# Scotland (from OA to OA)
temp_od11_scotoa_scotoa = od11_scotoa_scotoa[
    ["Area of usual residence", "Area of workplace", "travel21"]
]
od21_scot_oa = pd.concat([temp_df, temp_od11_scotoa_scotoa], axis=0, ignore_index=True)
od21_scot_oa.rename(columns={"travel21": "Car21"}, inplace=True)

# %%
# England and Wales (from OA to OA: real observation data)
od21_enw_oa = pd.read_csv(os.path.join(base_path, "od_enw_2021\\ODWP01EW_OA.csv"))
od21_enw_oa.rename(
    columns={
        "Output Areas code": "Area of usual residence",
        "OA of workplace code": "Area of workplace",
        "Count": "Car21",
    },
    inplace=True,
)
od21_enw_oa = od21_enw_oa[
    (od21_enw_oa["Place of work indicator (4 categories) code"] == 3)
    & (od21_enw_oa["Area of usual residence"] != od21_enw_oa["Area of workplace"])
]
od21_enw_oa = od21_enw_oa[["Area of usual residence", "Area of workplace", "Car21"]]

# remove North Irland
od21_enw_oa = od21_enw_oa[~od21_enw_oa["Area of workplace"].str.startswith("N")]
od21_enw_oa.reset_index(drop=True, inplace=True)

# %%
# Combined OD 2021 at OA level
od21_gb_oa = pd.concat([od21_scot_oa, od21_enw_oa], axis=0, ignore_index=True)

# %%
# export file
od21_gb_oa.to_csv(base_path / "od_gb_oa_2021.csv", index=False)
