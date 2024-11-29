# %%
from pathlib import Path
from nird.utils import load_config
import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])

# %%
capt_minor = []
capt_moderate = []
capt_extensive = []
capt_severe = []
minor = 0
moderate = 0
extensive = 0
severe = 0
for d in range(1, 111):  # 1-110
    if d <= 9:
        capt_minor.append(minor)
        capt_moderate.append(moderate)
        capt_extensive.append(extensive)
        capt_severe.append(severe)
    elif 9 < d <= 15:
        minor += 0.167
        capt_minor.append(minor)
        capt_moderate.append(moderate)
        capt_extensive.append(extensive)
        capt_severe.append(severe)
    elif 15 < d <= 20:
        capt_minor.append(minor)
        capt_moderate.append(moderate)
        capt_extensive.append(extensive)
        capt_severe.append(severe)
    elif 20 < d <= 35:
        moderate += 0.04
        capt_minor.append(minor)
        capt_moderate.append(moderate)
        capt_extensive.append(extensive)
        capt_severe.append(severe)
    elif 35 < d <= 45:
        moderate += 0.04
        extensive += 0.0263
        capt_minor.append(minor)
        capt_moderate.append(moderate)
        capt_extensive.append(extensive)
        capt_severe.append(severe)
    elif 45 < d <= 50:
        extensive += 0.0263
        capt_minor.append(minor)
        capt_moderate.append(moderate)
        capt_extensive.append(extensive)
        capt_severe.append(severe)
    elif 50 < d <= 73:
        extensive += 0.0263
        severe += 0.0167
        capt_minor.append(minor)
        capt_moderate.append(moderate)
        capt_extensive.append(extensive)
        capt_severe.append(severe)
    else:  # 73 - 110
        severe += 0.0167
        capt_minor.append(minor)
        capt_moderate.append(moderate)
        capt_extensive.append(extensive)
        capt_severe.append(severe)


capt_minor = [round(i, 2) for i in capt_minor]
capt_moderate = [round(i, 2) for i in capt_moderate]
capt_extensive = [round(i, 2) for i in capt_extensive]
capt_severe = [round(i, 2) for i in capt_severe]

# %%
with open(base_path / "parameters" / "capt_minor.json", "w") as f:
    json.dump(capt_minor, f)
with open(base_path / "parameters" / "capt_moderate.json", "w") as f:
    json.dump(capt_moderate, f)
with open(base_path / "parameters" / "capt_extensive.json", "w") as f:
    json.dump(capt_extensive, f)
with open(base_path / "parameters" / "capt_severe.json", "w") as f:
    json.dump(capt_severe, f)
