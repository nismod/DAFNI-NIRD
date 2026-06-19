"""This is to predict PA in 2030 and 2050"""

# %%
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# %%
base_dir = Path(r"C:\Oxford\Research\NIST\DfT Model")

# regression analysis for 2030 and 2050 production (total outflows) and attraction (total inflows) estimates
# Read Excel file
df = pd.read_csv(
    r"C:\Oxford\Research\NIST\DfT Model\processed_data\highs\high_productions.csv"
)
# Extract year columns
year_cols = [col for col in df.columns if col.startswith("YR")]

# Convert column names to numeric years
years = np.array([int(col.replace("YR", "")) for col in year_cols]).reshape(-1, 1)

# Lists to store predictions
pred_2030 = []
pred_2050 = []

# Fit regression for each row
for _, row in df.iterrows():
    y = row[year_cols].values.astype(float)

    # Handle missing values
    mask = ~np.isnan(y)

    model = LinearRegression()
    model.fit(years[mask], y[mask])

    pred_2030.append(model.predict([[2030]])[0])
    pred_2050.append(model.predict([[2050]])[0])

# Add predictions to dataframe
df["YR2030"] = pred_2030
df["YR2050"] = pred_2050

df.to_csv(
    r"C:\Oxford\Research\NIST\DfT Model\processed_data\highs\high_productions_with_predictions.csv",
    index=False,
)
