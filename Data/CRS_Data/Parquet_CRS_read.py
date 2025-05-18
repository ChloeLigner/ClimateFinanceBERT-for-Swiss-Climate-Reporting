# parquet downloaded from:
# https://data-explorer.oecd.org/vis?lc=en&fs[0]=Topic%2C1%7CDevelopment%23DEV%23%7COfficial%20Development%20Assistance%20%28ODA%29%23DEV_ODA%23&pg=0&fc=Topic&bp=true&snb=27&vw=ov&df[ds]=dsDisseminateFinalCloud&df[id]=DSD_CRS%40DF_CRS&df[ag]=OECD.DCD.FSD&df[vs]=1.4&dq=..1000.100._T._T.D.Q._T..&lom=LASTNPERIODS&lo=5&to[TIME_PERIOD]=false

import pandas as pd
import os

# === File paths ===
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
input_file = os.path.join(dir_path, "Data", "CRS_Data", "CRS.parquet")
output_file = os.path.join(dir_path, "Data", "CRS_Data", "CRS_Data_allyears.csv")

# Read the Parquet file
df = pd.read_parquet(input_file)

# Filter the data
filtered_df = df[
    (df['donor_code'] == 11)
]

# Save the filtered result to CSV
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to: {output_file}")
