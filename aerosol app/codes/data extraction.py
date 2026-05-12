import pandas as pd
import numpy as np
from calendar import monthrange

# === CONFIG ===
excel_path = "maqs-SMPS-CPC year 2022.xlsx"
output_csv = "lognormal_parameters_2022.csv"

# === READ ALL SHEETS ===
xls = pd.ExcelFile(excel_path)
all_results = []

print("=== DAILY ESTIMATED LOGNORMAL PARAMETERS ===")
print(f"{'Date':>10} | {'μ(logD)':>10} | {'σ(logD)':>10}")
print("-" * 40)

for sheet_name in xls.sheet_names:

    df = pd.read_excel(xls, sheet_name=sheet_name)

    # ensure datetime column
    if "datetime" not in df.columns:
        df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])

    # infer year, month from sheet name, e.g. Jan-2022
    year = int(sheet_name[-4:])
    month = pd.to_datetime(sheet_name[:3], format="%b").month

    # size bins: 106 SMPS channels in your Excel file
    size_cols = df.columns[1:107].astype(float)
    size_nm = size_cols.values
    logD = np.log(size_nm)

    days_in_month = monthrange(year, month)[1]
    dates = [
        f"{year}-{month:02d}-{day:02d}"
        for day in range(1, days_in_month + 1)
    ]

    # daily loop
    for day in dates:

        df_day = df[df["datetime"].dt.date.astype(str) == day]

        if df_day.empty:
            continue

        # daily mean distribution over all 106 SMPS channels
        mean_conc = df_day.iloc[:, 1:107].mean().values

        # avoid bad days with zero or missing total concentration
        total = np.nansum(mean_conc)

        if total <= 0 or np.isnan(total):
            continue

        # empirical normalized distribution
        pdf = mean_conc / total

        # numerical lognormal parameters
        mu = np.sum(pdf * logD)
        sigma = np.sqrt(np.sum(pdf * (logD - mu) ** 2))

        print(f"{day} | {mu:10.4f} | {sigma:10.4f}")

        all_results.append({
            "Date": day,
            "mu_log": mu,
            "sigma_log": sigma
        })

# === SAVE RESULTS ===
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(output_csv, index=False)

print(f"\n✅ Saved: {output_csv}")
print("Rows:", len(summary_df))
print("Columns:", list(summary_df.columns))
