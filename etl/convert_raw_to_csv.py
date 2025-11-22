import pandas as pd
import numpy as np
import re

INPUT_FILE  = "./features_raw.xlsx"
OUTPUT_FILE = "features.csv"


def clean_numeric_column(series):
    s = series.astype(str)
    s = s.str.strip()
    s = s.str.replace('"', '', regex=False)
    s = s.str.replace(r'\.0$', '', regex=True)
    s = s.str.replace(",", ".", regex=False)
    cleaned = pd.to_numeric(s, errors="coerce")

    return cleaned

def clean_id(series):
    s = series.astype(str)
    s = s.str.strip()
    s = s.str.replace(r"[^0-9\.,\-]", "", regex=True)
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace("[,.]", "", regex=True)

    cleaned = pd.to_numeric(s, errors="coerce").astype("Int64")

    return cleaned


def main():
    if INPUT_FILE.endswith(".xlsx") or INPUT_FILE.endswith(".xls"):
        df = pd.read_excel(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE)

    df = df.iloc[1:].reset_index(drop=True)

    if "cst_dim_id" in df.columns:
        df["cst_dim_id"] = clean_id(df["cst_dim_id"])

    for col in df.columns:
        if col == "cst_dim_id":
            continue

        col_data = df[col].astype(str)

        numeric_like = col_data.str.contains(r'^[\d\s\.,\-]+$', regex=True)
        if numeric_like.mean() > 0.3:
            df[col] = clean_numeric_column(df[col])

    df.to_csv(OUTPUT_FILE, index=False)



if __name__ == "__main__":
    main()
