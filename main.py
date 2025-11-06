import os
import boto3
import pandas as pd
from io import BytesIO
import pyarrow.parquet as pq
from rapidfuzz import process, fuzz
import traceback
from fuzzywuzzy import fuzz, process
import pandas as pd

# -----------------------------
# Step 1: Load Data from S3
# -----------------------------
s3 = boto3.client("s3")
bucket_name = "ad-dl-dev-sandboxzone"

# Actual S3 keys (object paths)
csv_key = "LG/2025-03-05/lg_combined_data.csv"  # specify file name, not folder
parquet_key = "IEG/universal_visits/2025-03-05/ieg-ars-visits.parquet"

def read_csv_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"], sep="|", low_memory=False)
    except Exception as e:
        print(f"Error reading CSV from s3://{bucket}/{key}: {e}")
        return pd.DataFrame()

def read_parquet_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        buffer = BytesIO(obj["Body"].read())
        return pq.read_table(buffer).to_pandas()
    except Exception as e:
        print(f"Error reading Parquet from s3://{bucket}/{key}: {e}")
        return pd.DataFrame()

# Load data
lg_data = read_csv_from_s3(bucket_name, csv_key)
ieg_data = read_parquet_from_s3("ad-dl-prod-rawzone", parquet_key)

#main passenger and priority pass filter on ieg
ieg_data = ieg_data[(ieg_data['Card'] == 'Priority Pass') & (ieg_data['Type'] == 'Main Passenger')]

print(f"LG data loaded: {lg_data.shape}")
print(f"IEG data loaded: {ieg_data.shape}")

if lg_data.empty or ieg_data.empty:
    raise ValueError("❌ One or both datasets failed to load properly.")

# -----------------------------
# Step 2: Normalize Names
# -----------------------------
def normalize_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

ieg_data["Lounge_clean"] = ieg_data["Lounge"].apply(normalize_name)

if "Name" in lg_data.columns:
    lg_data["Lounge_clean"] = lg_data["Name"].apply(normalize_name)
elif "Lounge_Code" in lg_data.columns:
    lg_data["Lounge_clean"] = lg_data["Lounge_Code"].astype(str).apply(normalize_name)
else:
    raise KeyError("LG data missing Lounge_Name or Lounge_Code column.")

# -----------------------------
# Step 3: Find Common Lounges (Exact + Fuzzy)
# -----------------------------
print("\n🔍 Finding common lounges between IEG and LG...")

# --- Step 3a: Exact lounge code intersection ---
ieg_codes = (
    ieg_data["Lounge"].astype(str).str.extract(r"([A-Z]{3}\d*)", expand=False).dropna().unique()
)
lg_codes = lg_data["Lounge_Code"].astype(str).unique() if "Lounge_Code" in lg_data.columns else []
exact_common = set(lg_codes) & set(ieg_codes)

print(f"Exact common lounges found: {len(exact_common)} → {list(exact_common)[:10]}")

# --- Step 3b: Fuzzy match lounge names ---
print("\n🧠 Performing fuzzy match on lounge names (threshold ≥85)...")

fuzzy_matches = []
ieg_unique = ieg_data["Lounge_clean"].dropna().unique().tolist()
lg_unique = lg_data["Lounge_clean"].dropna().unique().tolist()

for ieg_name in ieg_unique:
    best_match = process.extractOne(ieg_name, lg_unique, scorer=fuzz.token_sort_ratio)
    if best_match and best_match[1] >= 85:
        fuzzy_matches.append((ieg_name, best_match[0], best_match[1]))

fuzzy_df = pd.DataFrame(fuzzy_matches, columns=["IEG_Lounge", "LG_Lounge", "Match_Score"])
print(f"fuzzy lounge matches found: {len(fuzzy_df)}")
print(fuzzy_df.head())

# Create mapping dictionary for later lookups
fuzzy_map = dict(zip(fuzzy_df["IEG_Lounge"], fuzzy_df["LG_Lounge"]))
ieg_data["Lounge_Matched"] = ieg_data["Lounge_clean"].apply(lambda x: fuzzy_map.get(x, x))

# --- Combine all common lounges ---
all_common = set(exact_common) | set(fuzzy_df["LG_Lounge"].unique())
print(f"\n🎯 Total common lounges (including fuzzy): {len(all_common)}")
print(list(all_common)[:15])

# -----------------------------
# Step 4: Loop Over Common Lounges and Filter
# -----------------------------
output_dir = "./filtered_lounges"
os.makedirs(output_dir, exist_ok=True)

print("\n🔁 Filtering both datasets for each common lounge...")
ieg_filtered_all = []
lg_filtered_all = []

for lounge_name in all_common:
    lounge_name_clean = normalize_name(lounge_name).upper()
    print(lounge_name_clean)

    if lounge_name_clean in fuzzy_map.values():
        ieg_tmp = ieg_data[ieg_data["Lounge_clean"] == lounge_name_clean].copy()
        lg_tmp = lg_data[lg_data["Lounge_clean"] == lounge_name_clean].copy()
    else:
        ieg_tmp = ieg_data[ieg_data["Lounge"].astype(str).str.contains(lounge_name_clean, na=False)].copy()
        lg_tmp = lg_data[lg_data["Lounge_Code"].astype(str).str.contains(lounge_name_clean, na=False)].copy()

    if ieg_tmp.empty and lg_tmp.empty:
        print("empty")
        continue

    print(f"➡️ Lounge: {lounge_name} | IEG: {len(ieg_tmp)} | LG: {len(lg_tmp)}")

    ieg_filtered_all.append(ieg_tmp)
    lg_filtered_all.append(lg_tmp)

# Combine all into one DataFrame
ieg_filtered = pd.concat(ieg_filtered_all, ignore_index=True)
lg_filtered = pd.concat(lg_filtered_all, ignore_index=True)

print(ieg_filtered.head())
print(lg_filtered.head())
print(f"Combined IEG filtered: {len(ieg_filtered)} rows")
print(f"Combined LG filtered: {len(lg_filtered)} rows")



def standardize_name(name):
    if pd.isna(name):
        return ""
    return (
        str(name)
        .strip()
        .lower()
        .replace("&", "and")
        .replace(".", "")
        .replace(",", "")
        .replace("  ", " ")
    )

# Step 1: Standardize both datasets
ieg_filtered = ieg_filtered.copy()
lg_filtered = lg_filtered.copy()
print('1')
print(ieg_filtered['Name'].value_counts())
print(lg_filtered['Name'].value_counts())
print('2')
import pandas as pd
from datetime import timedelta
from rapidfuzz import fuzz, process
import re

# -----------------------------
# Step 1: Normalize names
# -----------------------------
def normalize_name(name):
    """Normalize and reorder names like 'LAST/FIRST' → 'FIRST LAST'."""
    if pd.isna(name):
        return ""
    
    name = str(name).upper().strip()
    name = re.sub(r'[^A-Z\s/]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name)

    # Convert "LAST/FIRST" → "FIRST LAST"
    if '/' in name:
        parts = name.split('/')
        name = ' '.join(parts[::-1]).strip()

    return name.strip()

# -----------------------------
# Step 2: Define fuzzy matching function
# -----------------------------
def find_best_match(ieg_name, lg_names):
    """Find the best fuzzy match for an IEG name in LG dataset."""
    ieg_norm = normalize_name(ieg_name)
    lg_norms = [normalize_name(n) for n in lg_names]

    best_match = None
    best_score = 0

    for lg_name, lg_norm in zip(lg_names, lg_norms):
        score_set = fuzz.token_set_ratio(ieg_norm, lg_norm)
        score_sort = fuzz.token_sort_ratio(ieg_norm, lg_norm)
        score_partial = fuzz.partial_ratio(ieg_norm, lg_norm)
        composite_score = (score_set * 0.5) + (score_sort * 0.3) + (score_partial * 0.2)

        if composite_score > best_score:
            best_match = lg_name
            best_score = composite_score

    return best_match, best_score

# -----------------------------
# Step 3: Convert date + time to datetime
# -----------------------------
ieg_filtered['Entry_DateTime'] = pd.to_datetime(
    ieg_filtered['Entry_Date'].astype(str) + ' ' + ieg_filtered['Entry_Time'].astype(str),
    errors='coerce'
)

lg_filtered['Visit_DateTime'] = pd.to_datetime(
    lg_filtered['Visit_Date'].astype(str) + ' ' + lg_filtered['Visit_Time'].astype(str),
    errors='coerce'
)

# -----------------------------
# Step 4: Match IEG -> LG with ±15 min window
# -----------------------------
matches = []

for _, ieg_row in ieg_filtered.iterrows():
    ieg_name = ieg_row['Name']
    ieg_dt = ieg_row['Entry_DateTime']
    
    # Filter LG rows within ±15 minutes
    lg_candidates = lg_filtered[
        (lg_filtered['Visit_DateTime'] >= ieg_dt - timedelta(minutes=15)) &
        (lg_filtered['Visit_DateTime'] <= ieg_dt + timedelta(minutes=15))
    ]
    
    if not lg_candidates.empty:
        best_match, score = find_best_match(ieg_name, lg_candidates['Name'])
        
        if best_match and score >= 25:  # lenient threshold
            lg_row = lg_candidates.loc[lg_candidates['Name'] == best_match].iloc[0]
            matches.append({
                "IEG_Name": ieg_name,
                "LG_Name": best_match,
                "Match_Score": round(score, 1),
                "IEG_Entry_DateTime": ieg_dt,
                "LG_Visit_DateTime": lg_row['Visit_DateTime'],
                "Card_No": lg_row.get('Card_No.', None)
            })

# Final matched DataFrame
fuzzy_matches_df = pd.DataFrame(matches)


# Filter only high-confidence matches
fuzzy_matches_df = fuzzy_matches_df[fuzzy_matches_df["Match_Score"] > 62].reset_index(drop=True)

print("✅ Fuzzy matching with ±15 min window complete.")
print(f"Total matches found (score > 62): {len(fuzzy_matches_df)}")
print(fuzzy_matches_df.head(20))


# print("✅ Fuzzy matching with ±15 min window complete.")
# print(f"Total matches found: {len(fuzzy_matches_df)}")
# print(fuzzy_matches_df[
#     (fuzzy_matches_df['Match_Score'] >= 60) & 
#     (fuzzy_matches_df['Match_Score'] <= 70)
# ].head(20))



# --- Step 4: Find unmatched IEG names ---
matched_ieg_names = set(fuzzy_matches_df['IEG_Name'])
unmatched_ieg = ieg_filtered[~ieg_filtered['Name'].isin(matched_ieg_names)]

print("\n⚠️ Unmatched IEG names:")
print(unmatched_ieg[['Name']].head(50))
