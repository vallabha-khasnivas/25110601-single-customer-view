import sys
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
from datetime import timedelta
from rapidfuzz import fuzz
import re
import logging
import watchtower
from awsglue.utils import getResolvedOptions

# -----------------------------
# Glue Parameters
# -----------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
region_name = 'us-east-1'
log_group_name = '/AD-DL-Prod-ARS-Logs'
log_stream_name = "ad-dl-prod-glue-etl"

# -----------------------------
# Logging Setup (CloudWatch + Console)
# -----------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console log output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# CloudWatch logs
boto3_client = boto3.client('logs', region_name=region_name)
cw_handler = watchtower.CloudWatchLogHandler(
    log_group=log_group_name,
    stream_name=log_stream_name,
    boto3_client=boto3_client
)
cw_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(cw_handler)

logger.info(f"🚀 Starting Glue Job: {args['JOB_NAME']}")

# -----------------------------
# Step 1: S3 Setup & Load Data
# -----------------------------
s3 = boto3.client("s3")

bucket_ieg = "ad-dl-prod-rawzone"
bucket_lg = "ad-dl-dev-sandboxzone"

parquet_key_ieg = "IEG/universal_visits/2025-03-05/ieg-ars-visits.parquet"
# parquet_key_lg = "LG-Historical/LG_2024_2025.parquet"
parquet_key_lg="LG/2025-03-05/LG_Combined.parquet"
def read_parquet_from_s3(bucket, key):
    try:
        logger.info(f"📥 Reading Parquet from s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        buffer = BytesIO(obj["Body"].read())
        return pq.read_table(buffer).to_pandas()
    except Exception as e:
        logger.error(f"❌ Error reading {bucket}/{key}: {e}")
        return pd.DataFrame()

lg_data = read_parquet_from_s3(bucket_lg, parquet_key_lg)
ieg_data = read_parquet_from_s3(bucket_ieg, parquet_key_ieg)

logger.info(f"✅ LG data loaded: {lg_data.shape}")
logger.info(f"✅ IEG data loaded: {ieg_data.shape}")

if lg_data.empty or ieg_data.empty:
    logger.error("❌ One or both datasets failed to load properly.")
    sys.exit(1)

# -----------------------------
# Step 2: Clean and Normalize
# -----------------------------
def normalize_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

ieg_data = ieg_data[(ieg_data['Card'] == 'Priority Pass') & (ieg_data['Type'] == 'Main Passenger')]
ieg_data["Lounge_clean"] = ieg_data["Lounge"].apply(normalize_name)

if "airport_code" in lg_data.columns and "Lounge_Code" not in lg_data.columns:
    lg_data.rename(columns={"airport_code": "Lounge_Code"}, inplace=True)
if "name" in lg_data.columns and "Name" not in lg_data.columns:
    lg_data.rename(columns={"name": "Name"}, inplace=True)

if "Lounge_Code" in lg_data.columns:
    lg_data["Lounge_clean"] = lg_data["Lounge_Code"].astype(str).apply(normalize_name)
elif "Name" in lg_data.columns:
    lg_data["Lounge_clean"] = lg_data["Name"].apply(normalize_name)
else:
    logger.error("LG data missing Lounge_Name or Lounge_Code column.")
    sys.exit(1)

# -----------------------------
# Step 3: Fuzzy Lounge Matching
# -----------------------------
logger.info("🔍 Performing fuzzy lounge match...")

from rapidfuzz import process
ieg_unique = ieg_data["Lounge_clean"].dropna().unique().tolist()
lg_unique = lg_data["Lounge_clean"].dropna().unique().tolist()

fuzzy_matches = []
for ieg_name in ieg_unique:
    best_match = process.extractOne(ieg_name, lg_unique, scorer=fuzz.token_sort_ratio)
    if best_match and best_match[1] >= 85:
        fuzzy_matches.append((ieg_name, best_match[0], best_match[1]))

fuzzy_df = pd.DataFrame(fuzzy_matches, columns=["IEG_Lounge", "LG_Lounge", "Match_Score"])
fuzzy_map = dict(zip(fuzzy_df["IEG_Lounge"], fuzzy_df["LG_Lounge"]))
logger.info(f"✅ Found {len(fuzzy_df)} lounge matches ≥85")

# -----------------------------
# Step 4: Filter & Merge
# -----------------------------
def normalize_person_name(name):
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    name = re.sub(r'[^A-Z\s/]', '', name)
    name = re.sub(r'\s+', ' ', name)
    if '/' in name:
        parts = name.split('/')
        name = ' '.join(parts[::-1]).strip()
    return name.strip()

def find_best_match(ieg_name, lg_names):
    ieg_norm = normalize_person_name(ieg_name)
    lg_norms = [normalize_person_name(n) for n in lg_names]
    best_match, best_score = None, 0
    for lg_name, lg_norm in zip(lg_names, lg_norms):
        score_set = fuzz.token_set_ratio(ieg_norm, lg_norm)
        score_sort = fuzz.token_sort_ratio(ieg_norm, lg_norm)
        score_partial = fuzz.partial_ratio(ieg_norm, lg_norm)
        composite_score = (score_set * 0.5) + (score_sort * 0.3) + (score_partial * 0.2)
        if composite_score > best_score:
            best_match, best_score = lg_name, composite_score
    return best_match, best_score

# Convert datetime fields
ieg_data['Entry_DateTime'] = pd.to_datetime(
    ieg_data['Entry_Date'].astype(str) + ' ' + ieg_data['Entry_Time'].astype(str),
    errors='coerce'
)
lg_data["Visit_DateTime"] = pd.to_datetime(lg_data["time"], unit="ms", errors='coerce')

# Fuzzy matching within ±15 mins
matches = []
for _, ieg_row in ieg_data.iterrows():
    ieg_name = ieg_row['Name']
    ieg_dt = ieg_row['Entry_DateTime']
    if pd.isna(ieg_dt):
        continue
    ieg_date = ieg_dt.date()
    lg_candidates = lg_data[
        (lg_data['Visit_DateTime'].dt.date == ieg_date) &
        (lg_data['Visit_DateTime'] >= ieg_dt - timedelta(minutes=15)) &
        (lg_data['Visit_DateTime'] <= ieg_dt + timedelta(minutes=15))
    ]
    if not lg_candidates.empty:
        best_match, score = find_best_match(ieg_name, lg_candidates['Name'])
        if best_match and score >= 25:
            lg_row = lg_candidates.loc[lg_candidates['Name'] == best_match].iloc[0]
            matches.append({
                "IEG_Name": ieg_name,
                "LG_Name": best_match,
                "Match_Score": round(score, 1),
                "IEG_Entry_DateTime": ieg_dt,
                "LG_Visit_DateTime": lg_row['Visit_DateTime'],
                "Card_No": lg_row.get('priority_pass_number', None)
            })

fuzzy_matches_df = pd.DataFrame(matches)
fuzzy_matches_df = fuzzy_matches_df[fuzzy_matches_df["Match_Score"] > 62].reset_index(drop=True)
logger.info(f"✅ Fuzzy matching complete → {len(fuzzy_matches_df)} high-confidence matches")

# Map back Card_No to IEG
card_map = fuzzy_matches_df.set_index("IEG_Name")["Card_No"].to_dict()
ieg_data["Card_No"] = ieg_data["Name"].map(card_map)

logger.info("✅ Card_No successfully mapped back to IEG data")

# -----------------------------
# Step 5: Write Output to S3
# -----------------------------
output_key = "IEG-LG/fuzzy_matched_results/fuzzy_matches.parquet"
output_buffer = BytesIO()

pq.write_table(pa.Table.from_pandas(ieg_data), output_buffer)
s3.put_object(Bucket=bucket_lg, Key=output_key, Body=output_buffer.getvalue())

logger.info(f"📤 Results successfully saved to s3://{bucket_lg}/{output_key}")
logger.info("🎯 Glue Job completed successfully ✅")
