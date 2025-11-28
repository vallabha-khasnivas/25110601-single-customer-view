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
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import to_timestamp, col, year, to_date, lit
from pyspark.sql.functions import regexp_replace

# -----------------------------
# Glue / Spark Setup
# -----------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# -----------------------------
# Logging (console + CloudWatch)
# -----------------------------
region_name = 'us-east-1'
log_group_name = '/AD-DL-Prod-ARS-Logs'
log_stream_name = "ad-dl-prod-glue-etl"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

boto3_client = boto3.client('logs', region_name=region_name)
cw_handler = watchtower.CloudWatchLogHandler(log_group=log_group_name, stream_name=log_stream_name, boto3_client=boto3_client)
cw_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(cw_handler)

logger.info(f"🚀 Starting Glue Spark Job: {args['JOB_NAME']}")

# -----------------------------
# S3 / Paths
# -----------------------------
s3 = boto3.client("s3")
bucket_ieg = "ad-dl-prod-rawzone"
bucket_lg = "ad-dl-dev-sandboxzone"
parquet_key_ieg = "IEG/universal_visits/historical/universal_visits_final.parquet"
parquet_key_lg = "LG-Historical/combined_all.parquet"
output_base = "IEG-Mapped"

# -----------------------------
# 0) Read both files with Spark (distributed)
# -----------------------------
lg_s3_path = f"s3://{bucket_lg}/{parquet_key_lg}"
ieg_s3_path = f"s3://{bucket_ieg}/{parquet_key_ieg}"

logger.info(f"📥 Reading LG via Spark: {lg_s3_path}")
lg_spark = spark.read.parquet(lg_s3_path)

logger.info(f"📥 Reading IEG via Spark: {ieg_s3_path}")
ieg_spark = spark.read.parquet(ieg_s3_path)

logger.info(f"LG columns: {lg_spark.columns}")
logger.info(f"IEG columns: {ieg_spark.columns}")

# -----------------------------
# 1) Parse timestamps in Spark and find common year (dynamic)
# -----------------------------
# Normalize IEG date:
if "Entry_Date" in ieg_spark.columns:
    # If Entry_Date has time part or not; try parsing with to_timestamp or to_date
    # ieg_spark = ieg_spark.withColumn("Entry_DateTS", to_timestamp(col("Entry_Date")))
    # ieg_spark = ieg_spark.withColumn("Entry_Date_only", to_date(col("Entry_DateTS")))
    from pyspark.sql.functions import substring, to_date, col, to_timestamp
    
    # Extract first 10 characters as YYYY-MM-DD
    ieg_spark = ieg_spark.withColumn(
        "Entry_Date_str",
        substring(col("Entry_Date"), 1, 10)
    )
    
    # Convert just the date part into timestamp (00:00 Timestamp)
    ieg_spark = ieg_spark.withColumn(
        "Entry_DateTS",
        to_timestamp(col("Entry_Date_str"), "yyyy-MM-dd")
    )
    
    # Create date-only column
    ieg_spark = ieg_spark.withColumn(
        "Entry_Date_only",
        to_date(col("Entry_DateTS"))
    )




else:
    logger.error("IEG missing Entry_Date")
    raise SystemExit(1)

# LG: parse `time` column — if epoch ms, detect numeric type and convert; else parse string
if "time" in lg_spark.columns:
    # try to guess numeric vs string on driver by checking schema type
    lg_time_type = [f for f in lg_spark.schema.fields if f.name == "time"][0].dataType.simpleString()
    if lg_time_type.startswith("long") or lg_time_type.startswith("int"):
        # epoch ms -> convert from ms to timestamp
        lg_spark = lg_spark.withColumn("Visit_DateTime", to_timestamp((col("time")/1000).cast("timestamp")))
    else:
        # try parsing string
        lg_spark = lg_spark.withColumn("Visit_DateTime", to_timestamp(col("time")))
else:
    # fallback: try other names
    for candidate in ["Visit_DateTime", "visit_time", "timestamp"]:
        if candidate in lg_spark.columns:
            lg_spark = lg_spark.withColumn("Visit_DateTime", to_timestamp(col(candidate)))
            break
    if "Visit_DateTime" not in lg_spark.columns:
        logger.error("LG missing time-like column")
        raise SystemExit(1)

lg_spark = lg_spark.withColumn("Visit_Date_only", to_date(col("Visit_DateTime")))

# Extract distinct years from cluster (distributed)
ieg_years = sorted([r[0] for r in ieg_spark.select(year(col("Entry_DateTS")).alias("y")).distinct().collect() if r[0] is not None])
lg_years = sorted([r[0] for r in lg_spark.select(year(col("Visit_DateTime")).alias("y")).distinct().collect() if r[0] is not None])

logger.info(f"📆 IEG years present: {ieg_years}")
logger.info(f"📆 LG years present: {lg_years}")

common_years = sorted(list(set(ieg_years).intersection(set(lg_years))))
if not common_years:
    logger.error("No common years found between IEG and LG")
    raise SystemExit(1)

year_filter = 2025
logger.info(f"🔎 Selected dynamic common year for processing: {year_filter}")

# -----------------------------
# 2) Filter BOTH datasets by the year USING SPARK (distributed, fast)
# -----------------------------
ieg_spark = ieg_spark.filter(year(col("Entry_DateTS")) == lit(year_filter)).cache()
lg_spark = lg_spark.filter(year(col("Visit_DateTime")) == lit(year_filter)).cache()

logger.info(f"📊 After year filter: IEG count (approx) = {ieg_spark.count()}, LG count (approx) = {lg_spark.count()}")

# -----------------------------
# 3) Apply light IEG filters in Spark (as in original)
# -----------------------------
# Keep original filter: Card == 'Priority Pass' & Type == 'Main Passenger'
if set(["Card", "Type"]).issubset(set(ieg_spark.columns)):
    ieg_spark = ieg_spark.filter((col("Card") == "Priority Pass") & (col("Type") == "Main Passenger"))
else:
    logger.warning("IEG missing Card/Type; continuing without those filters")

# Build a small lookup of unique lounges from LG in Spark (optional)
# Normalize lounge codes in pandas later; we will do fuzzy lounge match in pandas per-day

# -----------------------------
# 4) Determine which days to process (from IEG) — distributed
# -----------------------------
days = [r[0].isoformat() for r in ieg_spark.select(col("Entry_Date_only")).distinct().collect() if r[0] is not None]
logger.info(f"🗓 Days to process (count={len(days)}): sample={days[:5]}")

ieg_spark.select("Entry_Date_only").distinct().orderBy("Entry_Date_only").show(2000)

# -----------------------------
# Helper functions (pandas-side, unchanged)
# -----------------------------
def normalize_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

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

# -----------------------------
# 4) Determine which days to process (from IEG) — distributed
# -----------------------------
# collect as actual date objects (not strings) so Spark comparisons are type-safe
days = [r[0] for r in ieg_spark.select(col("Entry_Date_only")).distinct().collect() if r[0] is not None]
logger.info(f"🗓 Days to process (count={len(days)}): sample={[d.isoformat() for d in days][:5]}")

# -----------------------------
# 5) Day-by-day processing: collect only that day's data to pandas
# -----------------------------
final_matches_list = []

for day in days:
    day_str = day.isoformat()
    logger.info(f"🔁 Processing day {day_str}")

    # Spark filters for the day (distributed) -> then collect to pandas (small)
    ieg_day_spark = ieg_spark.filter(col("Entry_Date_only") == lit(day))
    lg_day_spark  = lg_spark.filter(col("Visit_Date_only") == lit(day))

    # collect to pandas; these should be small per-day slices
    ieg_day_pd = ieg_day_spark.toPandas()
    lg_day_pd  = lg_day_spark.toPandas()

    # If there are NO IEG rows for the day — nothing to write (but log)
    if ieg_day_pd.empty:
        logger.info(f"⏭ Skipping day {day_str} because no IEG rows for that day")
        # if you want an empty file for days with zero IEG rows, create one here (optional)
        continue

    # Normalize / prepare lounge/name columns (unchanged)
    ieg_day_pd["Lounge_clean"] = ieg_day_pd.get("Lounge", "").apply(normalize_name)
    if "airport_code" in lg_day_pd.columns and "Lounge_Code" not in lg_day_pd.columns:
        lg_day_pd.rename(columns={"airport_code": "Lounge_Code"}, inplace=True)
    if "name" in lg_day_pd.columns and "Name" not in lg_day_pd.columns:
        lg_day_pd.rename(columns={"name": "Name"}, inplace=True)

    if "Lounge_Code" in lg_day_pd.columns:
        lg_day_pd["Lounge_clean"] = lg_day_pd["Lounge_Code"].astype(str).apply(normalize_name)
    elif "Name" in lg_day_pd.columns:
        lg_day_pd["Lounge_clean"] = lg_day_pd["Name"].apply(normalize_name)
    else:
        # LG chunk empty of lounge/name columns - continue but we will still write IEG file with null Card_No
        lg_day_pd["Lounge_clean"] = []

    # Fuzzy lounge matches at day level (smaller universe)
    from rapidfuzz import process as rf_process
    ieg_unique = ieg_day_pd["Lounge_clean"].dropna().unique().tolist()
    lg_unique = lg_day_pd["Lounge_clean"].dropna().unique().tolist() if len(lg_day_pd) > 0 else []
    fuzzy_matches = []
    for ieg_name in ieg_unique:
        if not lg_unique:
            break
        best_match = rf_process.extractOne(ieg_name, lg_unique, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= 85:
            fuzzy_matches.append((ieg_name, best_match[0], best_match[1]))
    fuzzy_map = dict((a,b) for a,b,_ in fuzzy_matches)
    ieg_day_pd["Matched_Lounge"] = ieg_day_pd["Lounge_clean"].map(fuzzy_map).fillna("")

    # Prepare datetimes in pandas (as in your original)
    ieg_day_pd["Entry_Date"] = pd.to_datetime(ieg_day_pd["Entry_Date"], errors="coerce")
    if "Entry_Time" in ieg_day_pd.columns:
        ieg_day_pd['Entry_DateTime'] = pd.to_datetime(
            ieg_day_pd["Entry_Date"].dt.strftime('%Y-%m-%d') + ' ' + ieg_day_pd["Entry_Time"].astype(str),
            errors='coerce'
        )
    else:
        ieg_day_pd['Entry_DateTime'] = pd.to_datetime(ieg_day_pd["Entry_Date"], errors='coerce')

    # Ensure LG visit datetime is datetime dtype (if not already)
    if not pd.api.types.is_datetime64_any_dtype(lg_day_pd.get("Visit_DateTime", pd.Series(dtype='datetime64[ns]'))):
        if "Visit_DateTime" in lg_day_pd.columns:
            lg_day_pd["Visit_DateTime"] = pd.to_datetime(lg_day_pd["Visit_DateTime"], errors='coerce')

    # Run your fuzzy person matching within ±15 minutes for that day
    matches = []
    for _, ieg_row in ieg_day_pd.iterrows():
        ieg_name = ieg_row.get('Name', "")
        ieg_dt = ieg_row.get('Entry_DateTime', pd.NaT)
        if pd.isna(ieg_dt) or not ieg_name:
            # keep row, but no match; mapping to Card_No will be None later
            continue

        # If LG day is empty, window_df will be empty and we will not find matches — that's OK
        window_start = ieg_dt - timedelta(minutes=15)
        window_end = ieg_dt + timedelta(minutes=15)
        if not lg_day_pd.empty and "Visit_DateTime" in lg_day_pd.columns:
            window_df = lg_day_pd[
                (lg_day_pd["Visit_DateTime"] >= window_start) &
                (lg_day_pd["Visit_DateTime"] <= window_end)
            ]
        else:
            window_df = pd.DataFrame()  # empty

        if window_df.empty:
            # no candidates in ±15 min — that's fine, we will store null Card_No for this IEG row
            continue

        best_match, score = find_best_match(ieg_name, window_df["Name"].tolist())
        if best_match and score >= 62:
            lg_row = window_df.loc[window_df["Name"] == best_match].iloc[0]
            matches.append({
                "IEG_Name": ieg_name,
                "LG_Name": best_match,
                "Match_Score": round(score, 1),
                "IEG_Entry_DateTime": ieg_dt,
                "LG_Visit_DateTime": lg_row["Visit_DateTime"],
                "Card_No": lg_row.get("priority_pass_number", None),
                "Entry_Day": day_str
            })

    # Accumulate matches globally (if any)
    if matches:
        final_matches_list.extend(matches)

    # Map Card_No back to ieg_day_pd: if matches empty -> Card_No stays None for every row
    # if matches:
    #     fuzzy_matches_df = pd.DataFrame(matches)
    #     card_map = fuzzy_matches_df.set_index("IEG_Name")["Card_No"].to_dict()
    #     # Map by Name; preserve None where not found
    #     ieg_day_pd["Card_No"] = ieg_day_pd["Name"].map(card_map)
    # else:
    #     ieg_day_pd["Card_No"] = None

    # -----------------------------
    # Map Card_No back into full IEG df
    # -----------------------------   
    # if matches:
    #     fuzzy_matches_df = pd.DataFrame(matches)
    #     card_map = fuzzy_matches_df.set_index("IEG_Name")["Card_No"].to_dict()
    #     ieg_day_pd["Card_No"] = ieg_day_pd["Name"].map(card_map).fillna(None)
    # else:
    #     # no matches => entire column is None
    #     ieg_day_pd["Card_No"] = None

    # Correct row-level Card_No mapping
    if matches:
        # Convert matches to DataFrame
        fuzzy_matches_df = pd.DataFrame(matches)
    
        # Add row index (needed for exact mapping)
        fuzzy_matches_df["row_index"] = fuzzy_matches_df.index
    
        # Create an empty Card_No column first
        ieg_day_pd["Card_No"] = None
    
        # Map Card_No row-by-row
        for m in matches:
            row_idx = ieg_day_pd[ieg_day_pd["Name"] == m["IEG_Name"]].index
    
            # If multiple rows have same name, map based on Entry_DateTime match too
            row_idx = ieg_day_pd[
                (ieg_day_pd["Name"] == m["IEG_Name"]) &
                (ieg_day_pd["Entry_DateTime"] == m["IEG_Entry_DateTime"])
            ].index
    
            if len(row_idx) > 0:
                ieg_day_pd.loc[row_idx, "Card_No"] = m["Card_No"]
    
    else:
        # No matches found for the day
        ieg_day_pd["Card_No"] = None


    # Write this day's IEG (with Card_No) back to S3 as parquet (day folder)
    s3_key = f"{output_base}/Entry_Date={day_str}/IEG_MAPPED_{day_str}.parquet"
    try:
        buf = BytesIO()
        pq.write_table(pa.Table.from_pandas(ieg_day_pd.reset_index(drop=True)), buf)
        s3.put_object(Bucket=bucket_lg, Key=s3_key, Body=buf.getvalue())
        logger.info(f"📤 Written day {day_str} rows={len(ieg_day_pd)} -> s3://{bucket_lg}/{s3_key}")
    except Exception as e:
        logger.error(f"❌ Failed to write day {day_str} to S3: {e}")

# -----------------------------
# 5) Day-by-day processing: collect only that day's data to pandas
# -----------------------------
# final_matches_list = []

# for day_iso in days:
#     logger.info(f"🔁 Processing day {day_iso}")

#     # Spark filters for the day (distributed) -> then collect to pandas (small)
#     ieg_day_spark = ieg_spark.filter(col("Entry_Date_only") == lit(day_iso))
#     lg_day_spark  = lg_spark.filter(col("Visit_Date_only") == lit(day_iso))

#     # collect to pandas; these should be small per-day slices
#     ieg_day_pd = ieg_day_spark.toPandas()
#     lg_day_pd = lg_day_spark.toPandas()

#     if ieg_day_pd.empty or lg_day_pd.empty:
#         logger.info(f"⏭ Skipping day {day_iso} because no rows in one of the datasets")
#         continue

#     # preserve original columns / rename if needed (same as your logic)
#     ieg_day_pd["Lounge_clean"] = ieg_day_pd.get("Lounge", "").apply(normalize_name)
#     if "airport_code" in lg_day_pd.columns and "Lounge_Code" not in lg_day_pd.columns:
#         lg_day_pd.rename(columns={"airport_code": "Lounge_Code"}, inplace=True)
#     if "name" in lg_day_pd.columns and "Name" not in lg_day_pd.columns:
#         lg_day_pd.rename(columns={"name": "Name"}, inplace=True)

#     if "Lounge_Code" in lg_day_pd.columns:
#         lg_day_pd["Lounge_clean"] = lg_day_pd["Lounge_Code"].astype(str).apply(normalize_name)
#     elif "Name" in lg_day_pd.columns:
#         lg_day_pd["Lounge_clean"] = lg_day_pd["Name"].apply(normalize_name)
#     else:
#         logger.warning("LG day chunk missing Name/Lounge_Code; continuing")

#     # Fuzzy lounge matches at day level (smaller universe)
#     from rapidfuzz import process as rf_process
#     ieg_unique = ieg_day_pd["Lounge_clean"].dropna().unique().tolist()
#     lg_unique = lg_day_pd["Lounge_clean"].dropna().unique().tolist()
#     fuzzy_matches = []
#     for ieg_name in ieg_unique:
#         best_match = rf_process.extractOne(ieg_name, lg_unique, scorer=fuzz.token_sort_ratio)
#         if best_match and best_match[1] >= 85:
#             fuzzy_matches.append((ieg_name, best_match[0], best_match[1]))
#     fuzzy_map = dict((a,b) for a,b,_ in fuzzy_matches)
#     ieg_day_pd["Matched_Lounge"] = ieg_day_pd["Lounge_clean"].map(fuzzy_map).fillna("")

#     # Prepare datetimes in pandas (as in your original)
# # Ensure Entry_Date is datetime BEFORE using .dt
#     ieg_day_pd["Entry_Date"] = pd.to_datetime(ieg_day_pd["Entry_Date"], errors="coerce")
    
#     if "Entry_Time" in ieg_day_pd.columns:
#         ieg_day_pd['Entry_DateTime'] = pd.to_datetime(
#             ieg_day_pd["Entry_Date"].dt.strftime('%Y-%m-%d') + ' ' + ieg_day_pd["Entry_Time"].astype(str),
#             errors='coerce'
#         )
#     else:
#         ieg_day_pd['Entry_DateTime'] = pd.to_datetime(ieg_day_pd["Entry_Date"], errors='coerce')


#     # Ensure LG visit datetime is datetime dtype (if not already)
#     if not pd.api.types.is_datetime64_any_dtype(lg_day_pd.get("Visit_DateTime", pd.Series())):
#         lg_day_pd["Visit_DateTime"] = pd.to_datetime(lg_day_pd["Visit_DateTime"], errors='coerce')

#     # Run your fuzzy person matching within ±15 minutes for that day
#     matches = []
#     for _, ieg_row in ieg_day_pd.iterrows():
#         ieg_name = ieg_row.get('Name', "")
#         ieg_dt = ieg_row.get('Entry_DateTime', pd.NaT)
#         if pd.isna(ieg_dt) or not ieg_name:
#             continue

#         window_start = ieg_dt - timedelta(minutes=15)
#         window_end = ieg_dt + timedelta(minutes=15)
#         window_df = lg_day_pd[
#             (lg_day_pd["Visit_DateTime"] >= window_start) &
#             (lg_day_pd["Visit_DateTime"] <= window_end)
#         ]
#         if window_df.empty:
#             continue

#         best_match, score = find_best_match(ieg_name, window_df["Name"].tolist())
#         if best_match and score >= 62:
#             lg_row = window_df.loc[window_df["Name"] == best_match].iloc[0]
#             # logger.info(
#             #     f"🎯 MATCH | Day={day_iso} | IEG_Name='{ieg_name}' "
#             #     f"→ LG_Name='{best_match}' | Score={round(score,1)} "
#             #     f"| IEG_Time={ieg_dt} | LG_Time={lg_row['Visit_DateTime']}"
#             # )

#             matches.append({
#                 "IEG_Name": ieg_name,
#                 "LG_Name": best_match,
#                 "Match_Score": round(score, 1),
#                 "IEG_Entry_DateTime": ieg_dt,
#                 "LG_Visit_DateTime": lg_row["Visit_DateTime"],
#                 "Card_No": lg_row.get("priority_pass_number", None),
#                 "Entry_Day": day_iso
#             })

#     # Accumulate matches
#     if matches:
#         final_matches_list.extend(matches)

#     # Map Card_No back to ieg_day_pd and write day output to S3
#     # create a small map
#     if matches:
#         fuzzy_matches_df = pd.DataFrame(matches)
#         card_map = fuzzy_matches_df.set_index("IEG_Name")["Card_No"].to_dict()
#         ieg_day_pd["Card_No"] = ieg_day_pd["Name"].map(card_map)
#     else:
#         ieg_day_pd["Card_No"] = None

#     # Write this day's IEG (with Card_No) back to S3 as parquet (day folder)
#     day_str = day_iso
#     s3_key = f"{output_base}/Entry_Date={day_str}/IEG_MAPPED_{day_str}.parquet"
#     try:
#         buf = BytesIO()
#         pq.write_table(pa.Table.from_pandas(ieg_day_pd.reset_index(drop=True)), buf)
#         s3.put_object(Bucket=bucket_lg, Key=s3_key, Body=buf.getvalue())
#         logger.info(f"📤 Written day {day_str} rows={len(ieg_day_pd)} -> s3://{bucket_lg}/{s3_key}")
#     except Exception as e:
#         logger.error(f"❌ Failed to write day {day_str} to S3: {e}")

# -----------------------------
# After loop: summary and optional global file of matches
# -----------------------------
fuzzy_matches_df_all = pd.DataFrame(final_matches_list)
logger.info(f"✅ Total matches across all days: {len(fuzzy_matches_df_all)}")

# Optionally write aggregated matches
if not fuzzy_matches_df_all.empty:
    s3_key_all = f"{output_base}/ALL_MATCHES_{year_filter}.parquet"
    buf = BytesIO()
    pq.write_table(pa.Table.from_pandas(fuzzy_matches_df_all.reset_index(drop=True)), buf)
    s3.put_object(Bucket=bucket_lg, Key=s3_key_all, Body=buf.getvalue())
    logger.info(f"📤 Written aggregated matches -> s3://{bucket_lg}/{s3_key_all}")

logger.info("🎯 Glue Spark Job completed successfully ✅")
job.commit()



