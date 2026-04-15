"""
utils.py
--------
Handles everything BEFORE the ML model sees the data:
  1. Loading the file (CSV or Excel)
  2. Cleaning + standardising columns
  3. Engineering fraud-signal features

Each function is small, does ONE job, and is easy to test independently.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOADING
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = {
    "transaction_id", "user_id", "amount", "timestamp",
    "payment_method", "device_id", "ip_address", "location"
}

OPTIONAL_COLS = {"is_fraud"}   # present for training, absent for inference


def load_file(uploaded_file) -> pd.DataFrame:
    """
    Read a CSV or Excel file uploaded via Streamlit's file_uploader.

    Why we handle both formats:
      - SMEs often export from Excel; tech teams export CSV.
      - We want the app to "just work" for either.

    Returns
    -------
    Raw DataFrame (not yet cleaned).
    Raises ValueError if required columns are missing.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Only CSV and Excel (.xlsx/.xls) files are supported.")

    # Normalise column names: lowercase + strip spaces
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Check required columns exist
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Your file is missing these required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise types, fill missing values, remove obvious garbage rows.

    Business logic behind each decision:
      - Negative amounts → data entry errors, drop them
      - Missing IP/device → fill with 'unknown' so rules can still fire
      - Duplicate transaction IDs → keep first occurrence (idempotency)
    """
    df = df.copy()

    # ── Timestamp ─────────────────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_ts = df["timestamp"].isna().sum()
    if n_bad_ts > 0:
        print(f"⚠️  Dropped {n_bad_ts} rows with unparseable timestamps.")
    df = df.dropna(subset=["timestamp"])

    # ── Amount ────────────────────────────────────────────────────────────────
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[df["amount"] > 0].copy()      # negative / zero amounts are invalid

    # ── String columns — fill missing with 'unknown' ──────────────────────────
    for col in ["ip_address", "device_id", "location", "payment_method"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    df["user_id"] = df["user_id"].astype(str).str.strip()

    # ── Deduplicate ───────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    dupes = before - len(df)
    if dupes:
        print(f"ℹ️  Removed {dupes} duplicate transaction IDs.")

    # ── Sort chronologically (needed for rolling-window features) ─────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TIME FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose the timestamp into individual signals.

    WHY EACH FEATURE MATTERS FOR FRAUD:
      hour_of_day  → Fraudsters often strike late at night (1 AM – 5 AM)
      day_of_week  → Weekends have lower bank staff = slower fraud response
      is_weekend   → Simple binary flag for the above
      is_night     → Transactions between 11 PM and 5 AM are higher risk
    """
    df = df.copy()

    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek   # 0 = Monday
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]     = df["hour_of_day"].apply(
        lambda h: 1 if (h >= 23 or h <= 5) else 0
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BEHAVIOUR FEATURES  (the heart of fraud detection)
# ─────────────────────────────────────────────────────────────────────────────

def add_user_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, look BACKWARDS in time to compute what that
    specific user has been doing recently.

    This is called "historical aggregation" — the key idea:
      A fraudster might make 20 transactions in 10 minutes.
      A normal user makes 2–3 per day.
      So counting past transactions catches velocity attacks.

    Features created:
      txn_count_1h      — how many txns this user made in the past hour
      txn_count_24h     — same, but past 24 hours
      avg_amount_user   — this user's average spend (lifetime in our dataset)
      amount_deviation  — how far THIS transaction deviates from their average
                          > 3.0 means unusually large = suspicious
      is_first_txn      — 1 if this is the user's very first transaction
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    txn_count_1h  = []
    txn_count_24h = []
    avg_amt       = []
    deviation     = []
    is_first      = []

    for idx, row in df.iterrows():
        user = row["user_id"]
        ts   = row["timestamp"]

        # All PREVIOUS transactions by this user
        past = df[(df["user_id"] == user) & (df["timestamp"] < ts)]

        # Velocity counts
        one_hour_ago   = ts - timedelta(hours=1)
        one_day_ago    = ts - timedelta(hours=24)
        cnt_1h  = past[past["timestamp"] >= one_hour_ago].shape[0]
        cnt_24h = past[past["timestamp"] >= one_day_ago].shape[0]

        # Amount stats
        if len(past) == 0:
            user_avg = row["amount"]   # no history → use current amount
            dev      = 0.0
            first    = 1
        else:
            user_avg = past["amount"].mean()
            std      = past["amount"].std()
            dev      = (row["amount"] - user_avg) / (std + 1e-9)
            first    = 0

        txn_count_1h.append(cnt_1h)
        txn_count_24h.append(cnt_24h)
        avg_amt.append(round(user_avg, 2))
        deviation.append(round(dev, 4))
        is_first.append(first)

    df["txn_count_1h"]     = txn_count_1h
    df["txn_count_24h"]    = txn_count_24h
    df["avg_amount_user"]  = avg_amt
    df["amount_deviation"] = deviation
    df["is_first_txn"]     = is_first

    return df


def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Look for suspicious SHARING patterns across users.

    WHY THIS MATTERS:
      One stolen device or IP used by 5 different "users" = fraud ring.
      Legitimate devices are almost always tied to one person.

    Features created:
      ip_user_count      — how many distinct users share this IP
      device_user_count  — how many distinct users share this device
      ip_txn_count       — total transactions from this IP
    """
    df = df.copy()

    ip_user_count    = df.groupby("ip_address")["user_id"].nunique()
    device_user_count = df.groupby("device_id")["user_id"].nunique()
    ip_txn_count     = df.groupby("ip_address")["transaction_id"].count()

    df["ip_user_count"]     = df["ip_address"].map(ip_user_count).fillna(1)
    df["device_user_count"] = df["device_id"].map(device_user_count).fillna(1)
    df["ip_txn_count"]      = df["ip_address"].map(ip_txn_count).fillna(1)

    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect when a user suddenly transacts from a different city.

    HOW IT WORKS:
      We find each user's "home location" = the city they use most often.
      If a transaction comes from a different city, that's a mismatch flag.

    Feature created:
      location_mismatch — 1 if this city differs from the user's usual city
    """
    df = df.copy()

    # Most common location per user
    home_loc = (
        df.groupby("user_id")["location"]
        .agg(lambda x: x.mode()[0] if len(x) > 0 else "unknown")
    )
    df["home_location"]     = df["user_id"].map(home_loc)
    df["location_mismatch"] = (df["location"] != df["home_location"]).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ENCODING  (ML needs numbers, not strings)
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text columns into numbers the ML model can read.

    We use simple frequency encoding for high-cardinality columns (IP, device)
    because one-hot encoding would create thousands of columns.

    For low-cardinality columns (payment_method) we use label encoding.
    """
    df = df.copy()

    # Payment method → integer label
    payment_map = {"upi": 0, "card": 1, "cod": 2, "wallet": 3, "unknown": 4}
    df["payment_method_enc"] = (
        df["payment_method"].str.lower().map(payment_map).fillna(4)
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run ALL feature engineering steps in the correct order.

    Calling this ONE function gives you a fully prepared DataFrame
    ready to feed into the rule engine and ML model.
    """
    print("🔧 Running feature pipeline...")

    df = clean_data(df)
    print(f"  ✅ Cleaned: {len(df)} rows remain")

    df = add_time_features(df)
    print("  ✅ Time features added")

    df = add_user_behaviour_features(df)
    print("  ✅ User behaviour features added (this may take a moment...)")

    df = add_network_features(df)
    print("  ✅ Network features added")

    df = add_location_features(df)
    print("  ✅ Location features added")

    df = encode_categoricals(df)
    print("  ✅ Categoricals encoded")

    print(f"\n📦 Feature pipeline complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_generator import generate_dataset

    raw = generate_dataset(n=200)
    processed = run_feature_pipeline(raw)

    print("\nNew columns added by pipeline:")
    original_cols = set(raw.columns)
    new_cols = [c for c in processed.columns if c not in original_cols]
    for c in new_cols:
        print(f"  {c}: {processed[c].dtype}  |  sample: {processed[c].iloc[0]}")