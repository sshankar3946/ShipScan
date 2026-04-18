"""
utils.py — complete updated version with address risk detection
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import re
import warnings
warnings.filterwarnings("ignore")

def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Only CSV and Excel files supported.")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[df["amount"] > 0].copy()
    for col in ["ip_address", "device_id", "location", "payment_method"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]     = df["hour_of_day"].apply(
        lambda h: 1 if (h >= 23 or h <= 5) else 0)
    return df


def add_user_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    txn_count_1h, txn_count_24h, avg_amt, deviation, is_first = [], [], [], [], []
    for idx, row in df.iterrows():
        user = row["user_id"]
        ts   = row["timestamp"]
        past = df[(df["user_id"] == user) & (df["timestamp"] < ts)]
        cnt_1h  = past[past["timestamp"] >= ts - timedelta(hours=1)].shape[0]
        cnt_24h = past[past["timestamp"] >= ts - timedelta(hours=24)].shape[0]
        if len(past) == 0:
            user_avg, dev, first = row["amount"], 0.0, 1
        else:
            user_avg = past["amount"].mean()
            dev = (row["amount"] - user_avg) / (past["amount"].std() + 1e-9)
            first = 0
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
    df = df.copy()
    df["ip_user_count"]     = df["ip_address"].map(
        df.groupby("ip_address")["user_id"].nunique()).fillna(1)
    df["device_user_count"] = df["device_id"].map(
        df.groupby("device_id")["user_id"].nunique()).fillna(1)
    df["ip_txn_count"]      = df["ip_address"].map(
        df.groupby("ip_address")["transaction_id"].count()).fillna(1)
    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home_loc = df.groupby("user_id")["location"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else "unknown")
    df["home_location"]     = df["user_id"].map(home_loc)
    df["location_mismatch"] = (df["location"] != df["home_location"]).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ADDRESS RISK DETECTION
# Directly addresses the fake address problem Indian Amazon sellers face
# ─────────────────────────────────────────────────────────────────────────────

LANDMARK_KEYWORDS = [
    "ke paas", "ke saamne", "ke samne", "ke pass",
    "near ", "opposite ", "opp ", "behind ",
    "next to", "beside ", "adjacent",
    "water tank", "overhead tank",
    "gas agency", "petrol pump", "cng pump",
    "temple", "mandir", "masjid", "church", "gurudwara",
    "school ke", "college ke", "hospital ke",
    "market ke", "bazaar", "chowk",
    "wala ghar", "waali gali", "wale ghar",
]


def score_address_quality(address: str) -> dict:
    """
    Score a delivery address for fraud risk.
    Higher score = more suspicious.
    """
    if not isinstance(address, str) or address.strip() == "":
        return {
            "address_risk_score": 80,
            "address_flags": ["Address is missing or empty"],
            "is_landmark_only": False
        }

    addr = address.lower().strip()
    flags = []
    score = 0

    # Too short to be a real address
    if len(addr) < 12:
        score += 35
        flags.append(f"Address too short to be genuine ({len(addr)} characters)")

    # Landmark-only — no actual street info
    found = [kw for kw in LANDMARK_KEYWORDS if kw in addr]
    if found:
        score += 40
        flags.append(
            f"Landmark-only address — '{found[0].strip()}' "
            f"found but no street or house number"
        )

    # No numbers at all — real addresses almost always have one
    if not re.search(r'\d', addr):
        score += 20
        flags.append("No house, plot, or flat number in address")

    # Placeholder text
    if addr.strip() in ["unknown", "test", "na", "n/a", "none", "xyz", "abc"]:
        score += 60
        flags.append(f"Placeholder address value: '{address}'")

    return {
        "address_risk_score": min(score, 100),
        "address_flags": flags,
        "is_landmark_only": bool(found)
    }


def add_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add address risk signals to every transaction row."""
    df = df.copy()

    if "location" not in df.columns:
        df["address_risk_score"] = 0
        df["address_flags"]      = [[] for _ in range(len(df))]
        df["is_landmark_only"]   = False
        df["address_user_count"] = 1
        return df

    results = df["location"].apply(score_address_quality)
    df["address_risk_score"] = results.apply(lambda x: x["address_risk_score"])
    df["address_flags"]      = results.apply(lambda x: x["address_flags"])
    df["is_landmark_only"]   = results.apply(lambda x: x["is_landmark_only"])

    # Multiple users sharing same delivery address = possible fraud ring
    df["address_user_count"] = df["location"].map(
        df.groupby("location")["user_id"].nunique()).fillna(1)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    payment_map = {"upi": 0, "card": 1, "cod": 2, "wallet": 3, "unknown": 4}
    df["payment_method_enc"] = (
        df["payment_method"].str.lower().map(payment_map).fillna(4))
    return df


def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering. Returns DataFrame ready for detection."""
    df = clean_data(df)
    df = add_time_features(df)
    df = add_user_behaviour_features(df)
    df = add_network_features(df)
    df = add_location_features(df)
    df = add_address_features(df)
    df = encode_categoricals(df)
    return df
