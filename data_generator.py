"""
data_generator.py
-----------------
Generates a realistic sample dataset of UPI/eCommerce transactions
with known fraud patterns baked in.

WHY THIS EXISTS:
  - We need labeled data (is_fraud = 0 or 1) to train our ML model
  - We control exactly what fraud looks like, so we can verify detection works
  - Clients will replace this with their own CSV later
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ── Configuration ────────────────────────────────────────────────────────────
N_TRANSACTIONS  = 2000   # total rows in the dataset
FRAUD_RATE      = 0.08   # 8% of transactions are fraudulent (realistic for UPI)

# Pools of realistic-looking fake data
USER_IDS    = [f"USR{str(i).zfill(4)}" for i in range(1, 201)]   # 200 users
DEVICE_IDS  = [f"DEV{str(i).zfill(4)}" for i in range(1, 151)]   # 150 devices
IP_POOL     = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
               for _ in range(100)]                                # 100 IPs
LOCATIONS   = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"
]
PAYMENT_METHODS = ["UPI", "Card", "COD", "Wallet"]


def _random_timestamp(start: datetime, end: datetime) -> datetime:
    """Pick a random datetime between start and end."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def generate_dataset(
    n: int = N_TRANSACTIONS,
    fraud_rate: float = FRAUD_RATE,
    save_path: str = None
) -> pd.DataFrame:
    """
    Build a DataFrame of synthetic transactions.

    Parameters
    ----------
    n          : number of rows to generate
    fraud_rate : fraction of rows that are fraudulent
    save_path  : if provided, save the CSV to this path

    Returns
    -------
    pd.DataFrame with columns matching our expected schema
    """

    n_fraud   = int(n * fraud_rate)
    n_legit   = n - n_fraud

    end_ts    = datetime.now()
    start_ts  = end_ts - timedelta(days=30)   # last 30 days of activity

    rows = []

    # ── 1. LEGITIMATE transactions ────────────────────────────────────────────
    for _ in range(n_legit):
        user   = random.choice(USER_IDS)
        device = random.choice(DEVICE_IDS)
        ip     = random.choice(IP_POOL)
        loc    = random.choice(LOCATIONS)

        row = {
            "transaction_id" : f"TXN{random.randint(100000, 999999)}",
            "user_id"        : user,
            "amount"         : round(np.random.lognormal(mean=7.5, sigma=1.2), 2),
            # lognormal gives realistic skew: most txns small, some large
            "timestamp"      : _random_timestamp(start_ts, end_ts),
            "payment_method" : random.choice(PAYMENT_METHODS),
            "device_id"      : device,
            "ip_address"     : ip,
            "location"       : loc,
            "is_fraud"       : 0,
        }
        rows.append(row)

    # ── 2. FRAUDULENT transactions ────────────────────────────────────────────
    #
    # We embed 4 real-world fraud patterns so the model has something to learn:
    #
    # Pattern A — Velocity attack: same user hammers many txns in minutes
    # Pattern B — Account takeover: new device + new IP for existing user
    # Pattern C — High-value first hit: brand-new user, huge amount
    # Pattern D — Shared device abuse: one device used by many users

    fraud_patterns = ["velocity", "takeover", "high_value_new", "shared_device"]

    # Shared device for Pattern D
    shared_device = "DEV9999"
    shared_ip     = "10.0.0.99"

    for i in range(n_fraud):
        pattern = fraud_patterns[i % len(fraud_patterns)]
        ts      = _random_timestamp(start_ts, end_ts)

        if pattern == "velocity":
            # Same user, same IP, burst of transactions within seconds of each other
            user = random.choice(USER_IDS[:20])   # pick from a small pool
            ip   = random.choice(IP_POOL[:5])     # few IPs
            ts   = end_ts - timedelta(minutes=random.randint(0, 60))
            amount = round(np.random.uniform(500, 3000), 2)

        elif pattern == "takeover":
            # Existing user, but device + IP don't match their history
            user   = random.choice(USER_IDS[20:50])
            device = f"DEV{random.randint(9000, 9500)}"  # never-seen device
            ip     = f"203.{random.randint(0,255)}.{random.randint(0,255)}.1"
            amount = round(np.random.uniform(2000, 15000), 2)

        elif pattern == "high_value_new":
            # Brand-new user (not in main pool), very large first transaction
            user   = f"USR{random.randint(9000, 9999)}"
            amount = round(np.random.uniform(10000, 50000), 2)
            device = f"DEV{random.randint(8000, 8500)}"
            ip     = random.choice(IP_POOL)

        else:  # shared_device
            user   = f"USR{random.randint(200, 300)}"  # outside normal pool
            device = shared_device
            ip     = shared_ip
            amount = round(np.random.uniform(300, 5000), 2)

        # Fill in any variables not set by the pattern branch above
        if "device" not in dir() or pattern == "velocity":
            device = random.choice(DEVICE_IDS)
        if pattern not in ("takeover", "high_value_new", "shared_device"):
            device = random.choice(DEVICE_IDS)

        loc = random.choice(LOCATIONS)

        row = {
            "transaction_id" : f"TXN{random.randint(100000, 999999)}",
            "user_id"        : user,
            "amount"         : amount,
            "timestamp"      : ts,
            "payment_method" : random.choice(PAYMENT_METHODS),
            "device_id"      : device,
            "ip_address"     : ip,
            "location"       : loc,
            "is_fraud"       : 1,
        }
        rows.append(row)

    # ── 3. Assemble + shuffle ─────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Ensure amount is always positive
    df["amount"] = df["amount"].abs().clip(lower=1.0)

    # ── 4. Optionally save ────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Dataset saved → {save_path}")
        print(f"   Rows      : {len(df)}")
        print(f"   Fraud rows: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")

    return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset(save_path="data/sample_transactions.csv")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
    print("\nFraud breakdown:")
    print(df["is_fraud"].value_counts())