"""
model.py — complete updated version with address risk rules
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

RULES = {
    "high_amount":          10_000,
    "velocity_1h":          5,
    "velocity_24h":         15,
    "amount_deviation":     3.0,
    "shared_ip_users":      3,
    "shared_device_users":  3,
    "address_risk_high":    50,
    "address_user_count":   3,
}

ML_FEATURES = [
    "amount", "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "txn_count_1h", "txn_count_24h", "avg_amount_user", "amount_deviation",
    "is_first_txn", "ip_user_count", "device_user_count", "ip_txn_count",
    "location_mismatch", "payment_method_enc",
    "address_risk_score", "address_user_count",
]


def apply_rules(row: pd.Series) -> tuple:
    score   = 0
    reasons = []

    # ── Existing rules ────────────────────────────────────────────────────────

    if row["amount"] > RULES["high_amount"]:
        score += 2
        reasons.append(
            f"Amount Rs.{row['amount']:,.0f} exceeds safe threshold "
            f"of Rs.{RULES['high_amount']:,}"
        )

    if row.get("is_first_txn", 0) == 1 and row["amount"] > 5_000:
        score += 2
        reasons.append(
            f"First-ever transaction is unusually large "
            f"(Rs.{row['amount']:,.0f})"
        )

    if row.get("txn_count_1h", 0) > RULES["velocity_1h"]:
        score += 2
        reasons.append(
            f"User made {row['txn_count_1h']} transactions in the last hour"
        )

    if row.get("txn_count_24h", 0) > RULES["velocity_24h"]:
        score += 1
        reasons.append(
            f"User made {row['txn_count_24h']} transactions in last 24 hours"
        )

    if row.get("amount_deviation", 0) > RULES["amount_deviation"]:
        score += 1
        reasons.append(
            f"Amount is {row['amount_deviation']:.1f}x above this "
            f"user's usual spend"
        )

    if row.get("ip_user_count", 1) >= RULES["shared_ip_users"]:
        score += 2
        reasons.append(
            f"IP address used by {int(row['ip_user_count'])} different users"
        )

    if row.get("device_user_count", 1) >= RULES["shared_device_users"]:
        score += 2
        reasons.append(
            f"Device used by {int(row['device_user_count'])} different users"
        )

    if row.get("location_mismatch", 0) == 1:
        score += 1
        reasons.append(
            f"Location '{row.get('location', '?')}' differs from "
            f"user's usual location"
        )

    if row.get("is_night", 0) == 1:
        score += 1
        reasons.append(
            f"Transaction at {int(row.get('hour_of_day', 0))}:00 "
            f"(night-time window)"
        )

    # ── NEW: Address risk rules ───────────────────────────────────────────────

    # Rule — Suspicious or fake-looking address
    if row.get("address_risk_score", 0) >= RULES["address_risk_high"]:
        score += 2
        addr_flags = row.get("address_flags", [])
        if addr_flags:
            reasons.append(f"Suspicious address: {addr_flags[0]}")
        else:
            reasons.append("Delivery address quality is too low to be genuine")

    # Rule — Address shared by multiple different users
    if row.get("address_user_count", 1) >= RULES["address_user_count"]:
        score += 2
        reasons.append(
            f"Delivery address shared by "
            f"{int(row['address_user_count'])} different customers "
            f"— possible fraud ring"
        )

    # Rule — Triple threat: new user + landmark address + COD
    # This combination is the highest risk pattern in Indian eCommerce
    if (row.get("is_first_txn", 0) == 1 and
            row.get("is_landmark_only", False) and
            str(row.get("payment_method", "")).lower() == "cod"):
        score += 3
        reasons.append(
            "Highest risk combination detected: "
            "new buyer + landmark-only address + COD payment"
        )

    return score, reasons


def run_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    results = df.apply(apply_rules, axis=1)
    df["rule_score"]   = results.apply(lambda x: x[0])
    df["rule_reasons"] = results.apply(lambda x: x[1])
    max_possible = 16
    df["rule_prob"] = (df["rule_score"] / max_possible).clip(0, 1)
    return df


class FraudDetector:
    def __init__(self):
        self.rf_model  = None
        self.iso_model = None
        self.scaler    = StandardScaler()
        self.mode      = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> dict:
        features = [f for f in ML_FEATURES if f in df.columns]
        X = df[features].fillna(0)
        if "is_fraud" in df.columns:
            self.mode = "supervised"
            return self._fit_rf(df, X, features)
        else:
            self.mode = "unsupervised"
            return self._fit_iso(X)

    def _fit_rf(self, df, X, features):
        y = df["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1)
        self.rf_model.fit(X_train, y_train)
        y_pred = self.rf_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        self.is_fitted = True
        return {
            "mode":      "supervised (RandomForest)",
            "precision": round(report["1"]["precision"], 3),
            "recall":    round(report["1"]["recall"], 3),
            "f1":        round(report["1"]["f1-score"], 3),
            "features":  features,
        }

    def _fit_iso(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.iso_model = IsolationForest(
            n_estimators=200, contamination=0.08,
            random_state=42, n_jobs=-1)
        self.iso_model.fit(X_scaled)
        self.is_fitted = True
        return {"mode": "unsupervised (IsolationForest)", "features": list(X.columns)}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        features = [f for f in ML_FEATURES if f in df.columns]
        X = df[features].fillna(0)
        if self.mode == "supervised":
            return self.rf_model.predict_proba(X)[:, 1]
        else:
            X_scaled = self.scaler.transform(X)
            raw = self.iso_model.score_samples(X_scaled)
            return 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def compute_final_scores(df: pd.DataFrame, ml_probs: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["ml_prob"]         = ml_probs
    df["fraud_prob"]      = (0.45 * df["rule_prob"] + 0.55 * df["ml_prob"]).clip(0, 1)
    df["fraud_score_pct"] = (df["fraud_prob"] * 100).round(1)
    df["risk_label"]      = pd.cut(
        df["fraud_prob"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Medium", "High"]
    )
    return df


def build_explanation(row: pd.Series) -> str:
    label   = row.get("risk_label", "Low")
    score   = row.get("fraud_score_pct", 0)
    reasons = row.get("rule_reasons", [])
    emoji   = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(str(label), "❓")
    lines   = [f"{emoji} {str(label).upper()} RISK — Fraud probability: {score:.0f}%"]
    if reasons:
        lines.append("Reasons:")
        for r in reasons:
            lines.append(f"  • {r}")
    else:
        lines.append("Flagged by ML pattern detection.")
    return "\n".join(lines)


def run_detection(df: pd.DataFrame) -> tuple:
    df = run_rule_engine(df)
    detector = FraudDetector()
    metrics  = detector.fit(df)
    ml_probs = detector.predict_proba(df)
    df       = compute_final_scores(df, ml_probs)
    df["explanation"] = df.apply(build_explanation, axis=1)
    return df, metrics, detector
