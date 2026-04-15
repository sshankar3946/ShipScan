"""
model.py
--------
TWO detection systems that work together:

  (A) Rule Engine  — fast, transparent, always runs
  (B) ML Model     — learns patterns, fills gaps the rules miss

Final fraud score = weighted combination of both.

Why use BOTH?
  Rules alone: catch known patterns but miss novel fraud.
  ML alone: works but is a black box — hard to explain to clients.
  Together: best recall AND human-readable explanations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — RULE-BASED ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds — these are tunable per client
RULES = {
    "high_amount"         : 10_000,   # ₹ — single txn above this is risky
    "velocity_1h"         : 5,        # more than 5 txns in 1 hour = suspicious
    "velocity_24h"        : 15,       # more than 15 txns in 24 hours
    "amount_deviation"    : 3.0,      # 3 std deviations above personal average
    "shared_ip_users"     : 3,        # IP used by 3+ different users
    "shared_device_users" : 3,        # device used by 3+ different users
}


def apply_rules(row: pd.Series) -> tuple[int, list[str]]:
    """
    Check a SINGLE transaction row against all rules.

    Returns
    -------
    rule_score : int   — count of rules that fired (0 = clean, 5+ = very risky)
    reasons    : list  — plain-English explanation of WHY it was flagged
    """
    score   = 0
    reasons = []

    # Rule 1 — Unusually large amount
    if row["amount"] > RULES["high_amount"]:
        score += 2   # weighted double — amount is the strongest signal
        reasons.append(
            f"Transaction amount ₹{row['amount']:,.0f} exceeds safe threshold "
            f"of ₹{RULES['high_amount']:,}"
        )

    # Rule 2 — High-value first transaction (new user + big spend)
    if row.get("is_first_txn", 0) == 1 and row["amount"] > 5_000:
        score += 2
        reasons.append(
            f"First-ever transaction from this user is unusually large "
            f"(₹{row['amount']:,.0f})"
        )

    # Rule 3 — Velocity: too many transactions in 1 hour
    if row.get("txn_count_1h", 0) > RULES["velocity_1h"]:
        score += 2
        reasons.append(
            f"User made {row['txn_count_1h']} transactions in the last hour "
            f"(limit: {RULES['velocity_1h']})"
        )

    # Rule 4 — Velocity: too many in 24 hours
    if row.get("txn_count_24h", 0) > RULES["velocity_24h"]:
        score += 1
        reasons.append(
            f"User made {row['txn_count_24h']} transactions in the last 24 hours "
            f"(limit: {RULES['velocity_24h']})"
        )

    # Rule 5 — Amount much higher than this user's average
    if row.get("amount_deviation", 0) > RULES["amount_deviation"]:
        score += 1
        reasons.append(
            f"Amount is {row['amount_deviation']:.1f}x above this user's "
            f"usual spend (avg: ₹{row.get('avg_amount_user', 0):,.0f})"
        )

    # Rule 6 — Shared IP across multiple users
    if row.get("ip_user_count", 1) >= RULES["shared_ip_users"]:
        score += 2
        reasons.append(
            f"IP address {row['ip_address']} has been used by "
            f"{int(row['ip_user_count'])} different users"
        )

    # Rule 7 — Shared device across multiple users
    if row.get("device_user_count", 1) >= RULES["shared_device_users"]:
        score += 2
        reasons.append(
            f"Device {row['device_id']} has been used by "
            f"{int(row['device_user_count'])} different users"
        )

    # Rule 8 — Location mismatch
    if row.get("location_mismatch", 0) == 1:
        score += 1
        reasons.append(
            f"Transaction location '{row.get('location', '?')}' differs from "
            f"user's usual location '{row.get('home_location', '?')}'"
        )

    # Rule 9 — Night-time transaction
    if row.get("is_night", 0) == 1:
        score += 1
        reasons.append(
            f"Transaction occurred at {int(row.get('hour_of_day', 0))}:00 "
            f"(high-risk night-time window)"
        )

    return score, reasons


def run_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rules to ALL rows and return the DataFrame with new columns."""
    df = df.copy()

    results = df.apply(apply_rules, axis=1)
    df["rule_score"]  = results.apply(lambda x: x[0])
    df["rule_reasons"] = results.apply(lambda x: x[1])

    # Normalise rule_score to 0–100 probability
    max_possible_score = 13   # sum of all weights above
    df["rule_prob"] = (df["rule_score"] / max_possible_score).clip(0, 1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — ML MODEL
# ─────────────────────────────────────────────────────────────────────────────

# Columns the ML model uses as input features
ML_FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "txn_count_1h",
    "txn_count_24h",
    "avg_amount_user",
    "amount_deviation",
    "is_first_txn",
    "ip_user_count",
    "device_user_count",
    "ip_txn_count",
    "location_mismatch",
    "payment_method_enc",
]


class FraudDetector:
    """
    Wraps two models:
      - RandomForest  : used when 'is_fraud' labels exist in the data
      - IsolationForest: used when NO labels exist (unsupervised)

    After fitting, call .predict(df) to get fraud probabilities.
    """

    def __init__(self):
        self.rf_model    = None
        self.iso_model   = None
        self.scaler      = StandardScaler()
        self.mode        = None    # "supervised" or "unsupervised"
        self.is_fitted   = False
        self.feature_importances = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Train the appropriate model.

        If 'is_fraud' column exists → RandomForest (supervised).
        Else                        → IsolationForest (unsupervised).

        Returns a dict with training metrics.
        """
        available_features = [f for f in ML_FEATURES if f in df.columns]
        X = df[available_features].fillna(0)

        if "is_fraud" in df.columns:
            self.mode = "supervised"
            return self._fit_random_forest(df, X, available_features)
        else:
            self.mode = "unsupervised"
            return self._fit_isolation_forest(X)

    def _fit_random_forest(
        self, df: pd.DataFrame, X: pd.DataFrame, features: list
    ) -> dict:
        y = df["is_fraud"]

        # Split into train/test so we can report accuracy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",   # handles imbalanced fraud/legit ratio
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)

        # Feature importance for explainability
        self.feature_importances = dict(
            zip(features, self.rf_model.feature_importances_)
        )

        # Metrics
        y_pred = self.rf_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.is_fitted = True
        return {
            "mode"     : "supervised (RandomForest)",
            "precision": round(report["1"]["precision"], 3),
            "recall"   : round(report["1"]["recall"], 3),
            "f1"       : round(report["1"]["f1-score"], 3),
            "features" : features,
        }

    def _fit_isolation_forest(self, X: pd.DataFrame) -> dict:
        X_scaled = self.scaler.fit_transform(X)

        self.iso_model = IsolationForest(
            n_estimators=200,
            contamination=0.08,   # assume ~8% of data is anomalous
            random_state=42,
            n_jobs=-1
        )
        self.iso_model.fit(X_scaled)

        self.is_fitted = True
        return {
            "mode"    : "unsupervised (IsolationForest)",
            "note"    : "No fraud labels found. Using anomaly detection.",
            "features": list(X.columns),
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return fraud probability for each row (0.0 → 1.0).
        Works in both supervised and unsupervised mode.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting. Call .fit() first.")

        available_features = [f for f in ML_FEATURES if f in df.columns]
        X = df[available_features].fillna(0)

        if self.mode == "supervised":
            return self.rf_model.predict_proba(X)[:, 1]   # probability of class 1 (fraud)

        else:  # unsupervised
            X_scaled = self.scaler.transform(X)
            # IsolationForest returns negative scores: more negative = more anomalous
            raw_scores = self.iso_model.score_samples(X_scaled)
            # Convert to 0–1 probability-like score
            min_s, max_s = raw_scores.min(), raw_scores.max()
            normalized = 1 - (raw_scores - min_s) / (max_s - min_s + 1e-9)
            return normalized


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — COMBINE RULES + ML → FINAL SCORE
# ─────────────────────────────────────────────────────────────────────────────

RULE_WEIGHT = 0.45     # rules are highly interpretable → give them good weight
ML_WEIGHT   = 0.55     # ML catches subtle patterns rules miss


def compute_final_scores(df: pd.DataFrame, ml_probs: np.ndarray) -> pd.DataFrame:
    """
    Blend rule-based and ML scores into one final fraud probability.

    Also assigns a human-readable risk label:
      0–30%  → Low
      30–60% → Medium
      60%+   → High
    """
    df = df.copy()

    df["ml_prob"]   = ml_probs
    df["fraud_prob"] = (
        RULE_WEIGHT * df["rule_prob"] + ML_WEIGHT * df["ml_prob"]
    ).clip(0, 1)

    # Convert to percentage for display
    df["fraud_score_pct"] = (df["fraud_prob"] * 100).round(1)

    # Risk label
    df["risk_label"] = pd.cut(
        df["fraud_prob"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Medium", "High"]
    )

    return df


def build_explanation(row: pd.Series) -> str:
    """
    Produce a single plain-English explanation for a flagged transaction.
    This is what gets shown in the UI and reports.

    Example output:
      "⚠️ HIGH RISK — Fraud probability: 78%
       Reasons: IP address used by 5 different users.
                Transaction amount ₹45,000 exceeds safe threshold."
    """
    label = row.get("risk_label", "Low")
    score = row.get("fraud_score_pct", 0)
    reasons = row.get("rule_reasons", [])

    emoji = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(str(label), "❓")

    lines = [f"{emoji} {str(label).upper()} RISK — Fraud probability: {score:.0f}%"]

    if reasons:
        lines.append("Reasons detected:")
        for r in reasons:
            lines.append(f"  • {r}")
    else:
        lines.append("No specific rule violations — flagged by ML pattern detection.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — MASTER DETECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, FraudDetector]:
    """
    Full pipeline: rules → ML → final scores → explanations.

    Parameters
    ----------
    df : feature-engineered DataFrame (output of utils.run_feature_pipeline)

    Returns
    -------
    scored_df  : DataFrame with all original + detection columns
    metrics    : dict of training/model info
    detector   : fitted FraudDetector (for later inspection)
    """
    # Step 1: rules
    df = run_rule_engine(df)

    # Step 2: ML
    detector = FraudDetector()
    metrics  = detector.fit(df)
    ml_probs = detector.predict_proba(df)

    # Step 3: blend
    df = compute_final_scores(df, ml_probs)

    # Step 4: human explanations
    df["explanation"] = df.apply(build_explanation, axis=1)

    return df, metrics, detector


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_generator import generate_dataset
    from utils import run_feature_pipeline

    print("Generating data...")
    raw  = generate_dataset(n=500)
    df   = run_feature_pipeline(raw)

    print("\nRunning detection...")
    scored, metrics, detector = run_detection(df)

    print("\nModel metrics:", metrics)
    print("\nRisk label distribution:")
    print(scored["risk_label"].value_counts())

    print("\nSample HIGH risk explanation:")
    high = scored[scored["risk_label"] == "High"]
    if len(high):
        print(high["explanation"].iloc[0])