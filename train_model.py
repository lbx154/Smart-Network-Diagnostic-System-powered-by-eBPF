#!/usr/bin/env python3
from __future__ import annotations

"""Train anomaly detector or future-health forecaster on SmartNetDiag telemetry."""
import argparse
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type hints
    import pandas as pd


DEFAULT_FEATURES = [
    "avg_rtt_us",
    "p95_rtt_us",
    "retrans_count",
    "rolling_avg_rtt_us",
    "rolling_p95_rtt_us",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Isolation Forest or future-health forecaster for SmartNetDiag")
    parser.add_argument("--data", default="net_data.csv", help="input CSV path")
    parser.add_argument("--model", default="network_model.pkl", help="output model path")
    parser.add_argument("--mode", choices=["iforest", "gb_forecast"], default="iforest", help="training mode")
    parser.add_argument("--contamination", type=float, default=0.2, help="expected anomaly ratio")
    parser.add_argument("--estimators", type=int, default=150, help="number of trees")
    parser.add_argument("--test-size", type=float, default=0.2, help="hold-out ratio for quick validation")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES, help="feature columns to train on")
    parser.add_argument("--lags", type=int, default=5, help="temporal lags for forecasting mode")
    parser.add_argument("--gb-learning-rate", type=float, default=0.05, help="learning rate for Gradient Boosting forecaster")
    parser.add_argument("--gb-estimators", type=int, default=300, help="trees for Gradient Boosting forecaster")
    parser.add_argument("--gb-depth", type=int, default=3, help="tree depth for Gradient Boosting forecaster")
    parser.add_argument("--baseline-rtt-us", type=float, default=40_000.0, help="healthy p95 RTT (microseconds) for health score")
    parser.add_argument("--retrans-budget", type=float, default=2.0, help="acceptable retransmissions per interval")
    return parser.parse_args()


def load_data(path: str, features):
    df = pd.read_csv(path)
    df = df.dropna(subset=features)
    X = df[features]
    return df, X


def compute_health_score(df: pd.DataFrame, baseline_rtt_us: float, retrans_budget: float) -> pd.Series:
    """Derive a 0-1 health score where 1.0 = healthy link."""
    rtt_ratio = df["p95_rtt_us"].fillna(baseline_rtt_us) / max(baseline_rtt_us, 1e-6)
    rtt_score = 1 / (1 + rtt_ratio)  # smooth decay as RTT increases

    retrans_ratio = df["retrans_count"].fillna(0.0) / max(retrans_budget, 1e-6)
    retrans_score = 1 / (1 + retrans_ratio)

    return (0.6 * rtt_score + 0.4 * retrans_score).clip(0.0, 1.0)


def train_model(X, contamination: float, n_estimators: int, random_state: int = 42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler, X_scaled


def train_forecaster(
    df: pd.DataFrame,
    features,
    lags: int,
    baseline_rtt_us: float,
    retrans_budget: float,
    learning_rate: float,
    n_estimators: int,
    depth: int,
):
    df = df.copy()
    df["health_score"] = compute_health_score(df, baseline_rtt_us, retrans_budget)

    for lag in range(1, lags + 1):
        for feat in features:
            df[f"{feat}_lag{lag}"] = df[feat].shift(lag)
    df["target_health"] = df["health_score"].shift(-1)

    df = df.dropna()
    lag_suffixes = tuple(f"lag{lag}" for lag in range(1, lags + 1))
    feature_cols = [col for col in df.columns if col.endswith(lag_suffixes)]

    X = df[feature_cols]
    y = df["target_health"]

    # chronological split to respect temporal order
    split_idx = int(len(df) * (1 - 0.2))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=depth,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
    }

    return model, scaler, feature_cols, metrics, df


def evaluate(model, scaler, df: pd.DataFrame, features, test_size: float):
    _, X_test = train_test_split(df[features], test_size=test_size, random_state=42, shuffle=True)
    X_test_scaled = scaler.transform(X_test)

    df_test = df.loc[X_test.index].copy()
    df_test["prediction"] = model.predict(X_test_scaled)
    df_test["score"] = model.decision_function(X_test_scaled)

    # Assume labels are optional; if absent, skip confusion matrix
    metrics = {}
    if "label" in df.columns:
        y_true = df_test["label"].apply(lambda x: -1 if str(x).lower() == "anomaly" or str(x) == "-1" else 1)
        y_pred = df_test["prediction"]
        cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
        metrics["confusion_matrix"] = cm.tolist()
        metrics["normal_precision"] = cm[0][0] / max(cm[0].sum(), 1)
        metrics["anomaly_precision"] = cm[1][1] / max(cm[1].sum(), 1)
    return df_test, metrics


def plot_results(df: pd.DataFrame, features, out_path: Path):
    plt.figure(figsize=(10, 6))
    colors = df.get("prediction", pd.Series([1] * len(df))).map({1: "green", -1: "red"})
    plt.scatter(df[features[0]], df["retrans_count"], c=colors, alpha=0.6, marker="o")
    plt.xlabel(features[0])
    plt.ylabel("retrans_count")
    title = "Isolation Forest predictions" if "prediction" in df.columns else "Network telemetry"
    plt.title(title + " (green=normal, red=anomaly)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)


def save_artifacts(model, scaler, model_path: str):
    bundle = {"model": model, "scaler": scaler}
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)


def main():
    args = parse_args()
    global pd, plt, IsolationForest, GradientBoostingRegressor, confusion_matrix
    global mean_absolute_error, mean_squared_error, train_test_split, StandardScaler

    import matplotlib.pyplot as plt  # pylint: disable=import-error
    import pandas as pd  # pylint: disable=import-error
    from sklearn.ensemble import GradientBoostingRegressor, IsolationForest  # pylint: disable=import-error
    from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error  # pylint: disable=import-error
    from sklearn.model_selection import train_test_split  # pylint: disable=import-error
    from sklearn.preprocessing import StandardScaler  # pylint: disable=import-error

    df, X = load_data(args.data, args.features)
    print(f"Loaded {len(df)} samples from {args.data} with features {args.features}")

    metrics_payload = {"samples": len(df), "features": args.features}

    if args.mode == "iforest":
        model, scaler, X_scaled = train_model(X, args.contamination, args.estimators)
        df_pred = df.copy()
        df_pred["prediction"] = model.predict(X_scaled)
        df_pred["score"] = model.decision_function(X_scaled)
        df_test, metrics = evaluate(model, scaler, df, args.features, args.test_size)
        metrics_payload.update(
            {
                "mode": "iforest",
                "contamination": args.contamination,
                "estimators": args.estimators,
                "metrics": metrics,
            }
        )

        save_artifacts(model, scaler, args.model)
        print(f"✅ Saved model + scaler to {args.model}")

        result_plot = Path("model_result.png")
        plot_results(df_pred, args.features, result_plot)
        print(f"✅ Saved visualization to {result_plot}")

    else:
        model, scaler, feature_cols, metrics, df_forecast = train_forecaster(
            df,
            args.features,
            args.lags,
            args.baseline_rtt_us,
            args.retrans_budget,
            args.gb_learning_rate,
            args.gb_estimators,
            args.gb_depth,
        )

        bundle = {
            "model": model,
            "scaler": scaler,
            "mode": "gb_forecast",
            "lags": args.lags,
            "feature_columns": feature_cols,
            "baseline_rtt_us": args.baseline_rtt_us,
            "retrans_budget": args.retrans_budget,
        }
        with open(args.model, "wb") as f:
            pickle.dump(bundle, f)
        print(f"✅ Saved forecaster bundle to {args.model}")

        metrics_payload.update(
            {
                "mode": "gb_forecast",
                "lags": args.lags,
                "gb_learning_rate": args.gb_learning_rate,
                "gb_estimators": args.gb_estimators,
                "gb_depth": args.gb_depth,
                "health_metrics": metrics,
            }
        )

        plt.figure(figsize=(10, 5))
        plt.plot(df_forecast.index, df_forecast["health_score"], label="health_score")
        plt.title("Health score timeline")
        plt.xlabel("sample index")
        plt.ylabel("health (0-1)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("model_result.png")
        print("✅ Saved health score plot to model_result.png")

    metrics_path = Path("training_metrics.json")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"✅ Wrote training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
