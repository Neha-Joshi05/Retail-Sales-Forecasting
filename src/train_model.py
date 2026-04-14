"""
train_model.py
--------------
Trains a Random Forest Regressor on retail sales data.
Saves model, scaler, metrics and charts.
"""

import os, sys, joblib, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import load_and_preprocess, get_features_target


def train():
    # ── 1. Load data ─────────────────────────────────────
    df = load_and_preprocess("data/retail_sales.csv")
    X, y = get_features_target(df)

    # ── 2. Train / test split ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. Scale ──────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 4. Train model ────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    print("🔄 Training Random Forest...")
    model.fit(X_train_s, y_train)
    print("✅ Training done!")

    # ── 5. Evaluate ───────────────────────────────────────
    preds = model.predict(X_test_s)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    print(f"\n📊 Results:")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R²   : {r2:.4f}")

    # ── 6. Save model & scaler ────────────────────────────
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model,  "models/sales_forecast_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("💾 Model saved → models/sales_forecast_model.pkl")

    # ── 7. Save metrics ───────────────────────────────────
    with open("outputs/metrics.txt", "w") as f:
        f.write(f"MAE  : {mae:.2f}\n")
        f.write(f"RMSE : {rmse:.2f}\n")
        f.write(f"R²   : {r2:.4f}\n")

    # ── 8. Actual vs Predicted chart ──────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_test.values[:200], label="Actual",    color="#2196F3", lw=1.5)
    ax.plot(preds[:200],         label="Predicted", color="#FF5722", lw=1.5, alpha=0.8)
    ax.set_title("Actual vs Predicted Sales")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
    plt.close()

    # ── 9. Feature importance chart ───────────────────────
    feat_names = X.columns.tolist()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(range(len(feat_names)),
            importances[indices],
            color="#4CAF50", alpha=0.85)
    ax2.set_xticks(range(len(feat_names)))
    ax2.set_xticklabels([feat_names[i] for i in indices], rotation=45)
    ax2.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()

    print("🖼️  Charts saved → outputs/")
    return model, scaler, mae, rmse, r2


if __name__ == "__main__":
    train()