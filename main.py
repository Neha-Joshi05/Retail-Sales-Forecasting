"""
main.py  –  Retail Sales Forecasting Pipeline
----------------------------------------------
Run this to execute the full pipeline:
  1. Generate dataset
  2. Train model
  3. Save outputs
"""

import os
from src.train_model import train

print("="*55)
print("  🛒 Retail Sales Forecasting & Inventory System")
print("="*55)

# Step 1: Generate dataset
if not os.path.exists("data/retail_sales.csv"):
    print("\n📦 Generating dataset...")
    exec(open("data/generate_dataset.py").read())
else:
    print("\n✅ retail_sales.csv already exists — skipping")

# Step 2: Train model
print("\n🤖 Training model...")
model, scaler, mae, rmse, r2 = train()

print("\n" + "="*55)
print("  ✅ Pipeline Complete!")
print(f"     MAE  : {mae:.2f} units")
print(f"     RMSE : {rmse:.2f} units")
print(f"     R²   : {r2:.4f}")
print("\n  Run dashboard:  streamlit run dashboard.py")
print("="*55)