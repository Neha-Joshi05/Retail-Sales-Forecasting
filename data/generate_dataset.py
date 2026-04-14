"""
generate_dataset.py
-------------------
Generates a realistic synthetic retail sales dataset.
Simulates daily sales for 10 products across 3 stores for 2 years.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# --- Configuration ---
stores   = ["Store_A", "Store_B", "Store_C"]
products = [
    "Rice", "Sugar", "Oil", "Soap", "Shampoo",
    "Biscuits", "Chips", "Juice", "Milk", "Bread"
]
date_range = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")

rows = []
for store in stores:
    for product in products:
        base_sales = np.random.randint(20, 100)
        for date in date_range:
            # Seasonal effect
            seasonal = 10 * np.sin(2 * np.pi * date.month / 12)
            # Weekend boost
            weekend  = 15 if date.dayofweek >= 5 else 0
            # Random noise
            noise    = np.random.normal(0, 5)
            sales    = max(0, int(base_sales + seasonal + weekend + noise))
            price    = round(np.random.uniform(10, 500), 2)
            rows.append([date, store, product, sales, price])

df = pd.DataFrame(rows, columns=["Date","Store","Product","Sales","Price"])
df["Revenue"] = df["Sales"] * df["Price"]
df.to_csv("data/retail_sales.csv", index=False)

print(f"✅ Dataset created: {len(df):,} rows")
print(df.head())