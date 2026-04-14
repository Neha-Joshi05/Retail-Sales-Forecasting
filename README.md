# 🛒 Retail Sales Forecasting & Inventory Optimization System

> AI-powered retail sales forecasting and inventory optimization using Random Forest + Streamlit Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Retail businesses lose crores every year due to:
- ❌ Overstocking — too much inventory = money wasted
- ❌ Stockouts — empty shelves = lost sales
- ❌ No demand forecasting — guessing instead of predicting

This project builds an AI system that **forecasts future sales** and **optimizes inventory** automatically.

---

## 🏭 Industry Relevance

Companies actively using this technology:

| Sector | Companies |
|---|---|
| Retail | D-Mart, Reliance Retail, Big Bazaar |
| E-Commerce | Amazon, Flipkart, Meesho |
| Global | Walmart, Target, Zara |
| Tech | TCS, Infosys, Accenture, Wipro |

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| ML Model | Random Forest Regressor |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Persistence | Joblib |

---

## 📁 Project Structure 
# 🛒 Retail Sales Forecasting & Inventory Optimization System

> AI-powered retail sales forecasting and inventory optimization using Random Forest + Streamlit Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Retail businesses lose crores every year due to:
- ❌ Overstocking — too much inventory = money wasted
- ❌ Stockouts — empty shelves = lost sales
- ❌ No demand forecasting — guessing instead of predicting

This project builds an AI system that **forecasts future sales** and **optimizes inventory** automatically.

---

## 🏭 Industry Relevance

Companies actively using this technology:

| Sector | Companies |
|---|---|
| Retail | D-Mart, Reliance Retail, Big Bazaar |
| E-Commerce | Amazon, Flipkart, Meesho |
| Global | Walmart, Target, Zara |
| Tech | TCS, Infosys, Accenture, Wipro |

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| ML Model | Random Forest Regressor |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Persistence | Joblib |

---

## 📁 Project Structure
Retail-Sales-Forecasting/
│
├── data/
│   ├── generate_dataset.py
│   └── retail_sales.csv
│
├── src/
│   ├── preprocess.py
│   └── train_model.py
│
├── models/
│   ├── sales_forecast_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── metrics.txt
│
├── dashboard.py
├── main.py
├── requirements.txt
└── README.md
---

## 🚀 Setup & Installation

```bash
git clone https://github.com/Neha-Joshi05/Retail-Sales-Forecasting.git
cd Retail-Sales-Forecasting

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
# Full pipeline
python main.py

# Launch dashboard
streamlit run dashboard.py
```

---

## 📊 Dashboard Pages

| Page | What you see |
|---|---|
| 📊 Overview | Revenue trend, store comparison, top products |
| 🔮 Sales Forecast | Live predictions with demand category |
| 📦 Inventory Optimizer | Safety stock, reorder points, EOQ |
| 💡 Business Insights | Heatmap, weekday vs weekend, best/worst products |

---

## 📈 Model Results

| Metric | Value |
|---|---|
| Model | Random Forest Regressor |
| MAE | ~5 units |
| RMSE | ~7 units |
| R² | ~0.95 |

---

## 🎓 Learning Outcomes

- Retail data analysis & feature engineering
- Random Forest for sales forecasting
- Inventory optimization (Safety Stock, Reorder Point, EOQ)
- Streamlit dashboard development
- GitHub project documentation

---

## 👤 Author

**Neha Joshi**
[LinkedIn](https://linkedin.com/in/your-profile) · 
[GitHub](https://github.com/Neha-Joshi05)

---

## 📜 License

MIT License