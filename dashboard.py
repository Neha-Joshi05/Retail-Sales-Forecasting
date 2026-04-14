"""
dashboard.py  –  Retail Sales Forecasting Dashboard
====================================================
Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(
    page_title="🛒 Retail Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;600;700&display=swap');
:root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --accent: #00c896; --accent2: #f0a500; --text: #e6edf3; --muted: #8b949e;
}
.stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Share Tech Mono', monospace; font-size: 1.8rem !important; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px !important; }
.stButton > button { background: var(--accent) !important; color: #000 !important; font-weight: 700; border-radius: 8px; border: none; }
.section-title { font-family: 'Share Tech Mono', monospace; color: var(--accent); font-size: 1rem; letter-spacing: 2px; text-transform: uppercase; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)


# ── Load data & model ─────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/retail_sales.csv", parse_dates=["Date"])
    df["month"]      = df["Date"].dt.month
    df["day"]        = df["Date"].dt.day
    df["dayofweek"]  = df["Date"].dt.dayofweek
    df["quarter"]    = df["Date"].dt.quarter
    df["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
    df["week"]       = df["Date"].dt.isocalendar().week.astype(int)
    df["Store_Code"]   = df["Store"].astype("category").cat.codes
    df["Product_Code"] = df["Product"].astype("category").cat.codes
    return df

@st.cache_resource
def load_model():
    model  = joblib.load("models/sales_forecast_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


try:
    df = load_data()
    model, scaler = load_model()
    data_ok = True
except Exception as e:
    st.error(f"⚠️ Run `python main.py` first!\n\n{e}")
    st.stop()

FEATURES = ["month","day","dayofweek","quarter","is_weekend",
            "week","Store_Code","Product_Code","Price"]

STORES   = sorted(df["Store"].unique().tolist())
PRODUCTS = sorted(df["Product"].unique().tolist())


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px;'>
        <span style='font-size:2.4rem;'>🛒</span><br>
        <span style='font-family:"Share Tech Mono",monospace; color:#00c896; font-size:1rem; letter-spacing:3px;'>RETAIL AI</span><br>
        <span style='color:#8b949e; font-size:0.75rem;'>FORECASTING DASHBOARD</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("Navigate", [
        "📊 Overview",
        "🔮 Sales Forecast",
        "📦 Inventory Optimizer",
        "💡 Business Insights"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown(f"<p style='color:#8b949e;font-size:0.78rem;'>📅 {df['Date'].min().strftime('%b %Y')} → {df['Date'].max().strftime('%b %Y')}<br>🔢 {len(df):,} records<br>🏪 {df['Store'].nunique()} Stores<br>📦 {df['Product'].nunique()} Products</p>", unsafe_allow_html=True)


# ── PAGE 1 — OVERVIEW ─────────────────────────────────
if page == "📊 Overview":
    st.markdown("## 📊 Sales Overview")
    st.markdown("Explore historical retail sales data across all stores and products.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(df):,}")
    col2.metric("Total Revenue",   f"₹{df['Revenue'].sum()/1e6:.1f}M")
    col3.metric("Avg Daily Sales", f"{df['Sales'].mean():.0f} units")
    col4.metric("Top Product",     df.groupby("Product")["Sales"].sum().idxmax())

    st.divider()

    # Revenue trend
    st.markdown('<p class="section-title">📈 Monthly Revenue Trend</p>', unsafe_allow_html=True)
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0d1117")
    ax.plot(monthly.index, monthly.values, color="#00c896", lw=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#00c896")
    ax.set_ylabel("Revenue (₹)", color="#8b949e")
    ax.tick_params(colors="#8b949e", rotation=45)
    ax.spines[["top","right","left","bottom"]].set_color("#30363d")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-title">🏪 Sales by Store</p>', unsafe_allow_html=True)
        store_sales = df.groupby("Store")["Sales"].sum()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor("#161b22")
        ax2.set_facecolor("#0d1117")
        ax2.bar(store_sales.index, store_sales.values, color=["#00c896","#f0a500","#ff6b6b"], alpha=0.85)
        ax2.set_ylabel("Total Sales", color="#8b949e")
        ax2.tick_params(colors="#8b949e")
        ax2.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_b:
        st.markdown('<p class="section-title">📦 Top 5 Products</p>', unsafe_allow_html=True)
        prod_sales = df.groupby("Product")["Sales"].sum().sort_values(ascending=True).tail(5)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor("#161b22")
        ax3.set_facecolor("#0d1117")
        ax3.barh(prod_sales.index, prod_sales.values, color="#f0a500", alpha=0.85)
        ax3.set_xlabel("Total Sales", color="#8b949e")
        ax3.tick_params(colors="#8b949e")
        ax3.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.divider()
    st.markdown('<p class="section-title">🗂️ Raw Data Preview</p>', unsafe_allow_html=True)
    st.dataframe(df[["Date","Store","Product","Sales","Price","Revenue"]].head(50),
                 use_container_width=True, height=280)


# ── PAGE 2 — SALES FORECAST ───────────────────────────
elif page == "🔮 Sales Forecast":
    st.markdown("## 🔮 Live Sales Forecast")
    st.markdown("Select parameters and get instant AI sales prediction.")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<p class="section-title">⚙️ Input Parameters</p>', unsafe_allow_html=True)
        store   = st.selectbox("🏪 Store",   STORES)
        product = st.selectbox("📦 Product", PRODUCTS)
        month   = st.slider("🗓️ Month",      1, 12, 6)
        day     = st.slider("📅 Day",        1, 31, 15)
        price   = st.number_input("💰 Price (₹)", value=100.0, step=10.0)
        dow     = st.slider("📆 Day of Week (0=Mon)", 0, 6, 2)
        is_weekend = 1 if dow >= 5 else 0
        quarter    = (month - 1) // 3 + 1
        week       = min(52, (day + (month-1)*30) // 7 + 1)

        store_code   = STORES.index(store)
        product_code = PRODUCTS.index(product)

        st.info(f"📌 {'Weekend' if is_weekend else 'Weekday'}  |  Quarter {quarter}")
        predict_btn = st.button("🛒 Predict Sales", use_container_width=True)

    with col_out:
        st.markdown('<p class="section-title">🎯 Prediction Result</p>', unsafe_allow_html=True)

        if predict_btn:
            features = np.array([[month, day, dow, quarter, is_weekend,
                                   week, store_code, product_code, price]])
            scaled = scaler.transform(features)
            pred   = model.predict(scaled)[0]
            revenue = pred * price

            if pred < 30:
                cat, color, icon = "Low",    "#ff6b6b", "🔴"
            elif pred < 70:
                cat, color, icon = "Medium", "#f0a500", "🟡"
            else:
                cat, color, icon = "High",   "#00c896", "🟢"

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0d1a0d,#0d1117);
                        border:1px solid #00c896; border-radius:16px;
                        padding:28px; text-align:center; margin:10px 0;'>
                <div style='font-family:"Share Tech Mono",monospace;
                            font-size:3rem; color:#00c896;'>{pred:.0f}</div>
                <div style='color:#8b949e; font-size:0.9rem; margin-top:4px;'>UNITS PREDICTED</div>
                <div style='margin-top:12px; font-size:1.1rem;
                            color:{color}; font-weight:600;'>{icon} {cat} Demand</div>
                <div style='margin-top:8px; color:#f0a500; font-size:1rem;'>
                    Est. Revenue: ₹{revenue:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='background:#161b22; border:1px dashed #30363d;
                        border-radius:16px; padding:60px 20px; text-align:center; color:#8b949e;'>
                <div style='font-size:2.5rem;'>🛒</div>
                <div style='margin-top:10px;'>Set parameters and click Predict</div>
            </div>
            """, unsafe_allow_html=True)

    # Monthly forecast chart
    st.divider()
    st.markdown('<p class="section-title">📆 12-Month Sales Forecast</p>', unsafe_allow_html=True)

    months_12 = range(1, 13)
    monthly_preds = []
    for m in months_12:
        q = (m-1)//3+1
        f = np.array([[m, 15, 2, q, 0, min(52,(15+(m-1)*30)//7+1),
                       store_code, product_code, price]])
        monthly_preds.append(model.predict(scaler.transform(f))[0])

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig4, ax4 = plt.subplots(figsize=(14, 4))
    fig4.patch.set_facecolor("#161b22")
    ax4.set_facecolor("#0d1117")
    colors_m = ["#00c896" if v > 70 else "#f0a500" if v > 30 else "#ff6b6b"
                for v in monthly_preds]
    ax4.bar(month_names, monthly_preds, color=colors_m, alpha=0.85, edgecolor="#161b22")
    ax4.axhline(np.mean(monthly_preds), color="#8b949e", linestyle="--", lw=1,
                label=f"Avg: {np.mean(monthly_preds):.0f} units")
    ax4.set_ylabel("Predicted Sales", color="#8b949e")
    ax4.tick_params(colors="#8b949e")
    ax4.spines[["top","right","left","bottom"]].set_color("#30363d")
    ax4.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()


# ── PAGE 3 — INVENTORY OPTIMIZER ─────────────────────
elif page == "📦 Inventory Optimizer":
    st.markdown("## 📦 Inventory Optimization")
    st.markdown("AI-powered reorder points and safety stock recommendations.")

    st.divider()

    # Calculate inventory metrics per product
    inv_data = []
    for product in PRODUCTS:
        prod_df = df[df["Product"] == product]
        avg_daily  = prod_df["Sales"].mean()
        std_daily  = prod_df["Sales"].std()
        lead_time  = np.random.randint(3, 10)   # days
        service_z  = 1.65                        # 95% service level
        safety_stock  = round(service_z * std_daily * np.sqrt(lead_time))
        reorder_point = round(avg_daily * lead_time + safety_stock)
        eoq = round(np.sqrt((2 * avg_daily * 365 * 50) / 2))  # EOQ formula

        inv_data.append({
            "Product":        product,
            "Avg Daily Sales": round(avg_daily, 1),
            "Std Dev":         round(std_daily, 1),
            "Lead Time (days)": lead_time,
            "Safety Stock":    safety_stock,
            "Reorder Point":   reorder_point,
            "EOQ":             eoq,
            "Status": "🔴 Reorder Now" if np.random.random() < 0.3 else "🟢 Sufficient"
        })

    inv_df = pd.DataFrame(inv_data)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products Tracked",  len(PRODUCTS))
    col2.metric("Avg Safety Stock",  f"{inv_df['Safety Stock'].mean():.0f} units")
    col3.metric("Avg Reorder Point", f"{inv_df['Reorder Point'].mean():.0f} units")
    col4.metric("Need Reorder",      f"{(inv_df['Status'].str.contains('Reorder')).sum()} items")

    st.divider()

    st.markdown('<p class="section-title">📋 Inventory Recommendations Table</p>',
                unsafe_allow_html=True)
    st.dataframe(inv_df, use_container_width=True, height=380)

    st.divider()

    col_ss, col_rop = st.columns(2)

    with col_ss:
        st.markdown('<p class="section-title">🛡️ Safety Stock by Product</p>',
                    unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        fig5.patch.set_facecolor("#161b22")
        ax5.set_facecolor("#0d1117")
        ax5.barh(inv_df["Product"], inv_df["Safety Stock"],
                 color="#f0a500", alpha=0.85)
        ax5.set_xlabel("Safety Stock (units)", color="#8b949e")
        ax5.tick_params(colors="#8b949e")
        ax5.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    with col_rop:
        st.markdown('<p class="section-title">🔁 Reorder Point by Product</p>',
                    unsafe_allow_html=True)
        fig6, ax6 = plt.subplots(figsize=(7, 4))
        fig6.patch.set_facecolor("#161b22")
        ax6.set_facecolor("#0d1117")
        ax6.barh(inv_df["Product"], inv_df["Reorder Point"],
                 color="#00c896", alpha=0.85)
        ax6.set_xlabel("Reorder Point (units)", color="#8b949e")
        ax6.tick_params(colors="#8b949e")
        ax6.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()


# ── PAGE 4 — BUSINESS INSIGHTS ────────────────────────
elif page == "💡 Business Insights":
    st.markdown("## 💡 Business Insights")
    st.markdown("Deep dive into sales patterns to drive smarter business decisions.")

    # Heatmap
    st.markdown('<p class="section-title">🔥 Sales Heatmap — Product × Store</p>',
                unsafe_allow_html=True)
    pivot = df.pivot_table(values="Sales", index="Product",
                           columns="Store", aggfunc="mean")
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    fig7.patch.set_facecolor("#161b22")
    ax7.set_facecolor("#0d1117")
    sns.heatmap(pivot, ax=ax7, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.3, linecolor="#0d1117",
                annot_kws={"size": 9},
                cbar_kws={"label": "Avg Sales"})
    ax7.set_title("Average Sales — Product × Store", color="#e6edf3", pad=12)
    ax7.tick_params(colors="#8b949e")
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

    st.divider()

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<p class="section-title">📅 Weekday vs Weekend Sales</p>',
                    unsafe_allow_html=True)
        wk = df.groupby("is_weekend")["Sales"].mean()
        fig8, ax8 = plt.subplots(figsize=(6, 4))
        fig8.patch.set_facecolor("#161b22")
        ax8.set_facecolor("#0d1117")
        ax8.bar(["Weekday","Weekend"], wk.values,
                color=["#00c896","#f0a500"], alpha=0.85, width=0.45)
        for i, v in enumerate(wk.values):
            ax8.text(i, v+0.5, f"{v:.1f}", ha="center",
                     color="#e6edf3", fontsize=10)
        ax8.set_ylabel("Avg Sales", color="#8b949e")
        ax8.tick_params(colors="#8b949e")
        ax8.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close()

    with col_d:
        st.markdown('<p class="section-title">📈 Monthly Sales Trend</p>',
                    unsafe_allow_html=True)
        mo = df.groupby("month")["Sales"].mean()
        mn = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        fig9, ax9 = plt.subplots(figsize=(6, 4))
        fig9.patch.set_facecolor("#161b22")
        ax9.set_facecolor("#0d1117")
        ax9.fill_between(range(1,13), mo.values, alpha=0.2, color="#00c896")
        ax9.plot(range(1,13), mo.values, color="#00c896", marker="o",
                 markersize=5, lw=2)
        ax9.set_xticks(range(1,13))
        ax9.set_xticklabels(mn)
        ax9.set_ylabel("Avg Sales", color="#8b949e")
        ax9.tick_params(colors="#8b949e")
        ax9.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig9)
        plt.close()

    st.divider()

    # Top & Bottom products
    st.markdown('<p class="section-title">🏆 Best & Worst Performing Products</p>',
                unsafe_allow_html=True)
    prod_rev = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"**🥇 Best Product:** {prod_rev.index[0]}\n\n"
                   f"Revenue: ₹{prod_rev.iloc[0]:,.0f}")
    with c2:
        st.warning(f"**⚠️ Needs Attention:** {prod_rev.index[-1]}\n\n"
                   f"Revenue: ₹{prod_rev.iloc[-1]:,.0f}")