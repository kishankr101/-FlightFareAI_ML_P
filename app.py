
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlightFareAI – Indian Domestic Price Predictor",
    page_icon="✈",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main { background: #060b14; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Metric cards */
.metric-card {
    background: #0c1829;
    border: 1px solid #1a2e4a;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.metric-label { font-size: 11px; color: #4a8caa; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 800; color: #00f5d4; font-family: 'Syne', sans-serif; }
.metric-sub   { font-size: 12px; color: #4a6080; margin-top: 2px; }

/* Tag badges */
.tag { display: inline-block; background: rgba(0,245,212,0.1); border: 1px solid rgba(0,245,212,0.25);
       border-radius: 20px; padding: 3px 12px; font-size: 12px; color: #00f5d4; margin: 3px; }
.tag-yellow { background: rgba(247,183,49,0.1); border-color: rgba(247,183,49,0.3); color: #f7b731; }
.tag-grey   { background: rgba(74,96,128,0.15); border-color: #1e3050; color: #8aa0bb; }

/* Result box */
.result-box {
    background: linear-gradient(135deg, #0c1829, #0d1f38);
    border: 1px solid #1a2e4a;
    border-radius: 16px;
    padding: 24px;
}
.price-big { font-size: 48px; font-weight: 800; color: #00f5d4; font-family: 'Syne', sans-serif; letter-spacing: -1px; }

/* Tip box */
.tip-box {
    background: rgba(247,183,49,0.06);
    border: 1px solid rgba(247,183,49,0.2);
    border-radius: 10px;
    padding: 14px;
    margin-top: 12px;
}
.tip-label { font-size: 11px; color: #f7b731; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
.tip-text  { font-size: 13px; color: #c8a84b; line-height: 1.5; }

/* Section headers */
.section-title {
    font-size: 11px; color: #4a8caa; text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 10px; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading & model training ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_and_train(csv_path: str):
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    df.drop(columns=["Unnamed: 0", "flight", "days_left"], errors="ignore", inplace=True)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    label_maps = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        label_maps[col] = {cls: idx for idx, cls in enumerate(le.classes_)}

    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics.append({
            "Model": name,
            "R² Score": round(r2_score(y_test, preds), 4),
            "MAE (₹)": round(mean_absolute_error(y_test, preds), 2),
        })

    return df, encoders, label_maps, models, pd.DataFrame(metrics), X.columns.tolist()


# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown('<div style="font-size:40px;margin-top:6px">✈</div>', unsafe_allow_html=True)
with col_title:
    st.markdown(
        '<h1 style="margin:0;font-size:28px">FlightFare<span style="color:#00f5d4">AI</span></h1>'
        '<p style="margin:0;color:#4a6080;font-size:12px;letter-spacing:0.5px;text-transform:uppercase">'
        'Indian Domestic Flight Price Predictor · Trained on 300 K+ Records</p>',
        unsafe_allow_html=True,
    )

st.markdown('<hr style="border-color:#1a2e4a;margin:12px 0 20px">', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict Price", "📊 Model Insights", "📖 About Project"])


# ════════════════════════════════════════════════════════════════
#  TAB 1 – PREDICT
# ════════════════════════════════════════════════════════════════
with tab_predict:
    CSV_PATH = st.text_input(
        "Path to Clean_Dataset.csv",
        value="Clean_Dataset.csv",
        help="Provide the path to your dataset so the model can train.",
    )

    try:
        df, encoders, label_maps, models, metrics_df, feature_cols = load_and_train(CSV_PATH)
        data_loaded = True
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Upload the CSV or update the path above.")
        data_loaded = False

    if data_loaded:
        AIRLINES   = sorted(encoders["airline"].classes_.tolist())
        CITIES     = sorted(encoders["source_city"].classes_.tolist())
        TIMES      = sorted(encoders["departure_time"].classes_.tolist())
        STOPS      = sorted(encoders["stops"].classes_.tolist())
        CLASSES    = sorted(encoders["class"].classes_.tolist())

        st.markdown("### Configure Your Flight")

        c1, c2, c3 = st.columns(3)
        with c1:
            airline = st.selectbox("✈ Airline", AIRLINES)
        with c2:
            flight_class = st.selectbox("💺 Class", CLASSES)
        with c3:
            stops = st.selectbox("🔄 Stops", STOPS)

        c4, c5 = st.columns(2)
        with c4:
            source_city = st.selectbox("🛫 Source City", CITIES)
            departure_time = st.selectbox("🕐 Departure Time", TIMES)
        with c5:
            destination_city = st.selectbox("🛬 Destination City", CITIES)
            arrival_time = st.selectbox("🕔 Arrival Time", TIMES)

        duration = st.slider("⏱ Flight Duration (hours)", 0.5, 50.0, 2.5, step=0.25)

        st.markdown("")
        predict_btn = st.button("✈  Predict Ticket Price", use_container_width=True, type="primary")

        if predict_btn:
            # Build input row in correct column order
            input_dict = {
                "airline":          label_maps["airline"].get(airline, 0),
                "source_city":      label_maps["source_city"].get(source_city, 0),
                "departure_time":   label_maps["departure_time"].get(departure_time, 0),
                "stops":            label_maps["stops"].get(stops, 0),
                "arrival_time":     label_maps["arrival_time"].get(arrival_time, 0),
                "destination_city": label_maps["destination_city"].get(destination_city, 0),
                "class":            label_maps["class"].get(flight_class, 0),
                "duration":         duration,
            }
            X_input = pd.DataFrame([input_dict])[feature_cols]

            rf_model  = models["Random Forest"]
            dt_model  = models["Decision Tree"]
            lr_model  = models["Linear Regression"]

            rf_pred = rf_model.predict(X_input)[0]
            dt_pred = dt_model.predict(X_input)[0]
            lr_pred = lr_model.predict(X_input)[0]

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            # Main result cards
            r1, r2, r3 = st.columns(3)
            for col, name, pred, clr in [
                (r1, "Random Forest 🏆", rf_pred, "#00f5d4"),
                (r2, "Decision Tree",    dt_pred, "#7b2ff7"),
                (r3, "Linear Regression",lr_pred, "#f7b731"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">{name}</div>'
                        f'<div class="metric-value" style="color:{clr}">₹{int(pred):,}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Summary box
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="metric-label">Best Estimate (Random Forest · R² = 97.5%)</div>
                    <div class="price-big">₹{int(rf_pred):,}</div>
                    <div class="metric-sub">
                        Range approx: ₹{int(rf_pred * 0.88):,} – ₹{int(rf_pred * 1.12):,}
                    </div>
                    <br>
                    <span class="tag">{airline}</span>
                    <span class="tag-yellow">{flight_class}</span>
                    <span class="tag-grey">{source_city} → {destination_city}</span>
                    <span class="tag-grey">{stops} stop(s) · {duration}h</span>
                    <div class="tip-box">
                        <div class="tip-label">💡 Money-Saving Tip</div>
                        <div class="tip-text">
                            {'Book 3–4 weeks in advance and prefer Early Morning flights — they are consistently cheaper across all airlines.' if flight_class == 'Economy' else 'Business class fares drop slightly on weekday morning departures. Compare Vistara vs Air India for the best premium deal.'}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Mini comparison bar chart
            st.markdown("#### Model Comparison for This Flight")
            fig, ax = plt.subplots(figsize=(7, 2.8))
            fig.patch.set_facecolor("#0c1829")
            ax.set_facecolor("#0c1829")
            model_names = ["Random Forest", "Decision Tree", "Linear Regression"]
            preds_vals  = [rf_pred, dt_pred, lr_pred]
            colors      = ["#00f5d4", "#7b2ff7", "#f7b731"]
            bars = ax.barh(model_names, preds_vals, color=colors, height=0.5)
            for bar, val in zip(bars, preds_vals):
                ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                        f"₹{int(val):,}", va="center", color="white", fontsize=10)
            ax.set_xlabel("Predicted Price (₹)", color="#4a6080")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#1a2e4a")
            st.pyplot(fig)
            plt.close()


# ════════════════════════════════════════════════════════════════
#  TAB 2 – INSIGHTS
# ════════════════════════════════════════════════════════════════
with tab_insights:
    CSV_PATH2 = st.text_input(
        "Path to Clean_Dataset.csv ",
        value="Clean_Dataset.csv",
        key="csv2",
    )
    try:
        df2, _, _, _, metrics_df2, _ = load_and_train(CSV_PATH2)
        ins_loaded = True
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found.")
        ins_loaded = False

    if ins_loaded:
        st.markdown("### 📊 Model Performance")
        styled = metrics_df2.style \
            .background_gradient(subset=["R² Score"], cmap="YlGn") \
            .format({"R² Score": "{:.4f}", "MAE (₹)": "₹{:,.2f}"})
        st.dataframe(styled, use_container_width=True)

        # R² bar chart
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor("#0c1829")
        ax2.set_facecolor("#0c1829")
        clrs = ["#00f5d4", "#7b2ff7", "#f7b731"]
        ax2.bar(metrics_df2["Model"], metrics_df2["R² Score"], color=clrs, width=0.5)
        ax2.set_ylim(0.85, 1.0)
        ax2.set_ylabel("R² Score", color="#4a6080")
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#1a2e4a")
        ax2.set_title("Which Model is Most Accurate?", color="white", fontsize=13)
        for i, v in enumerate(metrics_df2["R² Score"]):
            ax2.text(i, v + 0.002, f"{v:.4f}", ha="center", color="white", fontsize=10)
        st.pyplot(fig2)
        plt.close()

        st.markdown("---")
        st.markdown("### ✈ Airline vs Ticket Price")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        fig3.patch.set_facecolor("#0c1829")
        ax3.set_facecolor("#0c1829")
        sns.boxplot(
            x="airline", y="price", data=df2,
            palette="coolwarm", ax=ax3,
            order=sorted(df2["airline"].unique()),
        )
        ax3.set_title("Airline vs Ticket Price (Encoded)", color="white")
        ax3.tick_params(colors="white")
        ax3.set_xlabel("Airline (encoded)", color="#4a6080")
        ax3.set_ylabel("Price (₹)", color="#4a6080")
        ax3.spines[:].set_color("#1a2e4a")
        st.pyplot(fig3)
        plt.close()

        st.markdown("### 💺 Class vs Ticket Price")
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        fig4.patch.set_facecolor("#0c1829")
        ax4.set_facecolor("#0c1829")
        sns.barplot(x="class", y="price", data=df2, palette=["#00f5d4", "#f7b731"], ax=ax4)
        ax4.set_title("Economy vs Business Price", color="white")
        ax4.tick_params(colors="white")
        ax4.set_xlabel("Class (encoded)", color="#4a6080")
        ax4.set_ylabel("Avg Price (₹)", color="#4a6080")
        ax4.spines[:].set_color("#1a2e4a")
        st.pyplot(fig4)
        plt.close()

        st.markdown("### 📈 Price Distribution")
        fig5, ax5 = plt.subplots(figsize=(10, 3))
        fig5.patch.set_facecolor("#0c1829")
        ax5.set_facecolor("#0c1829")
        ax5.hist(df2["price"], bins=80, color="#00f5d4", alpha=0.75, edgecolor="#0c1829")
        ax5.set_title("Flight Price Distribution", color="white")
        ax5.tick_params(colors="white")
        ax5.set_xlabel("Price (₹)", color="#4a6080")
        ax5.set_ylabel("Count", color="#4a6080")
        ax5.spines[:].set_color("#1a2e4a")
        st.pyplot(fig5)
        plt.close()


# ════════════════════════════════════════════════════════════════
#  TAB 3 – ABOUT
# ════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## 📖 About This Project")
    st.markdown("""
**Problem Statement**

This project predicts Indian domestic flight ticket prices based on multiple factors using supervised machine learning. It helps travelers estimate fair ticket prices and understand the key drivers of pricing differences.

---

**Dataset**
- **Source**: Indian domestic flight records
- **Size**: 300,153 rows, 12 columns
- **Target**: `price` (₹1,105 – ₹1,23,071)

**Features Used (after cleaning)**
| Feature | Type | Description |
|---|---|---|
| `airline` | Categorical | SpiceJet, AirAsia, Vistara, GO_FIRST, Indigo, Air_India |
| `source_city` | Categorical | Departure city |
| `destination_city` | Categorical | Arrival city |
| `departure_time` | Categorical | Time of day |
| `arrival_time` | Categorical | Time of day |
| `stops` | Categorical | zero / one / two_or_more |
| `class` | Categorical | Economy / Business |
| `duration` | Numeric | Flight duration in hours |

**Dropped columns**: `Unnamed: 0`, `flight` (ID), `days_left` (not used in final model)

---

**Pipeline**
1. Load & explore data (`df.head()`, `df.info()`, `df.describe()`)
2. Check & handle missing values → **zero missing values**
3. Drop irrelevant columns
4. Label encode all categorical features
5. Train/test split (80/20)
6. Train 3 models and compare

**Model Results**
| Model | R² Score | MAE |
|---|---|---|
| 🏆 Random Forest | **0.9754** | ₹2,163 |
| Decision Tree | 0.9749 | ₹2,159 |
| Linear Regression | 0.8983 | ₹4,911 |

**Winner: Random Forest** — explains 97.5% of price variance with an average error of just ₹2,163.

---
""")

    st.markdown(
        '<span class="tag">Regression</span>'
        '<span class="tag">EDA</span>'
        '<span class="tag">Label Encoding</span>'
        '<span class="tag">Scikit-learn</span>'
        '<span class="tag">300K Samples</span>'
        '<span class="tag">Random Forest</span>',
        unsafe_allow_html=True,
    )
    st.markdown("""
---
**How to Run**
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
streamlit run flight_price_predictor.py
```
Make sure `Clean_Dataset.csv` is in the same folder (or update the path in the app).
""")


