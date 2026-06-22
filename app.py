# FlightFareAI Pro - Streamlit app

# Single-file Streamlit application implementing the "Predict Flights" experience.
# - Attempts to load models/model.pkl and models/encoder.pkl via joblib.
# - If model files are missing or loading/prediction fails, runs in demo mode (synthetic predictions).

import streamlit as st
import pandas as pd
import numpy as np
#import joblib  # moved to lazy import inside loader
import os
import time
from typing import Tuple, List, Dict
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# App configuration & styles
# -----------------------------
st.set_page_config(page_title="FlightFareAI Pro", layout="wide", initial_sidebar_state="expanded")

# Initialize session state keys used for navigation
if 'nav_to' not in st.session_state:
    st.session_state['nav_to'] = None

# Load local CSS for dark glassmorphism UI
def load_css():
    css_path = os.path.join('assets', 'styles.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Minimal fallback styles
        st.markdown(
            """
            <style>
            .stApp { background-color: #0f1724; color: #e6eef8; }
            .glass { background: rgba(255,255,255,0.03); border-radius:12px; padding:16px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

load_css()

# -----------------------------
# Model service
# -----------------------------
MODEL_PATH = os.path.join('models', 'model.pkl')
ENCODER_PATH = os.path.join('models', 'encoder.pkl')

@st.cache_resource
def load_model_and_encoder() -> Tuple[object, object, bool]:
    """Try to load model and encoder. Returns (model, encoder, success_flag).
    If files are missing or loading fails, returns (None, None, False).
    Uses lazy imports so the app can run in demo mode even if heavy ML packages are not installed.
    Additionally falls back to models/demo_model.DemoModel and DemoEncoder when available.
    """
    try:
        # Lazy import joblib to avoid ModuleNotFoundError on import time in environments where
        # scikit-learn/joblib aren't installed. If joblib is missing we fall back to demo model.
        try:
            import joblib
            have_joblib = True
        except Exception:
            have_joblib = False

        # If model files exist and joblib is available, try to load them
        if have_joblib and os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                encoder = joblib.load(ENCODER_PATH)
                return model, encoder, True
            except Exception as e:
                st.warning(f"Failed to load model/encoder: {e}")
                # fall through to demo model

        # Next fallback: if a demo_model module exists in models/, import it and use
        try:
            from models.demo_model import DemoModel, DemoEncoder
            demo_encoder = DemoEncoder()
            demo_model = DemoModel()
            return demo_model, demo_encoder, True
        except Exception:
            # Final fallback: indicate no model available; app will use pure demo_predict_all
            return None, None, False
    except Exception as e:
        # Unexpected errors should not crash the app; fall back to demo mode.
        st.warning(f"Error while checking model files: {e}")
        return None, None, False


# -----------------------------
# Prediction logic
# -----------------------------

DEFAULT_AIRLINES = [
    'IndiGo', 'Air India', 'SpiceJet', 'Vistara', 'GoAir', 'AirAsia India', 'Alliance Air', 'TruJet'
]

CLASS_OPTIONS = ['Economy', 'Business', 'Premium Economy']


def demo_predict_all(input_features: Dict) -> List[Dict]:
    """Generate deterministic synthetic predictions for demo mode.
    input_features: dict with keys: source, destination, dep_time, arr_time, stops, travel_class, duration
    Returns list of dict {'airline': name, 'price': float}
    """
    base = 2500  # base price currency INR
    results = []
    np.random.seed(sum(map(ord, input_features.get('source','') + input_features.get('destination','')))%1000)
    for i, al in enumerate(DEFAULT_AIRLINES):
        airline_modifier = (i+1) * 150 + (len(al) % 5) * 20
        stops = int(input_features.get('stops', 0))
        duration = float(input_features.get('duration', 60))
        class_mod = {'Economy': 0, 'Premium Economy': 800, 'Business': 2000}.get(input_features.get('travel_class','Economy'), 0)
        # Simple heuristic
        price = base + duration * 8 + stops * 600 + airline_modifier + class_mod
        # small deterministic noise
        price *= 1 + ((i - 2) * 0.02)
        price = max(500, round(price, 0))
        results.append({'airline': al, 'price': float(price)})
    return results


def predict_with_model(model, encoder, input_features: Dict) -> List[Dict]:
    """Attempt to predict for each airline using the provided model and encoder.
    The exact feature engineering must match how model was trained.
    We attempt a best-effort mapping using common features; if that fails we raise an exception to trigger demo mode.
    """
    airlines = []
    # Try to get list of airlines from encoder (supports scikit-learn LabelEncoder or similar)
    try:
        if hasattr(encoder, 'classes_'):
            airlines = list(encoder.classes_)
        elif isinstance(encoder, list) or isinstance(encoder, tuple):
            airlines = list(encoder)
        else:
            # fallback
            airlines = DEFAULT_AIRLINES
    except Exception:
        airlines = DEFAULT_AIRLINES

    results = []
    # Prepare a DataFrame template. THIS MUST MATCH training features used for the model.
    # We make a conservative guess for common column names. If the model expects a different shape,
    # model.predict will raise and we fall back to demo.
    rows = []
    for al in airlines:
        row = {
            'Source': input_features.get('source', ''),
            'Destination': input_features.get('destination', ''),
            'Airline': al,
            'Dep_Time': input_features.get('dep_time', ''),
            'Arrival_Time': input_features.get('arr_time', ''),
            'Stops': int(input_features.get('stops', 0)),
            'Class': input_features.get('travel_class', 'Economy'),
            'Duration': float(input_features.get('duration', 60)),
        }
        rows.append(row)
    X = pd.DataFrame(rows)

    # Try to run model.predict. Many production models require preprocessing pipeline that handles strings.
    preds = model.predict(X)
    for al, p in zip(airlines, preds):
        results.append({'airline': al, 'price': float(np.round(float(p), 0))})
    return results


def get_ranked_results(input_features: Dict) -> Tuple[List[Dict], bool]:
    """Return sorted list of {'airline','price'} and a boolean 'is_demo' indicating whether demo mode used."""
    model, encoder, loaded = load_model_and_encoder()
    if loaded and model is not None and encoder is not None:
        try:
            results = predict_with_model(model, encoder, input_features)
            # Sort ascending
            results = sorted(results, key=lambda x: x['price'])
            return results, False
        except Exception as e:
            # Fallback to demo mode
            st.warning(f"Model prediction failed; switching to demo mode. ({e})")
            results = demo_predict_all(input_features)
            results = sorted(results, key=lambda x: x['price'])
            return results, True
    else:
        # Demo mode
        results = demo_predict_all(input_features)
        results = sorted(results, key=lambda x: x['price'])
        return results, True


# -----------------------------
# UI helpers & components
# -----------------------------

def currency(x):
    return f"₹{int(x):,}"


def price_category(price, mean_price):
    if price <= 0.8 * mean_price:
        return 'Cheap'
    elif price >= 1.3 * mean_price:
        return 'Expensive'
    else:
        return 'Average'


# -----------------------------
# Pages
# -----------------------------

def page_predict():
    st.title("Predict Flights — FlightFareAI Pro")
    st.markdown("""
    <div class='glass'>
    <p style='color:#cfe9ff'>Enter flight details below and click <b>Find Best Flights</b> to get ranked airline price predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    with st.form(key='predict_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            source = st.text_input('Source City', value='Mumbai')
            dep_time = st.time_input('Departure Time')
            stops = st.number_input('Number of Stops', min_value=0, max_value=5, value=0, step=1)
        with col2:
            destination = st.text_input('Destination City', value='Delhi')
            arr_time = st.time_input('Arrival Time')
            travel_class = st.selectbox('Travel Class', CLASS_OPTIONS)
        with col3:
            duration = st.number_input('Flight Duration (minutes)', min_value=30, max_value=1440, value=120, step=5)
            st.markdown('<br/>', unsafe_allow_html=True)
            find_btn = st.form_submit_button('Find Best Flights')

    if find_btn:
        # Prepare input features
        input_features = {
            'source': source.strip(),
            'destination': destination.strip(),
            'dep_time': dep_time.strftime('%H:%M'),
            'arr_time': arr_time.strftime('%H:%M'),
            'stops': stops,
            'travel_class': travel_class,
            'duration': duration,
        }

        # Loading spinner while predicting
        with st.spinner('Finding best flights...'):
            time.sleep(0.7)  # small UX pause
            results, is_demo = get_ranked_results(input_features)

        # Display summary metrics
        prices = [r['price'] for r in results]
        cheapest = results[0]
        most_expensive = results[-1]
        mean_price = np.mean(prices)

        # Cheapest Flight Card
        col1, col2, col3 = st.columns([3,2,2])
        with col1:
            st.markdown("""
            <div class='glass' style='padding:18px'>
            <h3 style='margin:0'>Cheapest Flight</h3>
            <h1 style='margin:6px 0'>{airline}</h1>
            <h2 style='margin:6px 0; color:#7ee787'>{price}</h2>
            <p style='color:#cfe9ff'>Savings: <b>{savings}</b> vs average</p>
            </div>
            """.format(airline=cheapest['airline'], price=currency(cheapest['price']), savings=currency(int(mean_price - cheapest['price']))), unsafe_allow_html=True)
        with col2:
            st.metric('Most Expensive', value=currency(most_expensive['price']), delta=most_expensive['airline'])
        with col3:
            st.metric('Average Price', value=currency(int(mean_price)))

        # Top 5 Cheapest
        st.subheader('Top 5 Cheapest Flights')
        top5 = results[:5]
        top5_df = pd.DataFrame(top5).reset_index().rename(columns={'index':'Rank', 'airline':'Airline','price':'Price'})
        top5_df['Rank'] = top5_df['Rank'] + 1
        top5_df['Price'] = top5_df['Price'].apply(currency)
        st.table(top5_df)

        # Premium & Best Value recommendations
        st.subheader('Recommendations')
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.markdown(f"**Premium Recommendation** (Most expensive): {most_expensive['airline']} — {currency(most_expensive['price'])}")
        with rcol2:
            # Best value: pick the airline closest to median price but below mean
            med = np.median(prices)
            best_value = min(results, key=lambda x: abs(x['price'] - med))
            st.markdown(f"**Best Value Recommendation**: {best_value['airline']} — {currency(best_value['price'])}")

        # Price comparison chart
        st.subheader('Price Comparison')
        df_plot = pd.DataFrame(results)
        fig = px.bar(df_plot, x='price', y='airline', orientation='h', labels={'price':'Price (INR)','airline':'Airline'}, text='price', color='price', color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Airline ranking table
        st.subheader('Airline Rankings')
        rank_df = pd.DataFrame(results)
        rank_df['Diff_From_Cheapest'] = rank_df['price'] - cheapest['price']
        rank_df['Price'] = rank_df['price'].apply(currency)
        rank_df = rank_df[['airline','Price','Diff_From_Cheapest']].rename(columns={'airline':'Airline'})
        rank_df['Diff_From_Cheapest'] = rank_df['Diff_From_Cheapest'].apply(lambda x: currency(x))
        st.dataframe(rank_df)

        # Savings insights
        st.subheader('Savings Insights')
        percent_saving = int(round((mean_price - cheapest['price'])/mean_price * 100, 0))
        st.metric('Potential Savings vs Average', f"{percent_saving}%", delta=currency(int(mean_price - cheapest['price'])))
        st.markdown(f"**Price Category of cheapest option**: {price_category(cheapest['price'], mean_price)}")

        if is_demo:
            st.info('Running in demo mode because pre-trained model/encoder files were not found or prediction failed. For production use, upload models/model.pkl and models/encoder.pkl that were used during training.')


def page_dashboard():
    # Modern hero and metric layout
    st.markdown(
        """
        <div class='hero'>
          <div style='flex:1'>
            <div style='display:flex; gap:12px; align-items:center'>
              <div class='icon-circle'>✈️</div>
              <div>
                <div class='title'>FlightFareAI Pro</div>
                <div class='subtitle'>Predict domestic flight prices and find the best deals across airlines.</div>
              </div>
            </div>
            <div style='margin-top:14px'>
              <span class='badge'>Dark • Modern • AI-backed</span>
            </div>
          </div>
          <div style='display:flex; gap:10px; align-items:center'>
            <a class='cta-btn' href=''>Open Predict Page</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics grid
    model_exists = os.path.exists(MODEL_PATH)
    demo_mode = not model_exists

    metrics_html = f"""
    <div class='metrics-grid'>
      <div class='metric-card'>
        <div class='metric-title'>Total Airlines</div>
        <div class='metric-value'>8</div>
        <div class='metric-delta'>Top carriers included</div>
      </div>
      <div class='metric-card'>
        <div class='metric-title'>Demo Mode</div>
        <div class='metric-value'>{'Enabled' if demo_mode else 'Disabled'}</div>
        <div class='metric-delta'>{'Using demo model' if demo_mode else 'Using uploaded model'}</div>
      </div>
      <div class='metric-card'>
        <div class='metric-title'>Default Branch</div>
        <div class='metric-value'>main</div>
        <div class='metric-delta'>Repository</div>
      </div>
      <div class='metric-card'>
        <div class='metric-title'>Last Update</div>
        <div class='metric-value'>See repo</div>
        <div class='metric-delta'>Pushes to main</div>
      </div>
    </div>
    """

    st.markdown(metrics_html, unsafe_allow_html=True)

    st.markdown('<br/>', unsafe_allow_html=True)
    st.markdown("""
    <div class='glass'>
      <div style='display:flex; justify-content:space-between; align-items:center'>
        <div>
          <strong>Quick actions</strong>
          <div style='color:var(--muted); margin-top:6px'>Jump to predictions or upload your model files.</div>
        </div>
        <div style='display:flex; gap:10px'>
          <a href='' class='cta-btn'>Find Best Flights</a>
          <a href='https://github.com/kishankr101/-FlightFareAI_ML_P' target='_blank' class='cta-btn' style='background:linear-gradient(90deg,#2dd4bf,#60a5fa);'>Open Repo</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def page_recommendations():
    st.title('Recommendations')
    st.markdown('AI-generated booking tips and insights (demo).')
    st.markdown('- Book early for best prices')
    st.markdown('- Consider flights with 1 stop for significant savings on some routes')
    st.markdown('- Use weekday travel to lower ticket costs')


def page_about():
    st.title('About FlightFareAI Pro')
    st.markdown('''
    FlightFareAI Pro is a Streamlit-based application that predicts domestic Indian flight prices across airlines and recommends the cheapest options.

    Tech stack: Streamlit, Pandas, Scikit-learn, Joblib, Plotly, Seaborn

    Project structure (single-file app):
    - app.py (this file)
    - models/ (place your model.pkl and encoder.pkl here)
    - assets/ (styles, icons)

    To run:
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    ''')


# -----------------------------
# Router
# -----------------------------

def main():
    st.sidebar.title('FlightFareAI Pro')
    st.sidebar.markdown('Dark • Modern • AI-backed')

    options = ['Dashboard', 'Predict Flights', 'Recommendations', 'About']

    # If a page navigation override was set in session_state (e.g., from a page button), use it as the default index
    try:
        nav_target = st.session_state.get('nav_to')
    except Exception:
        nav_target = None

    try:
        default_index = options.index(nav_target) if nav_target in options else 0
    except Exception:
        default_index = 0

    try:
        radio_choice = st.sidebar.radio('Navigation', options, index=default_index)
    except Exception:
        radio_choice = st.sidebar.radio('Navigation', options)

    # Clear the nav override once consumed
    try:
        if nav_target in options:
            st.session_state['nav_to'] = None
    except Exception:
        pass

    page = radio_choice

    if page == 'Dashboard':
        page_dashboard()
    elif page == 'Predict Flights':
        page_predict()
    elif page == 'Recommendations':
        page_recommendations()
    elif page == 'About':
        page_about()


if __name__ == '__main__':
    main()
