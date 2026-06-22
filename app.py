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
# Prediction / demo feed logic
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


def simulate_live_feed(input_features: Dict, jitter: int = 150) -> List[Dict]:
    """Create a simulated live feed with additional metadata (times, logos, stops).
       jitter controls price variability.
    """
    base_results = demo_predict_all(input_features)
    simulated = []
    now = pd.Timestamp.now()
    for r in base_results:
        airline = r['airline']
        price = max(300, int(r['price'] + np.random.randint(-jitter, jitter)))
        # simple time generation
        dep = input_features.get('dep_time', '08:00')
        arr = input_features.get('arr_time', '10:00')
        duration = input_features.get('duration', 120)
        stops = input_features.get('stops', 0)
        logo_name = airline.lower().split()[0]
        logo_path = f"assets/logos/{logo_name}.svg"
        simulated.append({
            'airline': airline,
            'price': price,
            'dep_time': dep,
            'arr_time': arr,
            'duration': duration,
            'stops': stops,
            'logo': logo_path,
            'updated_at': now.strftime('%H:%M:%S')
        })
    # sort by price ascending
    simulated = sorted(simulated, key=lambda x: x['price'])
    return simulated


# -----------------------------
# UI helpers & components
# -----------------------------

def currency(x):
    return f"₹{int(x):,}"


# -----------------------------
# Pages
# -----------------------------

def page_predict():
    st.title("Predict Flights — FlightFareAI Pro")
    st.markdown("""
    <div class='glass'>
    <p style='color:#cfe9ff'>Enter flight details below and click <b>Find Best Flights</b> to get attractive predicted flight cards (simulated live feed).</p>
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
        input_features = {
            'source': source.strip(),
            'destination': destination.strip(),
            'dep_time': dep_time.strftime('%H:%M'),
            'arr_time': arr_time.strftime('%H:%M'),
            'stops': stops,
            'travel_class': travel_class,
            'duration': duration,
        }

        with st.spinner('Finding best flights...'):
            time.sleep(0.7)
            # simulated live feed
            feed = simulate_live_feed(input_features)

        st.markdown(f"**Last updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown('<br/>', unsafe_allow_html=True)

        # Render attractive flight cards
        for item in feed:
            cols = st.columns([0.8, 3, 1])
            with cols[0]:
                # logo fallback
                logo = item.get('logo')
                if os.path.exists(logo):
                    st.image(logo, width=64)
                else:
                    st.markdown(f"<div class='flight-logo'>{item['airline'][:2]}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div class='flight-meta'><div style='display:flex; justify-content:space-between'><strong>{item['airline']}</strong><div class='flight-price'>{currency(item['price'])}</div></div><div class='flight-sub'>{item['dep_time']} → {item['arr_time']} • {int(item['duration'])} mins • {item['stops']} stop(s)</div></div>", unsafe_allow_html=True)
            with cols[2]:
                if st.button('Book', key=f"book_{item['airline']}_{item['price']}"):
                    st.success(f"Booking flow not implemented — selected {item['airline']} at {currency(item['price'])}")


def page_dashboard():
    # Modern hero and metric layout with About / Features
    st.markdown(
        """
        <div class='hero'>
          <div style='flex:1'>
            <div style='display:flex; gap:12px; align-items:center'>
              <img src='assets/icons/logo.svg' style='width:56px; height:56px; border-radius:12px; background:transparent;' />
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
        <div class='metric-title'>Model Mode</div>
        <div class='metric-value'>{'Demo' if demo_mode else 'Production'}</div>
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
          <strong>About & Features</strong>
          <div style='color:var(--muted); margin-top:6px'>FlightFareAI Pro helps you estimate flight prices across major carriers using a trained model or the built-in demo model. Key features below.</div>
        </div>
        <div style='display:flex; gap:10px'>
          <a href='' class='cta-btn'>Find Best Flights</a>
          <a href='https://github.com/kishankr101/-FlightFareAI_ML_P' target='_blank' class='cta-btn' style='background:linear-gradient(90deg,#2dd4bf,#60a5fa);'>Open Repo</a>
        </div>
      </div>
      <div style='margin-top:14px'>
        <ul>
          <li>Fast demo predictions using deterministic heuristic model</li>
          <li>Attractive, responsive UI with live-like flight cards</li>
          <li>Supports uploading your trained model (joblib) and encoder</li>
          <li>Lightweight CSS and assets optimized for Streamlit Cloud</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)


def page_recommendations():
    # Repurposed as Features / Project Details
    st.title('Project Features')
    st.markdown('FlightFareAI Pro — features and usage')
    st.markdown('''
    - Realistic demo model for quick demos
    - Clean, modern dashboard with metric cards
    - Responsive flight results with airline logos and booking CTA
    - Easy model upload workflow (future enhancement)
    ''')


def page_about():
    st.title('About FlightFareAI Pro')
    st.markdown('''
    FlightFareAI Pro is a Streamlit-based application that predicts domestic Indian flight prices across airlines and recommends the cheapest options.

    Tech stack: Streamlit, Pandas, simple demo model

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
    st.sidebar.markdown("<div class='sidebar-brand'>FlightFareAI Pro</div><div class='sidebar-sub'>Dark • Modern • AI-backed</div>", unsafe_allow_html=True)

    options = ['Dashboard', 'Predict Flights', 'Project Features', 'About']

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
    elif page == 'Project Features':
        page_recommendations()
    elif page == 'About':
        page_about()


if __name__ == '__main__':
    main()
