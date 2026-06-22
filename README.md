# FlightFareAI Pro

This repository now contains a single-file Streamlit app `app.py` that implements the "FlightFareAI Pro" UI and demo prediction flow.

Important:
- This app expects two files to be present in the `models/` directory:
  - `models/model.pkl` — a pre-trained scikit-learn model (RandomForest or a pipeline that accepts DataFrame rows matching the training features)
  - `models/encoder.pkl` — an encoder object (e.g., LabelEncoder or pipeline component listing airline classes)

If those files are not present, the app will run in demo mode with synthetic deterministic predictions so you can test the UI and features.

How to run:

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

Adding model files:
- Place your pre-trained model at `models/model.pkl`.
- Place your encoder at `models/encoder.pkl`.
- Restart the app; it will attempt to use the model. If the model input schema differs from the app's assumed columns, adjust your model/pipeline or modify `app.py`'s `predict_with_model` function to match your feature columns.

Notes:
- The app is designed as a single-file Streamlit app for simplicity but contains modular functions and comments for maintainability.
- I committed these initial UI and app scaffolding directly to the default branch `main` as requested.
