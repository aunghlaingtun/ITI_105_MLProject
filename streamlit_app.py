# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="HDB Price Predictor", page_icon="🏠", layout="centered")

# Custom CSS for better night visibility
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e2130;
    }
    
    /* Input fields styling */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
        border: 1px solid #4a4a4a !important;
        border-radius: 6px !important;
    }
    
    /* Input field focus state */
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 1px #00d4ff !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div > div {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #00d4ff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #00b8e6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #1a4d3a !important;
        border: 1px solid #28a745 !important;
        color: #ffffff !important;
    }
    
    /* Error message styling */
    .stError {
        background-color: #4d1a1a !important;
        border: 1px solid #dc3545 !important;
        color: #ffffff !important;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #4d3d1a !important;
        border: 1px solid #ffc107 !important;
        color: #ffffff !important;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #1a3d4d !important;
        border: 1px solid #17a2b8 !important;
        color: #ffffff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
        border: 1px solid #4a4a4a !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1e1e1e !important;
        border: 1px solid #4a4a4a !important;
        color: #ffffff !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #ffffff !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background-color: #2b2b2b !important;
        border: 2px dashed #4a4a4a !important;
        border-radius: 6px !important;
    }
    
    .stFileUploader > div > div > div {
        color: #ffffff !important;
    }
    
    /* Sidebar text styling */
    .css-1d391kg .stTextInput > label,
    .css-1d391kg .stSelectbox > label,
    .css-1d391kg .stFileUploader > label {
        color: #ffffff !important;
    }
    
    /* Main content labels */
    .stNumberInput > label,
    .stTextInput > label,
    .stSelectbox > label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #1e1e1e !important;
    }
    
    /* Caption styling */
    .stCaption {
        color: #a0a0a0 !important;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #2b2b2b !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #4a4a4a !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 HDB Resale Price Prediction (Linear Pipeline)")

# ----------- DEFAULT PATHS (ensure these exist in your repo for cloud) -----------
DEFAULT_MODEL_PATH = "ITI105/hdb_price_pipeline_cloud.pkl"
DEFAULT_DATA_PATH  = "ITI105/dataset/hdb_processed_data.csv"

# ----------- SIDEBAR: Uploads & Metadata -----------
st.sidebar.header("⚙️ Options")
uploaded_model = st.sidebar.file_uploader("Model pipeline (.pkl)", type=["pkl"])
uploaded_data  = st.sidebar.file_uploader("Training CSV (for feature schema)", type=["csv"])

st.sidebar.header("👤 Student Metadata")
student_name = st.sidebar.text_input("Name", value="Aung Hlaing Tun")
student_id = st.sidebar.text_input("Student ID", value="6319250G")
course = st.sidebar.text_input("Course", value="ITI-105")
team_id = st.sidebar.text_input("Project Group ID", value="AlogoRiddler")
project_date = st.sidebar.text_input("Project Date", value="25 Aug 2025")
sg_now = datetime.now(ZoneInfo("Asia/Singapore"))
inference_ts = st.sidebar.text_input("Model Inference Date (SGT)", value=sg_now.strftime("%Y-%m-%d %H:%M:%S %Z"))

# ----------- Handle Uploads -----------
MODEL_PATH = DEFAULT_MODEL_PATH
DATA_PATH  = DEFAULT_DATA_PATH

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(uploaded_model.read())
        MODEL_PATH = tmp.name

if uploaded_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_data.read())
        DATA_PATH = tmp.name

# ----------- Helpers -----------
def clamp(val, lo, hi, fallback=None):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback if fallback is not None else lo
    try:
        v = float(val)
    except Exception:
        return fallback if fallback is not None else lo
    return max(min(v, hi), lo)

def clamp_int(val, lo, hi, fallback=None):
    return int(round(clamp(val, lo, hi, fallback)))

@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load model at {path}: {e}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_template(path):
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception as e:
        st.error(f"❌ Failed to read training CSV at {path}: {e}")
        st.stop()
    X = df.drop(columns=["resale_price"]) if "resale_price" in df.columns else df.copy()
    return X, list(X.columns)

@st.cache_data(show_spinner=False)
def get_category_choices(path, available_cols, cols=("town","flat_type","flat_model","storey_range")):
    usecols = [c for c in cols if c in available_cols]
    if not usecols:
        return {}
    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        return {}
    return {c: sorted(df[c].dropna().astype(str).unique()) for c in df.columns}

# ----------- Load Model & Template -----------
if not os.path.exists(MODEL_PATH) and not uploaded_model:
    st.warning(f"📁 Model not found at `{DEFAULT_MODEL_PATH}`. Please upload a .pkl in the sidebar.")
    st.stop()
if not os.path.exists(DATA_PATH) and not uploaded_data:
    st.warning(f"📁 Training CSV not found at `{DEFAULT_DATA_PATH}`. Please upload a CSV in the sidebar.")
    st.stop()

pipe = load_model(MODEL_PATH)
X_template, required_cols = load_template(DATA_PATH)
choices = get_category_choices(DATA_PATH, required_cols)

# ----------- Metadata Pill (Enhanced for night visibility) -----------
st.markdown(
    f"""
    <div style="padding:12px;border-radius:8px;background:linear-gradient(135deg, #2b2b2b 0%, #3a3a3a 100%);border:1px solid #4a4a4a;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.3);">
      <div style="color:#ffffff;font-size:14px;line-height:1.6;">
        <strong style="color:#00d4ff;">Name:</strong> <span style="color:#e0e0e0;">{student_name}</span> &nbsp;|&nbsp;
        <strong style="color:#00d4ff;">ID:</strong> <span style="color:#e0e0e0;">{student_id}</span> &nbsp;|&nbsp;
        <strong style="color:#00d4ff;">Course:</strong> <span style="color:#e0e0e0;">{course}</span> &nbsp;|&nbsp;
        <strong style="color:#00d4ff;">Group:</strong> <span style="color:#e0e0e0;">{team_id}</span> &nbsp;|&nbsp;
        <strong style="color:#00d4ff;">Project Date:</strong> <span style="color:#e0e0e0;">{project_date}</span> &nbsp;|&nbsp;
        <strong style="color:#00d4ff;">Inference:</strong> <span style="color:#e0e0e0;">{inference_ts}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------- UI Inputs -----------
def get_num(name, default): return float(X_template[name].iloc[0]) if name in X_template else default
def get_str(name, default): return str(X_template[name].iloc[0]) if name in X_template else default

col1, col2 = st.columns(2)
with col1:
    floor_area_sqm = st.number_input("Floor Area (sqm)", 20.0, 250.0, clamp(get_num("floor_area_sqm", 90.0), 20.0, 250.0), step=1.0)
    remaining_lease_years = st.number_input("Remaining Lease (years)", 1.0, 99.0, clamp(get_num("remaining_lease_years", 70.0), 1.0, 99.0), step=1.0)
    min_storey = st.number_input("Min Storey", 1, 60, clamp_int(get_num("min_storey", 1), 1, 60), step=1)
    max_storey = st.number_input("Max Storey", 1, 60, clamp_int(get_num("max_storey", 3), 1, 60), step=1)
with col2:
    latitude = st.number_input("Latitude", 1.20, 1.50, clamp(get_num("latitude", 1.3500), 1.20, 1.50), step=0.0001, format="%.6f")
    longitude = st.number_input("Longitude", 103.60, 104.10, clamp(get_num("longitude", 103.8500), 103.60, 104.10), step=0.0001, format="%.6f")
    cpi = st.number_input("CPI", 80.0, 140.0, clamp(get_num("cpi", 105.0), 80.0, 140.0), step=0.1)
    distance_to_mrt = st.number_input("Distance to MRT (km)", 0.0, 20.0, clamp(get_num("distance_to_mrt", 1.2), 0.0, 20.0), step=0.1)

year = st.number_input("Transaction Year", 2015, 2025, clamp_int(get_num("year", 2024), 2015, 2025), step=1)
month_num = st.selectbox("Transaction Month (1–12)", list(range(1, 13)), index=clamp_int(get_num("month_num", 1), 1, 12) - 1)

if max_storey < min_storey:
    st.info("Max Storey adjusted to match Min Storey.")
    max_storey = min_storey
mid_storey_val = (min_storey + max_storey) / 2.0

def cat_input(label, col_name, default):
    opts = choices.get(col_name, [])
    # Using training choices avoids "unknown category" issues if the encoder doesn't ignore unknowns
    return st.selectbox(label, opts, index=opts.index(default) if default in opts else 0) if opts else st.text_input(label, value=default)

town         = cat_input("Town", "town", get_str("town", "ANG MO KIO"))
flat_type    = cat_input("Flat Type", "flat_type", get_str("flat_type", "3 ROOM"))
flat_model   = cat_input("Flat Model", "flat_model", get_str("flat_model", "Model A"))
storey_range = cat_input("Storey Range", "storey_range", get_str("storey_range", "01 TO 03"))

show_row = st.checkbox("Show feature row sent to model", value=False)

# ----------- Prediction Logic -----------
if st.button("Predict Price"):
    # Start with the template single row so all required columns exist
    X_input = X_template.copy()

    def set_if_present(col, val):
        if col in X_input.columns:
            X_input[col] = val

    # Set values from UI
    for col, val in {
        "floor_area_sqm": floor_area_sqm,
        "remaining_lease_years": remaining_lease_years,
        "min_storey": min_storey,
        "max_storey": max_storey,
        "mid_storey": mid_storey_val,
        "latitude": latitude,
        "longitude": longitude,
        "cpi": cpi,
        "distance_to_mrt": distance_to_mrt,
        "year": year,
        "month_num": int(month_num),
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "storey_range": storey_range
    }.items():
        set_if_present(col, val)

    # (Optional) numeric coercion – safe if your pipeline expects numerics
    numeric_like = [
        "floor_area_sqm", "remaining_lease_years", "min_storey", "max_storey", "mid_storey",
        "latitude", "longitude", "cpi", "distance_to_mrt", "year", "month_num",
    ]
    for col in numeric_like:
        if col in X_input.columns:
            X_input[col] = pd.to_numeric(X_input[col], errors="coerce")

    # Sanity check: shape 1 x n_features
    if len(X_input) != 1:
        X_input = X_input.iloc[:1, :]

    try:
        y_pred = float(pipe.predict(X_input)[0])
    except Exception as e:
        st.error("❌ Prediction failed. Common causes:\n"
                 "- Unknown categories not seen in training (ensure your encoder uses handle_unknown='ignore').\n"
                 "- Mismatched column names/order vs. the pipeline's ColumnTransformer.\n"
                 "- Missing required columns in the training CSV template.\n\n"
                 f"Error: {e}")
        if show_row:
            st.subheader("Feature row (debug)")
            st.dataframe(X_input)
        st.stop()

    # Enhanced success message with better styling
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:8px;background:linear-gradient(135deg, #1a4d3a 0%, #28a745 100%);border:1px solid #28a745;margin:16px 0;box-shadow:0 4px 12px rgba(40,167,69,0.3);">
            <div style="color:#ffffff;text-align:center;">
                <h3 style="margin:0;color:#ffffff;font-size:24px;">💰 Estimated Resale Price</h3>
                <h2 style="margin:8px 0 0 0;color:#00ff88;font-size:32px;font-weight:bold;">S${y_pred:,.0f}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("See prediction inputs"):
        st.write({
            "floor_area_sqm": floor_area_sqm,
            "remaining_lease_years": remaining_lease_years,
            "min_storey": min_storey,
            "max_storey": max_storey,
            "mid_storey": mid_storey_val,
            "latitude": latitude,
            "longitude": longitude,
            "cpi": cpi,
            "distance_to_mrt": distance_to_mrt,
            "year": year,
            "month_num": int(month_num),
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
            "storey_range": storey_range
        })

    if show_row:
        st.subheader("Feature row sent to model")
        st.dataframe(X_input, use_container_width=True)

st.caption(f"✅ Runtime: Python {st.__version__ and __import__('sys').version.split()[0]} · Streamlit {__import__('streamlit').__version__}")

