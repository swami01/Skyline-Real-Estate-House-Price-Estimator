"""
Streamlit UI -- a CLIENT of the FastAPI service, not a model host.

Run (with the API already running separately):
  uvicorn api:app --reload --port 8000
  streamlit run app.py

This is the key architectural difference from the fraud detection
project: this file never imports joblib, never loads a model, and
never calls engineer_features directly. It only makes HTTP requests to
api.py and renders whatever comes back. The prediction logic lives in
exactly one place (the API), which could be swapped, scaled, or reused
by a completely different frontend without this file changing at all.
"""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("HOUSE_PRICE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="House Price Estimator", page_icon="\U0001F3E1", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0b0f19 0%, #10162b 100%); }
h1, h2, h3, p, label, .stMarkdown { color: #e8ebf5 !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }
[data-testid="stMetricLabel"] { color: #9aa4c4 !important; }
div[data-testid="stForm"] {
    background: #141a2e; border: 1px solid #262e4a; border-radius: 14px;
    padding: 1.6rem 1.8rem;
}
.result-card {
    background: #141a2e; border: 1px solid #262e4a; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.stButton>button, button[kind="formSubmit"], div[data-testid="stFormSubmitButton"] button {
    background-color: #4c5ef5 !important; color: white !important;
    border-radius: 10px !important; border: none !important;
    font-weight: 600 !important; padding: 0.55rem 1.4rem !important;
}
.stButton>button:hover, button[kind="formSubmit"]:hover, div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #3a4bd1 !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## \U0001F3E1 House Price Estimator")
st.markdown(
    "<p style='color:#9aa4c4;margin-top:-8px'>California district median house value "
    "prediction \u2014 served via a FastAPI backend</p>",
    unsafe_allow_html=True,
)

OCEAN_OPTIONS = ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]


def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=60)
        return r.json()
    except requests.exceptions.RequestException:
        return None


health = check_health()
with st.sidebar:
    st.markdown("### API status")
    if health is None:
        st.error(f"Can't reach the API at {API_URL}. Is it running?\n\n`uvicorn api:app --reload --port 8000`")
    elif health.get("model_ready"):
        st.success(f"Connected \u2014 {API_URL}")
        try:
            meta = requests.get(f"{API_URL}/metadata", timeout=3).json()
            st.markdown("### Model info")
            st.write(f"**Model:** {meta['model_name']}")
            st.write(f"**Trained on:** {meta['n_train']:,} districts")
            st.write(f"**Tested on:** {meta['n_test']:,} districts")
            st.markdown("### Held-out test metrics")
            for k, v in meta["test_metrics"].items():
                st.write(f"**{k.upper()}:** {v:,.3f}")
        except requests.exceptions.RequestException:
            pass
    else:
        st.warning("API is up but no trained model found. Run `python3 train.py` first.")

left, right = st.columns([1.1, 1])

with left:
    with st.form("house_form"):
        st.markdown("#### District details")
        c1, c2 = st.columns(2)
        longitude = c1.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-122.23, format="%.4f")
        latitude = c2.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.88, format="%.4f")

        c3, c4 = st.columns(2)
        housing_median_age = c3.number_input("Median house age (years)", min_value=0.0, max_value=60.0, value=41.0)
        median_income = c4.number_input("Median income (10k USD units, e.g. 8.3 = ₹83,000)", min_value=0.1, value=8.3252)

        c5, c6 = st.columns(2)
        total_rooms = c5.number_input("Total rooms in district", min_value=1.0, value=880.0)
        total_bedrooms = c6.number_input("Total bedrooms in district", min_value=1.0, value=129.0)

        c7, c8 = st.columns(2)
        population = c7.number_input("Population", min_value=1.0, value=322.0)
        households = c8.number_input("Households", min_value=1.0, value=126.0)

        ocean_proximity = st.selectbox("Ocean proximity", OCEAN_OPTIONS, index=0)

        submitted = st.form_submit_button("Estimate price")

with right:
    if submitted:
        payload = {
            "longitude": longitude, "latitude": latitude,
            "housing_median_age": housing_median_age, "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms, "population": population,
            "households": households, "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't reach the API: {e}")
        else:
            if resp.status_code == 422:
                st.error(f"The API rejected this input: {resp.json()['detail']}")
            elif resp.status_code != 200:
                st.error(f"API error ({resp.status_code}): {resp.text}")
            else:
                result = resp.json()
                price = result["predicted_price"]

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.metric("Estimated median house value", f"₹{price:,.0f}")
                st.caption(f"Model: {result['model_name']}")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### Engineered features the model actually used")
                ef = result["engineered_features"]
                fig = go.Figure(go.Bar(
                    x=list(ef.values()), y=list(ef.keys()), orientation="h",
                    marker_color="#4c5ef5",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e8ebf5", height=220, margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "These ratios (rooms per household, bedrooms per room, "
                    "population per household) are computed server-side by the "
                    "API from your raw inputs \u2014 the same transformation used "
                    "during training."
                )
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="result-card"><p style="color:#9aa4c4">'
            'Fill in the district details and click <b>Estimate price</b>.'
            '</p></div>',
            unsafe_allow_html=True,
        )
