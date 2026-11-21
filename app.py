import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import available_dates, available_wards, load_assets, predict, predict_all

st.set_page_config(page_title="Dengue Early Warning", layout="wide")
COLORS = {"Low": "#2e9e5b", "Medium": "#e6b800", "High": "#f07d00", "Critical": "#d62728"}


@st.cache_resource
def get_assets():
    return load_assets()  # model + data loaded once, then cached


@st.cache_data
def get_prediction(ward, date_str):
    return predict(ward, date_str, get_assets())


@st.cache_data
def get_ranking(date_str):
    return predict_all(date_str, get_assets())


assets = get_assets()
wards = available_wards(assets)
dates = available_dates(assets)

st.sidebar.header("Choose area and date")
city = st.sidebar.selectbox("City", sorted({w["city"] for w in wards}))
ward = st.sidebar.selectbox("Ward", [w["ward_id"] for w in wards if w["city"] == city])
date = st.sidebar.date_input(
    "Forecast from date",
    value=pd.Timestamp("2025-08-11").date(),
    min_value=pd.Timestamp(dates[0]).date(),
    max_value=pd.Timestamp(dates[-1]).date(),
)
st.sidebar.caption("Uses weather up to this date to predict cases 2 to 4 weeks ahead.")

res = get_prediction(ward, str(date))

st.title("Dengue Early-Warning Dashboard")
c1, c2, c3 = st.columns(3)
c1.markdown("**Risk tier**")
c1.markdown(
    f"<span style='color:{COLORS[res['tier']]};font-size:2.4rem;font-weight:700'>{res['tier']}</span>",
    unsafe_allow_html=True,
)
c2.metric("Predicted cases (3-week window)", res["predicted_cases"])
c3.metric("Forecast window", f"{res['window_start']} to {res['window_end']}")
st.caption(f"{res['ward_id']} ({res['city']}). Based on weather data up to {res['date_used']}.")
if res["actual_cases"] is not None:
    st.info(f"Backtest: actual cases in this window were {res['actual_cases']}.")

left, right = st.columns(2)

dd = pd.DataFrame(res["drivers"])
fig = px.bar(
    dd, x="share_pct", y="driver", orientation="h", color="effect",
    color_discrete_map={"raises risk": "#d62728", "lowers risk": "#2e9e5b"},
    labels={"share_pct": "Share of the model's weighing (%)", "driver": ""},
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320)
left.subheader("Main drivers")
left.plotly_chart(fig)
left.caption("These are the factors the model weighed most for this prediction. They show what the model used, not proof of cause.")

df = assets["df"]
hist = df[(df["ward_id"] == ward) & (df["date"] <= pd.Timestamp(res["date_used"]))].tail(10)
fig2 = px.bar(hist, x="date", y="rainfall_mm", labels={"rainfall_mm": "Rainfall (mm)", "date": "Week"})
fig2.update_layout(height=320)
right.subheader("Last 10 weeks of rainfall")
right.plotly_chart(fig2)
right.caption("Rain takes about 2 to 4 weeks to turn into hospital cases.")

st.subheader(f"Highest-risk wards for week of {res['date_used']}")
rank = get_ranking(res["date_used"]).sort_values("predicted_cases", ascending=False).head(10)
st.dataframe(rank[["ward_id", "city", "tier", "predicted_cases"]], hide_index=True)
import json

TIER_NAMES = ["Low", "Medium", "High", "Critical"]


@st.cache_data
def get_test_metrics():
    path = Path(__file__).parent / "models" / "metrics.json"
    return json.loads(path.read_text())["test"]


st.divider()
st.subheader("Model error on unseen test data (2024 to 2025)")
m = get_test_metrics()
e1, e2, e3, e4 = st.columns(4)
e1.metric("MAE (cases per window)", round(m["mae"], 2), f"baseline {m['baseline_mae']:.2f}", delta_color="off")
e2.metric("RMSE (cases per window)", round(m["rmse"], 2))
e3.metric("Macro-F1 (tiers)", round(m["macro_f1"], 2))
e4.metric("Critical precision", f"{m['critical_precision'] * 100:.0f}%")

cm = pd.DataFrame(m["confusion_matrix"], index=TIER_NAMES, columns=TIER_NAMES)
fig3 = px.imshow(
    cm, text_auto=True, color_continuous_scale="Blues",
    labels={"x": "Predicted tier", "y": "Actual tier", "color": "Rows"},
)
fig3.update_layout(height=380)
st.plotly_chart(fig3)
st.caption(
    "Test data was used once, after thresholds were fixed on validation. "
    "Critical alerts have many false alarms: cut-offs were tuned on 2023, a high-dengue year."
)