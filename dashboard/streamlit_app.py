import json
import numpy as np
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="RideWise Dashboard", layout="wide")
st.title("RideWise Customer Churn Dashboard")

# Sidebar controls
st.sidebar.header("Settings")
data_path = st.sidebar.text_input("CSV path", "../data/processed/model_df_segmented.csv")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Segments", "Customer Explorer", "Predict"],
    index=0
)

# Load data
df = pd.read_csv(data_path)
def build_payload_from_features(row: pd.DataFrame, feature_cols_path: str):
    # row is a 1-row dataframe
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    record = row.iloc[0].to_dict()

    # Keep only features the model expects
    payload = {}
    for col in feature_cols:
        val = record.get(col, None)

        # Convert NaN to None (so JSON is valid)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            payload[col] = None
        else:
            # Convert numpy types to normal Python types
            if hasattr(val, "item"):
                val = val.item()
            payload[col] = val

    return payload

# -------------------- OVERVIEW --------------------
if page == "Overview":
    st.subheader("Overview")

    customers = df["user_id"].nunique() if "user_id" in df.columns else len(df)
    churn_rate = df["churn"].mean() if "churn" in df.columns else None
    total_trips = df["total_trips"].sum() if "total_trips" in df.columns else None
    total_spend = df["total_spend"].sum() if "total_spend" in df.columns else None
    arpu = (total_spend / customers) if total_spend is not None else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{customers:,}")
    c2.metric("Churn rate", f"{churn_rate*100:.1f}%" if churn_rate is not None else "—")
    c3.metric("Trips", f"{int(total_trips):,}" if total_trips is not None else "—")
    c4.metric("Revenue", f"{total_spend:,.0f}" if total_spend is not None else "—")
    c5.metric("ARPU", f"{arpu:,.2f}" if arpu is not None else "—")

    st.divider()
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Churn Distribution")

    if "churn" in df.columns:
        st.bar_chart(df["churn"].value_counts().sort_index())

    st.markdown("### Average Spend by Churn")
    if "total_spend" in df.columns and "churn" in df.columns:
        avg_spend = df.groupby("churn")["total_spend"].mean()
        st.bar_chart(avg_spend)

    st.markdown("### Average Trips by Churn")
    if "total_trips" in df.columns and "churn" in df.columns:
        avg_trips = df.groupby("churn")["total_trips"].mean()
        st.bar_chart(avg_trips)

    st.markdown("### Key Insight")
    if "churn" in df.columns and "total_spend" in df.columns:
        churn_spend = df[df["churn"] == 1]["total_spend"].mean()
        retain_spend = df[df["churn"] == 0]["total_spend"].mean()
        if churn_spend < retain_spend:
            st.success("Retained customers spend more on average than churned customers.")
        else:
            st.warning("Churned customers spend more on average than retained customers.")
# -------------------- SEGMENTS --------------------
elif page == "Segments":
    st.subheader("Segments")

    seg_col = "segment_name" if "segment_name" in df.columns else (
        "segment" if "segment" in df.columns else None
    )

    if seg_col is None:
        st.info("No segment column found (segment or segment_name).")
    else:
        if "churn" in df.columns:
            churn_by_seg = df.groupby(seg_col)["churn"].mean().sort_values(ascending=False)
            st.subheader("Churn rate by segment")
            st.bar_chart(churn_by_seg)

        # Highlight highest churn segment
        worst_segment = churn_by_seg.index[0]
        worst_rate = churn_by_seg.iloc[0] * 100

        st.warning(f"Highest churn segment: {worst_segment} ({worst_rate:.1f}%) — Prioritise retention strategy here.")

        st.markdown("### Recommended actions for this segment")

        st.write("""
        - Offer a loyalty reward (e.g., free ride credits after X trips)
        - Send personalised notifications for peak commute times
        - Fix friction points: app performance, payment issues, bike availability
        - Run a short survey to learn why they might leave
        """)

        st.subheader("Segment profile (average metrics)")
        metric_cols = [c for c in ["total_trips", "total_spend", "avg_fare", "days_since_last_trip", "conversion_rate", "total_sessions"]
                       if c in df.columns]

        if metric_cols:
            profile = df.groupby(seg_col)[metric_cols].mean().round(2)
            st.dataframe(profile, use_container_width=True)
        else:
            st.info("No common metric columns found to profile segments.")

# -------------------- CUSTOMER EXPLORER --------------------
elif page == "Customer Explorer":
    st.subheader("Customer Explorer")

    seg_col = "segment_name" if "segment_name" in df.columns else (
        "segment" if "segment" in df.columns else None
    )

    left, right = st.columns([1, 3])

    with left:
        segments = []
        if seg_col:
            segments = st.multiselect("Segment", sorted(df[seg_col].dropna().unique().tolist()))

        churn_vals = st.multiselect("Churn", [0, 1], default=[0, 1]) if "churn" in df.columns else []
        min_trips = st.number_input("Min total_trips", 0, 100000, 0, 1) if "total_trips" in df.columns else 0
        min_spend = st.number_input("Min total_spend", 0.0, 1e9, 0.0, 10.0) if "total_spend" in df.columns else 0.0

    filtered = df.copy()
    if segments and seg_col:
        filtered = filtered[filtered[seg_col].isin(segments)]
    if churn_vals and "churn" in df.columns:
        filtered = filtered[filtered["churn"].isin(churn_vals)]
    if "total_trips" in df.columns:
        filtered = filtered[filtered["total_trips"].fillna(0) >= min_trips]
    if "total_spend" in df.columns:
        filtered = filtered[filtered["total_spend"].fillna(0) >= min_spend]

    with right:
        st.write(f"Rows: {len(filtered):,}")
        st.dataframe(filtered.head(500), use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, "ridewise_filtered_customers.csv", "text/csv")

# -------------------- PREDICT --------------------
elif page == "Predict":
    st.subheader("Predict churn")

    api_url = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")

    if "user_id" not in df.columns:
        st.info("No user_id column found.")
    else:
        user_id = st.selectbox("Choose user_id", df["user_id"].astype(str).unique())

        row = df[df["user_id"].astype(str) == str(user_id)].head(1)
        payload = row.to_dict(orient="records")[0]

        feature_cols_path = "../models/feature_cols.json"  # adjust if your path is different
        payload = build_payload_from_features(row, feature_cols_path)

        # remove label if present
        payload.pop("churn", None)
        payload.pop("segment_name", None)

        st.caption("Payload preview")
        st.json(payload)

if st.button("Predict with API"):
    r = requests.post(
        api_url.rstrip("/") + "/predict", 
        json={"features": payload}, 
        timeout=15
    )
    st.write("Status:", r.status_code)
    result = r.json()
    st.json(result)

    if r.status_code != 200:
        st.error("Prediction failed — your API rejected the payload. Check the 'detail' output above.")
        st.stop()

    # --- Risk interpretation ---
    if "churn_probability" in result:
        p = float(result["churn_probability"])

        if p >= 0.7:
            st.error(f"High risk customer (prob={p:.2f}).")
            st.write("- Action: immediate retention offer + proactive support follow-up.")
        elif p >= 0.4:
            st.warning(f"Medium risk customer (prob={p:.2f}).")
            st.write("- Action: targeted nudges (discount, reminders, personalised route suggestions).")
        else:
            st.success(f"Low risk customer (prob={p:.2f}).")
            st.write("- Action: keep engaged (loyalty points, upsell subscription).")
    else:
        st.info("No churn_probability returned from API. Check your FastAPI response keys.")

st.sidebar.divider()
st.sidebar.caption("Built by Blessing • RideWise churn & insights dashboard")