import streamlit as st
import pandas as pd
import requests
import joblib
import time

# ── Load saved model, scaler, columns, and label encoder ──
model = joblib.load("air_quality_model.pkl")
scaler = joblib.load("aq_scaler.pkl")
expected_columns = joblib.load("aq_columns.pkl")
encoder = joblib.load("aq_label_encoder.pkl")

st.set_page_config(page_title="Karachi Air Quality Forecast", page_icon="🌫️")
st.title("🌫️ Karachi Air Quality Forecast (Next 3 Hours)")
st.markdown("Live prediction using real-time pollutant readings from OpenWeatherMap.")

# ── Settings (sidebar) ──
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")
lat = st.sidebar.number_input("Latitude", value=24.8607, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=67.0011, format="%.4f")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60 seconds", value=False)

# AQI category labels for nicer display
aqi_labels = {
    1: "Good 🟢",
    2: "Fair 🟢",
    3: "Moderate 🟡",
    4: "Poor 🟠",
    5: "Very Poor 🔴"
}

def fetch_live_pollution(lat, lon, api_key):
    """Call the OpenWeatherMap Current Air Pollution API and return components dict."""
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    entry = data["list"][0]
    components = entry["components"]
    current_aqi = entry["main"]["aqi"]
    return components, current_aqi

def make_prediction(components):
    """Build a feature row matching training columns, scale it, and predict."""
    raw_input = {
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": components.get("pm2_5"),
        "nh3": components.get("nh3"),
        "pm10": components.get("pm10"),
        # NOTE: pm10 was dropped during training (duplicate of pm2_5),
        # and today's aqi is NOT a feature (that was the leakage fix)
    }

    input_df = pd.DataFrame([raw_input])
    

    # Ensure all expected columns exist, in the right order
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    encoded_prediction = model.predict(scaled_input)[0]
    real_aqi = encoder.inverse_transform([encoded_prediction])[0]
    return real_aqi

# ── Main logic ──
if not api_key:
    st.info("Enter your OpenWeatherMap API key in the sidebar to get a live forecast.")
else:
    try:
        components, current_aqi = fetch_live_pollution(lat, lon, api_key)

        st.subheader("📡 Live Readings Right Now")
        cols = st.columns(4)
        cols[0].metric("PM2.5", f"{components.get('pm2_5')} µg/m³")
        cols[1].metric("PM10", f"{components.get('pm10')} µg/m³")
        cols[2].metric("O3", f"{components.get('o3')} µg/m³")
        cols[3].metric("Current AQI", f"{current_aqi} ({aqi_labels.get(current_aqi, '')})")

        predicted_aqi = make_prediction(components)

        st.subheader("🔮 Forecast — Air Quality in ~3 Hours")
        label = aqi_labels.get(int(predicted_aqi), str(predicted_aqi))

        if predicted_aqi <= 2:
            st.success(f"Predicted AQI: {predicted_aqi} — {label}\n\nAir quality is expected to stay safe.")
        elif predicted_aqi == 3:
            st.warning(f"Predicted AQI: {predicted_aqi} — {label}\n\nModerate air quality expected — sensitive groups should take care.")
        else:
            st.error(f"Predicted AQI: {predicted_aqi} — {label}\n\nUnhealthy air expected — consider limiting outdoor activity.")

        st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch live data: {e}")
    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")

# ── Auto-refresh ──
if auto_refresh:
    time.sleep(60)
    st.rerun()



