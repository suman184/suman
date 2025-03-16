import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import tensorflow as tf
import time

# Load the trained model
model = tf.keras.models.load_model("hyperbolic_ride_demand_model_final.keras")

# Set page title and layout
st.set_page_config(page_title="Peak Ride Demand Predictor (Driver)", layout="wide")

# Initialize session state for data persistence
if "data" not in st.session_state:
    st.session_state.data = None
if "selected_date" not in st.session_state:
    st.session_state.selected_date = datetime.today()
if "hour_index" not in st.session_state:
    st.session_state.hour_index = 0
if "location" not in st.session_state:
    st.session_state.location = ""
if "notifications" not in st.session_state:
    st.session_state.notifications = []

# Title and welcome message
st.title("🚖 Peak Ride Demand Predictor (Driver)")
st.write("Welcome, Driver!")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Earnings", "Notifications", "Settings"])

# Function to predict high-demand areas
def predict_high_demand(data, model):
    high_demand_areas = []
    if data is not None and "Region" in data.columns:
        for region in data["Region"].unique():
            input_data = np.zeros((1, 1, 48))  # Adjust input shape as needed
            input_data[0, 0, st.session_state.hour_index] = 1  # Set the hour index feature
            predicted_demand = model.predict(input_data)[0][0]
            high_demand_areas.append((region, predicted_demand))
        # Sort by predicted demand and return top 3
        high_demand_areas.sort(key=lambda x: x[1], reverse=True)
        return high_demand_areas[:3]
    return []

# Function to update notifications
def update_notifications():
    if st.session_state.data is not None:
        high_demand_areas = predict_high_demand(st.session_state.data, model)
        if high_demand_areas:
            st.session_state.notifications = [
                f"High demand predicted in {region} with demand {int(demand)}."
                for region, demand in high_demand_areas
            ]
        else:
            st.session_state.notifications = ["No high-demand areas predicted."]
    else:
        st.session_state.notifications = ["Upload a dataset to get predictions."]

# Home/Dashboard Page
if page == "Home":
    st.header("📍 Home/Dashboard")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)

    if st.session_state.data is not None:
        data = st.session_state.data
        st.write("### Uploaded Data Preview:")
        st.dataframe(data.head())

        # Check if "Region" column exists
        if "Region" in data.columns:
            location_options = sorted(data["Region"].unique())
        else:
            st.warning("⚠️ No 'Region' column found in dataset!")
            location_options = []

        # Input fields for user parameters
        selected_date = st.date_input("Select Date", st.session_state.selected_date)
        # hour_of_day = st.selectbox("Select Hourly Range:", [f"{i}:00 - {i+1}:00" for i in range(24)], index=st.session_state.hour_index)
        # st.session_state.hour_index = int(hour_of_day.split(":")[0])

        # Location selection
        location = st.selectbox("Select Location:", location_options) if location_options else st.text_input("Enter Location:")
        st.session_state.location = location

        # Prepare input data for model prediction
        input_data = np.zeros((1, 1, 48))  # Initialize with zeros, adjust as needed
        input_data[0, 0, st.session_state.hour_index] = 1  # Set the hour index feature
        predicted_demand = model.predict(input_data)[0][0]

        # Simulate "actual demand" using the predicted demand with some noise
        actual_demand = predicted_demand * np.random.uniform(0.9, 1.1)  # Add slight variation

        # Line chart for demand trends
        st.write("### 📈 Actual vs Predicted Demand")
        hours = [f"{i}:00" for i in range(24)]
        actual_demand_data = [actual_demand * np.random.uniform(0.9, 1.1) for _ in range(24)]  # Simulate actual demand over 24 hours
        predicted_demand_data = [predicted_demand * np.random.uniform(0.95, 1.05) for _ in range(24)]  # Simulate predicted demand over 24 hours

        # Create a DataFrame for plotting
        demand_df = pd.DataFrame({
            "Hour": hours,
            "Actual Demand": actual_demand_data,
            "Predicted Demand": predicted_demand_data
        })

        # Plot the comparison graph
        fig_line = px.line(demand_df, x="Hour", y=["Actual Demand", "Predicted Demand"],
                           labels={'value': "Demand", 'x': "Hour"},
                           title=f"Actual vs Predicted Ride Demand on {selected_date} in {location}")
        st.plotly_chart(fig_line, use_container_width=True)

        # Top high-demand areas
        st.write("### 🚦 Top High-Demand Areas")
        if location_options:
            top_areas = data.groupby("Region")["Searches"].sum().nlargest(3)
            st.write(top_areas)
        else:
            st.warning("No data available for high-demand areas.")

        # Bangalore Map with Lanes and Demand Zones
        st.write("### 🗺️ Bangalore Map with High-Demand Areas")
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)

        if location_options:
            for loc in location_options:
                color = "red" if loc == location else "blue"
                folium.Marker(
                    location=[12.9716 + np.random.uniform(-0.05, 0.05), 77.5946 + np.random.uniform(-0.05, 0.05)],
                    popup=f"High demand in {loc}",
                    icon=folium.Icon(color=color, icon="info-sign")
                ).add_to(m)
        
        folium_static(m)

# Earnings Page
elif page == "Earnings":
    st.header("💰 My Earnings")

    # Placeholder earnings data
    earnings_data = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Earnings": [100, 150, 200, 180, 220, 250, 300]
    })

    # Bar chart for earnings
    st.write("### 📊 Weekly Earnings")
    fig_bar = px.bar(earnings_data, x="Day", y="Earnings",
                     labels={'x': "Day", 'y': "Earnings (INR)"},
                     title="Weekly Earnings")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Trip history
    st.write("### 📜 Trip History")
    trip_history = pd.DataFrame({
        "Date": ["2023-10-01", "2023-10-02", "2023-10-03"],
        "Time": ["07:00", "08:00", "09:00"],
        "Fare (INR)": [200, 250, 300],
        "Distance (km)": [5, 7, 10]
    })
    st.dataframe(trip_history)

# Notifications Page
elif page == "Notifications":
    st.header("🔔 Notifications")

    # Update notifications every 10 seconds
    if "last_update" not in st.session_state:
        st.session_state.last_update = datetime.now()

    if (datetime.now() - st.session_state.last_update).seconds >= 10:
        update_notifications()
        st.session_state.last_update = datetime.now()

    # Display notifications
    for notification in st.session_state.notifications:
        st.write(f"- {notification}")

# Settings Page
elif page == "Settings":
    st.header("⚙️ Settings")

    # Input fields for profile settings
    name = st.text_input("Name", "[Driver Name]")
    vehicle = st.text_input("Vehicle", "[Vehicle Info]")

    if st.button("Save"):
        st.success("✅ Settings saved successfully!")