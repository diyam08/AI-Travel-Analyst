import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px

st.set_page_config(page_title="AI Travel Analyst", layout="wide", page_icon="✈️")

st.markdown("<h1 style='text-align: center; color: #1E88E5;'>✈️ AI Travel Analyst Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555555;'>Data-Driven Flight Price Analysis & Search Engine</h4>", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_artifacts():
    model = joblib.load('model_artifacts/flight_model.pkl.gz')
    with open('model_artifacts/feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_cleaned_data():
    return pd.read_csv('model_artifacts/cleaned_flight_data.csv')

model, feature_columns = load_artifacts()
df = load_cleaned_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Fare Predictor", 
    "2. 10 Interactive Insights", 
    "3. Flight Search Engine", 
    "4. Model Explainability"
])

# TAB 1: FARE PREDICTOR
with tab1:
    st.subheader("Predict Flight Ticket Fare")
    col1, col2, col3 = st.columns(3)
    with col1:
        airline = st.selectbox("Airline Carrier", sorted(df['Airline'].unique()))
        source = st.selectbox("Origin City", sorted(df['Source'].unique()))
        destination = st.selectbox("Destination City", sorted(df['Destination'].unique()))
        travel_class = st.selectbox("Travel Class", sorted(df['Travel_Class'].unique()))
    with col2:
        journey_date = st.date_input("Date of Departure")
        dep_time = st.time_input("Departure Time")
        arr_time = st.time_input("Arrival Time")
        days_before = st.slider("Days Before Departure", 1, 90, 15)
    with col3:
        stops = st.selectbox("Number of Stops", [0, 1, 2, 3])
        distance = st.number_input("Distance (km)", min_value=100.0, value=1000.0, step=100.0)
        passengers = st.slider("Passenger Count", 1, 6, 1)

    if st.button("Calculate Estimated Fare", type="primary"):
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        input_df['Total_Stops'] = stops
        input_df['Journey_Day'] = journey_date.day
        input_df['Journey_Month'] = journey_date.month
        input_df['Dep_Hour'] = dep_time.hour
        input_df['Dep_Min'] = dep_time.minute
        input_df['Arrival_Hour'] = arr_time.hour
        input_df['Arrival_Min'] = arr_time.minute
        input_df['Distance_km'] = distance
        input_df['Days_Before_Departure'] = days_before
        input_df['Passenger_Count'] = passengers
        
        for key in [f"Airline_{airline}", f"Source_{source}", f"Destination_{destination}", f"Travel_Class_{travel_class}"]:
            if key in input_df.columns:
                input_df[key] = 1
                
        pred_price = model.predict(input_df)[0]
        st.success(f"### Estimated Flight Price: ₹ {pred_price:,.2f}")

# TAB 2: 10 INTERACTIVE CHARTS
with tab2:
    st.subheader("Interactive Price Factors Dashboard")
    c1, c2 = st.columns(2)
    sample_df = df.sample(3000, random_state=42)
    with c1:
        st.plotly_chart(px.histogram(sample_df, x="Price", nbins=30, title="1. Flight Price Distribution"), use_container_width=True)
        st.plotly_chart(px.box(sample_df, x="Airline", y="Price", color="Airline", title="2. Price Comparison by Airline"), use_container_width=True)
        st.plotly_chart(px.box(sample_df, x="Travel_Class", y="Price", color="Travel_Class", title="3. Fare by Travel Class"), use_container_width=True)
        st.plotly_chart(px.scatter(sample_df, x="Distance_km", y="Price", color="Travel_Class", title="4. Distance (km) vs Price"), use_container_width=True)
        st.plotly_chart(px.line(df.groupby('Days_Before_Departure')['Price'].mean().reset_index(), x='Days_Before_Departure', y='Price', title="5. Booking Lead Time Trend"), use_container_width=True)
    with c2:
        st.plotly_chart(px.box(sample_df, x="Total_Stops", y="Price", color="Total_Stops", title="6. Price vs Total Stops"), use_container_width=True)
        st.plotly_chart(px.bar(df.groupby('Season')['Price'].mean().reset_index(), x='Season', y='Price', title="7. Seasonal Fare Trends"), use_container_width=True)
        st.plotly_chart(px.bar(df.groupby('Booking_Channel')['Price'].mean().reset_index(), x='Booking_Channel', y='Price', title="8. Price by Booking Channel"), use_container_width=True)
        st.plotly_chart(px.line(df.groupby('Dep_Hour')['Price'].mean().reset_index(), x='Dep_Hour', y='Price', title="9. Fluctuation by Departure Hour"), use_container_width=True)
        st.plotly_chart(px.bar(df.groupby('Dep_Time_Window')['Price'].median().reset_index(), x='Dep_Time_Window', y='Price', title="10. Median Price by Time Window"), use_container_width=True)

# TAB 3: RECOMMENDATION SEARCH ENGINE
with tab3:
    st.subheader("Budget & Route Flight Search Engine")
    max_budget = st.number_input("Maximum Budget (₹)", min_value=1000, value=15000, step=1000)
    rec_src = st.selectbox("Origin City", sorted(df['Source'].unique()), key="r_src")
    rec_dst = st.selectbox("Destination City", sorted(df['Destination'].unique()), key="r_dst")
    
    matches = df[(df['Source'] == rec_src) & (df['Destination'] == rec_dst) & (df['Price'] <= max_budget)].sort_values(by='Price')
    if not matches.empty:
        st.success(f"Found {len(matches)} matching flights under ₹{max_budget:,}:")
        st.dataframe(matches[['Airline', 'Source', 'Destination', 'Travel_Class', 'Total_Stops', 'Distance_km', 'Price']].head(10), use_container_width=True)
    else:
        st.warning("No flights available matching your selected budget and route criteria.")

# TAB 4: FEATURE IMPORTANCE
with tab4:
    st.subheader("Predictive Drivers Influencing Flight Fares")
    importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    fig_imp = px.bar(importance_df.head(10), x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis', title="Top 10 Model Feature Importances")
    st.plotly_chart(fig_imp, use_container_width=True)
