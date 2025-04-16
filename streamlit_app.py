import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the data
df = pd.read_csv("water_consumption.csv")

# Streamlit layout
st.title("Clean Water Consumption Predictor")
st.write("### 💧Water: the only drink that falls from the sky for free.")
st.write("##### Every drop counts! Curious how much water we’ll use? Let’s find out together! ")

# Dropdown selections
years = sorted(df['date'].unique())
states = sorted(df['state'].unique())
categories = ['All'] + sorted(df['sector'].unique())

selected_state = st.selectbox("Select State", states)
selected_category = st.selectbox("Select Sector", categories)

# Data filtering
if selected_category == 'All':
    temp_df = df[df['state'] == selected_state]
    grouped_df = temp_df.groupby(['date', 'state'], as_index=False)['value'].sum()
    grouped_df['sector'] = 'All'
    filtered_df = grouped_df[['date', 'state', 'sector', 'value']]
else:
    filtered_df = df[
        (df['state'] == selected_state) &
        (df['sector'] == selected_category)
    ]

if filtered_df.empty:
    st.warning("No data available for this combination. Please choose another state or sector.")
    st.stop()

st.write("Filtered Data", filtered_df)

# Prepare time series
filtered_df = filtered_df.sort_values('date')
filtered_df['date'] = pd.to_numeric(filtered_df['date'])  # Ensure numeric year
ts_data = filtered_df.set_index('date')['value']

# Train Double Exponential Smoothing model
@st.cache_resource
def train_des_model(series):
    model = ExponentialSmoothing(series, trend="add", seasonal=None)
    fitted_model = model.fit()
    return fitted_model

fitted_model = train_des_model(ts_data)

# Year slider
selected_year = st.slider("Select a Year", min_value=min(years), max_value=2040, value=2025)

# Forecast
forecast_years = np.arange(ts_data.index.min(), selected_year + 1)
forecast = fitted_model.predict(start=ts_data.index.min(), end=selected_year)
predicted_value = forecast[selected_year]

st.success(f"💡 Predicted daily water consumption for the year {selected_year}, state '{selected_state}', and sector '{selected_category}': **{predicted_value:.2f} million litres**")

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts_data.index, ts_data.values, label='Actual Data', marker='o')
ax.plot(forecast.index, forecast.values, label='Forecast (DES)', linestyle='--', color='orange')
ax.scatter(selected_year, predicted_value, color='green', label='Prediction', s=100, zorder=5)

ax.set_xlabel('Year')
ax.set_ylabel('Water Consumption (million)')
ax.set_title('Double Exponential Smoothing Forecast')
ax.legend()
st.pyplot(fig)
