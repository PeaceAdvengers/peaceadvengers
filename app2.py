import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load your dataset
df = pd.read_csv("water_consumption.csv")

# Streamlit layout
st.title("Clean Water Consumption Predictor")
st.write("### 💧Water: the only drink that falls from the sky for free.")
st.write("##### Every drop counts! Curious how much water we’ll use? Let’s find out together!")

# Dropdown selections
years = sorted(df['date'].unique())
states = sorted(df['state'].unique())
categories = ['All'] + sorted(df['sector'].unique())

selected_state = st.selectbox("Select State", states)
selected_category = st.selectbox("Select Sector", categories)

if selected_category == 'All':
    temp_df = df[df['state'] == selected_state]
    grouped_df = temp_df.groupby(['date'], as_index=False)['value'].sum()
    grouped_df['sector'] = 'All'
    filtered_df = grouped_df[['date', 'sector', 'value']]
else:
    filtered_df = df[
        (df['state'] == selected_state) &
        (df['sector'] == selected_category)
    ][['date', 'sector', 'value']]

if filtered_df.empty:
    st.warning("No data available for this combination. Please choose another state or sector.")
    st.stop()

# Sort by date and ensure numeric
filtered_df = filtered_df.sort_values('date')
filtered_df['date'] = pd.to_numeric(filtered_df['date'])

# Fit Exponential Smoothing model
@st.cache_resource
def fit_expo_model(data):
    model = ExponentialSmoothing(data['value'], trend='add', seasonal=None)
    fit = model.fit()
    return fit

fit = fit_expo_model(filtered_df)

# Year slider
selected_year = st.slider("Select a Year", min_value=min(years), max_value=2040, value=2025)

# Predict
prediction_index = selected_year
last_year = filtered_df['date'].max()

if selected_year <= last_year:
    predicted_value = filtered_df[filtered_df['date'] == selected_year]['value'].values[0]
else:
    forecast_years = selected_year - last_year
    future_forecast = fit.forecast(forecast_years)
    predicted_value = future_forecast.iloc[-1]

st.success(f"💡 Predicted daily water consumption for the year {selected_year}, state '{selected_state}', and sector '{selected_category}': **{predicted_value:.2f} million litres**")

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting actual data
ax.plot(filtered_df['date'], filtered_df['value'], label='Actual Data', color='blue', marker='o', linestyle='-', linewidth=2)

# Plotting the forecast data
forecast_range = list(range(int(last_year) + 1, selected_year + 1))
if forecast_range:
    forecast_vals = fit.forecast(len(forecast_range))
    ax.plot(forecast_range, forecast_vals, label='Forecast', color='red', linestyle='--', linewidth=2)

# Mark the prediction point
ax.scatter(selected_year, predicted_value, color='green', label='Prediction', s=100, zorder=5)

# Adding title and labels
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Water Consumption (million litres)', fontsize=12)
ax.set_title('Double Exponential Smoothing Forecast', fontsize=14)

# Adding grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Adding a legend
ax.legend(loc='upper left', fontsize=12)

# Display the plot
st.pyplot(fig)
