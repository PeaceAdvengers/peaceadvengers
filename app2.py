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
# Double Exponential Smoothing Forecast
model = ExponentialSmoothing(train['value'], trend='add', seasonal=None)
fit = model.fit()
forecast = fit.forecast(1)

fit = fit_expo_model(filtered_df)

# Year slider
selected_year = st.slider("Select Year", min_value=min(years), max_value=2040, value=2025)

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


# Combined figure
plt.figure(figsize=(14, 6))

# --- First chart: Double Exponential Smoothing Forecast ---
plt.subplot(1, 2, 1)
plt.plot(train['Year'], train['Count'], label='Actual Data', color='blue')
plt.plot([train['Year'].iloc[-1], train['Year'].iloc[-1] + 1],
         [train['Count'].iloc[-1], forecast.values[0]],
         label='Forecast', color='red')
plt.scatter(train['Year'].iloc[-1] + 1, forecast.values[0], color='green', label='Prediction', s=100)
plt.xlabel('Year')
plt.ylabel('Water Consumption (million litres)')
plt.title('Double Exponential Smoothing Forecast')
plt.legend()

# --- Second chart: Seasonal Decomposition ---
plt.subplot(1, 2, 2)
decompose_result = sm.tsa.seasonal_decompose(train['Count'], model='additive', period=1)  # Adjust period if needed
decompose_result.plot()
plt.title('Seasonal Decomposition of Water Consumption')

plt.tight_layout()
plt.show()
