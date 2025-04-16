import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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
if selected_category == 'All':
    # Filter the selected state
    temp_df = df[df['state'] == selected_state]

    # Group by date and state, then sum the value
    grouped_df = temp_df.groupby(['date', 'state'], as_index=False)['value'].sum()

    # Add a 'sector' column so the structure matches
    grouped_df['sector'] = 'All'

    # Reorder columns to match expected order
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

# Train Linear Regression model
@st.cache_resource
def train_model(dataframe, is_all):
    if is_all:
        X = dataframe[['date']]
        y = dataframe['value']
        model = LinearRegression().fit(X, y)
        return model, dataframe, ['date']
    else:
        df_encoded = pd.get_dummies(dataframe, columns=['state', 'sector'], drop_first=True)
        X = df_encoded[['date'] + [col for col in df_encoded.columns if col.startswith('state_') or col.startswith('sector_')]]
        y = df_encoded['value']
        model = LinearRegression().fit(X, y)
        return model, df_encoded, X.columns.tolist()


is_all = selected_category == 'All'
model, df_encoded, model_features = train_model(filtered_df, is_all)

# Year slider
selected_year = st.slider("Select a Year", min_value=min(years), max_value=2040, value=2025)

# Prepare input for prediction
input_dict = {'date': selected_year}

if not is_all:
    for feature in model_features:
        if feature.startswith("state_"):
            input_dict[feature] = 1 if feature == f"state_{selected_state}" else 0
        elif feature.startswith("sector_"):
            input_dict[feature] = 1 if feature == f"sector_{selected_category}" else 0

# Fill in any missing expected features
for feature in model_features:
    input_dict.setdefault(feature, 0)

X_new = pd.DataFrame([input_dict])[model_features]
predicted_proportion = model.predict(X_new)[0]


st.success(f"💡 Predicted daily water consumption for the year {selected_year}, state '{selected_state}', and sector '{selected_category}': **{predicted_proportion:.2f} million litres**")

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_encoded['date'], df_encoded['value'], color='blue', label='Actual Data')
y_pred = model.predict(df_encoded[model_features])
ax.plot(df_encoded['date'], y_pred, color='red', label='Regression Line')
ax.scatter(selected_year, predicted_proportion, color='green', label='Prediction', s=100, zorder=5)

ax.set_xlabel('Year')
ax.set_ylabel('Water Consumption (million)')
ax.set_title('Linear Regression: Water Consumption over Time')
ax.legend()
st.pyplot(fig)

# --- SARIMA Forecasting ---
st.subheader("📈 SARIMA Forecasting (Seasonal ARIMA)")

# Prepare data for SARIMA
df_sarima = filtered_df.copy()
df_sarima['date'] = pd.to_datetime(df_sarima['date'], format='%Y')  # Assumes 'date' is year only
df_sarima = df_sarima.set_index('date').sort_index()
df_sarima = df_sarima.resample('YS').sum()  # Year start frequency, adjust if needed

# Check if enough data for SARIMA
if len(df_sarima) >= 24:  # Need enough points for seasonal modeling
    # Fit SARIMA model
    sarima_model = SARIMAX(df_sarima['value'],
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    sarima_results = sarima_model.fit(disp=False)

    # Forecast 12 steps ahead
    forecast_steps = 12
    forecast = sarima_results.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=df_sarima.index[-1] + pd.offsets.YearBegin(1),
                                   periods=forecast_steps,
                                   freq='YS')
    forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
    conf_int = forecast.conf_int()

    # Plot SARIMA forecast
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df_sarima['value'].plot(ax=ax2, label='Observed', color='blue')
    forecast_series.plot(ax=ax2, label='Forecast', color='green')
    ax2.fill_between(forecast_index,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='pink', alpha=0.3, label='95% Confidence Interval')
    ax2.set_title('SARIMA Forecast: Water Consumption')
    ax2.set_ylabel('Water Consumption (million)')
    ax2.set_xlabel('Year')
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("Not enough historical data to perform SARIMA forecasting. At least 24 data points are recommended.")
