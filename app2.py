import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt

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

# Filter data
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

# Preprocess data (you can add your feature engineering steps here)
filtered_df = filtered_df.sort_values('date')
filtered_df['date'] = pd.to_numeric(filtered_df['date'])

# Add features like 'year' or 'month' if applicable
filtered_df['year'] = pd.to_datetime(filtered_df['date'], format='%Y').dt.year
filtered_df['month'] = pd.to_datetime(filtered_df['date'], format='%Y').dt.month

# Train-test split (e.g., last 5 years for testing)
split_year = filtered_df['date'].max() - 5
train = filtered_df[filtered_df['date'] <= split_year]
test = filtered_df[filtered_df['date'] > split_year]

# Fit Holt’s Linear Trend Model
fit = Holt(np.asarray(train['value'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
forecast_vals = fit.forecast(len(test))

# Merge forecast with test for prediction display
y_hat_avg = test.copy()
y_hat_avg['forecast'] = forecast_vals

# Fit Linear Regression Model with multiple features
X_train = train[['date', 'year', 'month']]  # Using date, year, and month as features
y_train = train['value'].values

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Linear Regression forecast for the test set
X_test = test[['date', 'year', 'month']]  # Same features used for the test
test['lr_forecast'] = linear_model.predict(X_test)

# Prediction slider for selected year
selected_year = st.slider("Select a Year", min_value=min(years), max_value=2040, value=2025)
last_year = filtered_df['date'].max()

# Predict future using Linear Regression
X_future = np.array([[selected_year, selected_year, 1]])  # Future year prediction
predicted_lr = linear_model.predict(X_future)[0]

# Holt's prediction for the selected year
if selected_year <= filtered_df['date'].max():
    try:
        predicted_value_holt = filtered_df[filtered_df['date'] == selected_year]['value'].values[0]
    except IndexError:
        predicted_value_holt = np.nan
else:
    future_steps = selected_year - filtered_df['date'].max()
    predicted_value_holt = float(fit.forecast(future_steps)[-1])

# Display both predictions
st.success(f"🟢 Holt's forecast for {selected_year}: **{predicted_value_holt:.2f} million litres**")

# First plot for Holt's Forecast
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(train['date'], train['value'], label='Train Data', color='blue')
ax1.plot(test['date'], test['value'], label='Test Data', color='orange')
ax1.plot(y_hat_avg['date'], y_hat_avg['forecast'], label='Holt Forecast', color='green', linestyle='--')

# Future predictions for Holt
if selected_year > filtered_df['date'].max():
    ax1.scatter(selected_year, predicted_value_holt, color='green', s=100, marker='o', label=f'Holt Prediction for {selected_year}')

# Set labels and title for Holt
ax1.set_xlabel('Year')
ax1.set_ylabel('Water Consumption (million litres)')
ax1.set_title('📈 Holt Forecast for Water Consumption')
ax1.legend(loc='best')
ax1.grid(True)

# Display Holt's plot
st.pyplot(fig1)

st.success(f"🔵 Linear Regression forecast for {selected_year}: **{predicted_lr:.2f} million litres**")
# Second plot for Linear Regression Forecast
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot actual data points
ax2.scatter(filtered_df['date'], filtered_df['value'], color='blue', label='Actual Data')
ax2.plot(filtered_df['date'], linear_model.predict(filtered_df[['date', 'year', 'month']]), color='red', label='Regression Line')
ax2.scatter(selected_year, predicted_lr, color='green', label='Prediction', s=100, zorder=5)

# Label the axes and add the title
ax2.set_xlabel('Year')
ax2.set_ylabel('Water Consumption (million litres)')
ax2.set_title('Linear Regression Forecast for Water Consumption')
ax2.legend(loc='best')
ax2.grid(True)

# Display Linear Regression plot
st.pyplot(fig2)

