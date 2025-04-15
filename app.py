import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("water_consumption.csv")

# Streamlit layout
st.title("Clean Water Consumption Prediction")
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


st.success(f"💡 Predicted daily total consumption for the year {selected_year}, state '{selected_state}', and sector '{selected_category}': **{predicted_proportion:.2f} million litres**")

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
