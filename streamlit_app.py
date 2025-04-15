import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\Vivian Yip\Downloads\water_consumption.csv")

df = load_data()

# Create a DataFrame
df = pd.DataFrame(data)

# Streamlit app layout
st.title("💧Clean Water Consumption Prediction")
st.write("### Select Year and Area")

# Let users select a year and area
years = df['date'].unique()  # Get unique years from your 'data' column
states = df['state'].unique()  # Get unique states from your 'state' column
categories = df['sector'].unique()

# Now, create the selectbox for state selection
selected_state = st.selectbox("Select State", list(states), index=0)
selected_category = st.selectbox("Select Sector", list(categories), index=0)

filtered_df = df
if selected_state:
    filtered_df = filtered_df[filtered_df['state'] == selected_state]
if selected_category:
    filtered_df = filtered_df[filtered_df['sector'] == selected_category]
st.write(filtered_df)

# Linear Regression
# We'll assume 'data' (year) is the independent variable and 'value' is the dependent variable.
# First, we need to encode the categorical variable (state) into numeric format.

@st.cache_resource
def train_model(filtered_df):
    df_encoded = pd.get_dummies(filtered_df, columns=['state', 'sector'], drop_first=True)
    X = df_encoded[['date'] + [col for col in df_encoded.columns if col.startswith('state_')]]
    y = df_encoded['value']
    model = LinearRegression().fit(X, y)
    return model, df_encoded, y

model, df_encoded, y = train_model(filtered_df)

selected_year = st.slider("Select a Year", min(df['date']), 2040, 2025)
if selected_state:
    selected_state_encoded = pd.get_dummies([selected_state], columns=states, drop_first=True)
else:
    selected_state_encoded = pd.DataFrame(np.zeros((1, len(states) - 1)), columns=[f"state_{s}" for s in states[1:]])

if selected_category:
    selected_category_encoded = pd.get_dummies([selected_category], columns=categories, drop_first=True)
else:
    selected_category_encoded = pd.DataFrame(np.zeros((1, len(categories) - 1)), columns=[f"category_{c}" for c in categories[1:]])

# Prepare the data for prediction
X_new = np.array([[selected_year] + list(selected_state_encoded.iloc[0]) + list(selected_category_encoded.iloc[0])])

# Predict the proportion for the selected year, state, and category
predicted_proportion = model.predict(X_new)[0]

# Display the predicted proportion
st.write(f"Predicted total consumption for the year {selected_year}, state {selected_state}, and category {selected_category}: {predicted_proportion:.2f} millions ")

# Plot the regression line and prediction
plt.figure(figsize=(10, 6))
plt.scatter(df_encoded['date'], y, color='blue', label='Actual Data')

# Plot the regression line
y_pred = model.predict(X)
plt.plot(df_encoded['date'], y_pred, color='red', label='Regression Line')

# Highlight the predicted point
plt.scatter(selected_year, predicted_proportion, color='green', label=f"Prediction for {selected_year}", s=100, zorder=5)

plt.xlabel('Year')
plt.ylabel('value')
plt.title('Linear Regression of Proportion vs. Year with State and Strata')
plt.legend()
st.pyplot(plt)
