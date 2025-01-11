# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly.offline import plot
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the Dataset
file_path = r"C:\Users\Admin\Downloads\DailyDelhiClimateTest.csv"
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Data Overview
print(data.head())
print(data.describe())
print(data.info())

# Check for Missing Values
print("Missing Values:\n", data.isnull().sum())

# Parse Dates and Add New Columns
try:
    data["date"] = pd.to_datetime(data["date"], format='%d-%m-%Y')
except ValueError:
    print("Error: Incorrect date format in the dataset. Ensure it matches '%d-%m-%Y'.")
    exit()

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

# Function to Determine Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

data['season'] = data['month'].apply(get_season)

# Visualizations
# 1. Mean Temperature Over Time
figure = px.line(data, x="date", y="meantemp", title='Mean Temperature in Delhi Over the Years', markers=True)
figure.show()

# 2. Humidity Over Time
figure = px.line(data, x="date", y="humidity", title='Humidity in Delhi Over the Years', markers=True)
figure.show()

# 3. Wind Speed Over Time
figure = px.line(data, x="date", y="wind_speed", title='Wind Speed in Delhi Over the Years', markers=True)
figure.show()

# 4. Relationship Between Temperature and Humidity
figure = px.scatter(data_frame=data, x="humidity", y="meantemp", size="meantemp", 
                    trendline="ols", title="Relationship Between Temperature and Humidity")
figure.show()

# 5. Temperature Change by Year and Month
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data=data.sort_values(by=['year', 'month']), x='month', y='meantemp', hue='year')
plt.show()

# Forecasting with Facebook Prophet
# Prepare Data for Prophet
forecast_data = data.rename(columns={"date": "ds", "meantemp": "y"})
forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])

# Add Seasonal Components
model = Prophet()
model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

# Optional: Add Additional Regressors
if 'humidity' in forecast_data.columns and 'wind_speed' in forecast_data.columns:
    forecast_data['humidity'] = data['humidity']
    forecast_data['wind_speed'] = data['wind_speed']
    model.add_regressor('humidity')
    model.add_regressor('wind_speed')

# Fit the Model
model.fit(forecast_data)

# Make Future Predictions
future = model.make_future_dataframe(periods=365)
if 'humidity' in forecast_data.columns and 'wind_speed' in forecast_data.columns:
    future['humidity'] = forecast_data['humidity'].mean()  # Replace with actual values if available
    future['wind_speed'] = forecast_data['wind_speed'].mean()

predictions = model.predict(future)

# Plot Forecast
fig, ax = plt.subplots(figsize=(10, 6))
model.plot(predictions, ax=ax)
plt.title("Forecasted Mean Temperature")
plt.show()

fig = plot_plotly(model, predictions)
plot(fig)

# Evaluate the Model
actual = forecast_data['y'].values
predicted = predictions['yhat'][:len(actual)].values
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f"Model Evaluation:\nMAE: {mae}\nRMSE: {rmse}")
