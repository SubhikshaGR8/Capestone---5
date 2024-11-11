import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

# Load your data
sales_data = pd.read_csv('Pizza_Sale - pizza_sales.csv')  # Adjust the path as needed
ingredient_data = pd.read_csv('Pizza_ingredients - Pizza_ingredients.csv')  # Adjust the path as needed

# Data Preprocessing
# Convert order_date to datetime
sales_data['order_date'] = pd.to_datetime(sales_data['order_date'], errors='coerce')
sales_data['quantity'] = sales_data['quantity'].astype(int)

# Grouping sales data by week
sales_data.set_index('order_date', inplace=True)
weekly_sales = sales_data.resample('W').sum()['quantity']
weekly_sales.index.freq = 'W-SUN'  # Set frequency explicitly

# Train ARIMA model
model = ARIMA(weekly_sales, order=(1, 1, 1))  # Adjust the order if needed
model_fit = model.fit()

# Forecasting
forecast_steps = 7  # Forecasting for the next week
forecast = model_fit.get_forecast(steps=forecast_steps)
predicted_sales = forecast.predicted_mean

# Actual values for MAPE calculation (ensure you have this data)
# For demonstration, we'll assume you have actual sales values
# Replace the following line with your actual sales data for the forecasted period
actual_values = weekly_sales[-forecast_steps:].values  # Ensure this has actual sales for the forecast period

# Calculate MAPE
predicted_values = predicted_sales.values

# Check if actual values are available for MAPE calculation
if len(actual_values) == len(predicted_values):
    mape = mean_absolute_percentage_error(actual_values, predicted_values)
    print(f'MAPE: {mape:.2f}%')
else:
    print("Actual values not available for MAPE calculation or mismatch in length.")

print("Actual Values:", actual_values)
print("Predicted Values:", predicted_values)

# Purchase Order Generation
# Ingredient Calculation
ingredient_totals = []

# Iterate through the forecasted sales to calculate required ingredients
for index, qty in predicted_sales.items():
    week_data = sales_data[sales_data.index.isocalendar().week == index.isocalendar().week]
    for _, row in week_data.iterrows():
        ingredients = ingredient_data[ingredient_data['pizza_name_id'] == row['pizza_name_id']]
        for _, ing_row in ingredients.iterrows():
            total_needed = qty * ing_row['Items_Qty_In_Grams'] / 1000  # Convert grams to kg
            
            ingredient_totals.append({
                'Ingredient': ing_row['pizza_ingredients'],
                'Quantity Needed': total_needed
            })

# Convert the list to a DataFrame
purchase_order = pd.DataFrame(ingredient_totals)

# Group by ingredient and sum quantities
purchase_order = purchase_order.groupby('Ingredient').sum().reset_index()

# Display the purchase order
print("Purchase Order:")
print(purchase_order)

# Save results to CSV (optional)
purchase_order.to_csv('purchase_order.csv', index=False)

print("Project completed successfully!")

