import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

# Step 1: Load the datasets
sales_data = pd.read_csv('Pizza_Sale - pizza_sales.csv')  # Replace with your actual file path
ingredient_data = pd.read_csv('Pizza_ingredients - Pizza_ingredients.csv')  # Replace with your actual file path

# Step 2: Data Cleaning
sales_data['order_date'] = pd.to_datetime(sales_data['order_date'], format='%d-%m-%Y', errors='coerce')
sales_data.dropna(inplace=True)
sales_data['quantity'] = pd.to_numeric(sales_data['quantity'], errors='coerce')
sales_data = sales_data[(sales_data['quantity'] > 0) & (sales_data['quantity'] < 100)]
sales_data['month'] = sales_data['order_date'].dt.month.astype(int)

# Step 3: Exploratory Data Analysis (EDA)
monthly_sales = sales_data.groupby('month')['quantity'].sum()

plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title('Monthly Sales Distribution')
plt.xlabel('Month')
plt.ylabel('Quantity Sold')
plt.xticks(monthly_sales.index)
plt.show()

# Step 4: Feature Engineering
sales_data['day_of_week'] = sales_data['order_date'].dt.dayofweek
sales_data['week_of_year'] = sales_data['order_date'].dt.isocalendar().week

# Step 5: Model Selection and Training
weekly_sales = sales_data.resample('W-Mon', on='order_date').sum()['quantity']

# Fit an ARIMA model
try:
    model = ARIMA(weekly_sales, order=(1, 1, 1))
    model_fit = model.fit()
except Exception as e:
    print("Error fitting ARIMA model:", e)

# Step 6: Forecasting
forecast_steps = 7  # Predicting for the next week
forecast = model_fit.forecast(steps=forecast_steps)

# Debugging: Print forecast results
print("Forecasted Sales for Next Week:")
print(forecast)

# Step 7: Ingredient Calculation
total_ingredients = {}
for idx, qty in enumerate(forecast):
    if idx < len(ingredient_data):  # Assuming ingredient_data is structured appropriately
        pizza_id = ingredient_data.iloc[idx]['pizza_name_id']  # Use actual mapping logic
        ingredients = ingredient_data[ingredient_data['pizza_name_id'] == pizza_id]
        for _, row in ingredients.iterrows():
            ingredient = row['pizza_ingredients']
            required_qty = row['Items_Qty_In_Grams'] * qty
            total_ingredients[ingredient] = total_ingredients.get(ingredient, 0) + required_qty

# Step 8: Generate Purchase Order
if total_ingredients:
    purchase_order = pd.DataFrame(list(total_ingredients.items()), columns=['Ingredient', 'Quantity Needed'])
else:
    purchase_order = pd.DataFrame(columns=['Ingredient', 'Quantity Needed'])

purchase_order.to_csv('purchase_order.csv', index=False)

# Debugging: Print purchase order
print("Purchase Order:")
print(purchase_order)

# Step 9: Model Evaluation
# Example: Actual values for the last week (if available)
actual_values = weekly_sales[-forecast_steps:].values
predicted_values = forecast.values

if len(actual_values) == len(predicted_values):
    mape = mean_absolute_percentage_error(actual_values, predicted_values)
    print(f'MAPE: {mape:.2f}%')
else:
    print("Actual values not available for MAPE calculation.")

# Step 10: Save the cleaned datasets
sales_data.to_csv('cleaned_sales_data.csv', index=False)
ingredient_data.to_csv('cleaned_ingredient_data.csv', index=False)

print("Project completed successfully!")
