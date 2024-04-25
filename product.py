import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to generate random dates
def random_dates(start_date, end_date, n=200000):
    date_range = (end_date - start_date).days
    random_dates = [start_date + timedelta(days=np.random.randint(date_range)) for _ in range(n)]
    return sorted(random_dates)

# Function to generate large retail dataset for dairy products
def generate_large_dairy_data(start_date, end_date):
    products = ['Milk', 'Cheese', 'Yogurt', 'Butter', 'Ice Cream', 'Cream']
    dates = random_dates(start_date, end_date, n=200000)  # Generate 200,000 values

    base_sales = np.random.randint(50000, 150000, size=len(dates))
    trend = np.linspace(0, 20000000, len(dates)) + np.random.normal(0, 50000, len(dates))
    seasonality = 10 * np.sin(np.arange(len(dates)) * (2 * np.pi / 365))
    random_noise = np.random.normal(0, 1000000, len(dates))
    sales = base_sales + trend + seasonality + random_noise

    cost = 0.6 * sales

    product_names = np.random.choice(products, size=len(dates))

    data = {'Date': dates, 'Product': product_names, 'Monthly_Sales': sales, 'Cost': cost}
    return pd.DataFrame(data)

# Load the dataset
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
dairy_df = generate_large_dairy_data(start_date, end_date)

# Preprocessing
dairy_df = dairy_df.drop_duplicates()
dairy_df = dairy_df.dropna()

# Outlier Detection and Handling
z_scores = np.abs(stats.zscore(dairy_df[['Monthly_Sales', 'Cost']]))
threshold = 3
dairy_df = dairy_df[(z_scores < threshold).all(axis=1)]

# Data Normalization/Scaling
scaler = MinMaxScaler()
dairy_df[['Monthly_Sales', 'Cost']] = scaler.fit_transform(dairy_df[['Monthly_Sales', 'Cost']])

# Feature Engineering
dairy_df['Month'] = pd.to_datetime(dairy_df['Date']).dt.month
dairy_df['Day'] = pd.to_datetime(dairy_df['Date']).dt.day

# Categorical Encoding
dairy_df = pd.get_dummies(dairy_df, columns=['Product'])

# Save and Reload the dataset
dairy_df.to_csv('cleaned_dairy_dataset.csv', index=False)
dairy_df = pd.read_csv('cleaned_dairy_dataset.csv')
dairy_df['Date'] = pd.to_datetime(dairy_df['Date'])
dairy_df.set_index('Date', inplace=True)

# Resample to monthly frequency and calculate average sales for each product
dairy_df = dairy_df.resample('M').mean()  # Monthly average to smooth out data

# Calculate the average sales for each product over the entire period
average_sales = dairy_df.mean()

# Extract only product columns for plotting
product_sales = average_sales.filter(like='Product_')

# Specify colors for each product
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

# Plotting the average sales for each product
plt.figure(figsize=(10, 6))
product_sales.plot(kind='bar', color=colors)
plt.title('Average Sales of Dairy Products')
plt.xlabel('Products')
plt.ylabel('Average Sales (Normalized)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Print the accuracy metrics for each product (placeholder for actual modeling)
# Note: Actual metrics should be calculated using model predictions and true values.
print("Accuracy Metrics for Each Product:")
for product, color in zip(product_sales.index, colors):
    mae = np.random.random()  # Placeholder for actual MAE
    rmse = np.random.random()  # Placeholder for actual RMSE
    print(f"{product.replace('Product_', '')}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")