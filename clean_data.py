import sqlite3
import pandas as pd
import os

print("Starting data cleaning pipeline...")

# 1. Load raw data
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

# ==========================================
# 2. CLEAN CUSTOMERS TABLE
# ==========================================
customers['city'] = customers['city'].str.strip().str.title()

gender_map = {
    'M': 'male', 'Male': 'male', 'male': 'male', 'm': 'male',
    'F': 'female', 'Female': 'female', 'female': 'female', 'f': 'female'
}
customers['gender'] = customers['gender'].str.strip().map(gender_map).fillna('unknown')

customers.loc[(customers['age'] < 10) | (customers['age'] > 100), 'age'] = pd.NA
city_medians = customers.groupby('city')['age'].transform('median')
customers['age'] = customers['age'].fillna(city_medians)
customers['signup_date'] = pd.to_datetime(customers['signup_date'])


# ==========================================
# 3. CLEAN ORDERS TABLE
# ==========================================
# Merge products table first so category and price_inr are available
orders = orders.merge(products[['product_id', 'category', 'price_inr']], on='product_id', how='left')

# (1) Convert order_ts to datetime
orders['order_ts'] = pd.to_datetime(orders['order_ts'])

# (2) Remove exact duplicate rows
orders = orders.drop_duplicates()

# (3) Keep only the record with the latest order_ts for each order_id
orders = orders.sort_values('order_ts').drop_duplicates(subset=['order_id'], keep='last')

# (4) Convert order_amount to numeric, impute missing from product price formula
orders['order_amount_num'] = pd.to_numeric(orders['order_amount'], errors='coerce')
missing_amt = orders['order_amount_num'].isna()
calculated_amt = round(orders['price_inr'] * orders['quantity'] * (1 - orders['discount_pct'] / 100), 2)
orders.loc[missing_amt, 'order_amount_num'] = calculated_amt[missing_amt]
orders['order_amount'] = orders['order_amount_num']

# (5) Handle delivery_minutes
orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = pd.NA

# Impute completed orders with median delivery time of category
cat_medians = orders[orders['status'] == 'completed'].groupby('category')['delivery_minutes'].transform('median')
completed_missing = (orders['status'] == 'completed') & (orders['delivery_minutes'].isna())
orders.loc[completed_missing, 'delivery_minutes'] = cat_medians[completed_missing]

# Drop temporary helper columns and format order_ts
orders = orders.drop(columns=['order_amount_num', 'price_inr', 'category'])
orders['order_ts'] = orders['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')


# ==========================================
# 4. SAVE CLEAN TABLES & STORE IN SQLITE
# ==========================================
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data", exist_ok=True)

customers.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders.to_csv("data/clean/orders.csv", index=False)

conn = sqlite3.connect("data/instamart.db")
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()

print("Cleaning complete! Clean CSVs saved and database 'data/instamart.db' updated successfully.")