import sqlite3
import pandas as pd
import os

print("Building customer profiles and auditing tickets...")

# 1. Load raw data
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")
tickets = pd.read_csv("data/raw/tickets.csv")

print("=== RAW TICKETS AUDIT ===")
print("Total tickets:", len(tickets))
print("Unique ticket categories:", tickets['true_category'].nunique())

# 2. Save clean tickets and store in SQLite
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data", exist_ok=True)

tickets.to_csv("data/clean/tickets.csv", index=False)

conn = sqlite3.connect("data/instamart.db")
tickets.to_sql("tickets", conn, if_exists="replace", index=False)

# Reload clean tables from db for aggregation
customers = pd.read_sql("SELECT * FROM customers", conn)
orders = pd.read_sql("SELECT * FROM orders", conn)
tickets_db = pd.read_sql("SELECT * FROM tickets", conn)

# 3. Aggregate metrics per customer from orders
order_metrics = orders.groupby('customer_id').agg(
    total_orders=('order_id', 'count'),
    completed_orders=('status', lambda x: (x == 'completed').sum()),
    cancelled_orders=('status', lambda x: (x == 'cancelled').sum()),
    total_spend=('order_amount', 'sum'),
    avg_delivery_minutes=('delivery_minutes', 'mean')
).reset_index()

# 4. Aggregate metrics per customer from tickets
ticket_metrics = tickets_db.groupby('customer_id').agg(
    total_tickets=('ticket_id', 'count')
).reset_index()

# 5. Build unified customer_profiles table
profiles = customers.merge(order_metrics, on='customer_id', how='left') \
                    .merge(ticket_metrics, on='customer_id', how='left')

# Fill NaN values for customers with no orders or tickets
profiles['total_orders'] = profiles['total_orders'].fillna(0)
profiles['completed_orders'] = profiles['completed_orders'].fillna(0)
profiles['cancelled_orders'] = profiles['cancelled_orders'].fillna(0)
profiles['total_spend'] = profiles['total_spend'].fillna(0.0)
profiles['total_tickets'] = profiles['total_tickets'].fillna(0)

# Save profiles to CSV and SQLite
profiles.to_csv("data/clean/customer_profiles.csv", index=False)
profiles.to_sql("customer_profiles", conn, if_exists="replace", index=False)

conn.close()
print("Customer profiles built successfully and stored in SQLite!")