import pandas as pd

# Load raw tables (keep_default_na=False for tickets as required)
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

print("=== RAW ORDERS AUDIT ===")
print("Raw orders rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending rows:", (orders['status'] == 'pending').sum())
print("Distinct order_id values:", orders['order_id'].nunique())

# Unparseable order amounts
non_numeric_amounts = pd.to_numeric(orders['order_amount'], errors='coerce').isna().sum()
print("Unparseable order_amount entries:", non_numeric_amounts)

# Delivery minutes metrics
orders_del = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
print("Negative delivery_minutes:", (orders_del < 0).sum())
print("Blank delivery_minutes:", orders['delivery_minutes'].isna().sum() | (orders['delivery_minutes'] == "").sum())

print("\n=== RAW CUSTOMERS AUDIT ===")
print("Blank age count:", customers['age'].isna().sum())
print("Age outside 10–100:", ((customers['age'] < 10) | (customers['age'] > 100)).sum())
print("Distinct raw city spellings:", customers['city'].nunique())
print("Distinct non-blank raw gender spellings:", customers['gender'].dropna().nunique())
print("Blank gender count:", customers['gender'].isna().sum())