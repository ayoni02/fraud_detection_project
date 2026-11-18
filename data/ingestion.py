import os
import pandas as pd
import sqlite3

# 1. Define paths
db_path = "database/fraud_detection.db"
csv_path = "data/raw/nigerian_fraud_data.csv"

# Ensure database directory exists
os.makedirs("database", exist_ok=True)

# 2. Load the CSV into pandas
print("Loading CSV into memory...")
df = pd.read_csv(csv_path)

# 3. Connect to SQLite database (creates it if it doesn't exist)
print(f"Connecting to database at {db_path}...")
conn = sqlite3.connect(db_path)

# 4. Write dataframe to SQL table
print("Writing data to 'transactions' table...")
df.to_sql('transactions', conn, if_exists='replace', index=False)

# 5. Verify ingestion with a quick query
query = "SELECT COUNT(*) as total_rows FROM transactions;"
result = pd.read_sql(query, conn)
print(f"Success! Total rows loaded into database: {result['total_rows'].values[0]}")

conn.close()