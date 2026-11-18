import sqlite3
import pandas as pd

conn = sqlite3.connect("database/fraud_detection.db")
df = pd.read_sql("SELECT * FROM transactions LIMIT 1000", conn)
conn.close()

print("Column cardinalities (unique values in sample):")
for col in df.select_dtypes(include=['object', 'category']).columns:
    print(f" - {col}: {df[col].nunique()} unique values")