"""
Step 1: download the dataset, convert to Parquet, and PROFILE it before modelling.
Run:  python src/01_load_and_profile.py
Everything printed here should feed Chapter 3 (data description) and Chapter 4.
"""
import glob, os
import duckdb
from huggingface_hub import snapshot_download

REPO = "electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset"
OUT = "data/transactions.parquet"

os.makedirs("data", exist_ok=True)
if not os.path.exists(OUT):
    local = snapshot_download(repo_id=REPO, repo_type="dataset")
    files = glob.glob(f"{local}/**/*.csv", recursive=True) + glob.glob(f"{local}/**/*.parquet", recursive=True)
    print("Found files:", files)
    src = files[0]  # if there are several shards, change this to a glob pattern
    reader = f"read_csv_auto('{src}')" if src.endswith(".csv") else f"read_parquet('{src}')"
    duckdb.sql(f"COPY (SELECT * FROM {reader}) TO '{OUT}' (FORMAT PARQUET)")

con = duckdb.connect()
con.sql(f"CREATE VIEW t AS SELECT * FROM read_parquet('{OUT}')")

print("\n--- Row count & class balance ---")
print(con.sql("SELECT COUNT(*) AS n, SUM(is_fraud::INT) AS frauds, "
              "ROUND(100.0*AVG(is_fraud::INT),3) AS fraud_pct FROM t").df())

print("\n--- Fraud types ---")
print(con.sql("SELECT fraud_type, COUNT(*) n FROM t GROUP BY 1 ORDER BY 2 DESC").df())

print("\n--- Date range ---")
print(con.sql("SELECT MIN(timestamp) AS first, MAX(timestamp) AS last FROM t").df())

print("\n--- Accounts & history depth (needed for SQL features) ---")
print(con.sql("""
  SELECT COUNT(DISTINCT sender_account) AS senders,
         ROUND(AVG(c),1) AS avg_txns_per_sender, MEDIAN(c) AS median_txns
  FROM (SELECT sender_account, COUNT(*) c FROM t GROUP BY 1)
""").df())

print("\n--- Missing values (top 15) ---")
cols = [r[0] for r in con.sql("DESCRIBE t").fetchall()]
nulls = " UNION ALL ".join(f"SELECT '{c}' AS col, SUM(({c} IS NULL)::INT) AS nulls FROM t" for c in cols)
print(con.sql(f"SELECT * FROM ({nulls}) ORDER BY nulls DESC LIMIT 15").df())

print("\n--- Sanity check: negative time_since_last_transaction? ---")
print(con.sql("SELECT COUNT(*) AS negatives FROM t WHERE time_since_last_transaction < 0").df())

print("\n--- LEAKAGE CHECK: fraud rate by value of the pre-computed risk columns ---")
for c in ["merchant_fraud_rate", "persona_fraud_risk", "location_fraud_risk", "channel_risk_score"]:
    print(f"\n{c}")
    print(con.sql(f"SELECT ROUND({c},2) AS v, COUNT(*) n, ROUND(100*AVG(is_fraud::INT),2) AS fraud_pct "
                  f"FROM t GROUP BY 1 ORDER BY 1 LIMIT 12").df())