import sqlite3
import pandas as pd

# Connect to database and check exact class distribution
conn = sqlite3.connect("database/fraud_detection.db")
df1 = pd.read_sql("SELECT is_fraud, COUNT(*) as count FROM transactions GROUP BY is_fraud", conn)
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

print("Exact Fraud Distribution in Database:")
print(df1)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, classification_report


df = df.sample(n=500000, random_state=42)
target_column = 'is_fraud' 
cols_to_drop = [target_column, 'fraud_type', 'transaction_id', 'timestamp', 'ip_address', 'device_hash']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]

X = df.drop(columns=cols_to_drop)
y = df[target_column]

# 2. Encode categorical columns
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
X = X.fillna(0)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Model
print("Training model...")
model = RandomForestClassifier(
    random_state=42, 
    n_jobs=-1, 
    n_estimators=100,
    max_depth=15,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# 5. Find the Optimal Threshold automatically using F1-score
y_probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Avoid division by zero
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print(f"\n✨ Optimal Threshold Found: {best_threshold:.4f}")

# Evaluate with the scientifically optimal threshold
y_optimal_pred = (y_probs >= best_threshold).astype(int)
print(classification_report(y_test, y_optimal_pred))