import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Load data
csv_path = "data/raw/nigerian_fraud_data.csv"
print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)

target_column = 'is_fraud'
cols_to_drop = [target_column, 'fraud_type', 'transaction_id', 'timestamp', 'ip_address', 'device_hash']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]

X = df.drop(columns=cols_to_drop)
y = df[target_column].values

# 2. One-hot encoding & scaling
print("Preprocessing features...")
X = pd.get_dummies(X, drop_first=True).fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Undersample Majority Class on Training Set ONLY (Prevents data leakage)
print("Balancing training data via undersampling...")
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train

fraud_df = train_df[train_df['target'] == 1]
normal_df = train_df[train_df['target'] == 0]

# Sample normal transactions to be 3x the fraud count (3:1 ratio)
normal_sampled = normal_df.sample(n=len(fraud_df) * 3, random_state=42)
balanced_train = pd.concat([fraud_df, normal_sampled]).sample(frac=1, random_state=42)

X_train_balanced = balanced_train.drop(columns=['target']).values
y_train_balanced = balanced_train['target'].values

print(f"Balanced Training Rows: {len(X_train_balanced)}")

# 5. PyTorch Datasets & Loaders
class FraudDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(FraudDataset(X_train_balanced, y_train_balanced), batch_size=256, shuffle=True)
test_loader = DataLoader(FraudDataset(X_test, y_test), batch_size=1024, shuffle=False)

# 6. Define Neural Network
class FraudFNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

model = FraudFNN(input_dim=X_train_balanced.shape[1]).to(device)

# Standard BCE Loss (No extreme pos_weight required now)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training Loop (Watch the loss actually go down this time!)
epochs = 5
print("Training Balanced FNN Model...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f}")

# 8. Evaluation on True Test Set
model.eval()
y_preds, y_true = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        probs = torch.sigmoid(model(batch_x))
        y_preds.extend((probs >= 0.5).cpu().numpy().astype(int))
        y_true.extend(batch_y.numpy().astype(int))

print("\n--- Balanced Deep Learning Evaluation Report ---")
print(classification_report(y_true, y_preds))