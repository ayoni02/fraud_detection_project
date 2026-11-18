import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# 1. Load directly from CSV file in raw folder
csv_path = "data/raw/nigerian_fraud_data.csv"
print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)

target_column = 'is_fraud'
cols_to_drop = [target_column, 'fraud_type', 'transaction_id', 'timestamp', 'ip_address', 'device_hash']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]

X = df.drop(columns=cols_to_drop)
y = df[target_column].values

# 2. Encode categorical features and scale numerical features
print("Preprocessing features...")
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. PyTorch Dataset & DataLoader
class FraudDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FraudDataset(X_train, y_train), batch_size=512, shuffle=True)
test_loader = DataLoader(FraudDataset(X_test, y_test), batch_size=1024, shuffle=False)

# 5. Define Deep Feedforward Neural Network (FNN) Architecture
class FraudFNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = FraudFNN(input_dim=X_train.shape[1]).to(device)

# Handle class imbalance via pos_weight in BCEWithLogitsLoss
pos_count = sum(y_train == 1)
neg_count = sum(y_train == 0)
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
epochs = 5
print("Training Deep Learning FNN Model...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f}")

# 7. Model Evaluation
model.eval()
y_preds = []
y_true = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).cpu().numpy().astype(int)
        y_preds.extend(preds)
        y_true.extend(batch_y.numpy().astype(int))

print("\n--- Deep Learning (FNN) Evaluation Report ---")
print(classification_report(y_true, y_preds))