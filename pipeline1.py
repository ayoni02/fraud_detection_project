import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, precision_recall_curve

# 1. Load directly from CSV file in raw folder
csv_path = "data/raw/nigerian_fraud_data.csv"
print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)

target_column = 'is_fraud'
cols_to_drop = [target_column, 'fraud_type', 'transaction_id', 'timestamp', 'ip_address', 'device_hash']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]

X = df.drop(columns=cols_to_drop)
y = df[target_column].values

# 2. Preprocessing & Encoding
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

# 5. Define Neural Network Architecture
class FraudFNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

model = FraudFNN(input_dim=X_train.shape[1]).to(device)

# 6. Focal Loss for Imbalanced Data
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.85, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)

# 7. Training Loop
epochs = 10
print("Training FNN with Focal Loss...")
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
        
    epoch_loss = total_loss / len(train_loader)
    scheduler.step(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f}")

# 8. Evaluation & Threshold Tuning
model.eval()
y_probs = []
y_true = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.sigmoid(outputs)
        y_probs.extend(probs.cpu().numpy())
        y_true.extend(batch_y.numpy().astype(int))

y_probs = np.array(y_probs).flatten()
y_true = np.array(y_true).flatten()

precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"\n✨ Optimal Decision Threshold: {best_threshold:.4f}")
y_preds = (y_probs >= best_threshold).astype(int)

print("\n--- Optimized Deep Learning Evaluation Report ---")
print(classification_report(y_true, y_preds))