import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from encoding_utils import MLP

df = pd.read_csv('mpsa.csv') 
rna_encoding = {'A': 0, 'C': 1, 'U': 2, 'G': 3}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot_encode(seq, vocab=rna_encoding):
    arr = np.array([vocab[base] for base in seq])
    return np.eye(len(vocab))[arr]  

# Keep only df['set'] that has training and validation
df = df[df['set'].isin(['training', 'validation'])]
one_hot_seqs = np.array([one_hot_encode(seq).flatten() for seq in df['x']])
targets = df['y'].values.astype(np.float32)
X_tensor = torch.tensor(one_hot_seqs, dtype=torch.float32).to(device)
y_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1).to(device) 
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_size = X_tensor.shape[1]  # 9 bases * 4 = 36
model = MLP(input_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "mlp_model.pt")
print("Model saved to mlp_model.pt")

# input_size = X_tensor.shape[1]  # 9 bases * 4 = 36
# model = MLP(input_size).to(device)
# model.load_state_dict(torch.load("mlp_model.pt", map_location=device))
# model.eval()

# # Test on test set
# df = pd.read_csv('mpsa.csv') 
# rna_encoding = {'A': 0, 'C': 1, 'U': 2, 'G': 3}
# df = df[df['set'].isin(['test'])]
# one_hot_seqs = np.array([one_hot_encode(seq).flatten() for seq in df['x']])
# targets = df['y'].values.astype(np.float32)
# X_tensor = torch.tensor(one_hot_seqs, dtype=torch.float32).to(device)
# y_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1).to(device) 
# dataset = TensorDataset(X_tensor, y_tensor)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Concatenate all
# y_all = []
# for xb, yb in loader:
#     xb, yb = xb.to(device), yb.to(device)
#     pred = model(xb)
#     y_all.append(pred.detach().cpu().numpy())

# y_all = np.concatenate(y_all)
# print(y_all[0:190])

#     print(pred) 