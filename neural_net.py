import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv("dataset/train.csv")

X = data.drop("Target", axis=1)
y = data["Target"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print(X, y)


class NeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = X.shape[1]
hidden_size = 100
num_classes = len(np.unique(y))

model = NeuralNet(input_size, hidden_size, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    # Shuffle the data at the beginning of each epoch
    permutation = torch.randperm(X.size()[0])
    
    for i in range(0, X.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X[indices], y[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y).sum().item() / y.size(0)

print(f'Accuracy: {accuracy * 100:.2f}%')
