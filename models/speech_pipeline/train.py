import torch
from torch.utils.data import DataLoader
from dataset import TESSDataset
from model import SpeechModel

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../../dataset/archive/TESS Toronto emotional speech set data")
dataset = TESSDataset(dataset_path)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpeechModel().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# SAVE MODEL HERE
torch.save(model.state_dict(), "speech_model.pth")

print("Model saved successfully!")