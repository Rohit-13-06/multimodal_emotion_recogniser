import torch
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import TextModel

texts = [
    "I am very happy today",
    "I am angry"
]

labels = [3, 0]

dataset = TextDataset(texts, labels)

loader = DataLoader(dataset, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextModel().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(5):

    for batch in loader:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1} complete")