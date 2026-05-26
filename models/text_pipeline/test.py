import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Setup python path to be able to import relative modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import dataset from root folder since MultimodalTESSDataset maps TESS text nicely
root_dir = os.path.join(script_dir, "../..")
sys.path.append(root_dir)

from multimodal_dataset import MultimodalTESSDataset
from model import TextModel

# Define paths
dataset_path = os.path.join(root_dir, "dataset/archive/TESS Toronto emotional speech set data")
model_path = os.path.join(root_dir, "text_model.pth")

def test_text():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Text model on device: {device}")

    # Check weights exist
    if not os.path.exists(model_path):
        print(f"Error: Trained weights not found at {model_path}. Please run train_pipeline.py first.")
        return

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder not found at {dataset_path}")
        return

    print("Loading test dataset...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MultimodalTESSDataset(dataset_path, tokenizer, cache=True)
    # Split deterministic 20% test subset
    generator = torch.Generator().manual_seed(42)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = TextModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n========================================")
    print(f" Text Model Test Complete")
    print(f" Total Samples Evaluated: {total}")
    print(f" Test Accuracy: {accuracy:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    test_text()
