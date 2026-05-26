import os
import sys
import torch
from torch.utils.data import DataLoader

# Setup python path to be able to import relative modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from dataset import TESSDataset
from model import SpeechModel

# Define paths
root_dir = os.path.join(script_dir, "../..")
dataset_path = os.path.join(root_dir, "dataset/archive/TESS Toronto emotional speech set data")
model_path = os.path.join(root_dir, "speech_model.pth")

def test_speech():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Speech model on device: {device}")

    # Check weights exist
    if not os.path.exists(model_path):
        print(f"Error: Trained weights not found at {model_path}. Please run train_pipeline.py first.")
        return

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder not found at {dataset_path}")
        return

    print("Loading test dataset...")
    dataset = TESSDataset(dataset_path)
    # Split deterministic 20% test subset
    generator = torch.Generator().manual_seed(42)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = SpeechModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n========================================")
    print(f" Speech Model Test Complete")
    print(f" Total Samples Evaluated: {total}")
    print(f" Test Accuracy: {accuracy:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    test_speech()
