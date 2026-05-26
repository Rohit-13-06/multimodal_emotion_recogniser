import os
import sys
import torch
from torch.utils.data import DataLoader

# Setup python path to be able to import relative modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import dataset and encoders from root and neighbor folders
root_dir = os.path.join(script_dir, "../..")
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "models/speech_pipeline"))
sys.path.append(os.path.join(root_dir, "models/text_pipeline"))

from multimodal_dataset import get_multimodal_datasets
from speech_model import SpeechModel
from text_model import TextModel
from model import FusionModel

# Define paths
speech_path = os.path.join(root_dir, "speech_model.pth")
text_path = os.path.join(root_dir, "text_model.pth")
fusion_path = os.path.join(root_dir, "fusion_model.pth")
dataset_path = os.path.join(root_dir, "dataset/archive/TESS Toronto emotional speech set data")

def test_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Multimodal Fusion model on device: {device}")

    # Check weights exist
    if not (os.path.exists(speech_path) and os.path.exists(text_path) and os.path.exists(fusion_path)):
        print("Error: Trained model weights not found. Please train models first.")
        return

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder not found at {dataset_path}")
        return

    print("Loading test dataset...")
    _, val_dataset = get_multimodal_datasets(dataset_path)
    loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize and load speech model
    speech_model = SpeechModel().to(device)
    speech_model.load_state_dict(torch.load(speech_path, map_location=device))
    speech_model.eval()

    # Initialize and load text model
    text_model = TextModel().to(device)
    text_model.load_state_dict(torch.load(text_path, map_location=device))
    text_model.eval()

    # Initialize and load fusion model
    fusion_model = FusionModel().to(device)
    fusion_model.load_state_dict(torch.load(fusion_path, map_location=device))
    fusion_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            speech_input = batch["speech_mfcc"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Extract hidden features
            speech_emb = speech_model(speech_input, return_embeddings=True)
            text_emb = text_model(input_ids, attention_mask, return_embeddings=True)

            outputs = fusion_model(speech_emb, text_emb)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n========================================")
    print(f" Multimodal Fusion Model Test Complete")
    print(f" Total Samples Evaluated: {total}")
    print(f" Test Accuracy: {accuracy:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    test_fusion()
