import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup python path to be able to import relative modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import dataset and encoders from root and neighbor folders
root_dir = os.path.join(script_dir, "../..")
sys.path.append(root_dir)

from multimodal_dataset import get_multimodal_datasets

def load_module_from_path(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

speech_mod = load_module_from_path("speech_model", os.path.join(root_dir, "models/speech_pipeline/model.py"))
text_mod = load_module_from_path("text_model", os.path.join(root_dir, "models/text_pipeline/model.py"))

SpeechModel = speech_mod.SpeechModel
TextModel = text_mod.TextModel
from model import FusionModel

# Define paths
speech_path = os.path.join(root_dir, "speech_model.pth")
text_path = os.path.join(root_dir, "text_model.pth")
fusion_path = os.path.join(root_dir, "fusion_model.pth")
dataset_path = os.path.join(root_dir, "dataset/archive/TESS Toronto emotional speech set data")

def train_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Fusion model on device: {device}")

    # Check that speech and text models are already trained
    if not (os.path.exists(speech_path) and os.path.exists(text_path)):
        print("Error: Speech or Text model checkpoints not found. Please train Speech and Text models first.")
        return

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder not found at {dataset_path}")
        return

    print("Loading datasets...")
    train_dataset, val_dataset = get_multimodal_datasets(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize and load speech model
    speech_model = SpeechModel().to(device)
    speech_model.load_state_dict(torch.load(speech_path, map_location=device))
    speech_model.eval()
    for param in speech_model.parameters():
        param.requires_grad = False

    # Initialize and load text model
    text_model = TextModel().to(device)
    text_model.load_state_dict(torch.load(text_path, map_location=device))
    text_model.eval()
    for param in text_model.parameters():
        param.requires_grad = False

    # Initialize fusion model
    fusion_model = FusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)

    print("Beginning training loop (10 epochs)...")
    for epoch in range(10):
        fusion_model.train()
        total_loss = 0
        
        for batch in train_loader:
            speech_input = batch["speech_mfcc"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Extract embeddings from base networks
            with torch.no_grad():
                speech_emb = speech_model(speech_input, return_embeddings=True)
                text_emb = text_model(input_ids, attention_mask, return_embeddings=True)

            outputs = fusion_model(speech_emb, text_emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validate after each epoch
        fusion_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                speech_input = batch["speech_mfcc"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                speech_emb = speech_model(speech_input, return_embeddings=True)
                text_emb = text_model(input_ids, attention_mask, return_embeddings=True)

                outputs = fusion_model(speech_emb, text_emb)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch {epoch+1}/10 | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save weights
    torch.save(fusion_model.state_dict(), fusion_path)
    print(f"Fusion model weights saved to {fusion_path} successfully!")

if __name__ == "__main__":
    train_fusion()
