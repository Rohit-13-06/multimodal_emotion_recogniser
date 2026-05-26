import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import importlib.util

# Set random seeds for reproducibility
torch.manual_seed(42)

# Dynamic import utility to avoid name clashes
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load models
speech_mod = load_module_from_path("speech_model", os.path.join(script_dir, "models/speech_pipeline/model.py"))
text_mod = load_module_from_path("text_model", os.path.join(script_dir, "models/speech_pipeline/text_pipeline/model.py"))
fusion_mod = load_module_from_path("fusion_model", os.path.join(script_dir, "models/speech_pipeline/text_pipeline/fusion_pipeline/model.py"))

SpeechModel = speech_mod.SpeechModel
TextModel = text_mod.TextModel
FusionModel = fusion_mod.FusionModel

from multimodal_dataset import get_multimodal_datasets

# Global configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = os.path.join(script_dir, "dataset/archive/TESS Toronto emotional speech set data")
BATCH_SIZE = 32
SPEECH_EPOCHS = 10
TEXT_EPOCHS = 10
FUSION_EPOCHS = 10
LEARNING_RATE = 0.001

def evaluate_accuracy(model, loader, model_type="speech"):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(DEVICE)
            
            if model_type == "speech":
                x = batch["speech_mfcc"].to(DEVICE)
                outputs = model(x)
            elif model_type == "text":
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                outputs = model(input_ids, attention_mask)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return (correct / total) * 100

def train_speech(train_loader, val_loader):
    print("\n--- Training Speech Model ---")
    model = SpeechModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(SPEECH_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch["speech_mfcc"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_acc = evaluate_accuracy(model, val_loader, "speech")
        print(f"Epoch {epoch+1}/{SPEECH_EPOCHS} | Loss: {total_loss:.4f} | Test Acc: {val_acc:.2f}%")
        
    torch.save(model.state_dict(), os.path.join(script_dir, "speech_model.pth"))
    print("Speech model saved!")
    return model

def train_text(train_loader, val_loader):
    print("\n--- Training Text Model (BERT Classifier) ---")
    model = TextModel().to(DEVICE)
    
    # Freeze BERT parameters to make it lightning fast on CPU!
    for param in model.bert.parameters():
        param.requires_grad = False
        
    criterion = nn.CrossEntropyLoss()
    # Optimize only the classification head
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.005)
    
    for epoch in range(TEXT_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_acc = evaluate_accuracy(model, val_loader, "text")
        print(f"Epoch {epoch+1}/{TEXT_EPOCHS} | Loss: {total_loss:.4f} | Test Acc: {val_acc:.2f}%")
        
    torch.save(model.state_dict(), os.path.join(script_dir, "text_model.pth"))
    print("Text model saved!")
    return model

def train_fusion(train_loader, val_loader, speech_model, text_model):
    print("\n--- Training Fusion Model ---")
    
    # Ensure backbones are in eval mode and frozen
    speech_model.eval()
    text_model.eval()
    
    model = FusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(FUSION_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x_speech = batch["speech_mfcc"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            
            with torch.no_grad():
                # Extract embeddings
                speech_features = speech_model(x_speech, return_embeddings=True)
                text_features = text_model(input_ids, attention_mask, return_embeddings=True)
                
            optimizer.zero_grad()
            outputs = model(speech_features, text_features)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Evaluate fusion accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                x_speech = batch["speech_mfcc"].to(DEVICE)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                speech_features = speech_model(x_speech, return_embeddings=True)
                text_features = text_model(input_ids, attention_mask, return_embeddings=True)
                
                outputs = model(speech_features, text_features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = (correct / total) * 100
        print(f"Epoch {epoch+1}/{FUSION_EPOCHS} | Loss: {total_loss:.4f} | Test Acc: {val_acc:.2f}%")
        
    torch.save(model.state_dict(), os.path.join(script_dir, "fusion_model.pth"))
    print("Fusion model saved!")
    return model

def main():
    print(f"Loading datasets from: {DATASET_DIR}")
    train_dataset, val_dataset = get_multimodal_datasets(DATASET_DIR)
    
    if train_dataset is None:
        print("Error: Could not load TESS dataset. Check directory structure.")
        return
        
    print(f"Dataset successfully loaded! Train samples: {len(train_dataset)}, Test samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 1. Train speech pipeline
    speech_model = train_speech(train_loader, val_loader)
    
    # 2. Train text pipeline
    text_model = train_text(train_loader, val_loader)
    
    # 3. Train fusion pipeline
    fusion_model = train_fusion(train_loader, val_loader, speech_model, text_model)
    
    # Final evaluations
    speech_acc = evaluate_accuracy(speech_model, val_loader, "speech")
    text_acc = evaluate_accuracy(text_model, val_loader, "text")
    
    # Fusion final eval
    fusion_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            x_speech = batch["speech_mfcc"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            speech_features = speech_model(x_speech, return_embeddings=True)
            text_features = text_model(input_ids, attention_mask, return_embeddings=True)
            
            outputs = fusion_model(speech_features, text_features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    fusion_acc = (correct / total) * 100
    
    print("\n" + "="*40)
    print("      FINAL PERFORMANCE COMPARISON")
    print("="*40)
    print(f" Speech-Only Accuracy : {speech_acc:.2f}%")
    print(f" Text-Only Accuracy   : {text_acc:.2f}%")
    print(f" Multimodal Fusion    : {fusion_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
