import os
import sys
import torch
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer

# Append speech pipeline folder to path to reuse preprocessing
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "models/speech_pipeline"))
from preprocess import preprocess_audio

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "ps": 6
}

class MultimodalTESSDataset(Dataset):
    def __init__(self, root_dir, tokenizer, cache=True):
        self.files = []
        self.tokenizer = tokenizer
        self.cache = cache
        self.cached_mfccs = {}

        if not os.path.exists(root_dir):
            print(f"Warning: root_dir {root_dir} does not exist.")
            return

        # Traversal including nested directory check
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                # Ignore nested TESS folders to avoid duplicates
                if "TESS Toronto emotional" in folder:
                    continue
                for file in os.listdir(folder_path):
                    if file.endswith(".wav"):
                        self.files.append(
                            os.path.join(folder_path, file)
                        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        # 1. Load/Preprocess audio
        if self.cache and idx in self.cached_mfccs:
            mfcc = self.cached_mfccs[idx]
        else:
            mfcc = preprocess_audio(file_path)
            if self.cache:
                self.cached_mfccs[idx] = mfcc

        # 2. Extract and tokenize text
        basename = os.path.basename(file_path)
        name_no_ext = basename.replace(".wav", "")
        parts = name_no_ext.split("_")
        
        # Word extraction
        if len(parts) >= 3:
            word = parts[1]
        elif len(parts) == 2:
            word = parts[0]
        else:
            word = name_no_ext

        # 3. Parse emotion label
        emotion = parts[-1].lower()
        if emotion not in emotion_map:
            # Fallback checks
            matched = False
            for k in emotion_map:
                if k in basename.lower():
                    label = emotion_map[k]
                    matched = True
                    break
            if not matched:
                label = emotion_map["neutral"]
        else:
            label = emotion_map[emotion]

        # Tokenize text
        encoding = self.tokenizer(
            word,
            truncation=True,
            padding='max_length',
            max_length=16, # Words are short, 16 is plenty!
            return_tensors='pt'
        )

        return {
            "speech_mfcc": torch.tensor(mfcc, dtype=torch.float32),
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label),
            "raw_text": word,
            "file_path": file_path
        }

def get_multimodal_datasets(root_dir, val_split=0.2, seed=42):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MultimodalTESSDataset(root_dir, tokenizer, cache=True)
    
    if len(dataset) == 0:
        return None, None

    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    return train_dataset, val_dataset
