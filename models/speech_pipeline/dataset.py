import os
import torch
from torch.utils.data import Dataset
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

class TESSDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []

        if not os.path.exists(root_dir):
            print(f"Warning: root_dir {root_dir} does not exist.")
            return

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".wav"):
                        self.files.append(
                            os.path.join(folder_path, file)
                        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        mfcc = preprocess_audio(file_path)

        # Robustly parse emotion
        basename = os.path.basename(file_path)
        parts = basename.replace(".wav", "").split("_")
        emotion = parts[-1].lower()

        if emotion not in emotion_map:
            # Fallback checks if formatting is slightly different (e.g. OAF_pleasant_surprise)
            matched = False
            for k in emotion_map:
                if k in basename.lower():
                    label = emotion_map[k]
                    matched = True
                    break
            if not matched:
                label = emotion_map["neutral"] # Fallback default
        else:
            label = emotion_map[emotion]

        return (
            torch.tensor(mfcc, dtype=torch.float32),
            torch.tensor(label)
        )