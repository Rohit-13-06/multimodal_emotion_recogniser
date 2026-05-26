import torch
import torch.nn as nn

class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(256 + 768, 256)

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 7)

    def forward(self, speech_features, text_features):

        fused = torch.cat(
            (speech_features, text_features),
            dim=1
        )

        x = self.fc1(fused)

        x = self.dropout(x)

        output = self.fc2(x)

        return output