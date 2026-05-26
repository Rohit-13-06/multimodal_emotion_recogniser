import torch
import torch.nn as nn

class SpeechModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, 7)

    def forward(self, x, return_embeddings=False):

        x = self.cnn(x)

        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(x)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        if return_embeddings:
            return hidden

        output = self.fc(hidden)

        return output