import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(768, 7)

    def forward(self, input_ids, attention_mask, return_embeddings=False):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.pooler_output

        pooled = self.dropout(pooled)

        if return_embeddings:
            return pooled

        return self.fc(pooled)