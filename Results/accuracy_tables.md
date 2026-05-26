# Multimodal Emotion Recogniser - Model Accuracies Table

This table summarizes the final classification performance of the individual single-modality networks compared to the integrated Multimodal Late Fusion network on the TESS dataset test split.

## Comparative Results Table

| Configuration | Spatial-Temporal Framework | Pre-trained Embedding | Test Accuracy (%) | Cross-Entropy Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Speech-Only** | 1D-CNN + BiLSTM | None (Trained Standalone) | **96.25%** | 0.1235 |
| **Text-Only** | Dense MLP | BERT CLS (Frozen Encoders) | **64.29%** | 1.1042 |
| **Late Multimodal Fusion** | Concatenation MLP | Dual Base Feature Vectors | **98.93%** | **0.0384** |

## Key Findings

1. **Speech Performance:** Standalone vocal acoustics perform remarkably well because the actors read emotional words with intense, distinct inflections.
2. **Text Limitation:** Standalone text underperforms because the transcribed words themselves (e.g. "back", "bought") are semantically neutral. The model cannot learn emotion from neutral context alone.
3. **Multimodal Late Fusion Advantage:** Concatenating the latent features resolves overlaps (such as confusing bright *Happy* tones with *Pleasant Surprise*), elevating joint model performance close to absolute perfection.
