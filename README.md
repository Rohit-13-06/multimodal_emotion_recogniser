# Multimodal Emotion Recogniser 🧠🎙️📝

An advanced PyTorch and Hugging Face Transformers machine learning framework designed to perform comparative emotion classification on human speech and text transcriptions. It features a custom **1D-CNN + Bidirectional LSTM** speech network, a **BERT** text classification model, and a **Late Fusion Neural Network** that fuses both modalities to achieve superior classification confidence.

The project is wrapped in an ultra-premium, **glassmorphic dark-mode web application dashboard** that supports real-time pipeline training with interactive stdout streaming logs, detailed model evaluation gauges, a real-time inference playground with canvas-rendered audio spectrograms, a dynamic bouncing emoji showcase, and high-fidelity test accuracy cards.

---

## 📂 Project Directory Structure

Following professional software engineering principles, the codebase is strictly organized into modular pipelines:

```text
multimodal_emotion_recogniser/
│
├── models/
│   ├── speech_pipeline/       # 1D-CNN + BiLSTM Speech Model
│   │   ├── dataset.py         # Standard raw audio data loader
│   │   ├── model.py           # Speech CNN-BiLSTM network architecture definition
│   │   ├── preprocess.py      # Mel-Frequency Cepstral Coefficients (MFCC) feature extraction
│   │   ├── train.py           # Standalone Speech Model training loop
│   │   └── test.py            # Standalone Speech Model testing & validation evaluator
│   │
│   ├── text_pipeline/         # Pre-trained BERT Semantic Model
│   │   ├── dataset.py         # Word Tokenizer data loader
│   │   ├── model.py           # BERT-pooler output hidden classification network definition
│   │   ├── train.py           # Standalone Text Model training head optimizer
│   │   └── test.py            # Standalone Text Model testing & validation evaluator
│   │
│   └── fusion_pipeline/       # Multimodal Concatenation Late-Fusion Model
│       ├── model.py           # Late-fusion MLP classifier block definition
│       ├── train.py           # Standalone Fusion Model training head (extracts base embeddings)
│       └── test.py            # Standalone Fusion Model testing & validation evaluator
│
├── Results/
│   ├── accuracy_tables.md     # Comparative accuracy tables of all 3 variants
│   └── plots/                 # Subdirectory ready to house plots
│
├── static/
│   ├── app.js                 # Frontend script: SSE log stream, canvas heatmap drawing, dynamic glows
│   └── style.css              # Cosmic space dark-mode theme, pulsating auroras, and keyframe animations
│
├── templates/
│   └── index.html             # Premium visual layout with glassmorphic cards and circular gauges
│
├── app.py                     # High-speed production Flask backend server (in-memory persistent cache)
├── multimodal_dataset.py      # Core unified speech-and-text dataset class (RAM MFCC caching)
├── train_pipeline.py          # Unified master pipeline orchestrator (sequentially trains all 3 models)
├── requirements.txt           # Main python project dependency packages
└── README.md                  # System overview and detailed execution instructions
```

---

## 🌟 Key Visual & UI Features

* **Dynamic Bouncing Emotion Emoji Showcase:** A glassmorphic bouncing container beneath the spectrogram that decodes the winner's prediction in real-time. It maps outcomes to highly emotive animated emojis (`😡`, `🤢`, `😨`, `😊`, `😐`, `😢`, `😲`), bathing the layout in a custom high-intensity glow matching the emotion's tone (e.g. vibrant red shadow for `Angry`, emerald green for `Happy`, gold for `Surprise`).
* **Model Test Accuracies Grid:** Displays side-by-side metric badges for all three models (Speech: **96.3%**, Text: **64.3%**, Fusion: **98.9%**) right under the spectrogram, instantly proving late-fusion superiority.
* **Canvas Heatmap spectrogram Generator:** Decodes 40x160 MFCC audio feature matrices returned by the API and renders beautiful cyan-to-purple heatmaps on an HTML5 canvas pixel-by-pixel.
* **Asset Cache-Busters:** Standardizes asset queries (`style.css?v=2` and `app.js?v=2`) and injects strict, development-friendly caching headers via Flask's `@app.after_request` hook, completely preventing local browser cache retention and guaranteeing instant page reloads.

---

## ⚡ Production Optimizations (10x Speedup)

To deploy this deep learning framework in a responsive, real-time environment, we optimized the back-end infrastructure to reduce request latency by over **10x** (from ~2,000ms down to a blazing-fast **~180-240ms**):
1. **Global In-Memory Caching:** Established persistent global registries (`GLOBAL_MODELS`, `GLOBAL_TOKENIZER`, `GLOBAL_DATASET`) in `app.py`. Instead of reloading massive weights files (including the 438 MB BERT-based `text_model.pth`) from disk on every single click, the server loads them **exactly once** into active RAM on start.
2. **In-Memory RAM Caching:** Configured `cache=True` inside `MultimodalTESSDataset` to cache preprocessed audio MFCC matrices directly in RAM, bypassing slow disk traverses.
3. **Dynamic Hot-Swapping:** Configured a `clear_model_cache()` trigger within the web terminal's Server-Sent Events stream. When the training pipeline finishes, the server immediately flushes outdated weights from RAM and loads the newly generated checkpoints without requiring a server reboot!
4. **Evaluation Graph Tracking Exclusions:** Wrapped predictions inside PyTorch's `with torch.no_grad():` block to discard gradient computational trees during evaluations, shrinking RAM footprints and speeding up inferences.

---

## 🚀 Getting Started

### 📋 Prerequisites
- **Python 3.13.0** (or Python 3.10+)
- **Git**

### 💻 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Rohit-13-06/multimodal_emotion_recogniser.git
   cd multimodal_emotion_recogniser
   ```

2. **Initialize and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install flask
   ```

4. **Dataset Setup:**
   * Download the **TESS (Toronto Emotional Speech Set)** dataset.
   * Organize the raw wav files inside the root directory under the path:
     `dataset/archive/TESS Toronto emotional speech set data/`
   * *(The code expects directories like `OAF_angry/`, `YAF_fear/` containing files named `OAF_back_angry.wav`, etc.)*

---

## 🕹️ Execution Instructions

### A. Running the Premium Web Dashboard
To launch the interactive dashboard serve backend:
```bash
python app.py
```
Open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

* **Training Dashboard:** Click **"Run Training Pipeline"** to watch model compilation logs stream live into the custom visual terminal console. On completion, circular SVG progress rings will animate to show final comparative accuracies.
* **Interactive Playground:** Select any balanced test audio clip from the dropdown. The app will immediately play the wav, render its spectrogram, bounce the matching emoji, and display the confidence distributions for the Speech, Text, and Multimodal Fusion layers side-by-side!

---

### B. Running the Unified Training Pipeline
To train all three pipelines sequentially and export weight checkpoints (`speech_model.pth`, `text_model.pth`, `fusion_model.pth`) via a unified script:
```bash
python train_pipeline.py
```

---

### C. Running Standalone Pipeline Scripts
Each neural network pipeline has standalone, modular training and testing scripts. Because multiple directories house files named `model.py`, these scripts utilize dynamic path spec imports to prevent Python naming collisions.

#### 1. Speech Pipeline (1D-CNN + BiLSTM)
* **To Train Standalone:**
  ```bash
  python models/speech_pipeline/train.py
  ```
  *(Trains the CNN + BiLSTM network standalone on raw TESS MFCC frames and exports `speech_model.pth`.)*
* **To Test Standalone:**
  ```bash
  python models/speech_pipeline/test.py
  ```
  *(Loads `speech_model.pth` and evaluates spatial-temporal validation accuracy on the TESS test split.)*

#### 2. Text Pipeline (BERT Classifier)
* **To Train Standalone:**
  ```bash
  python models/text_pipeline/train.py
  ```
  *(Freezes core pre-trained BERT layers and optimizes only the dense emotion head on tokenized word transcripts.)*
* **To Test Standalone:**
  ```bash
  python models/text_pipeline/test.py
  ```
  *(Loads `text_model.pth` and evaluates semantic validation accuracy on TESS transcripts.)*

#### 3. Fusion Pipeline (Concatenation MLP)
* **To Train Standalone:**
  ```bash
  python models/fusion_pipeline/train.py
  ```
  *(Loads the trained speech and text encoders, freezes them to extract hidden vectors, concatenates them into a 1024-d joint embedding, and trains the Late Fusion MLP.)*
* **To Test Standalone:**
  ```bash
  python models/fusion_pipeline/test.py
  ```
  *(Loads `fusion_model.pth`, extracts base speech and text embeddings, and outputs joint validation accuracy.)*

---

## 🧪 Mathematical Separability & late Fusion Logic

A classic challenge in speech emotion recognition is distinguishing between **Happy** and **Pleasant Surprise (PS)**. Both present near-identical acoustic profiles: rapid syllable articulation rates, elevated average pitch, and high energy amplitudes.

* **Acoustic Ambiguity:** Speech-only models frequently confuse these two classes. For example, if a speaker voices the neutral word `"talk"` with an intense high-pitched spike, a speech-only model might classify it as Surprise.
* **Semantic Ambiguity:** Standalone text models fail on neutral transcripts. The word `"talk"` itself is emotionally neutral. Lacking vocal context, the Text model must make a statistical guess, frequently predicting incorrect classes like `"Disgust"`.
* **Late Fusion Resolution:** The Late Fusion MLP concatenates the **256-d Speech embedding** and the **768-d Text embedding**. By joining the two vectors, the model leverages text semantics to push apart the acoustic overlaps of *Happy/Surprise*, and resolves linguistic ambiguity by checking vocal tone.

This joint vector space results in linearly separable emotional clusters, elevating classification accuracy from 64.3% (Text) and 96.3% (Speech) to an outstanding **98.9% (Fusion)**!
