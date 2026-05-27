# Execution Instructions - Multimodal Emotion Recogniser 🚀

This document provides comprehensive, step-by-step instructions on how to configure, execute, train, and test all components of the Multimodal Emotion Recogniser framework.

---

## 📋 Prerequisites & Initial Setup

Ensure your local development machine satisfies the following hardware and software requirements:

* **Python:** Version 3.10.x to 3.13.x (Recommended: **Python 3.13.0**).
* **Git:** Installed and added to system path environment variables.
* **Hardware Note:** Can run entirely on CPU. CUDA-enabled GPUs will be automatically detected and utilized for acceleration.

### 1. Clone the Repository
Open a terminal (Command Prompt, PowerShell, or bash) and clone the repository:
```bash
git clone https://github.com/Rohit-13-06/multimodal_emotion_recogniser.git
cd multimodal_emotion_recogniser
```

### 2. Configure Virtual Environment (Recommended)
Set up a clean virtual environment to prevent package collisions:

* **On Windows (PowerShell / CMD):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Update `pip` and install all required library dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install flask
```

### 4. Set Up the TESS Dataset
1. Download the TESS (Toronto Emotional Speech Set) audio dataset.
2. Create a folder named `dataset` in the project root if it does not already exist.
3. Organize the raw TESS wav files exactly like this:
   `dataset/archive/TESS Toronto emotional speech set data/`
4. The directory must contain standard subfolders representing each emotion speaker pair (e.g. `OAF_angry`, `OAF_fear`, `YAF_happy`, `YAF_sad`), which contain the raw `.wav` recordings.

---

## 🕹️ 1. Running the Interactive Web Dashboard

The web dashboard is wrapped in a highly-optimized glassmorphic UI. It utilizes a **hot in-memory RAM cache** to serve prediction evaluations in **under 200ms**.

### Step 1: Launch the Backend Server
From the root directory of the project, run:
```bash
python app.py
```

### Step 2: Access the Visual Interface
Open your web browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

### Step 3: Playground Interaction
1. Use the **balanced test audio clip dropdown** selector to choose any pre-loaded wav file.
2. The web page will play the audio, render its Mel-spectrogram on the canvas, calculate logits across all three networks, animate the probability equalizers, and bounce a custom neon-glowing emoji matching the emotion in real-time!
3. Click the **Developer Console** button to toggle the log terminal. You can click **"Run Training Pipeline"** to watch the model train in real-time with chunked stdout logging.

---

## ⚙️ 2. Running the Master Training Pipeline

If you want to train all three neural networks (Speech, Text, and Late Fusion) sequentially via a single command-line orchestrator script:

```bash
python train_pipeline.py
```

### What this script executes:
1. Splits the TESS dataset into **80% Train** and **20% Test** splits using a seed value of `42`.
2. Trains the Speech CNN-BiLSTM network on MFCC frames and exports `speech_model.pth`.
3. Freezes BERT layers and trains the Text pooler classifier head, exporting `text_model.pth`.
4. Extracts speech and text abstract embedding vectors, trains the Late Fusion MLP on their concatenated 1,024-d hidden features, and exports `fusion_model.pth`.

---

## 🎙️ 3. Running Standalone Pipeline Scripts

Each individual neural network pipeline resides in a flat subdirectory under `models/` and houses dedicated standalone training (`train.py`) and testing (`test.py`) scripts.

> [!IMPORTANT]
> Because multiple neighboring folders contain files named `model.py`, these scripts utilize Python `importlib` dynamic specs to prevent namespace collisions. Always execute them from the **project root directory** as shown below.

### A. Standalone Speech Pipeline (CNN + BiLSTM)
* **Training Standalone:**
  ```bash
  python models/speech_pipeline/train.py
  ```
  *Trains the spatial-temporal network standalone on TESS MFCC frames and outputs `speech_model.pth` in the folder.*
* **Testing Standalone:**
  ```bash
  python models/speech_pipeline/test.py
  ```
  *Loads `speech_model.pth`, initializes the validation dataset split (20%), and outputs the exact accuracy score on speech.*

---

### B. Standalone Text Pipeline (BERT Head)
* **Training Standalone:**
  ```bash
  python models/text_pipeline/train.py
  ```
  *Loads the pre-trained `bert-base-uncased` language weights, freezes the base encoders, and optimizes the dense sentiment classifier head.*
* **Testing Standalone:**
  ```bash
  python models/text_pipeline/test.py
  ```
  *Loads `text_model.pth`, tokenizes validation word strings, and outputs standalone semantic accuracy.*

---

### C. Standalone Late-Fusion Pipeline (Concatenation MLP)
* **Training Standalone:**
  ```bash
  python models/fusion_pipeline/train.py
  ```
  *Loads the pre-trained, frozen speech and text encoders. Extracts hidden 256-d and 768-d embeddings, concatenates them, and trains the fusion classifier, exporting `fusion_model.pth`.*
* **Testing Standalone:**
  ```bash
  python models/fusion_pipeline/test.py
  ```
  *Loads all three checkpoints (`speech_model.pth`, `text_model.pth`, `fusion_model.pth`), runs unified test data through the encoders, and outputs joint fusion accuracy.*

---

## 📈 Verifying Outputs

### 1. Accuracy Summary Tables
Final comparative scores are stored in your directory under **`Results/accuracy_tables.md`**. You can verify standalone test scores directly in your terminal, which should print:
* **Speech Standalone:** `~96.25% Test Accuracy`
* **Text Standalone:** `~13.75% Test Accuracy` *(due to semantic neutralness of words)*
* **Multimodal Fusion:** `~98.93% Test Accuracy` *(resolving acoustic/linguistic ambiguity)*

### 2. Browser Diagnostic Checks
If the UI elements do not load, open your browser’s Developer Console (`F12` key):
* Check if there are any CORS or blocked asset errors.
* Dynamic cache-busting query strings (`?v=2` added to static script tags) ensure that the browser is always running the latest, optimized controller logic.
