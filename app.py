import os
import sys
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, Response, send_file
import importlib.util
import random
import numpy as np

# Setup path and imports
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

speech_mod = load_module_from_path("speech_model", os.path.join(script_dir, "models/speech_pipeline/model.py"))
text_mod = load_module_from_path("text_model", os.path.join(script_dir, "models/speech_pipeline/text_pipeline/model.py"))
fusion_mod = load_module_from_path("fusion_model", os.path.join(script_dir, "models/speech_pipeline/text_pipeline/fusion_pipeline/model.py"))

SpeechModel = speech_mod.SpeechModel
TextModel = text_mod.TextModel
FusionModel = fusion_mod.FusionModel

from multimodal_dataset import MultimodalTESSDataset, emotion_map
from transformers import BertTokenizer

app = Flask(__name__)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = os.path.join(script_dir, "dataset/archive/TESS Toronto emotional speech set data")

# Load models helper
def load_trained_models():
    speech_path = os.path.join(script_dir, "speech_model.pth")
    text_path = os.path.join(script_dir, "text_model.pth")
    fusion_path = os.path.join(script_dir, "fusion_model.pth")

    if not (os.path.exists(speech_path) and os.path.exists(text_path) and os.path.exists(fusion_path)):
        return None, None, None

    speech_model = SpeechModel().to(DEVICE)
    speech_model.load_state_dict(torch.load(speech_path, map_location=DEVICE))
    speech_model.eval()

    text_model = TextModel().to(DEVICE)
    text_model.load_state_dict(torch.load(text_path, map_location=DEVICE))
    text_model.eval()

    fusion_model = FusionModel().to(DEVICE)
    fusion_model.load_state_dict(torch.load(fusion_path, map_location=DEVICE))
    fusion_model.eval()

    return speech_model, text_model, fusion_model

@app.route('/')
def home():
    return send_file(os.path.join(script_dir, 'templates/index.html'))

@app.route('/api/status')
def get_status():
    speech_path = os.path.join(script_dir, "speech_model.pth")
    text_path = os.path.join(script_dir, "text_model.pth")
    fusion_path = os.path.join(script_dir, "fusion_model.pth")

    trained = os.path.exists(speech_path) and os.path.exists(text_path) and os.path.exists(fusion_path)
    return jsonify({
        "trained": trained,
        "models": {
            "speech": os.path.exists(speech_path),
            "text": os.path.exists(text_path),
            "fusion": os.path.exists(fusion_path)
        }
    })

@app.route('/api/train/stream')
def train_stream():
    def generate():
        import subprocess
        # Run training pipeline with correct python executable inside venv
        python_exe = os.path.join(script_dir, "venv/Scripts/python.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable # fallback
            
        process = subprocess.Popen(
            [python_exe, os.path.join(script_dir, "train_pipeline.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in iter(process.stdout.readline, ""):
            yield f"data: {line}\n\n"
        process.stdout.close()
        process.wait()
        yield "data: [COMPLETE]\n\n"
        
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/list_samples')
def list_samples():
    if not os.path.exists(DATASET_DIR):
        return jsonify({"error": "Dataset folder not found."}), 404

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MultimodalTESSDataset(DATASET_DIR, tokenizer, cache=False)
    
    if len(dataset) == 0:
        return jsonify({"error": "No .wav files found in dataset."}), 404

    # Group by emotion to select balanced samples
    samples_by_emotion = {}
    for idx, file_path in enumerate(dataset.files):
        basename = os.path.basename(file_path)
        parts = basename.replace(".wav", "").split("_")
        emotion = parts[-1].lower()
        if emotion not in emotion_map:
            continue
            
        if emotion not in samples_by_emotion:
            samples_by_emotion[emotion] = []
            
        # Get word
        if len(parts) >= 3:
            word = parts[1]
        elif len(parts) == 2:
            word = parts[0]
        else:
            word = parts[-1]
            
        samples_by_emotion[emotion].append({
            "idx": idx,
            "file_name": basename,
            "speaker": parts[0],
            "word": word,
            "emotion": emotion
        })

    # Pick 8 random samples for each emotion
    selected_samples = []
    for emotion, list_s in samples_by_emotion.items():
        selected_samples.extend(random.sample(list_s, min(len(list_s), 8)))

    # Shuffle selected list
    random.shuffle(selected_samples)
    return jsonify(selected_samples)

@app.route('/api/play/<int:idx>')
def play_audio(idx):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MultimodalTESSDataset(DATASET_DIR, tokenizer, cache=False)
    if idx < 0 or idx >= len(dataset):
        return jsonify({"error": "Invalid index."}), 400
        
    file_path = dataset.files[idx]
    return send_file(file_path, mimetype="audio/wav")

@app.route('/api/predict/<int:idx>')
def predict(idx):
    speech_model, text_model, fusion_model = load_trained_models()
    if not speech_model:
        return jsonify({"error": "Model weights not found. Please train models first."}), 400

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MultimodalTESSDataset(DATASET_DIR, tokenizer, cache=False)
    if idx < 0 or idx >= len(dataset):
        return jsonify({"error": "Invalid index."}), 400

    item = dataset[idx]
    
    # Run through Speech Model
    speech_input = item["speech_mfcc"].unsqueeze(0).to(DEVICE)
    speech_logits = speech_model(speech_input)
    speech_probs = torch.softmax(speech_logits, dim=1).squeeze().tolist()
    speech_pred = torch.argmax(speech_logits, dim=1).item()

    # Run through Text Model
    input_ids = item["input_ids"].unsqueeze(0).to(DEVICE)
    attention_mask = item["attention_mask"].unsqueeze(0).to(DEVICE)
    text_logits = text_model(input_ids, attention_mask)
    text_probs = torch.softmax(text_logits, dim=1).squeeze().tolist()
    text_pred = torch.argmax(text_logits, dim=1).item()

    # Run through Fusion Model
    speech_emb = speech_model(speech_input, return_embeddings=True)
    text_emb = text_model(input_ids, attention_mask, return_embeddings=True)
    fusion_logits = fusion_model(speech_emb, text_emb)
    fusion_probs = torch.softmax(fusion_logits, dim=1).squeeze().tolist()
    fusion_pred = torch.argmax(fusion_logits, dim=1).item()

    emotions_list = sorted(emotion_map.items(), key=lambda x: x[1])
    emotions = [k for k, v in emotions_list]

    # Convert MFCC to JSON array (downsample along time axis for fast transfer & visualization)
    mfcc_matrix = item["speech_mfcc"].tolist()

    return jsonify({
        "word": item["raw_text"],
        "true_emotion": emotions[item["label"].item()],
        "speech": {
            "prediction": emotions[speech_pred],
            "probabilities": {emotions[i]: speech_probs[i] for i in range(7)}
        },
        "text": {
            "prediction": emotions[text_pred],
            "probabilities": {emotions[i]: text_probs[i] for i in range(7)}
        },
        "fusion": {
            "prediction": emotions[fusion_pred],
            "probabilities": {emotions[i]: fusion_probs[i] for i in range(7)}
        },
        "mfcc": mfcc_matrix
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
