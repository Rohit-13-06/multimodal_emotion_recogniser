import librosa
import numpy as np

SAMPLE_RATE = 16000
MAX_LENGTH = 160


def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Remove silence
    audio, _ = librosa.effects.trim(audio)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Padding
    if mfcc.shape[1] < MAX_LENGTH:
        pad_width = MAX_LENGTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :MAX_LENGTH]

    return mfcc