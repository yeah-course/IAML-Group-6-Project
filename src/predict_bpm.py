
# IMPORTS
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
import os


# 1. LOAD AND MATCH THE DATA

# Load the BPM spreadsheet

audio_folder = Path("data/raw")
spreadsheet_path = Path("data/bpm_spreadsheet.xlsx")

bpm_labels = pd.read_excel(spreadsheet_path)
print(bpm_labels.head())

audio_files = os.listdir(audio_folder)
print("Number of audio files found:", len(audio_files))

# Match the datasets, remove _clip from the audio file name and .wav suffix.

audio_data = pd.DataFrame({
    "Song Title": [
        os.path.splitext(f)[0].replace("_clip", "")
        for f in audio_files
    ],
    "filename": audio_files
})

print(audio_data.head())
print(len(audio_data))

data = pd.merge(audio_data, bpm_labels, on="Song Title")

print(data.head())

# load the audio data numerically into an array and store it in a new column called "audio_data"

audio_data_list = []

for song in os.listdir(audio_folder):

    filepath = os.path.join(audio_folder, song)

    signal, sr = librosa.load(filepath, sr=22050, mono=True, duration=4.0)

    audio_data_list.append(signal)

audio_data_list = np.array(audio_data_list)

print(audio_data_list.shape)

# We will now write a feature extraction function that takes in a path to an audio file and returns the extracted features.

def extract_features(signal, sr):

    features = {

        "RMS": np.mean(librosa.feature.rms(y=signal)),
        "Spectral Centroid": np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)),
        "Spectral Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)),
        "Spectral Rolloff": np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr)),
        "Zero Crossing Rate": np.mean(librosa.feature.zero_crossing_rate(y=signal)),
        "Onset Strength": np.mean(librosa.onset.onset_strength(y=signal, sr=sr)),


    }
    return features

features_list = []

for signal in audio_data_list:
    features_list.append(extract_features(signal, sr))

features_df = pd.DataFrame(features_list)

print(features_df.head())
print()
print(features_df.shape)

# combine features with original data

data = pd.concat([data.reset_index(drop=True),
                  features_df.reset_index(drop=True)], axis=1)

print(data.head())
print(data.isnull().sum())
print(data.describe())

# look at feature distribution

features_df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Scale the data so its comparable on a graph

features = ["RMS", "Spectral Centroid", "Spectral Bandwidth",
            "Spectral Rolloff", "Zero Crossing Rate", "Onset Strength"]

from sklearn.preprocessing import StandardScaler

variables = ["BPM"] + features

scaler = StandardScaler()

scaled_data = pd.DataFrame(scaler.fit_transform(data[variables]), columns=variables)

# plot the data

import matplotlib.pyplot as plt

plt.plot
plt.scatter(scaled_data["BPM"], scaled_data["Onset Strength"])
plt.legend()
plt.xlabel("BPM")
plt.ylabel("FEATURES")
plt.title("RMS vs BPM")
plt.show()