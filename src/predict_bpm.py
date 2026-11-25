
# IMPORTS
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import glob

input_folder_dubstep = Path("data/extracted_clips/Dubstep")
input_folder_jungle = Path("data/extracted_clips/Jungle")
input_folder_house = Path("data/extracted_clips/House")

audio_files_dubstep = [
    file for file in input_folder_dubstep.iterdir()
    if file.suffix.lower() in [".wav", ".mp3"]
]
audio_files_jungle = [
    file for file in input_folder_jungle.iterdir()
    if file.suffix.lower() in [".wav", ".mp3"]
]
audio_files_house = [
    file for file in input_folder_house.iterdir()
    if file.suffix.lower() in [".wav", ".mp3"]
]

# Define all of our extractable features

def extract_features(signal, sr):

        rms = float(np.mean(librosa.feature.rms(y=signal)))
        spec_cen = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)))
        spec_ban = float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)))
        spec_roll = float(np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)))
        onsets = librosa.onset.onset_detect(y=signal, sr=sr)
        onset_rate = float(len(onsets) / (len(signal) / sr))
        ons_str_std = float(np.std(librosa.onset.onset_strength(y=signal, sr=sr)))
        ons_str_mean = float(np.mean(librosa.onset.onset_strength(y=signal, sr=sr)))


        features_list = [rms, spec_cen, spec_ban, spec_roll, zcr, ons_str_std, onset_rate, ons_str_mean]
        return features_list

keys = {"Jungle": 0, "House": 1, "Dubstep": 2}
print(keys)

dict = {"key":[], "rms":[], "spec_cent":[], "spec_ban":[], "spec_roll":[], "zcr":[], "ons_str_std":[], "onset_rate":[], "ons_str_mean": []}

for file in audio_files_jungle:
        audio, sr = librosa.load(file, sr=44100, mono=True)
        features = extract_features(audio, 44100)
        dict["key"].append(0)
        dict["rms"].append(features[0])
        dict["spec_cent"].append(features[1])
        dict["spec_ban"].append(features[2])
        dict["spec_roll"].append(features[3])
        dict["zcr"].append(features[4])
        dict["ons_str_std"].append(features[5])
        dict["onset_rate"].append(features[6])
        dict["ons_str_mean"].append(features[7])


for file in audio_files_house:
        audio, sr = librosa.load(file, sr=44100, mono=True)
        features = extract_features(audio, 44100)
        dict["key"].append(1)
        dict["rms"].append(features[0])
        dict["spec_cent"].append(features[1])
        dict["spec_ban"].append(features[2])
        dict["spec_roll"].append(features[3])
        dict["zcr"].append(features[4])
        dict["ons_str_std"].append(features[5])
        dict["onset_rate"].append(features[6])
        dict["ons_str_mean"].append(features[7])

for file in audio_files_dubstep:
        audio, sr = librosa.load(file, sr=44100, mono=True)
        features = extract_features(audio, 44100)
        dict["key"].append(2)
        dict["rms"].append(features[0])
        dict["spec_cent"].append(features[1])
        dict["spec_ban"].append(features[2])
        dict["spec_roll"].append(features[3])
        dict["zcr"].append(features[4])
        dict["ons_str_std"].append(features[5])
        dict["onset_rate"].append(features[6])
        dict["ons_str_mean"].append(features[7])

df = pd.DataFrame(dict)

print(df)


#############
#############

# ANALYSE FEATURES

# Analyse RMS

plt.scatter(df[df["key"] == 0]["key"],
            df[df["key"] == 0]["rms"],
            color="green", label="Jungle")

plt.scatter(df[df["key"] == 1]["key"],
            df[df["key"] == 1]["rms"],
            color="blue", label="House")

plt.scatter(df[df["key"] == 2]["key"],
            df[df["key"] == 2]["rms"],
            color="red", label="Dubstep")

plt.xlabel("Genre (key)")
plt.ylabel("RMS")
plt.title("RMS by Genre")
plt.xticks([0, 1, 2], ["Jungle", "House", "Dubstep"])
plt.legend()
plt.show()


# Spectral Centroid Graphs point to an interesting observation

plt.scatter(df[df["key"] == 0]["key"],
            df[df["key"] == 0]["spec_cent"],
            color="green", label="Jungle")

plt.scatter(df[df["key"] == 1]["key"],
            df[df["key"] == 1]["spec_cent"],
            color="blue", label="House")

plt.scatter(df[df["key"] == 2]["key"],
            df[df["key"] == 2]["spec_cent"],
            color="red", label="Dubstep")

plt.xlabel("Genre (key)")
plt.ylabel("Spectral Centroid (Hz)")
plt.title("Spectral Centroid by Genre")
plt.xticks([0, 1, 2], ["Jungle", "House", "Dubstep"])
plt.legend()
plt.show()

# Analyze Spectral Bandwidth

plt.scatter(df[df["key"] == 0]["key"],
            df[df["key"] == 0]["spec_ban"],
            color="green", label="Jungle")

plt.scatter(df[df["key"] == 1]["key"],
            df[df["key"] == 1]["spec_ban"],
            color="blue", label="House")

plt.scatter(df[df["key"] == 2]["key"],
            df[df["key"] == 2]["spec_ban"],
            color="red", label="Dubstep")

plt.xlabel("Genre (key)")
plt.ylabel("Spectral Bandwidth (Hz)")
plt.title("Spectral Bandwidth by Genre")
plt.xticks([0, 1, 2], ["Jungle", "House", "Dubstep"])
plt.legend()
plt.show()

# Analyse Onset Strength

plt.scatter(df[df["key"] == 0]["key"],
            df[df["key"] == 0]["ons_str_mean"],
            color="green", label="Jungle")

plt.scatter(df[df["key"] == 1]["key"],
            df[df["key"] == 1]["ons_str_mean"],
            color="blue", label="House")

plt.scatter(df[df["key"] == 2]["key"],
            df[df["key"] == 2]["ons_str_mean"],
            color="red", label="Dubstep")

plt.xlabel("Genre (key)")
plt.ylabel("Onset Strength")
plt.title("Onset Strength by Genre")
plt.xticks([0, 1, 2], ["Jungle", "House", "Dubstep"])
plt.legend()
plt.show()

# Analyse Onset Rate

plt.scatter(df[df["key"] == 0]["key"],
            df[df["key"] == 0]["onset_rate"],
            color="green", label="Jungle")

plt.scatter(df[df["key"] == 1]["key"],
            df[df["key"] == 1]["onset_rate"],
            color="blue", label="House")

plt.scatter(df[df["key"] == 2]["key"],
            df[df["key"] == 2]["onset_rate"],
            color="red", label="Dubstep")

plt.xlabel("Genre (key)")
plt.ylabel("Onset Rate")
plt.title("Onset Rate by Genre")
plt.xticks([0, 1, 2], ["Jungle", "House", "Dubstep"])
plt.legend()
plt.show()

# Plot Onset Rate Mean against Onset Rate Standard Deviation

plt.scatter(df[df["key"] == 0]["ons_str_mean"],
            df[df["key"] == 0]["ons_str_std"],
            color="green", label="Jungle")

plt.scatter(df[df["key"] == 1]["ons_str_mean"],
            df[df["key"] == 1]["ons_str_std"],
            color="blue", label="House")

plt.scatter(df[df["key"] == 2]["ons_str_mean"],
            df[df["key"] == 2]["ons_str_std"],
            color="red", label="Dubstep")

plt.xlabel("Mean Onset Strength")
plt.ylabel("StDev Onset Strength")
plt.title("Onset Strength by Genre")
plt.legend()
plt.show()

# Plot Spectral Centroid against Spectral Bandwidth

plt.scatter(df[df["key"] == 0]["spec_cent"],
            df[df["key"] == 0]["ons_str_mean"],
            color="green", label="Jungle")

plt.scatter(df[df["key"] == 1]["spec_cent"],
            df[df["key"] == 1]["ons_str_mean"],
            color="blue", label="House")

plt.scatter(df[df["key"] == 2]["spec_cent"],
            df[df["key"] == 2]["ons_str_mean"],
            color="red", label="Dubstep")
plt.xlabel("Spectral Centroid (Hz)")
plt.ylabel("Mean Onset Strength")
plt.title("Spectral Centroid vs Mean Onset Strength")
plt.legend()
plt.show()

# Plot RMS against Spectral Centroid

plt.scatter(df[df["key"] == 0]["rms"],
            df[df["key"] == 0]["spec_cent"],
            color="green", label="Jungle")

plt.scatter(df[df["key"] == 1]["rms"],
            df[df["key"] == 1]["spec_cent"],
            color="blue", label="House")

plt.scatter(df[df["key"] == 2]["rms"],
            df[df["key"] == 2]["spec_cent"],
            color="red", label="Dubstep")

plt.xlabel("RMS")
plt.ylabel("Spectral Centroid")
plt.title("RMS vs Spectral Centroid")
plt.legend()
plt.show()

