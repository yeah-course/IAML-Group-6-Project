
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

dict = {"rms":[], "spec_cent":[], "spec_ban":[], "spec_roll":[], "zcr":[], "ons_str_std":[], "onset_rate":[], "ons_str_mean": []}
keys = []

for file in audio_files_jungle:
        audio, sr = librosa.load(file, sr=44100, mono=True)
        features = extract_features(audio, 44100)
        keys.append(0)
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
        keys.append(1)
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
        keys.append(2)
        dict["rms"].append(features[0])
        dict["spec_cent"].append(features[1])
        dict["spec_ban"].append(features[2])
        dict["spec_roll"].append(features[3])
        dict["zcr"].append(features[4])
        dict["ons_str_std"].append(features[5])
        dict["onset_rate"].append(features[6])
        dict["ons_str_mean"].append(features[7])

df = pd.DataFrame(dict)[["rms", "spec_cent", "ons_str_mean"]]
y = np.array(keys)



#############
#############

# ANALYSE FEATURES

# Analyse RMS

def scatter_features(x, y):

        fig, ax = plt.subplots()

        ax.scatter(x, y)
        ax.set_xlabel(f'{x}')
        ax.set_ylabel(f'{y}')
        plt.legend()
        ax.set_title(f'{x} vs {y}')

        return fig

fig = (scatter_features(df["spec_cent"], df["ons_str_mean"]))



###########
###########

# Important Features: Spectral Centroid vs Mean Onset Strength
# Import Features: RMS vs Spectral Centroid

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix


dict_train, dict_remaining, y_train, y_remaining = sklearn.model_selection.train_test_split(df, y, train_size = 0.7, random_state = 42, stratify = y)
dict_val, dict_test, y_val, y_test = sklearn.model_selection.train_test_split(dict_remaining, y_remaining, train_size = 0.5, random_state = 42, stratify = y_remaining)

print(dict_train)
print(y_train)


k_values = [1, 3, 5, 9, 15]
train_accuracy_knn = []
val_accuracy_log = []
val_accuracy_knn = []
test_accuracy_best_knn = []


# k-NN Algorithm Training

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

dict_train_scaled = scaler.fit_transform(dict_train)
dict_val_scaled = scaler.transform(dict_val)
dict_test_scaled = scaler.transform(dict_test)


for k in k_values:
    knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(dict_train_scaled, y_train)

    # test how well it performs on training data

    y_train_pred_knn = knn_model.predict(dict_train_scaled)
    train_accuracy_score_knn = sklearn.metrics.accuracy_score(y_train, y_train_pred_knn)
    train_accuracy_knn.append(train_accuracy_score_knn)

    # test how well it performs on validation data

    y_val_pred_knn = knn_model.predict(dict_val_scaled)
    val_accuracy_score_knn = sklearn.metrics.accuracy_score(y_val, y_val_pred_knn)
    val_accuracy_knn.append(val_accuracy_score_knn)

print()
print("k-NN training data accuracy (for k = [1, 3, 5, 9, 15]):\n", train_accuracy_knn)
print()
print("k-NN validation data accuracy (for k = [1, 3, 5, 9, 15]):\n", val_accuracy_knn)
print()

best_k = 5

best_knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=9)
best_knn_model.fit(dict_train_scaled, y_train)
y_test_pred_knn = best_knn_model.predict(dict_test_scaled)
test_accuracy_score_knn = sklearn.metrics.accuracy_score(y_test, y_test_pred_knn)
test_accuracy_best_knn.append(test_accuracy_score_knn)

print("Test Accuracy Score:", test_accuracy_score_knn)








