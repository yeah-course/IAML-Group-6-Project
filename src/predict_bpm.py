
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
        rms_std = float(np.std(librosa.feature.rms(y=signal)))
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfccs))
        mfcc_std = float(np.std(mfccs))


        features_list = [rms, 
                         spec_cen, 
                         spec_ban, 
                         spec_roll, 
                         zcr, ons_str_std, 
                         onset_rate, 
                         ons_str_mean, 
                         rms_std, 
                         mfcc_mean, 
                         mfcc_std]
        return features_list

keys = {"Jungle": 0, "House": 1, "Dubstep": 2}
print(keys)

dict = {"rms":[], 
        "spec_cent":[], 
        "spec_ban":[], 
        "spec_roll":[], 
        "zcr":[], 
        "ons_str_std":[], 
        "onset_rate":[], 
        "ons_str_mean": [], 
        "rms_std":[],
        "mfcc_mean":[],
        "mfcc_std":[]}
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
        dict["rms_std"].append(features[8])
        dict["mfcc_mean"].append(features[9])
        dict["mfcc_std"].append(features[10])


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
        dict["rms_std"].append(features[8])
        dict["mfcc_mean"].append(features[9])
        dict["mfcc_std"].append(features[10])

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
        dict["rms_std"].append(features[8])
        dict["mfcc_mean"].append(features[9])
        dict["mfcc_std"].append(features[10])

df = pd.DataFrame(dict)
y = np.array(keys)

#############

# Stratified 5-fold Cross-Validation

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(knn_model, df, y, cv=skf, scoring="accuracy")

print("Cross-validation accuracies:", scores)
print("Average accuracy:", scores.mean())
print("Standard deviation:", scores.std())


#############

# ANALYSE FEATURES

def scatter_features(x, y_feature, labels, x_name, y_name):

    fig, ax = plt.subplots()

    genres = {
        0: "Jungle",
        1: "House",
        2: "Dubstep"
    }

    for label, genre in genres.items():

        mask = (labels == label)

        ax.scatter(
            x[mask],
            y_feature[mask],
            label=genre
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{x_name} vs {y_name}")

    ax.legend(title="Genre")

    return fig

# Scatter Plot Function


fig = scatter_features(df["rms"].to_numpy(), df["mfcc_mean"].to_numpy(), y, "RMS", "AVERAGE MFCC")
plt.show()

fig = scatter_features(df["rms"].to_numpy(), df["mfcc_std"].to_numpy(), y, "RMS", "STDEV MFCC")
plt.show()

fig = scatter_features(df["spec_cent"].to_numpy(), df["mfcc_mean"].to_numpy(), y, "Spectral Centroid", "MEAN MFCC")
plt.show()

fig = scatter_features(df["spec_cent"].to_numpy(), df["mfcc_std"].to_numpy(), y, "Spectral Centroid", "STDEV MFCC")
plt.show()

###########
###########

# Important Features: Spectral Centroid vs Mean Onset Strength
# Import Features: RMS vs Spectral Centroid


# DATA SPLITTING

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix


dict_train, dict_remaining, y_train, y_remaining = sklearn.model_selection.train_test_split(df, y, train_size = 0.7, random_state=42, stratify = y)

dict_val, dict_test, y_val, y_test = sklearn.model_selection.train_test_split(dict_remaining, y_remaining, 
                                                                              train_size = 0.5, random_state=42, stratify = y_remaining)

print(dict_train)
print(y_train)

# ESTABLISH RELEVANT KNN LISTS

k_values = [1, 3, 5, 9, 15]
train_accuracy_knn = []
val_accuracy_log = []
val_accuracy_knn = []
test_accuracy_best_knn = []


# k-NN Algorithm Training

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Set up repeated stratified 5-fold cross-validation
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

# Test different values of k
k_values = [1, 3, 5, 9, 15]

knn_cv_results = []

for k in k_values:

    # Create a pipeline: scale data, then run k-NN
    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    # Run cross-validation
    scores = cross_val_score(
        knn_model,
        dict_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    # Store results
    knn_cv_results.append(scores)

    print(f"\nk = {k}")
    print("Average accuracy:", np.mean(scores))
    print("Standard deviation:", np.std(scores))

#Original Algorithm

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

# print()
# print("k-NN training data accuracy (for k = [1, 3, 5, 9, 15]):\n", train_accuracy_knn)
# print()
# print("k-NN validation data accuracy (for k = [1, 3, 5, 9, 15]):\n", val_accuracy_knn)
# print()


best_knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
best_knn_model.fit(dict_train_scaled, y_train)
y_test_pred_knn = best_knn_model.predict(dict_test_scaled)
test_accuracy_score_knn = sklearn.metrics.accuracy_score(y_test, y_test_pred_knn)
test_accuracy_best_knn.append(test_accuracy_score_knn)

print("Test Accuracy Score:", test_accuracy_score_knn)

# Run a Confusion Matrix

print()

cm = confusion_matrix(y_test, y_test_pred_knn)
print("Confusion Matrix for k-NN (n=5) on Test Data:\n", cm)
print()

# END OF K-NN CLASSIFIER

#######

#######

# RANDOM FOREST CLASSIFIER

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

rf_model = sklearn.ensemble.RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

# Run cross-validation
rf_scores = cross_val_score(
    rf_model,
    dict_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

# Display results
print("Random Forest Cross-Validation Results:")
print("Individual accuracies:", rf_scores)
print("Average accuracy:", np.mean(rf_scores))
print("Standard deviation:", np.std(rf_scores))


# NEURAL NETWORK

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np



# Same cross-validation setup
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

# Create neural network model

nn_model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(10,),
        solver="lbfgs",
        max_iter=5000,
        alpha=0.1,
        random_state=42
    )
)

# Run cross-validation
nn_scores = cross_val_score(
    nn_model,
    dict_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

# Display results
print("Neural Network Cross-Validation Results:")
print("Average accuracy:", np.mean(nn_scores))
print("Standard deviation:", np.std(nn_scores))





