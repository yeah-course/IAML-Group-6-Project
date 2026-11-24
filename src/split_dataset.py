import numpy as np
import pandas as pd
import sklearn.model_selection
import matplotlib.pyplot as plt
import librosa

# Right now, our dataset is a pandas DataFrame with a single column called "clips" that contains the paths to the audio clips.
# We can now split this data into training, validation, and test sets using sklearn's train_test_split function.

# Set the random seed, for reproducibility

np.random.seed(42)

#seperate 70% for training

dataset_train, dataset_remaining = sklearn.model_selection.train_test_split(dataset, train_size = 0.7, random_state=42)

#divide the remaining 30% equally (15% each)

dataset_val, dataset_test = sklearn.model_selection.train_test_split(dataset_remaining, test_size = 0.5, random_state=42)

print("Training:", len(dataset_train))
print("Validation:", len(dataset_val))
print("Testing:", len(dataset_test))
print("Total:", len(dataset))

# Save the three datasets as CSV files

dataset_train.to_csv("data/splits/train.csv", index=False)
dataset_val.to_csv("data/splits/validation.csv", index=False)
dataset_test.to_csv("data/splits/test.csv", index=False)

print("Datasets successfully saved!")

# We are now going to extract the audio data from the clips using librosa.

import librosa

data = {"training dataset": Path("data/splits/train.csv"), "validation dataset": Path("data/splits/validation.csv"), "testing dataset": Path("data/splits/test.csv")}

loaded_data = {}

for data_type, file_path in data.items():
    print(f"Extracting audio data for {data_type}...")

    # Load the dataset CSV file
    dataset = pd.read_csv(file_path)

    # Create a new column to store the audio data
    dataset["audio_data"] = None

    for index, row in dataset.iterrows():
        clip_path = row["clips"]
        audio, sr = librosa.load(clip_path, sr=None, mono=True)
        dataset.at[index, "audio_data"] = audio

    # Save the updated dataset with audio data
    output_file_path = file_path.with_name(f"{data_type.replace(' ', '_')}_with_audio.csv")
    dataset.to_csv(output_file_path, index=False)
    print(f"Audio data extracted and saved to {output_file_path}.")

    loaded_data[data_type] = dataset

    for data_type, dataset in loaded_data.items():
        print(f"{data_type}: {dataset.shape}")

# Our usable data is in loaded_data, which is a dictionary containing the training, validation, and testing datasets with the audio data included. 
# Each dataset is a pandas DataFrame with two columns: "clips" (the paths to the audio clips) and "audio_data" (the extracted audio data as numpy arrays).

# end of data split.

# WHAT FEATURES ARE WE GOING TO EXTRACT?
# 1) Onset Envelope Strength
# 2) Onset Intervals
# 3) RMS Energy
# 4) Low-frequency Energy
# 5) Spectral Flux
# 6) Autocorrelation

# Spectral Centroid

centroid_values = []

for file in dataset["audio_data"]:
    centroid = librosa.feature.spectral_centroid(y=file, sr=sr)
    average_centroid = np.mean(centroid)
    centroid_values.append(average_centroid)

dataset["Mean Spectral Centroid"] = centroid_values

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(dataset["Mean Spectral Centroid"], dataset["bpm"])
plt.xlabel("Mean Spectral Centroid")
plt.ylabel("BPM")
plt.title("Relationship between Spectral Centroid and BPM")
plt.show()