
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import os

# Folder containing the original songs on Google Drive
input_folder = Path("data/raw/Dubstep")


# Folder where extracted clips will be saved

output_folder = Path("data/extracted_clips/Dubstep")
output_folder.mkdir(parents=True, exist_ok=True)

# Settings
hop_length = 1024
frame_length = 4096

# Find all WAV and MP3 files
audio_files = [
    file for file in input_folder.iterdir()
    if file.suffix.lower() in [".wav", ".mp3"]
]

print(f"Found {len(audio_files)} audio files.")

for file in audio_files:

    # Load audio
    audio, sr = librosa.load(file, sr=22050, mono=True)

    # Calculate RMS for each frame
    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )[0]

    # Convert 4 seconds into number of RMS frames
    window_frames = int(4 * sr / 512)

    # Calculate rolling average RMS
    rolling_rms = np.convolve(
        rms,
        np.ones(window_frames) / window_frames,
        mode="valid"
    )

    # Find highest average RMS
    best_index = np.argmax(rolling_rms)
   

    # Convert frame index to audio sample index
    start_sample = best_index * 512
    end_sample = start_sample + int(4 * sr)

    # Extract 4-second clip
    clip = audio[start_sample:end_sample]

    filename = os.path.splitext(os.path.basename(file))[0]

    sf.write(f'data/extracted_clips/Dubstep/{filename}.wav', clip, sr)

    ############
    ############


    
input_folder = Path("data/raw/Jungle")


# Folder where extracted clips will be saved

output_folder = Path("data/extracted_clips/Jungle")
output_folder.mkdir(parents=True, exist_ok=True)

# Settings
hop_length = 1024
frame_length = 4096

# Find all WAV and MP3 files
audio_files = [
    file for file in input_folder.iterdir()
    if file.suffix.lower() in [".wav", ".mp3"]
]

print(f"Found {len(audio_files)} audio files.")

for file in audio_files:

    # Load audio
    audio, sr = librosa.load(file, sr=22050, mono=True)

    # Calculate RMS for each frame
    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )[0]

    # Convert 4 seconds into number of RMS frames
    window_frames = int(4 * sr / 512)

    # Calculate rolling average RMS
    rolling_rms = np.convolve(
        rms,
        np.ones(window_frames) / window_frames,
        mode="valid"
    )

    # Find highest average RMS
    best_index = np.argmax(rolling_rms)
   

    # Convert frame index to audio sample index
    start_sample = best_index * 512
    end_sample = start_sample + int(4 * sr)

    # Extract 4-second clip
    clip = audio[start_sample:end_sample]

    filename = os.path.splitext(os.path.basename(file))[0]

    sf.write(f'data/extracted_clips/Jungle/{filename}.wav', clip, sr)


########
#######


input_folder = Path("data/raw/House")


# Folder where extracted clips will be saved

output_folder = Path("data/extracted_clips/House")
output_folder.mkdir(parents=True, exist_ok=True)

# Settings
hop_length = 1024
frame_length = 4096

# Find all WAV and MP3 files
audio_files = [
    file for file in input_folder.iterdir()
    if file.suffix.lower() in [".wav", ".mp3"]
]

print(f"Found {len(audio_files)} audio files.")

for file in audio_files:

    # Load audio
    audio, sr = librosa.load(file, sr=22050, mono=True)

    # Calculate RMS for each frame
    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )[0]

    # Convert 4 seconds into number of RMS frames
    window_frames = int(4 * sr / 512)

    # Calculate rolling average RMS
    rolling_rms = np.convolve(
        rms,
        np.ones(window_frames) / window_frames,
        mode="valid"
    )

    # Find highest average RMS
    best_index = np.argmax(rolling_rms)
   

    # Convert frame index to audio sample index
    start_sample = best_index * 512
    end_sample = start_sample + int(4 * sr)

    # Extract 4-second clip
    clip = audio[start_sample:end_sample]

    filename = os.path.splitext(os.path.basename(file))[0]

    sf.write(f'data/extracted_clips/House/{filename}.wav', clip, sr)


print(f"Saved 4 second clip: {output_folder}")

print("Finished processing!")

print("yay")
