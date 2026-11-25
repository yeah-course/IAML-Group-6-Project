import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Folder containing the original songs on Google Drive
input_folder = Path(
    "/Users/egonfarber/Library/CloudStorage/GoogleDrive-egon.farber4@gmail.com/My Drive/University Courses/Year 2/Audio Machine Learning/IAML (songs)"
)

# Folder where extracted clips will be saved
output_folder = Path("data/full_songs")
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

    print(f"Processing {file.name}...")

    # Load audio
    audio, sr = librosa.load(file, sr=22050, mono=True)

    # Calculate RMS (overall audio energy)
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

 # Save the entire audio recording
output_path = output_folder / f"{file.stem}.wav"

sf.write(output_path, audio, sr)

print(f"Saved full song: {output_path}")

print("Finished processing!")
