
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Folder containing the original songs on Google Drive
input_folder = Path(
    "/Users/egonfarber/Library/CloudStorage/GoogleDrive-egon.farber4@gmail.com/My Drive/University Courses/Year 2/Audio Machine Learning/IAML (songs)"
)

# Folder where extracted clips will be saved
output_folder = Path("data/extracted_clips")
output_folder.mkdir(parents=True, exist_ok=True)

# Settings
duration = 4
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
    audio, sr = librosa.load(file, sr=None, mono=True)

    # Skip songs shorter than 4 seconds
    clip_samples = int(duration * sr)

    if len(audio) < clip_samples:
        print("Song too short, skipping.")
        continue

    # Calculate RMS (overall audio energy)
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Find the 4-second window with the highest average RMS
    window_frames = int(duration * sr / hop_length)

    window_energy = np.convolve(
        rms,
        np.ones(window_frames) / window_frames,
        mode="valid"
    )

    best_window = np.argmax(window_energy)

    # Convert the selected window into sample positions
    start_sample = best_window * hop_length

    # Ensure the clip stays within the original audio
    start_sample = min(start_sample, len(audio) - clip_samples)
    end_sample = start_sample + clip_samples

    clip = audio[start_sample:end_sample]

    # Save the extracted clip
    output_path = output_folder / f"{file.stem}_clip.wav"
    sf.write(output_path, clip, sr)

    print(f"Saved: {output_path}")

print("Finished processing!")
