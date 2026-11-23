print("Python is working!")

from pathlib import Path

audio_folder = Path(
    "/Users/egonfarber/Library/CloudStorage/GoogleDrive-egon.farber4@gmail.com/My Drive/University Courses/Year 2/Audio Machine Learning/IAML (songs)"
)

print("Folder exists:", audio_folder.exists())
print("Folder contents:", list(audio_folder.iterdir()))
