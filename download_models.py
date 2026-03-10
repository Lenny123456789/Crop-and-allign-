"""Download the dlib facial landmark predictor model required by align_and_crop.py."""

import bz2
import os
import urllib.request

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"


def download_shape_predictor(dest_path: str = MODEL_PATH) -> None:
    """Download and decompress the dlib 68-point shape predictor if not present.

    Args:
        dest_path: Where to save the decompressed ``.dat`` file.
    """
    if os.path.exists(dest_path):
        print(f"Model already exists at '{dest_path}'. Skipping download.")
        return

    bz2_path = dest_path + ".bz2"
    print(f"Downloading shape predictor from {MODEL_URL} …")
    urllib.request.urlretrieve(MODEL_URL, bz2_path)

    print("Decompressing …")
    with bz2.open(bz2_path, "rb") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())
    os.remove(bz2_path)
    print(f"Model saved to '{dest_path}'.")


if __name__ == "__main__":
    download_shape_predictor()
