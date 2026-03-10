# Crop-and-align

A Python script that **frontally aligns and crops faces** from selfie images.
It detects the largest face in each image, rotates and scales it so both eyes
land at canonical positions, and saves a square crop at a configurable size.

---

## Features

- Frontal alignment based on eye-landmark positions (corrects in-plane head tilt)
- Automatic scale normalisation (consistent inter-eye distance in every output)
- Batch mode: process an entire directory at once
- Configurable output crop size (default 256 × 256 px)
- Clear error messages when no face is detected

## Requirements

- Python 3.8+
- [dlib](http://dlib.net/) – face detection & 68-point facial landmark prediction
- OpenCV, NumPy, imutils

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Setup – download the landmark model

The alignment step requires dlib's 68-point shape predictor (~60 MB).
Download it once with the included helper script:

```bash
python download_models.py
```

This saves `shape_predictor_68_face_landmarks.dat` in the current directory.

## Usage

### Single image

```bash
python align_and_crop.py --input selfie.jpg --output aligned.jpg
```

### Batch (whole directory)

```bash
python align_and_crop.py --input ./photos --output ./aligned_photos
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | *(required)* | Input image or directory |
| `--output` / `-o` | *(required)* | Output image or directory |
| `--face-width` | `256` | Side length of the square output crop (px) |
| `--predictor` | `shape_predictor_68_face_landmarks.dat` | Path to the dlib `.dat` model |

## Running tests

```bash
pip install pytest
pytest tests/
```

Tests run without the model file (model-dependent tests are skipped automatically).
