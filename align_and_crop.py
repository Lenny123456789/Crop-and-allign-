"""Frontally align and crop faces from selfie images.

Usage
-----
Single image::

    python align_and_crop.py --input selfie.jpg --output aligned.jpg

Batch (directory)::

    python align_and_crop.py --input ./photos --output ./aligned_photos

Optional arguments
------------------
--face-width  INT   Width (and height) of the output square crop in pixels (default: 256).
--predictor   PATH  Path to the dlib 68-point shape predictor .dat file
                    (default: shape_predictor_68_face_landmarks.dat in the
                    current directory).  Run ``download_models.py`` first if
                    the file is missing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import dlib
import numpy as np

# ---------------------------------------------------------------------------
# Supported image extensions.
# Note: WebP support requires OpenCV to be built with libwebp; it is
# excluded here to avoid silent read failures on standard installations.
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Default desired positions of the eyes in the output image (as fractions of
# the output width / height).  These values produce a natural-looking crop.
_DESIRED_LEFT_EYE = (0.35, 0.35)


def load_models(predictor_path: str) -> tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
    """Load the dlib face detector and shape predictor.

    Args:
        predictor_path: Path to ``shape_predictor_68_face_landmarks.dat``.

    Returns:
        A ``(detector, predictor)`` tuple.

    Raises:
        FileNotFoundError: If *predictor_path* does not exist.
    """
    if not os.path.isfile(predictor_path):
        raise FileNotFoundError(
            f"Shape predictor not found at '{predictor_path}'. "
            "Run 'python download_models.py' to download it."
        )
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor


def _eye_center(shape: dlib.full_object_detection, start: int, end: int) -> np.ndarray:
    """Return the mean (x, y) of landmark points [start, end)."""
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(start, end)], dtype=np.float64)
    return pts.mean(axis=0)


def align_and_crop_face(
    image: np.ndarray,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
    desired_face_width: int = 256,
    desired_face_height: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Detect the largest face in *image*, align it frontally, and crop it.

    Alignment is performed by rotating and scaling the image so that both eyes
    land at pre-defined positions in the output crop.  This corrects for in-plane
    head tilt and brings selfies taken at different scales to a canonical size.

    Args:
        image:               Input BGR image (NumPy array as returned by OpenCV).
        detector:            dlib HOG-based frontal face detector.
        predictor:           dlib 68-point facial landmark predictor.
        desired_face_width:  Width of the output crop in pixels.
        desired_face_height: Height of the output crop in pixels.
                             Defaults to *desired_face_width* (square output).

    Returns:
        The aligned, cropped face as a BGR NumPy array, or ``None`` if no face
        was detected in the image.
    """
    if desired_face_height is None:
        desired_face_height = desired_face_width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upsampling (second argument = 1) helps detect smaller faces.
    rects = detector(gray, 1)
    if not rects:
        return None

    # Work with the largest detected face.
    rect = max(rects, key=lambda r: r.width() * r.height())
    shape = predictor(gray, rect)

    # Landmarks 36-41 → left eye, 42-47 → right eye.
    left_eye_center = _eye_center(shape, 36, 42)
    right_eye_center = _eye_center(shape, 42, 48)

    # Angle between the eye centres (degrees, positive = counter-clockwise).
    d_y = right_eye_center[1] - left_eye_center[1]
    d_x = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(d_y, d_x))

    # Desired horizontal eye separation in the output image.
    desired_right_eye_x = 1.0 - _DESIRED_LEFT_EYE[0]
    desired_dist = (desired_right_eye_x - _DESIRED_LEFT_EYE[0]) * desired_face_width

    # Scale so that the inter-eye distance matches the desired distance.
    current_dist = np.hypot(d_x, d_y)
    scale = desired_dist / current_dist

    # Midpoint between the eyes.
    eyes_center = (
        (left_eye_center[0] + right_eye_center[0]) / 2.0,
        (left_eye_center[1] + right_eye_center[1]) / 2.0,
    )

    # Build the rotation + scale matrix around the eye midpoint …
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # … then translate so that the eyes end up at the desired positions in the crop.
    t_x = desired_face_width * 0.5
    t_y = desired_face_height * _DESIRED_LEFT_EYE[1]
    M[0, 2] += t_x - eyes_center[0]
    M[1, 2] += t_y - eyes_center[1]

    aligned = cv2.warpAffine(
        image,
        M,
        (desired_face_width, desired_face_height),
        flags=cv2.INTER_CUBIC,
    )
    return aligned


def process_image(
    input_path: str,
    output_path: str,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
    desired_face_width: int = 256,
) -> bool:
    """Read one image, align-and-crop, and write the result.

    Args:
        input_path:         Path to the source image file.
        output_path:        Path where the result will be saved.
        detector:           dlib face detector.
        predictor:          dlib shape predictor.
        desired_face_width: Side length of the square output crop.

    Returns:
        ``True`` on success, ``False`` if no face was detected or the image
        could not be read.
    """
    image = cv2.imread(input_path)
    if image is None:
        print(f"[WARNING] Could not read image: {input_path}", file=sys.stderr)
        return False

    result = align_and_crop_face(image, detector, predictor, desired_face_width)
    if result is None:
        print(f"[WARNING] No face detected in: {input_path}", file=sys.stderr)
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"[OK] Saved aligned face to: {output_path}")
    return True


def process_directory(
    input_dir: str,
    output_dir: str,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
    desired_face_width: int = 256,
) -> tuple[int, int]:
    """Process all images in *input_dir* and write results to *output_dir*.

    Args:
        input_dir:          Directory containing input images.
        output_dir:         Directory where aligned images will be saved.
        detector:           dlib face detector.
        predictor:          dlib shape predictor.
        desired_face_width: Side length of the square output crop.

    Returns:
        A ``(success_count, total_count)`` tuple.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    success = 0
    for img_file in image_files:
        out_file = output_path / img_file.name
        if process_image(str(img_file), str(out_file), detector, predictor, desired_face_width):
            success += 1

    total = len(image_files)
    print(f"\nProcessed {success}/{total} image(s) successfully.")
    return success, total


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Frontally align and crop faces from selfie images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to an input image file or a directory of images.",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to the output image file (single) or directory (batch).",
    )
    parser.add_argument(
        "--face-width", type=int, default=256,
        help="Width and height (pixels) of the square output crop.",
    )
    parser.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to the dlib 68-point shape predictor .dat file.",
    )

    args = parser.parse_args(argv)

    try:
        detector, predictor = load_models(args.predictor)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if os.path.isdir(args.input):
        success, total = process_directory(
            args.input, args.output, detector, predictor, args.face_width
        )
        return 0 if success > 0 else 1
    else:
        ok = process_image(args.input, args.output, detector, predictor, args.face_width)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
