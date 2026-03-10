"""Unit tests for align_and_crop.py.

Tests that do not require the dlib shape predictor model file are always run.
Tests marked with ``@pytest.mark.requires_model`` are skipped automatically
when the model file is absent (useful in CI environments where the 60 MB
model has not been downloaded yet).
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Make sure the repo root is importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import align_and_crop  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "shape_predictor_68_face_landmarks.dat")
MODEL_AVAILABLE = os.path.isfile(MODEL_PATH)

requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="shape_predictor_68_face_landmarks.dat not found – run download_models.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blank_image(width: int = 320, height: int = 240) -> np.ndarray:
    """Return a plain grey BGR image."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _make_mock_shape(left_eye_x: int, left_eye_y: int, right_eye_x: int, right_eye_y: int):
    """Build a minimal mock dlib shape with 68 parts.

    Eye landmarks are placed at the supplied coordinates; all other points are
    at (0, 0).
    """
    parts = []
    for i in range(68):
        pt = MagicMock()
        if 36 <= i < 42:
            pt.x, pt.y = left_eye_x, left_eye_y
        elif 42 <= i < 48:
            pt.x, pt.y = right_eye_x, right_eye_y
        else:
            pt.x, pt.y = 0, 0
        parts.append(pt)

    shape = MagicMock()
    shape.part.side_effect = lambda idx: parts[idx]
    return shape


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------

class TestLoadModels:
    def test_missing_predictor_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Shape predictor not found"):
            align_and_crop.load_models(str(tmp_path / "nonexistent.dat"))

    @requires_model
    def test_loads_successfully(self):
        detector, predictor = align_and_crop.load_models(MODEL_PATH)
        assert detector is not None
        assert predictor is not None


# ---------------------------------------------------------------------------
# _eye_center
# ---------------------------------------------------------------------------

class TestEyeCenter:
    def test_returns_mean_of_landmarks(self):
        shape = _make_mock_shape(100, 50, 200, 50)
        # Left-eye landmarks 36-41 are all at (100, 50)
        center = align_and_crop._eye_center(shape, 36, 42)
        np.testing.assert_array_almost_equal(center, [100.0, 50.0])

    def test_mixed_positions(self):
        shape = MagicMock()
        pts = [(0, 0), (10, 10), (20, 0)]
        shape.part.side_effect = lambda i: MagicMock(x=pts[i][0], y=pts[i][1])
        center = align_and_crop._eye_center(shape, 0, 3)
        np.testing.assert_array_almost_equal(center, [10.0, 10.0 / 3])


# ---------------------------------------------------------------------------
# align_and_crop_face (mocked detector/predictor)
# ---------------------------------------------------------------------------

class TestAlignAndCropFaceMocked:
    def _make_detector_with_rect(self, image: np.ndarray, x1=80, y1=60, x2=240, y2=180):
        """Return a mock detector that yields one rectangle."""
        rect = MagicMock()
        rect.width.return_value = x2 - x1
        rect.height.return_value = y2 - y1
        detector = MagicMock(return_value=[rect])
        return detector, rect

    def test_returns_none_when_no_face_detected(self):
        image = _make_blank_image()
        detector = MagicMock(return_value=[])
        predictor = MagicMock()
        result = align_and_crop.align_and_crop_face(image, detector, predictor)
        assert result is None

    def test_output_shape_default_size(self):
        image = _make_blank_image(320, 240)
        detector, rect = self._make_detector_with_rect(image)
        shape = _make_mock_shape(100, 100, 200, 100)
        predictor = MagicMock(return_value=shape)

        result = align_and_crop.align_and_crop_face(image, detector, predictor, desired_face_width=256)

        assert result is not None
        assert result.shape == (256, 256, 3)

    def test_output_shape_custom_size(self):
        image = _make_blank_image(640, 480)
        detector, rect = self._make_detector_with_rect(image)
        shape = _make_mock_shape(150, 150, 300, 150)
        predictor = MagicMock(return_value=shape)

        result = align_and_crop.align_and_crop_face(image, detector, predictor, desired_face_width=128)

        assert result is not None
        assert result.shape == (128, 128, 3)

    def test_custom_face_height(self):
        image = _make_blank_image(640, 480)
        detector, rect = self._make_detector_with_rect(image)
        shape = _make_mock_shape(150, 150, 300, 150)
        predictor = MagicMock(return_value=shape)

        result = align_and_crop.align_and_crop_face(
            image, detector, predictor, desired_face_width=128, desired_face_height=160
        )

        assert result is not None
        assert result.shape == (160, 128, 3)

    def test_largest_face_chosen(self):
        """When two faces are detected the larger one should be used."""
        image = _make_blank_image(640, 480)

        small_rect = MagicMock()
        small_rect.width.return_value = 50
        small_rect.height.return_value = 50

        large_rect = MagicMock()
        large_rect.width.return_value = 200
        large_rect.height.return_value = 200

        detector = MagicMock(return_value=[small_rect, large_rect])

        # Track which rect predictor is called with.
        called_with = []

        def mock_predictor(gray, rect):
            called_with.append(rect)
            return _make_mock_shape(100, 100, 200, 100)

        result = align_and_crop.align_and_crop_face(image, detector, mock_predictor, 256)
        assert called_with[0] is large_rect


# ---------------------------------------------------------------------------
# process_image
# ---------------------------------------------------------------------------

class TestProcessImage:
    def test_missing_input_file_returns_false(self, tmp_path):
        detector = MagicMock(return_value=[])
        predictor = MagicMock()
        ok = align_and_crop.process_image(
            str(tmp_path / "ghost.jpg"),
            str(tmp_path / "out.jpg"),
            detector,
            predictor,
        )
        assert not ok

    def test_no_face_returns_false(self, tmp_path):
        img_path = str(tmp_path / "blank.jpg")
        cv2.imwrite(img_path, _make_blank_image())

        detector = MagicMock(return_value=[])
        predictor = MagicMock()

        ok = align_and_crop.process_image(img_path, str(tmp_path / "out.jpg"), detector, predictor)
        assert not ok

    def test_success_saves_file(self, tmp_path):
        img_path = str(tmp_path / "face.jpg")
        cv2.imwrite(img_path, _make_blank_image(320, 240))

        rect = MagicMock()
        rect.width.return_value = 160
        rect.height.return_value = 160
        detector = MagicMock(return_value=[rect])
        shape = _make_mock_shape(100, 100, 200, 100)
        predictor = MagicMock(return_value=shape)

        out_path = str(tmp_path / "aligned.jpg")
        ok = align_and_crop.process_image(img_path, out_path, detector, predictor, desired_face_width=128)

        assert ok
        assert os.path.isfile(out_path)
        saved = cv2.imread(out_path)
        assert saved is not None
        assert saved.shape == (128, 128, 3)


# ---------------------------------------------------------------------------
# process_directory
# ---------------------------------------------------------------------------

class TestProcessDirectory:
    def test_processes_images_in_directory(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        for name in ("a.jpg", "b.png", "c.txt"):  # c.txt should be ignored
            path = input_dir / name
            if name.endswith(".txt"):
                path.write_text("not an image")
            else:
                cv2.imwrite(str(path), _make_blank_image())

        rect = MagicMock()
        rect.width.return_value = 160
        rect.height.return_value = 160
        detector = MagicMock(return_value=[rect])
        shape = _make_mock_shape(100, 100, 200, 100)
        predictor = MagicMock(return_value=shape)

        success, total = align_and_crop.process_directory(
            str(input_dir), str(output_dir), detector, predictor, desired_face_width=128
        )

        assert total == 2  # only .jpg and .png
        assert success == 2
        assert len(list(output_dir.iterdir())) == 2


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestMain:
    def test_missing_predictor_exits_1(self, tmp_path):
        img = tmp_path / "face.jpg"
        cv2.imwrite(str(img), _make_blank_image())

        exit_code = align_and_crop.main([
            "--input", str(img),
            "--output", str(tmp_path / "out.jpg"),
            "--predictor", str(tmp_path / "nonexistent.dat"),
        ])
        assert exit_code == 1

    def test_no_face_exits_1(self, tmp_path):
        img_path = str(tmp_path / "blank.jpg")
        cv2.imwrite(img_path, _make_blank_image())

        # Create a dummy predictor file so load_models doesn't raise.
        fake_predictor = str(tmp_path / "fake.dat")
        with patch("align_and_crop.load_models") as mock_load, \
             patch("align_and_crop.process_image", return_value=False):
            mock_load.return_value = (MagicMock(), MagicMock())
            exit_code = align_and_crop.main([
                "--input", img_path,
                "--output", str(tmp_path / "out.jpg"),
                "--predictor", fake_predictor,
            ])
        assert exit_code == 1

    def test_success_exits_0(self, tmp_path):
        img_path = str(tmp_path / "face.jpg")
        cv2.imwrite(img_path, _make_blank_image())

        with patch("align_and_crop.load_models") as mock_load, \
             patch("align_and_crop.process_image", return_value=True):
            mock_load.return_value = (MagicMock(), MagicMock())
            exit_code = align_and_crop.main([
                "--input", img_path,
                "--output", str(tmp_path / "out.jpg"),
                "--predictor", str(tmp_path / "fake.dat"),
            ])
        assert exit_code == 0
