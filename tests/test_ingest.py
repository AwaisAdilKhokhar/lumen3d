"""Tests for the ingest stage (no GPU needed — just cv2)."""

import cv2
import numpy as np

from lumen3d.ingest import downscale_frames


def _write_image(path, width, height):
    """Write a solid gray image of the given size to `path`."""
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_downscale_shrinks_wide_frames_preserving_aspect(tmp_path):
    # Arrange: a 2000x1000 frame (wider than max_width).
    src = tmp_path / "src"
    src.mkdir()
    _write_image(src / "frame_00001.jpg", width=2000, height=1000)
    out = tmp_path / "small"

    # Act
    result = downscale_frames([src / "frame_00001.jpg"], max_width=1000, output_dir=out)

    # Assert: width capped at 1000, height scaled by the same factor (2:1 kept).
    assert len(result) == 1
    image = cv2.imread(str(result[0]))
    height, width = image.shape[:2]
    assert width == 1000
    assert height == 500


def test_downscale_leaves_small_frames_untouched(tmp_path):
    # Arrange: an 800-wide frame, already under the 1024 cap.
    src = tmp_path / "src"
    src.mkdir()
    _write_image(src / "frame_00001.jpg", width=800, height=600)
    out = tmp_path / "small"

    # Act
    result = downscale_frames([src / "frame_00001.jpg"], max_width=1024, output_dir=out)

    # Assert: size unchanged, but still written into output_dir.
    image = cv2.imread(str(result[0]))
    height, width = image.shape[:2]
    assert (width, height) == (800, 600)
    assert result[0].parent == out


def test_downscale_returns_sorted_paths(tmp_path):
    # Arrange: two frames handed in out of order.
    src = tmp_path / "src"
    src.mkdir()
    _write_image(src / "frame_00002.jpg", width=1200, height=600)
    _write_image(src / "frame_00001.jpg", width=1200, height=600)
    out = tmp_path / "small"

    # Act
    result = downscale_frames(
        [src / "frame_00002.jpg", src / "frame_00001.jpg"],
        max_width=1000,
        output_dir=out,
    )

    # Assert: output is name-sorted, not input order.
    assert [p.name for p in result] == ["frame_00001.jpg", "frame_00002.jpg"]
