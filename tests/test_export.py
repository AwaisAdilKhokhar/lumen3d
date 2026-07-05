"""Tests for export_ply (write a point cloud to a .ply file).

The binary path is the one that ships (the hosted demo serves it), so the core
test WRITES a tiny cloud, then READS the bytes back and checks every value
survived: header format, coord round-trip, color round-trip, and the >255 clip.
No GPU, no models -- a hand-built 3-point cloud whose values we already know.
"""

import numpy as np

from lumen3d.export import export_ply

# The on-disk record layout the binary writer promises: little-endian float32
# coords then uint8 colors, in property order. Reading with this dtype is the
# independent check that the writer packed exactly what the header declares.
_VERTEX = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def _read_ply(path):
    """Split a .ply into (header_text, vertex_records) by the end_header marker."""
    raw = path.read_bytes()
    marker = b"end_header\n"
    split = raw.index(marker) + len(marker)
    header = raw[:split].decode("ascii")
    body = np.frombuffer(raw[split:], dtype=_VERTEX)
    return header, body


def _sample_cloud():
    points = np.array([
        [1.5, -2.0, 3.25],
        [0.0, 0.1, -0.1],
        [-4.75, 8.0, 0.0],
    ], dtype=np.float32)
    colors = np.array([
        [255, 0, 128],
        [10, 20, 30],
        [0, 255, 64],
    ], dtype=np.uint8)
    return points, colors


def test_binary_round_trip_preserves_every_value(tmp_path):
    # Arrange
    points, colors = _sample_cloud()
    path = tmp_path / "scene.ply"

    # Act
    export_ply(path, points, colors, binary=True)
    header, body = _read_ply(path)

    # Assert -- header declares binary, right count, properties in order
    assert "format binary_little_endian 1.0" in header
    assert "element vertex 3" in header
    assert header.index("property float x") < header.index("property uchar red")

    # ...and every coordinate and color survived the round trip exactly.
    assert len(body) == 3
    assert np.array_equal(np.c_[body["x"], body["y"], body["z"]], points)
    assert np.array_equal(
        np.c_[body["red"], body["green"], body["blue"]], colors
    )


def test_binary_clips_out_of_range_colors(tmp_path):
    # A stray >255 (or negative) color must not wrap around under the uint8 cast;
    # the writer clips into 0..255 first.
    points = np.zeros((2, 3), dtype=np.float32)
    colors = np.array([[300, -5, 128], [255, 0, 256]], dtype=np.int32)
    path = tmp_path / "scene.ply"

    export_ply(path, points, colors, binary=True)
    _, body = _read_ply(path)

    assert np.array_equal(
        np.c_[body["red"], body["green"], body["blue"]],
        np.array([[255, 0, 128], [255, 0, 255]], dtype=np.uint8),
    )


def test_ascii_still_writes_readable_text(tmp_path):
    # binary=False keeps the human-readable form: an ascii header and one
    # "x y z r g b" line per point.
    points, colors = _sample_cloud()
    path = tmp_path / "scene.ply"

    export_ply(path, points, colors, binary=False)
    text = path.read_text()

    assert "format ascii 1.0" in text
    assert "element vertex 3" in text
    # header (10 lines incl. end_header) + 3 vertex lines, trailing newline
    body_lines = text.split("end_header\n")[1].strip().splitlines()
    assert len(body_lines) == 3
    assert body_lines[0].split() == ["1.5000", "-2.0000", "3.2500", "255", "0", "128"]
