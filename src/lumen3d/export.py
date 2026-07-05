import numpy as np


def export_ply(path, points, colors, binary=True) -> None:
    """Write a point cloud to a .ply file.

    binary=True (default) writes `binary_little_endian`: each vertex is a fixed
    15-byte record (three float32 coords + three uint8 colors) instead of a text
    line. That's ~2x smaller than ASCII and *much* faster for a browser's
    PLYLoader to parse (raw byte copy, not char-by-char parseFloat) -- which is
    what the hosted demo needs. binary=False keeps the human-readable ASCII form
    for eyeballing/debugging.
    """
    if binary:
        _export_ply_binary(path, points, colors)
    else:
        _export_ply_ascii(path, points, colors)


def _export_ply_binary(path, points, colors) -> None:
    n = len(points)
    # The header is ALWAYS ascii text, even for a binary body. Note we open the
    # file in "wb", so the header must be encoded to bytes before writing.
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # One structured record per vertex. The field ORDER and TYPES must match the
    # property lines above exactly: x,y,z as little-endian float32 ("<f4"), then
    # r,g,b as uint8 ("u1"). Filling this array and dumping it with .tobytes()
    # writes all n records in one shot -- no Python loop over millions of points.
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    verts = np.empty(n, dtype=dtype)
    pts = np.asarray(points, dtype="<f4")
    cols = np.asarray(colors)
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    # Clip into 0..255 before the uint8 cast so stray/negative values wrap safely.
    cols = np.clip(cols, 0, 255).astype("u1")
    verts["red"], verts["green"], verts["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


def _export_ply_ascii(path, points, colors) -> None:
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        n = len(points)

        f.write(f"element vertex {n}\n")
        f.write(f"property float x\n")
        f.write(f"property float y\n")
        f.write(f"property float z\n")
        f.write(f"property uchar red\n")
        f.write(f"property uchar green\n")
        f.write(f"property uchar blue\n")
        f.write(f"end_header\n")
        for point, color in zip(points, colors):
            x, y, z = point        # three floats
            r, g, b = color        # three uint8 values
            # Fixed 4-decimal coords: kills float32 repr noise
            # (2.2999999... -> 2.3000) and shrinks the file. 4 decimals is
            # 0.1 mm at meter scale -- far finer than the 504x280 depth
            # ceiling, so no real detail is lost. Colors are ints: no noise.
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
