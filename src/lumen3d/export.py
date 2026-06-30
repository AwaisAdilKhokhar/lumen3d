def export_ply(path, points, colors) -> None:
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
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")