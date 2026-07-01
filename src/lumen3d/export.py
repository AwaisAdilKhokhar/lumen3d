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
            # Fixed 4-decimal coords: kills float32 repr noise
            # (2.2999999... -> 2.3000) and shrinks the file. 4 decimals is
            # 0.1 mm at meter scale -- far finer than the 504x280 depth
            # ceiling, so no real detail is lost. Colors are ints: no noise.
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")