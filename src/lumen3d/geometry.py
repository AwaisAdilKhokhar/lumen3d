import numpy as np
import cv2




def pixel_to_camera_point(u, v, depth, fx, fy, cx, cy):
    dir_x = (u-cx)/fx
    dir_y = (v-cy)/fy
    dir_z=1
    X = dir_x * depth
    Y = dir_y * depth
    Z = depth
    return (X,Y,Z)


def camera_point_to_world(cam_point, c2w):
    # cam_point: a 3-element [X, Y, Z] (the output of pixel_to_camera_point)
    # c2w:       a 4x4 numpy array — the "camera -> world" note
    # 1) glue a 1 onto cam_point to make [X, Y, Z, 1]
    # 2) multiply by c2w using @
    # 3) return the first 3 numbers as the world point
    point=np.array([cam_point[0],cam_point[1],cam_point[2],1.0])
    result=c2w @ point
    xyz=result[:3]
    return xyz

def resize_mask(mask, shape):
    # mask: (Hm, Wm) bool array (from SAM2)
    # shape: the target (H, W) — i.e. depth.shape
    # -> returns an (H, W) bool array, nearest-neighbor resized
    H,W=shape
    uint8_mask = mask.astype(np.uint8)
    resized_mask = cv2.resize(uint8_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = resized_mask.astype(bool)
    return mask

def unproject_frame(depth, K, c2w, image, conf, conf_thr,mask=None):
    # depth: (H, W), K: (3,3), c2w: (4,4), image: (H, W, 3) uint8, conf: (H, W)
    # 1) pull fx, fy, cx, cy out of K
    # 2) read H, W from depth.shape
    # 3) loop every pixel (v, then u):
    #      - read d = depth[v, u]; skip if not finite / not > 0
    #      - skip if conf[v, u] < conf_thr
    #      - cam   = pixel_to_camera_point(u, v, d, fx, fy, cx, cy)
    #      - world = camera_point_to_world(cam, c2w)
    #      - collect world into a points list, and image[v, u] into a colors list
    # 4) return points, colors  (two lists)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    H,W = depth.shape
    points=[]
    colors=[]

    for v in range(H):
        for u in range(W):
            d = depth[v, u]
            if not (np.isfinite(d) and d > 0):   # bad depth? skip this pixel entirely
                continue
            if conf[v, u] < conf_thr:            # low confidence? skip
                continue
            if mask is not None and not mask[v, u]:
                continue

            # --- only valid pixels reach here ---
            cam   = pixel_to_camera_point(u, v, d, fx, fy, cx, cy)
            world = camera_point_to_world(cam, c2w)
            points.append(world)
            colors.append(image[v, u])
    return points, colors

def to_homogeneous(ext):
    # ext: (3,4) or (4,4) -> always return (4,4)
    # if already 4x4, return as-is
    # otherwise: start from a 4x4 identity (so the bottom row is [0,0,0,1]),
    #            drop the 3x4 block into the top, return it

    if ext.shape == (4,4):
        return ext
        
    else:
        H = np.eye(4, dtype=ext.dtype)   # bottom row is already [0,0,0,1]
        H[:3, :4] = ext                  # overwrite the top 3 rows with [R | t]
        return H




def unproject(depth, K, extrinsics, image, conf, conf_thr):
    # depth (N,H,W), K (N,3,3), extrinsics (N,4,4) w2c,
    # image (N,H,W,3) uint8, conf (N,H,W), conf_thr float
    # 1) N = number of frames
    # 2) for each frame i:
    #      - c2w = flip extrinsics[i]
    #      - pts, cols = unproject_frame(depth[i], K[i], c2w, image[i], conf[i], conf_thr)
    #      - extend a running points list and colors list
    # 3) convert both to np arrays: points -> float32, colors -> uint8
    # 4) return points, colors

    N= depth.shape[0]
    world_points=[]
    world_colors=[]
    for i in range(N):
        c2w = np.linalg.inv(to_homogeneous(extrinsics[i])) 
        pts, cols = unproject_frame(depth[i], K[i], c2w, image[i], conf[i], conf_thr)
        world_points.extend(pts)
        world_colors.extend(cols)
    points=np.array(world_points, dtype=np.float32)
    colors=np.array(world_colors, dtype=np.uint8)
    return points,colors
