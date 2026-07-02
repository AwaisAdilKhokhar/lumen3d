"""Tests for the unprojection geometry (pixel -> camera -> world)."""

import numpy as np

from lumen3d.geometry import (
    pixel_to_camera_point,
    camera_point_to_world,
    unproject_frame,
    unproject_frame_vectorized,
    unproject,
    resize_mask,
)


# --- pixel_to_camera_point -------------------------------------------------

def test_dead_center_pixel_points_straight_ahead():
    # Arrange: a 100x100 camera, center at (50, 50), zoom 100.
    # Act: unproject the dead-center pixel at depth 2.
    point = pixel_to_camera_point(u=50, v=50, depth=2, fx=100, fy=100, cx=50, cy=50)
    # Assert: no left/right or up/down lean -> straight ahead, 2 units out.
    assert point == (0.0, 0.0, 2)


def test_pixel_right_of_center_leans_right():
    # A pixel to the RIGHT of center (u=70 > cx=50) should get a positive X.
    point = pixel_to_camera_point(u=70, v=50, depth=2, fx=100, fy=100, cx=50, cy=50)
    assert point == (0.4, 0.0, 2)


# --- camera_point_to_world -------------------------------------------------

def test_identity_note_leaves_point_unchanged():
    # The "do-nothing" note (camera at origin, facing forward): world == camera.
    cam = (0.8, 0.0, 4.0)
    world = camera_point_to_world(cam, np.eye(4))
    assert np.allclose(world, [0.8, 0.0, 4.0])


def test_shift_note_translates_point():
    # A note that says "camera stood 10 to the right" adds 10 to X.
    cam = (0.8, 0.0, 4.0)
    shift = np.array([
        [1, 0, 0, 10],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    world = camera_point_to_world(cam, shift)
    assert np.allclose(world, [10.8, 0.0, 4.0])


# --- unproject_frame -------------------------------------------------------

def test_unproject_frame_skips_invalid_depth():
    # Arrange: a 2x2 frame. The bottom-right pixel has depth -1 (invalid)
    # and must be dropped. fx=fy=1, cx=cy=0 makes the math round numbers:
    # a valid pixel (u, v) at depth d -> (u*d, v*d, d).
    depth = np.array([[2, 2],
                      [2, -1]], dtype=float)
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    c2w = np.eye(4)
    image = np.array([[[10, 10, 10], [20, 20, 20]],
                      [[30, 30, 30], [40, 40, 40]]], dtype=np.uint8)
    conf = np.array([[1.0, 1.0],
                     [1.0, 1.0]])

    # Act
    pts, cols = unproject_frame(depth, K, c2w, image, conf, conf_thr=0.5)

    # Assert: exactly 3 points survive (the -1 pixel is gone), in scan order.
    assert len(pts) == 3
    assert np.allclose(pts, [[0, 0, 2],
                             [2, 0, 2],
                             [0, 2, 2]])
    # ...and colors stay row-aligned with the surviving points.
    assert np.array_equal(cols, [[10, 10, 10],
                                 [20, 20, 20],
                                 [30, 30, 30]])


def test_unproject_frame_drops_low_confidence():
    # Same frame, all depths valid, but one pixel's confidence is below the bar.
    depth = np.array([[2, 2],
                      [2, 2]], dtype=float)
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    c2w = np.eye(4)
    image = np.array([[[10, 10, 10], [20, 20, 20]],
                      [[30, 30, 30], [40, 40, 40]]], dtype=np.uint8)
    conf = np.array([[1.0, 0.1],    # top-right pixel is low-confidence
                     [1.0, 1.0]])

    pts, cols = unproject_frame(depth, K, c2w, image, conf, conf_thr=0.5)

    # The low-confidence pixel (u=1, v=0) is dropped -> 3 points survive.
    assert len(pts) == 3

def test_unproject_frame_mask():
    # Same frame, all depths valid, we test if the mask works
    depth = np.array([[2, 2],
                      [2, 2]], dtype=float)
    mask = np.array([[False, False],
                      [True, False]], dtype=bool)
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    c2w = np.eye(4)
    image = np.array([[[10, 10, 10], [20, 20, 20]],
                      [[30, 30, 30], [40, 40, 40]]], dtype=np.uint8)
    conf = np.array([[1.0, 1.0],    
                     [1.0, 1.0]])

    pts, cols = unproject_frame(depth, K, c2w, image, conf, conf_thr=0.5,mask=mask)

   
    assert len(pts) == 1
    assert np.allclose(pts, [[0, 2, 2]])
    assert np.array_equal(cols, [[30, 30, 30]])


def test_unproject_frame_mask_resize():
    
    
    mask = np.array([[False, False],
                      [True, False]], dtype=bool)
    

    result = resize_mask(mask, (4, 4))

    

   
    assert result.shape == (4, 4)
    assert result.dtype == bool
    assert result[3, 0]==True
    assert result[0, 0]==False
    assert result[0, 3]==False
    assert result[3, 3]==False


# --- unproject (all frames) ------------------------------------------------

def test_unproject_loops_frames_and_flips_the_note():
    # Two single-pixel frames. fx=fy=1, cx=cy=0 -> a pixel (0,0) at depth d
    # gives camera point (0, 0, d).
    depth = np.array([[[2.0]], [[3.0]]])                       # frame0 d=2, frame1 d=3
    K = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * 2, dtype=float)

    ext0 = np.eye(4)                    # frame0: camera at the origin
    ext1 = np.eye(4); ext1[0, 3] = -10  # frame1: w2c shifting x by -10 (camera stood at x=10)
    extrinsics = np.array([ext0, ext1])

    image = np.array([[[[10, 10, 10]]], [[[20, 20, 20]]]], dtype=np.uint8)
    conf = np.array([[[1.0]], [[1.0]]])

    pts, cols = unproject(depth, K, extrinsics, image, conf, conf_thr=0.5)

    # frame0 -> (0,0,2) unchanged; frame1 -> camera (0,0,3) shifted +10 in x -> (10,0,3)
    assert np.allclose(pts, [[0, 0, 2],
                             [10, 0, 3]])
    assert np.array_equal(cols, [[10, 10, 10],
                                 [20, 20, 20]])
    # the contract requires float32 points / uint8 colors
    assert pts.dtype == np.float32
    assert cols.dtype == np.uint8


# --- unproject_frame_vectorized (must equal the loop version) --------------

def test_vectorized_unproject_matches_the_loop():
    # A frame with a mix of good, invalid-depth, and low-confidence pixels and a
    # non-trivial pose. The fast NumPy path must return the exact same points and
    # colors, in the same order, as the hand-written loop.
    rng = np.random.default_rng(0)
    depth = np.array([[2.0, 3.0, -1.0],
                      [1.5, 4.0,  2.5],
                      [0.0, 5.0,  1.0]])
    K = np.array([[1.2, 0.0, 1.0],
                  [0.0, 1.1, 0.5],
                  [0.0, 0.0, 1.0]])
    c2w = np.array([[1, 0, 0, 2.0],
                    [0, 1, 0, -1.0],
                    [0, 0, 1, 0.5],
                    [0, 0, 0, 1.0]], dtype=float)
    image = rng.integers(0, 256, size=(3, 3, 3), dtype=np.uint8)
    conf = np.array([[1.0, 0.2, 1.0],
                     [1.0, 1.0, 0.9],
                     [1.0, 0.1, 1.0]])

    loop_pts, loop_cols = unproject_frame(depth, K, c2w, image, conf, conf_thr=0.5)
    vec_pts, vec_cols = unproject_frame_vectorized(depth, K, c2w, image, conf, conf_thr=0.5)

    assert np.allclose(np.array(loop_pts, dtype=np.float32), vec_pts)
    assert np.array_equal(np.array(loop_cols, dtype=np.uint8), vec_cols)


def test_vectorized_unproject_honors_the_mask():
    depth = np.full((2, 2), 2.0)
    mask = np.array([[False, False],
                     [True,  False]], dtype=bool)
    K = np.eye(3)
    c2w = np.eye(4)
    image = np.array([[[10, 10, 10], [20, 20, 20]],
                      [[30, 30, 30], [40, 40, 40]]], dtype=np.uint8)
    conf = np.ones((2, 2))

    pts, cols = unproject_frame_vectorized(depth, K, c2w, image, conf, conf_thr=0.5, mask=mask)

    assert pts.shape == (1, 3)
    assert np.allclose(pts, [[0, 2, 2]])
    assert np.array_equal(cols, [[30, 30, 30]])
