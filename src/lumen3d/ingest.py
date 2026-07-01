"""Ingest stage: turn raw input (a single image, an image folder, or a video) into frames."""

from pathlib import Path
import cv2

# The image file types we accept as input frames.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

def find_images(folder: str) -> list[Path]:
    """Find all image files in a folder and return them as a sorted list.

    Args:
        folder: Path to a directory that contains image files.

    Returns:
        A list of Path objects (one per image), sorted by name so the frames
        are always in a consistent, predictable order.

    Example:
        >>> find_images("assets/examples/SOH")
        [PosixPath('assets/examples/SOH/0001.jpg'), ...]
    """

    

    folder_path = Path(folder)

    images = [
        p
        for p in folder_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)



def extract_video_frames(video_path: str, output_dir: str, stride: int = 10) -> list[Path]:
    """Takes in a video path, output directory and stride to selectively return frames from the video depending on stride.

    Args:
        video_path: Path to the video
        output_dir: Path to the folder you want to create that will contain frames of the video
        stride: stride determines how many frames we skip from the video and how many you keep. if stride in 10 every 10th frame will be saved to be used and the rest will be discarded


    Returns:
        A list of Path objects (one per image), sorted by name so the frames
        are always in a consistent, predictable order.

   
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)


    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path!r}")

    frame_index = 0
    
    saved=[]
    while True:
        ok, frame = cap.read()
        if not ok:
            break  
        frame_index += 1
        if frame_index % stride == 0:
            
            out_path = Path(output_dir) / f"frame_{frame_index:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)

    cap.release()
    return sorted(saved)


def downscale_frames(frames: list[Path], max_width: int = 1024, output_dir: str = "frames_small") -> list[Path]:
    """Resize frames so none is wider than `max_width`, preserving aspect ratio.

    Heavy models (SAM2 especially) run out of GPU memory on full-resolution
    frames — a 4K frame OOMs an 8GB card. Shrinking the longest side to ~1024px
    before the build is mandatory on modest hardware. Frames already at or below
    `max_width` are copied through unchanged.

    Args:
        frames: List of frame image paths (from `load_frames`).
        max_width: Maximum output width in pixels. Height scales to match.
        output_dir: Folder to write the resized frames into (created if needed).

    Returns:
        A sorted list of Path objects, one per resized frame in `output_dir`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = []
    for frame_path in frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Could not read frame: {frame_path!r}")

        height, width = image.shape[:2]        # cv2 is (H, W, C) — height first
        if width > max_width:
            scale = max_width / width
            # cv2.resize wants (width, height); INTER_AREA is best for shrinking.
            new_size = (max_width, round(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        out_path = out / frame_path.name
        cv2.imwrite(str(out_path), image)
        saved.append(out_path)

    return sorted(saved)














def load_frames(input_path: str, output_dir: str ='frames', stride: int = 10) -> list[Path]:
    """Route an input path to the right handler and return a list of frames.

    Three kinds of input are accepted:
        - a folder of images -> find_images()
        - a video file       -> extract_video_frames()
        - a single image file -> returned as a one-element list (no extraction)

    Args:
        input_path: Path to an image folder, a video file, or a single image.
        output_dir: Folder for extracted video frames (unused for images).
        stride: For a video, keep every Nth frame. Ignored for images.

    Returns:
        A list of Path objects (one per image or frame), sorted by name so the
        frames are always in a consistent, predictable order.
    """

    path_object = Path(input_path)
    if path_object.is_dir():
        return find_images(input_path)

    elif path_object.is_file():
        suffix = path_object.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return extract_video_frames(input_path, output_dir, stride)

        elif suffix in IMAGE_EXTENSIONS:
            # A single image is already a usable frame — no extraction or copy
            # needed. Everything downstream (DA3, SAM2, fusion, embedding) loops
            # over N frames and works fine for N == 1, so we just hand it back as
            # a one-element list. Note: one viewpoint means a monocular / 2.5D
            # reconstruction — depth for the surfaces facing the camera only.
            return [path_object]

        else:
            raise ValueError(
                f"Unsupported file type: {path_object.suffix!r}. Accepted input: "
                "a single image, a folder of images, or a video file."
            )

    else:
        raise ValueError(f"Unsupported input: {input_path!r}. Please give  an image folder or a video file.")
    





    

    
    
