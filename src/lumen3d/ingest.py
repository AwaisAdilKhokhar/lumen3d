"""Ingest stage: turn raw input (an image folder or a video) into frames."""

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


    











def load_frames(input_path: str) -> list[Path]:
    """Takes in input path and routes to either find_images() or extract_video_frames()

    Args:
        input_path: Path to a directory that contains image files or video.

    Returns:
        A list of Path objects (one per image or frame), sorted by name so the frames
        are always in a consistent, predictable order.

    
    """

    path_object = Path(input_path)
    if path_object.is_dir():
        return find_images(input_path)

    elif path_object.is_file():
        if path_object.suffix.lower() in VIDEO_EXTENSIONS:
            raise NotImplementedError("function to take video not implemented yet")

        else:
            raise ValueError("Only video files or image folders are accepted")

    else:
        raise ValueError(f"Unsupported input: {input_path!r}. Please give  an image folder or a video file.")
    





    

    
    
