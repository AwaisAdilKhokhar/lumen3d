import numpy as np 
from .backbone import Reconstruction 
from .segmentation import ObjectMask 
from .geometry import resize_mask, to_homogeneous, unproject_frame 

def fuse_masks_to_3d(
    recon: Reconstruction,
    masks: list[list[ObjectMask]],
    conf_thr: float = 0.0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ''' we combine SAM2 and DA3 output and basically we return a dict whose keys are the mask ids and values are the point clouds of these objects'''

    buckets={}

    for i, frame_masks in enumerate(masks):
        c2w = np.linalg.inv(to_homogeneous(recon.extrinsics[i]))
        for obj in frame_masks:
            mask=resize_mask(obj.mask,recon.depth[i].shape)
            points, colors=unproject_frame(recon.depth[i], recon.intrinsics[i], c2w, recon.images[i], recon.conf[i], conf_thr, mask=mask)

            if obj.mask_id not in buckets:
                buckets[obj.mask_id] = ([], [])
            buckets[obj.mask_id][0].extend(points)  # pile points into list 0
            buckets[obj.mask_id][1].extend(colors)  # pile colors into list 1
            
    result = {}
    for mask_id, (plist, clist) in buckets.items():
        result[mask_id] = (
            np.array(plist, dtype=np.float32),
            np.array(clist, dtype=np.uint8),
        )
    return result
            



