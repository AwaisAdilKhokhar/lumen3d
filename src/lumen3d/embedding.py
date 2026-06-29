from abc import ABC, abstractmethod
from .geometry import resize_mask
from .segmentation import ObjectMask
import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed_regions(self, images, masks) -> dict[int, np.ndarray]:
        ...

class SigLIPEmbedder(Embedder):

    def __init__(self, weights="google/siglip-base-patch16-224", device=None):
        self.weights = weights   
        self.device = device     
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is None or self._processor is None:       # not loaded yet?
            from transformers import AutoProcessor, AutoModel
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            self._processor = AutoProcessor.from_pretrained(self.weights)   # the adapter
            self._model     = AutoModel.from_pretrained(self.weights).to(self.device)
        return self._processor,self._model

    def _embed_image(self, image: np.ndarray) -> np.ndarray:
        import torch
        processor, model = self._load_model()
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features[0].cpu().numpy()

    def embed_regions(self, images: np.ndarray, masks: list[list[ObjectMask]]) -> dict[int, np.ndarray]:
        
        buckets = {}
        for i, frame_masks in enumerate(masks):
            frame = images[i]
            for obj in frame_masks:
                mask = resize_mask(obj.mask, frame.shape[:2])
                if not mask.any():
                    continue
                rows= np.any(mask,axis=1)
                cols= np.any(mask,axis=0)
                ys = np.where(rows)[0]   # the indices of rows that are True, e.g. [12, 13, ..., 40]
                xs = np.where(cols)[0]
                ymin, ymax = ys[0], ys[-1]   # first & last row containing the object
                xmin, xmax = xs[0], xs[-1]
                
                crop = frame[ymin:ymax + 1, xmin:xmax + 1]
                vec = self._embed_image(crop)
                if obj.mask_id not in buckets:
                    buckets[obj.mask_id] = []
                buckets[obj.mask_id].append(vec)
        result = {}
        for mask_id, vlist in buckets.items():
            result[mask_id] = np.mean(vlist, axis=0).astype(np.float32)
        return result