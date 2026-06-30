import pickle


def save_scene(path, embeddings, geometry, scene=None) -> None:
    bundle = {
    "embeddings": embeddings,   # {mask_id: (768,) float32}   — meaning
    "geometry":   geometry,     # {mask_id: (points, colors)} — per-object 3D
    "scene":      scene,        # (points, colors) full cloud, or None — backdrop
}
    with open(path, "wb") as file:
        pickle.dump(bundle, file)

    

def load_scene(path) -> dict:
    with open(path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data
