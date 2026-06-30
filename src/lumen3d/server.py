from .cache import load_scene
from .embedding import SigLIPEmbedder
from fastapi import FastAPI
from pydantic import BaseModel
from .query import query_scene
from fastapi.staticfiles import StaticFiles


bundle = load_scene("scene.pkl")
embedder = SigLIPEmbedder()
app = FastAPI()



class Query(BaseModel):       # describes the JSON body: {"text": "..."}
    text: str


@app.get("/health")                   # "when someone GETs /health, run this"
def health():
    return {"status": "ok", "objects": len(bundle['embeddings'])}


@app.post("/query")
def query(q: Query):          # FastAPI parses the body into `q`
    user_text = q.text
    result=query_scene(user_text, bundle, embedder, top_k=1)
    result_dict={}
    result_dict['results']=[]
    for mask_id, score, points, colors in result:
        result_dict["results"].append(
            {"mask_id": int(mask_id), "score": float(score),
             "points": points.tolist(), "colors": colors.tolist()}
        )
    return result_dict

app.mount("/", StaticFiles(directory="viewer", html=True), name="viewer")