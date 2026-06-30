import numpy as np


def similarity(query_vec: np.ndarray,object_embeddings: dict[int, np.ndarray]) ->  list[tuple[int, float]]:


    buckets= []
    for mask_id,vec in object_embeddings.items():
        score = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)))
        buckets.append((mask_id,score))
    return sorted(buckets,key=lambda pair: pair[1],reverse=True)


def query_scene(query, bundle, embedder, top_k=1):
    
    embeddings,geometry=bundle['embeddings'],bundle['geometry']
    query_embedding=embedder.embed_text(query)
    ranked_matches=similarity(query_embedding, embeddings)
    results=[]
    for mask_id, score in ranked_matches[:top_k]:
        points, colors = geometry[mask_id]
        results.append((mask_id, score, points, colors))
    return results


