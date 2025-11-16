from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

app = FastAPI()

df = pd.read_csv("data.csv")
embeddings = np.load("book_embeddings.npy")
nn = joblib.load("nn_index.joblib")
model = SentenceTransformer("all-mpnet-base-v2")

class Query(BaseModel):
    text: str
    top_k: int = 5

@app.post("/recommend")
def recommend_endpoint(q: Query):
    user_emb = model.encode([q.text], convert_to_numpy=True)
    user_emb = normalize(user_emb)

    distances, indices = nn.kneighbors(user_emb, n_neighbors=q.top_k)

    output = []
    for rank, idx in enumerate(indices[0]):
        output.append({
            "rank": rank+1,
            "title": df.loc[idx, "title"],
            "summary": df.loc[idx, "summary"],
            "tone_vector": df.loc[idx, "tone_vector"],
            "distance": float(distances[0][rank])
        })

    return {"results": output}
