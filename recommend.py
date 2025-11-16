import numpy as np
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Load artifacts
df = pd.read_csv("data.csv")
embeddings = np.load("book_embeddings.npy")
nn = joblib.load("nn_index.joblib")
model = SentenceTransformer("all-mpnet-base-v2")

def recommend_books(user_text, top_k=5):
    # 1. Embed user input
    user_emb = model.encode([user_text], convert_to_numpy=True)
    user_emb = normalize(user_emb)

    # 2. Query the nearest neighbors
    distances, indices = nn.kneighbors(user_emb, n_neighbors=top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank+1,
            "title": df.loc[idx, "title"],
            "summary": df.loc[idx, "summary"],
            "tone_vector": df.loc[idx, "tone_vector"],
            "distance": float(distances[0][rank])
        })

    return results


if __name__ == "__main__":
    query = input("Describe what you want to read: ")
    recs = recommend_books(query)
    print("\nRecommended Books:\n")
    for r in recs:
        print(f"{r['rank']}. {r['title']} — {r['tone_vector']} (score={r['distance']:.4f})")
