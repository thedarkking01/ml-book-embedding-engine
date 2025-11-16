import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('./data.csv')

df['tone_vector'] = df['tone_vector'].apply(lambda x: x.split(','))

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['tone_vector'])
print(f"Labels: {mlb.classes_}")


model = SentenceTransformer('all-mpnet-base-v2')
texts = df['summary'].tolist()
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Normalize (for cosine similarity)
embeddings = normalize(embeddings)
np.save('book_embeddings.npy', embeddings)

# Build Nearest Neighbors index
emb = np.load('book_embeddings.npy')
nn = NearestNeighbors(n_neighbors=20, metric='cosine').fit(emb)
# save index (joblib)
joblib.dump(nn, 'nn_index.joblib')


