from langchain.embeddings import OpenAIEmbeddings
import pickle
import os

# Cache path
EMBEDDINGS_CACHE = "embeddings_cache.pkl"

# Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Function to cache and retrieve embeddings
def manage_embeddings(profiles, recache=False):
    if os.path.exists(EMBEDDINGS_CACHE) and not recache:
        with open(EMBEDDINGS_CACHE, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}
        for candidate_id, profile in profiles.items():
            text = f"{profile['Summary']} {json.dumps(profile)}"
            embeddings[candidate_id] = embedding_model.embed_query(text)
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump(embeddings, f)
    return embeddings
