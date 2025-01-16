import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings model
embedding_model = OpenAIEmbeddings()

def preprocess_profile_text(profile):
    """
    Preprocess a single profile to create a text representation.
    Combines 'Summary', 'Skills', and 'Experience' fields into a single text block.
    Converts all text to lowercase for uniformity.
    """
    summary = profile.get("Summary", "").lower()  # Convert to lowercase
    skills = " ".join(profile.get("Skills", [])).lower()  # Convert to lowercase
    experiences = " ".join(exp.get("description", "").lower() for exp in profile.get("Experience", []))
    return f"{summary} {skills} {experiences}"


def preprocess_and_embed_profiles(profiles):
    """
    Preprocess profiles and compute their embeddings.
    Returns a dictionary with candidate IDs as keys and embeddings as values.
    """
    embeddings = {}
    for candidate_id, profile in profiles.items():
        text_representation = preprocess_profile_text(profile)
        embedding = embedding_model.embed_query(text_representation)
        embeddings[candidate_id] = {
            "profile": profile,
            "embedding": embedding,
        }
    return embeddings

def embed_query(query):
    """
    Generate an embedding vector for the user's query.
    Converts query to lowercase for consistency with profile embeddings.
    """
    return embedding_model.embed_query(query.lower())


def search_profiles(query_embedding, profile_embeddings, top_k=1):
    """
    Search for the top_k most similar profiles based on cosine similarity.
    """
    candidate_ids = list(profile_embeddings.keys())
    embeddings = np.array([profile_embeddings[cid]["embedding"] for cid in candidate_ids])
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = {candidate_ids[i]: profile_embeddings[candidate_ids[i]]["profile"] for i in top_indices}
    return results
