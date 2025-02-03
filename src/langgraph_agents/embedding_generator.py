from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_candidate_embeddings(data):
    """Generates embeddings for all candidates."""
    embeddings = {}
    for candidate_id, details in data.items():
        # Combine Summary, Skills, and Experience descriptions for embedding
        text_to_embed = details['Summary']
        if 'Skills' in details and isinstance(details['Skills'], list):
            text_to_embed += " " + ", ".join(details['Skills'])
        if 'Experience' in details and isinstance(details['Experience'], list):
            experiences = [exp['description'] for exp in details['Experience'] if 'description' in exp]
            text_to_embed += " " + " ".join(experiences)

        # Generate embedding for the candidate
        embedding = embedding_model.encode(text_to_embed, convert_to_numpy=True)
        embeddings[candidate_id] = embedding

    return embeddings