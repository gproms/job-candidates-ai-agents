import numpy as np

# Function to query candidates based on embeddings
def query_candidates(query, embeddings, model, top_k=1):
    """Queries the top matching candidates for a given user query."""
    # Generate embedding for the query
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Compute similarity scores
    results = []
    for candidate_id, candidate_embedding in embeddings.items():
        similarity = np.dot(query_embedding, candidate_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
        )
        results.append((candidate_id, similarity))

    # Sort results by similarity score
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:top_k]