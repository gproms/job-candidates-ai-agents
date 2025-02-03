from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

def find_best_matches(query_embedding: List[float], profile_embeddings: Dict[str, List[float]], profiles: Dict, top_k: int = 1):
    """
    Find the top K matches for a query embedding.
    """
    similarities = {
        candidate_id: cosine_similarity([query_embedding], [embedding])[0][0]
        for candidate_id, embedding in profile_embeddings.items()
    }

    sorted_candidates = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_profiles = {candidate_id: profiles[candidate_id] for candidate_id, _ in sorted_candidates[:top_k]}

    return top_profiles, sorted_candidates[:top_k]
