import logging
from typing import Dict, List, Tuple
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EmbeddingsAgent:
    def __init__(self):
        # Initialize the embeddings model
        self.embeddings_model = OpenAIEmbeddings()

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a given text.
        """
        try:
            embedding = self.embeddings_model.embed_query(text)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding for text: {text}\n{e}")
            return np.zeros((1536,))  # Return zero-vector if embedding fails

    def search(self, query: str, profiles: Dict, top_k: int = 1) -> Dict:
        """
        Search for the top-k most relevant profiles based on the query using cosine similarity.

        Args:
            query (str): User search query.
            profiles (Dict): Candidate profiles.
            top_k (int): Number of top results to return.

        Returns:
            Dict: A dictionary of the top-k matching profiles.
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Generate embeddings for candidate summaries
        profile_embeddings = []
        profile_ids = []
        for candidate_id, profile in profiles.items():
            summary = profile.get("Summary", "")
            if summary:
                embedding = self.generate_embedding(summary)
                profile_embeddings.append(embedding)
                profile_ids.append(candidate_id)
            else:
                logger.warning(f"Profile {candidate_id} missing Summary field.")

        if not profile_embeddings:
            logger.warning("No valid profiles with embeddings.")
            return {}

        # Compute cosine similarities
        profile_embeddings = np.stack(profile_embeddings)
        similarities = cosine_similarity([query_embedding], profile_embeddings).flatten()

        # Get top-k matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_results = {profile_ids[i]: profiles[profile_ids[i]] for i in top_indices}

        logger.info(f"Top-{top_k} profiles retrieved based on embeddings.")
        return top_results
