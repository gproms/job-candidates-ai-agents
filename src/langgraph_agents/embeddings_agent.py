import logging
from typing import Dict, List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EmbeddingsAgent:
    """
    Agent for generating embeddings and matching profiles to user queries.
    """

    def __init__(self, vectorstore_path: str = None):
        """
        Initialize the EmbeddingsAgent.

        :param vectorstore_path: Path to an existing FAISS vectorstore. If None, a new store will be created.
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vectorstore = (
            FAISS.load_local(vectorstore_path, self.embeddings)
            if vectorstore_path
            else None
        )

    def _generate_profile_text(self, profile: Dict) -> str:
        """
        Generate a structured text from a candidate profile for embedding generation.

        :param profile: A dictionary representing the candidate's profile.
        :return: A string summarizing the candidate's profile.
        """
        try:
            name = profile.get("name", "Unknown")
            years_of_experience = sum(
                float(exp.get("duration_years", 0))
                for exp in profile.get("Experience", [])
                if exp.get("duration_years", " ") not in ["unknown", None]
            )
            companies = ", ".join(
                exp.get("company", "Unknown")
                for exp in profile.get("Experience", [])
                if "company" in exp
            )
            roles = ", ".join(
                exp.get("title", " ") for exp in profile.get("Experience", [])
            )
            degrees = ", ".join(
                edu.get("degree", " ") for edu in profile.get("Education", [])
            )
            skills = ", ".join(profile.get("Skills", []))

            return (
                f"{name} has {years_of_experience:.1f} years of experience, worked at {companies}, "
                f"held roles such as {roles}, has degrees like {degrees}, and skills including {skills}."
            )
        except Exception as e:
            logger.error(f"Error generating profile text: {e}")
            return "Unknown profile summary."

    def generate_embeddings(self, profiles: Dict) -> FAISS:
        """
        Generate embeddings for profiles and store them in a FAISS vectorstore.

        :param profiles: A dictionary of candidate profiles.
        :return: A FAISS vectorstore containing the embeddings.
        """
        profile_data = [
            {
                "id": candidate_id,
                "text": self._generate_profile_text(profile),
            }
            for candidate_id, profile in profiles.items()
        ]
        texts = [entry["text"] for entry in profile_data]
        metadatas = [{"id": entry["id"]} for entry in profile_data]

        self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas)
        self.vectorstore.save_local("vectorstore")
        logger.info("Embeddings successfully generated and stored in vectorstore.")
        return self.vectorstore

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for the user query.

        :param query: The user query as a string.
        :return: A list of floats representing the query embedding.
        """
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

    def find_best_match(self, profiles: Dict, query: str, top_k: int = 1) -> List[Dict]:
        """
        Find the best matching profiles for a user query using cosine similarity.

        :param profiles: A dictionary of candidate profiles.
        :param query: The user query as a string.
        :param top_k: The number of top matches to return.
        :return: A list of dictionaries containing the best matches and their similarity scores.
        """
        if not self.vectorstore:
            logger.info("Generating embeddings as vectorstore is not initialized.")
            self.generate_embeddings(profiles)

        query_embedding = self.generate_query_embedding(query)

        if not query_embedding:
            logger.error("Failed to generate query embedding.")
            return []

        # Perform similarity search
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        top_matches = [
            {"profile": profiles[res.metadata["id"]], "similarity": score}
            for res, score in results
        ]

        return top_matches
