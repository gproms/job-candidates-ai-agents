from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize models
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Cache file path
EMBEDDINGS_CACHE = "embeddings_cache.pkl"


def generate_and_cache_embeddings(profiles, force_generate_embeddings=False):
    """
    Generate and cache embeddings for candidate profiles.
    """
    if os.path.exists(EMBEDDINGS_CACHE) and not force_generate_embeddings:
        with open(EMBEDDINGS_CACHE, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}
        for candidate_id, profile in profiles.items():
            # profile_text = f"{profile['Summary']} {json.dumps(profile)}"
            profile_text = f"{json.dumps(profile)}"
            embeddings[candidate_id] = embedding_model.embed_query(profile_text)

        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump(embeddings, f)
    return embeddings


def shortlist_candidates(query, profiles, embeddings, top_n=10):
    """
    Shortlist candidates using cosine similarity of embeddings.
    """
    query_embedding = embedding_model.embed_query(query)
    similarities = {
        candidate_id: cosine_similarity(
            [query_embedding], [embedding]
        )[0][0]
        for candidate_id, embedding in embeddings.items()
    }
    shortlisted = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [candidate_id for candidate_id, _ in shortlisted]


def refine_shortlist_with_llm(query, profiles, shortlisted_ids):
    """
    Refine the shortlisted candidates using LLM.
    """
    refined_candidates = {}
    for candidate_id in shortlisted_ids:
        profile = profiles[candidate_id]
        messages = [
            SystemMessage(content="Answer True if the profile matches the query, else False."),
            HumanMessage(content=f"Query: {query}\nProfile: {json.dumps(profile)}")
        ]
        response = llm.invoke(messages)
        is_match = response.content.strip().lower() == "true"
        if is_match:
            refined_candidates[candidate_id] = profile
    return refined_candidates


def search_candidates(query, profiles, top_n=10):
    """
    Unified function to search candidates based on query.
    """
    embeddings = generate_and_cache_embeddings(profiles, force_generate_embeddings=True)
    shortlisted_ids = shortlist_candidates(query, profiles, embeddings, top_n=top_n)
    refined_results = refine_shortlist_with_llm(query, profiles, shortlisted_ids)
    return refined_results
