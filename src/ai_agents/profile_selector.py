from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

def create_embeddings(profiles):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [model.encode(profile['Summary']) for profile in profiles.values()]
    return embeddings

def vector_search(query, embeddings, profiles):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    index = faiss.IndexFlatL2(len(query_embedding))
    
    # Convert list of embeddings to a numpy array
    embeddings_array = np.array(embeddings)
    index.add(embeddings_array)
    
    _, indices = index.search(query_embedding.reshape(1, -1), 3)
    top_candidates = [list(profiles.keys())[i] for i in indices[0]]
    return top_candidates

def refine_with_llm(top_candidates, profiles):
    # Initialize LLM with API key from environment
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    refined_candidates = {}
    for candidate_id in top_candidates:
        profile = profiles[candidate_id]
        # Use LLM to refine the selection
        messages = [
            SystemMessage(content="Refine the candidate selection based on the profile."),
            HumanMessage(content=f"Profile: {json.dumps(profile)}")
        ]
        response = llm.invoke(messages)
        if response.content.strip().lower() == "true":  # Simplified logic for illustration
            refined_candidates[candidate_id] = profile
    
    return refined_candidates

def main():
    with open('/Users/jproms/projects/job-candidates-ai-agents/data/profiles_candidates2.json', 'r') as file:
        profiles = json.load(file)

    query = "Find a candidate with 4 years of work experience"
    embeddings = create_embeddings(profiles)
    top_candidates = vector_search(query, embeddings, profiles)
    refined_candidates = refine_with_llm(top_candidates, profiles)

    print("Refined Candidates:")
    for candidate_id in refined_candidates:
        print(f"Candidate ID: {candidate_id}")
        print(f"Summary: {profiles[candidate_id]['Summary']}")

if __name__ == "__main__":
    main()