import json
from typing import Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langgraph.graph import Graph
from typing import Annotated, Dict, List, Tuple
import operator
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize models
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_query_agent(state):
    """Agent that understands natural language query and converts it to structured format"""
    query = state['query']
    messages = [
        SystemMessage(content="""You are a query parser that converts natural language queries into JSON format.
        Convert the query into a structured JSON object with relevant search criteria.
        Return only valid JSON, no other text.
        Example: For "Find someone with 3 years experience in Python"
        Return: {"experience": 3, "skills": ["Python"]}"""),
        HumanMessage(content=f"Convert to JSON: {query}")
    ]
    response = llm.invoke(messages)
    try:
        state['parsed_query'] = json.loads(response.content)
        print("\nParse Query Step:")
        print(f"Input query: {query}")
        print(f"Parsed query: {state['parsed_query']}")
    except json.JSONDecodeError:
        state['parsed_query'] = {"raw_query": query}
    return state

def vector_search_agent(state):
    """Agent that performs vector search to find relevant candidates"""
    profiles = state['profiles']
    query = state['query']
    num_candidates = state.get('num_candidates', 5)  # Configurable, default to 5
    
    # Create embeddings for profiles and query
    profile_embeddings = [embedding_model.encode(p['Summary']) for p in profiles.values()]
    query_embedding = embedding_model.encode(query)
    
    # Perform vector search
    index = faiss.IndexFlatL2(len(query_embedding))
    index.add(np.array(profile_embeddings))
    D, I = index.search(query_embedding.reshape(1, -1), num_candidates)
    
    # Get top candidates
    state['candidates'] = {list(profiles.keys())[i]: profiles[list(profiles.keys())[i]] for i in I[0]}
    return state

def llm_refinement_agent(state):
    """Agent that uses LLM to refine candidate selection"""
    candidates = state['candidates']
    parsed_query = state['parsed_query']
    
    messages = [
        SystemMessage(content="""You are a candidate matching expert. Your task is to:
        1. Analyze the search criteria in the parsed query
        2. For experience requirements:
           - Handle exact matches (e.g., "4 years")
           - Handle ranges (e.g., "more than 3 years", "less than 5 years")
           - Look in both Summary and Experience sections
        3. For education requirements:
           - Check degrees (PhD, Masters, Bachelors, etc.)
           - Check fields of study
        4. For skills requirements:
           - Match required skills with candidate's skills
           - Consider both exact matches and related skills
        5. Handle multiple criteria with proper logical operations
        
        IMPORTANT: Return ONLY the JSON object with matching candidates, no markdown formatting, no ```json tags, no other text.
        Example format: {"candidate_id": {"name": "...", "Summary": "..."}}"""),
        HumanMessage(content=f"Find matching candidates.\nCriteria: {json.dumps(parsed_query)}\nCandidates: {json.dumps(candidates)}")
    ]
    
    print("\nRefinement Step:")
    print(f"Criteria being checked: {parsed_query}")
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Remove any markdown formatting if present
    if response_text.startswith('```'):
        response_text = response_text.split('\n', 1)[1]  # Remove first line
        response_text = response_text.rsplit('\n', 1)[0]  # Remove last line
        response_text = response_text.replace('```', '').strip()
    
    print(f"Cleaned LLM Response: {response_text[:200]}...")
    
    try:
        results = json.loads(response_text)
        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            state['results'] = results
        else:
            print("Response was valid JSON but not in expected format")
            state['results'] = {}
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        state['results'] = {}
    
    print(f"Final matches found: {len(state['results'])}")
    if state['results']:
        print("Matching candidates:")
        for cid in state['results'].keys():
            print(f"- {cid}")
    
    return state

def create_agent_workflow():
    """Create a workflow of agents using LangGraph"""
    workflow = Graph()
    
    # Define the workflow
    workflow.add_node("parse_query", parse_query_agent)
    workflow.add_node("vector_search", vector_search_agent)
    workflow.add_node("llm_refinement", llm_refinement_agent)
    
    # Define the channel and end state
    workflow = (
        workflow.add_edge("parse_query", "vector_search")
        .add_edge("vector_search", "llm_refinement")
        .set_entry_point("parse_query")
        .set_finish_point("llm_refinement")
    )
    
    return workflow.compile()

def execute_query(query: str, profiles: Dict, num_candidates: int = 5):
    """Execute the query using the agent workflow"""
    workflow = create_agent_workflow()
    
    state = {
        "query": query,
        "profiles": profiles,
        "parsed_query": None,
        "candidates": None,
        "results": None,
        "num_candidates": num_candidates
    }
    
    final_state = workflow.invoke(state)
    return final_state['results']

def load_profiles(file_path):
    """Load candidate profiles from JSON file"""
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    profiles = load_profiles('/Users/jproms/projects/job-candidates-ai-agents/data/profiles_candidates2.json')
    # natural_language_query = "Find a candidate with less than 4 years of work experience"
    natural_language_query = "Find just one candidate with Python skills"
    results = execute_query(natural_language_query, profiles)
    
    print("Query Results:")
    if not results:
        print("No matching candidates found.")
    else:
        for candidate_id, profile in results.items():
            print(f"Candidate ID: {candidate_id}")
            print(f"Summary: {profile['Summary']}")
            print(json.dumps(profile, indent=2))

if __name__ == "__main__":
    main()