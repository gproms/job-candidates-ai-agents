import streamlit as st
import json
from langgraph_agents.query_agent import interpret_and_filter_profiles
from langgraph_agents.embeddings_agent import EmbeddingsAgent
from utils.file_loader import load_json
from src import DATA_DIR

# Paths
PROFILES_JSON_PATH = f"{DATA_DIR}/profiles_candidates.json"


# Load candidate profiles
@st.cache_data
def load_profiles():
    with open(PROFILES_JSON_PATH, "r") as f:
        return json.load(f)


# Initialize embeddings agent
embeddings_agent = EmbeddingsAgent()

# Streamlit UI
st.title("Candidate Search App")
st.sidebar.title("Search Options")

# Select search method
search_method = st.sidebar.radio("Select Search Method", ["NLP Search", "Embeddings Search"])

# Input query
query = st.text_input("Enter your search query", placeholder="E.g., Find candidates with Python skills")

# Search button
if st.button("Search"):
    if not query:
        st.error("Please enter a search query!")
    else:
        profiles = load_profiles()

        if search_method == "NLP Search":
            st.info("Running NLP Search...")
            try:
                results = interpret_and_filter_profiles(query, profiles)
                if results:
                    st.success("Search Results:")
                    st.json(results)
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"Error during NLP Search: {e}")

        elif search_method == "Embeddings Search":
            st.info("Running Embeddings Search...")
            try:
                results = embeddings_agent.search(query, profiles)
                if results:
                    st.success("Search Results:")
                    st.json(results)
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"Error during Embeddings Search: {e}")
