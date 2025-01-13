import os
from utils.file_loader import load_json, load_files_from_directory, DATA_DIR

from langgraph_agents.profile_agent import create_profile_graph
from langgraph_agents.graph_visualiser import visualise_graph
from langgraph_agents.search_agent import search_profiles

def main():
    # Load data
    linkedin_data = load_json(os.path.join(DATA_DIR, "linkedin_profiles.json"))
    cv_data = load_files_from_directory(os.path.join(DATA_DIR, "cvs"), "txt")
    interview_data = load_files_from_directory(os.path.join(DATA_DIR, "interviews"), "txt")

    # Initialize graph
    profile_graph = create_profile_graph()
    compiled_graph = profile_graph.compile()
    initial_state = {
        "linkedin_data": linkedin_data,
        "cv_data": cv_data,
        "interview_data": interview_data,
    }

    # Run graph
    final_state = compiled_graph.invoke(initial_state)
    profiles_database = final_state["profiles"]

    # Visualize the graph
    visualise_graph(profile_graph, output_path="profile_graph.png")

    # Example user query
    query = "Find profiles with at least 3 years of experience."
    results = search_profiles(profiles_database, query)

    print("Search Results:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
