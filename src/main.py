import os
from dotenv import load_dotenv
from langgraph_agents.profile_agent import create_profile_graph, synthesis_node
from langgraph_agents.search_agent import search_profiles

# Load environment variables
load_dotenv()

# Sample data for integration testing
linkedin_data = [
    {"name": "John Doe", "Current Role": "Senior Developer", "Connections": 500, "Endorsements": ["Python", "Java"]},
    {"name": "Jane Smith", "Current Role": "Data Analyst", "Connections": 300, "Endorsements": ["Python", "R"]}
]

cv_data = {
    "john_doe.txt": "Experience: 5 years in Software Development\nEducation: BSc in Computer Science\nSkills: Python, Java, SQL",
    "jane_smith.txt": "Experience: 3 years in Data Analysis\nEducation: MSc in Data Science\nSkills: Python, R, SQL"
}

interview_data = {
    "john_doe_interview.txt": "John highlighted his ability to work in teams and solve complex problems.",
    "jane_smith_interview.txt": "Jane showcased strong skills in data visualization and statistical analysis."
}

# Initialize graph and data
profile_graph = create_profile_graph()
compiled_graph = profile_graph.compile()
initial_state = {
    "linkedin_data": linkedin_data,
    "cv_data": cv_data,
    "interview_data": interview_data,
}


def test_query():
    """Test query function for debugging search independently."""
    profiles_db = {
        "John Doe": {
            "Experience": {"Total Years": 5, "Current Role": "Senior Developer"},
            "Education": "BSc in Computer Science",
            "Skills": ["Python", "Java", "SQL"]
        },
        "Jane Smith": {
            "Experience": {"Total Years": 3, "Current Role": "Data Analyst"},
            "Education": "MSc in Data Science",
            "Skills": ["Python", "R", "SQL", "Visualization"]
        }
    }
    query = "Find profiles with at least 5 years of experience and skills in Python."
    search_results = search_profiles(profiles_db, query)
    print("Test Query Results:", search_results)


def test_synthesis_node():
    """Test synthesis node in isolation."""
    mock_state = {
        "cv_data": {
            "john_doe": {
                "Experience": "5 years in Software Development",
                "Education": "BSc in Computer Science",
                "Skills": ["Python", "Java", "SQL"]
            },
            "jane_smith": {
                "Experience": "3 years in Data Analysis",
                "Education": "MSc in Data Science",
                "Skills": ["Python", "R", "SQL"]
            }
        },
        "linkedin_data": {
            "John Doe": {
                "Experience": {"Current Role": "Senior Developer"},
                "Education": {},
                "Skills": ["Python", "Java"]
            },
            "Jane Smith": {
                "Experience": {"Current Role": "Data Analyst"},
                "Education": {},
                "Skills": ["Python", "R"]
            }
        },
        "interview_data": {
            "john_doe_interview": {
                "Experience": {"Teamwork": "Strong ability to work in teams"},
                "Education": {},
                "Skills": {"Problem Solving": "Skilled in solving complex problems"}
            },
            "jane_smith_interview": {
                "Experience": {
                    "Data Visualization": "Strong skills demonstrated",
                    "Statistical Analysis": "Strong skills demonstrated"
                },
                "Education": {},
                "Skills": {
                    "Data Visualization": "Strong",
                    "Statistical Analysis": "Strong"
                }
            }
        }
    }

    # Call the synthesis node
    result = synthesis_node(mock_state)

    # Debugging printouts
    print("Synthesis Node Result:")
    print(result)
    print("State After Synthesis:")
    print(mock_state)


def main():
    # Visualize the graph
    compiled_graph.get_graph().draw_png('graph_visualization.png')

    print("Processing profiles...")
    thread = {"configurable": {"thread_id": "1"}}
    global_state = initial_state.copy()

    # Process nodes using stream
    for state in compiled_graph.stream(initial_state, thread):
        global_state.update(state)
        print(f"State After Node Execution: {state}")
        print(f"Global State: {global_state}")

    # Debug global state at the end
    print("Final Global State:", global_state)

    # Query `profiles` from the final state
    profiles = global_state.get('profiles')
    if not profiles:
        print("No profiles found. Debugging full state:")
        print(global_state)
    else:
        print(f"Profiles: {profiles}")

        # Example query
        query = "Find profiles with at least 5 years of experience and skills in Python."
        results = search_profiles(profiles, query)
        print("Search Results:", results)

from langgraph.graph import StateGraph

# def test_synthesis_node_isolation():
#     # Create a simplified graph with only the synthesis node
#     test_graph = StateGraph(AgentState)
#     test_graph.add_node("synthesis", synthesis_node)
#     test_graph.set_entry_point("synthesis")
#     test_compiled_graph = test_graph.compile()
#
#     # Test input state
#     test_state = {
#         "cv_data": cv_data,
#         "linkedin_data": linkedin_data,
#         "interview_data": interview_data,
#     }
#
#     # Execute the test graph
#     test_final_state = test_compiled_graph.invoke(test_state)
#     print(f"Test Final State: {test_final_state}")
#
# # Call the test function
# test_synthesis_node_isolation()


if __name__ == "__main__":
    # Test query function (optional, for debugging search logic)
    # test_query()
    #
    # # Test synthesis node in isolation
    # test_synthesis_node()

    # Run the main function to process data through the graph and query results
    main()

    print("Done")
