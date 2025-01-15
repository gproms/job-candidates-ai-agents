import json
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Function to interpret the query
def interpret_query(query: str) -> str:
    """
    Uses the LLM to generate Python filter logic for a user query.
    Includes the schema of the profiles in the prompt for accurate generation.
    """
    schema_description = """
    Each profile is a dictionary with the following structure:
    {
        "Experience": {"Total Years": int, "Current Role": str},
        "Education": {"Degree": str},
        "Skills": list[str]
    }
    """
    prompt = (
        f"You are a Python expert. Create a Python function named `filter_func(profile)` "
        f"that evaluates if a profile matches the following query: {query}. "
        f"The function should use the following profile structure:\n{schema_description}\n"
        f"The function should return True if the profile matches and False otherwise. "
        f"Do not include extra explanations, just the code."
    )
    response = model.invoke([{"role": "user", "content": prompt}])
    return response.content


# Function to validate the generated query code
def validate_query_code(code: str) -> str:
    """
    Validates the generated Python code and extracts only the function definition.
    Ensures no additional explanations or text are included.
    """
    prompt = (
        f"You are a Python expert. Analyze the following Python code for correctness, potential bugs, and corner cases. "
        f"If the code is correct, return it as-is. If improvements are needed, return the improved code. "
        f"Return ONLY the Python code in your response. Do not include any explanations or commentary.\n\nCode:\n{code}"
    )
    response = model.invoke([{"role": "user", "content": prompt}])

    # Debug: Print the raw response from the LLM
    print(f"Raw LLM Response:\n{response.content}\n{'-'*40}")

    # Attempt to extract the function definition
    validated_code = response.content.strip()

    # Look for the function definition explicitly
    if "def filter_func" in validated_code:
        start_index = validated_code.find("def filter_func")
        validated_code = validated_code[start_index:].strip()
    elif "```" in validated_code:  # Handle markdown code blocks
        code_blocks = [
            block for block in validated_code.split("```") if "def filter_func" in block
        ]
        if code_blocks:
            validated_code = code_blocks[0].strip()
        else:
            raise ValueError(
                f"Validated code does not contain a valid 'filter_func': {validated_code}"
            )
    else:
        # Debug: Print what the code looks like before raising an error
        print(f"Code Missing 'filter_func':\n{validated_code}")
        raise ValueError(
            f"Validated code does not contain a valid 'filter_func': {validated_code}"
        )

    # Debug: Print the final validated code
    print(f"Final Validated Code:\n{validated_code}\n{'-'*40}")

    return validated_code

def execute_filter(profiles: Dict[str, Dict[str, Any]], filter_func_code: str) -> List[Dict[str, Any]]:
    """
    Executes the generated filter function on the profiles and returns matching profiles.
    """
    try:
        exec(filter_func_code, globals())
        if 'filter_func' not in globals():
            raise ValueError("filter_func is not defined in the generated code.")
    except Exception as e:
        raise RuntimeError(f"Failed to define filter_func: {e}")

    # Apply the filter function to profiles
    results = []
    for profile_name, profile_data in profiles.items():
        print(f"Applying filter to profile: {profile_name} -> {profile_data}")  # Debugging printout
        try:
            if 'filter_func' in globals() and filter_func(profile_data):
                results.append(profile_data)
        except Exception as e:
            print(f"Error applying filter to profile '{profile_name}': {e}")
    print(f"Filtered Profiles: {results}")  # Debugging printout
    return results



# Main search function
def search_profiles(profiles: Dict[str, Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Main search function to interpret, validate, and execute a query.
    """
    print(f"Processing Query: {query}")

    # Step 1: Interpret the query into Python filter logic
    filter_func_code = interpret_query(query)
    print(f"Generated Filter Function:\n{filter_func_code}")

    # Step 2: Validate and refine the generated code
    validated_code = validate_query_code(filter_func_code)
    print(f"Validated Filter Function:\n{validated_code}")

    # Step 3: Execute the filter function on profiles
    results = execute_filter(profiles, validated_code)

    return results

if __name__ == "__main__":
    # Sample profiles database
    profiles_db = {
        "John Doe": {
            "Experience": {"Total Years": 5, "Current Role": "Senior Developer"},
            "Education": {"Degree": "BSc in Computer Science"},
            "Skills": ["Python", "Java", "SQL", "Learning"]
        },
        "Jane Smith": {
            "Experience": {"Total Years": 3, "Current Role": "Data Analyst"},
            "Education": {"Degree": "MSc in Data Science"},
            "Skills": ["Python", "R", "SQL", "Visualization"]
        }
    }

    # Sample query
    user_query = "Find profiles with at least 7 years of experience and skills in Python, or having either Visualization or C++ skills."
    search_results = search_profiles(profiles_db, user_query)

    # Output results
    print("Search Results:")
    for result in search_results:
        print(result)
