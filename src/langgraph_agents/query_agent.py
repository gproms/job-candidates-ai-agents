import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize LLMs
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
critique_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
refinement_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def interpret_and_filter_profiles(query: str, profiles: Dict) -> Dict:
    """
    Interpret the query and apply a filter to the profiles using generated Python code.
    """
    profiles = validate_and_convert_profiles(profiles)

    # Add schema description and toy data example
    schema_description = """
    ### Data Schema:
    - Each profile is a dictionary with the following fields:
      - `Experience`: A list of dictionaries, each representing a work experience with:
        - `title` (string): The job title.
        - `company` (string): The name of the company.
        - `duration_years` (float or string): The duration in years.
        - `description` (string): A description of the role.
      - `Education`: A list of dictionaries, each representing an academic qualification with:
        - `degree` (string): The name of the degree.
        - `institution` (string): The name of the institution.
        - `year` (int): The year of graduation.
      - `Skills`: A list of strings representing the candidate's skills.
      - `Summary`: A string summarizing the candidate's background.

    ### Example Data:
    {
        "candidate_101": {
            "name": "Ethan Green",
            "Summary": "Experienced Robotics Engineer with expertise in automation and hardware design.",
            "Experience": [
                {
                    "title": "Robotics Engineer",
                    "company": "AutoTech Solutions",
                    "duration_years": 5.0,
                    "description": "Designed robotic systems for automated manufacturing processes."
                }
            ],
            "Education": [
                {
                    "degree": "MSc in Robotics",
                    "institution": "Tech Institute",
                    "year": 2019
                }
            ],
            "Skills": [
                "Robotics",
                "Automation",
                "Hardware Design",
                "Python"
            ]
        },
        "candidate_102": {
            "name": "Sophia Clark",
            "Summary": "Creative Graphic Designer specializing in branding and UI/UX design.",
            "Experience": [
                {
                    "title": "Graphic Designer",
                    "company": "DesignSphere",
                    "duration_years": 3.0,
                    "description": "Developed brand identities and UI/UX prototypes for clients."
                }
            ],
            "Education": [
                {
                    "degree": "BA in Graphic Design",
                    "institution": "Art and Design University",
                    "year": 2020
                }
            ],
            "Skills": [
                "Branding",
                "UI/UX Design",
                "Illustration",
                "Adobe Suite"
            ]
        }
    }
    """

    # Generate filtering code based on user query
    prompt = f"""
    You are an assistant that writes Python functions to filter profiles from a given dataset.
    The dataset is a JSON object where keys are candidate IDs and values are profiles.

    ### Rules for Writing Code:
    1. Always return a Python function named `filter_profiles`.
    2. The function should take one parameter: `profiles` (the dataset).
    3. Use proper Python syntax, including indentation.
    4. Ensure the function returns a dictionary of filtered profiles.
    5. Run the search against the 'Summary' field first (profile["Summary"]), and use the other fields like 'Experience', 'Skills', and 'Education' to refine the search if necessary.
    6. Follow the data schema provided below.

    {schema_description}

    ### Example Query:
    - Query: "Find candidates with experience in Agile methodologies."

    ### Example Output Code:
    ```python
    def filter_profiles(profiles):
        filtered_profiles = {{}}
        for candidate_id, profile in profiles.items():
            if <matching criteria found after analysing profile["Summary"]>:
                filtered_profiles[candidate_id] = profile
            elif any(<matching criteria> in exp.get("description", "") for exp in profile.get("Experience", [])):
                filtered_profiles[candidate_id] = profile
        return filtered_profiles
    ```

    Write Python code to filter profiles based on this query:
    Query: {query}
    Ensure the code is clean and follows Python standards.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    generated_code = response.content.strip()

    logger.info("\n=== Generated Code ===")
    logger.info(generated_code)

    # Step 1: Critique the Code
    critique = critique_generated_code(query, generated_code, schema_description)

    # Step 2: Refine the Code
    refined_code = refine_generated_code(query, generated_code, critique, schema_description)

    # Execute the refined code
    try:
        exec_globals = {}
        exec(refined_code, exec_globals)
        filter_profiles = exec_globals["filter_profiles"]

        # Apply the filter function to the profiles
        filtered_profiles = filter_profiles(profiles)

        logger.info("\n=== Filtered Profiles ===")
        logger.info(filtered_profiles)
    except Exception as e:
        logger.error(f"Failed to execute code. Error: {e}")
        return {}

    return filtered_profiles



def critique_generated_code(query:str, code: str, schema_description: str) -> str:
    """
    Use the critique LLM to analyze the generated code and provide feedback.
    """
    critique_prompt = f"""
    You are an assistant that critiques Python code to identify logical flaws, edge cases, and potential errors, given the query and the proposed solution.

    ### Data Schema:
    {schema_description}

    ### Instructions:
    1. Analyze the user query to understand exactly what the user wants to achieve, to determine if the proposed code will meet the requirements.
    1. Analyze the provided code for:
       - Logical errors or bugs.
       - Missing edge case handling.
       - Areas where the code might fail.
       - Analyse if the user is asking for one or N results, and verify that the code aligns with this requirement. If the user asks for "a" candidate or "any" candidate, or the best or worse candidate, the code should return just one candidate.
       - Verify if the function will return the number of users required by the user query.
       - Proper use of the data schema for filtering logic.
    2. Provide concise feedback on how to improve the code.
    
    ### User query:
    {query}

    ### Provided Code:
    {code}
    """
    critique_response = critique_llm.invoke([SystemMessage(content=critique_prompt)])
    critique = critique_response.content.strip()

    logger.info("\n=== Critique ===")
    logger.info(critique)

    return critique


def refine_generated_code(query: str, code: str, critique: str, schema_description: str) -> str:
    """
    Use the refinement LLM to improve the generated code based on the critique.
    """
    refinement_prompt = f"""
    You are an assistant that refines Python code based on feedback from a critique and making sure it will solve the user query.

    ### Data Schema:
    {schema_description}

    ### Instructions:
    1. Review the original code and the provided critique.
    2. Incorporate the critique to improve the code, addressing any logical errors, edge cases, or potential issues.
    3. Analyse if the user is asking for one or N results, and verify that the code aligns with this requirement. If the user asks for "a" candidate or "any" candidate, or the best or worse candidate, the code should return just one candidate.
    4 Verify if the function will return the number of users required by the user query. "a" candidate or "any" candidate, or the best or worse candidate, the code should return just one candidate.
    5. Check for case sensitivity in the search and fix accordingly.
    5. Ensure the refined code is ready to execute and follows Python standards and the function name is filter_profiles.
    6. Do not include comments, explanations, or markdowns in the output—only return executable Python code. Only runnable code.
    7. Strip any markdowns before returning the code. Make sure there are no triple backticks at the top or bottom of the code.
    8. The returned Python function MUST be named `filter_profiles`, no other name is allowed.
    9. Take a moment to reflect on the query, your resulting code, and the changes you made to improve it. If you find any issues, correct them before submitting the refined code.
    
    ### User query:
    {query}

    ### Original Code:
    {code}

    ### Critique:
    {critique}

    ### Refined Code:
    """
    refinement_response = refinement_llm.invoke([SystemMessage(content=refinement_prompt)])
    refined_code = refinement_response.content.strip()
    refined_code = refined_code.replace("```python", "").replace("```", "")

    logger.info("\n=== Refined Code ===")
    logger.info(refined_code)

    return refined_code


def validate_and_convert_profiles(profiles: Dict) -> Dict:
    """
    Ensure profiles contain consistent data types, particularly for `duration_years`.
    Converts `duration_years` to float where possible.
    """
    for candidate_id, profile in profiles.items():
        for experience in profile.get("Experience", []):
            if "duration_years" in experience:
                try:
                    experience["duration_years"] = float(experience["duration_years"])
                except (ValueError, TypeError):
                    experience["duration_years"] = 0.0  # Default to 0.0 if conversion fails
    return profiles
