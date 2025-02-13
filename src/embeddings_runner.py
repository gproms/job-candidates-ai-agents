import openai
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Function to parse the user query dynamically
def parse_query(query):
    """Parses the user query to extract key attributes."""
    parsed_query = {
        "years_of_experience": None,
        "experience_condition": None,  # 'less_than', 'greater_than', or 'exact'
        "skills": [],
        "education": None
    }
    if "years" in query:
        tokens = query.split()
        for i, token in enumerate(tokens):
            if token.isdigit() and "years" in tokens[i + 1]:
                parsed_query["years_of_experience"] = int(token)
                if "less" in query:
                    parsed_query["experience_condition"] = "less_than"
                elif "more" in query or "greater" in query:
                    parsed_query["experience_condition"] = "greater_than"
                else:
                    parsed_query["experience_condition"] = "exact"

    if "skills" in query:
        parsed_query["skills"] = [skill.strip() for skill in query.split("skills")[1].split(",")]

    if "PhD" in query or "degree" in query or "education" in query:
        parsed_query["education"] = "PhD"

    return parsed_query


# Function to generate candidate embeddings
def create_candidate_embeddings(data):
    """Generate embeddings for all candidates."""
    embeddings = {}
    for candidate_id, details in data.items():
        text_to_embed = ""
        if 'Summary' in details:
            text_to_embed += f"summary={details['Summary']}; "
        if 'Skills' in details:
            skills = ", ".join(details['Skills'])
            text_to_embed += f"skills={skills}; "
        if 'Experience' in details:
            experiences = []
            for exp in details['Experience']:
                duration = exp.get('duration_years', 'unknown')
                experiences.append(
                    f"title={exp.get('title', 'unknown')}, company={exp.get('company', 'unknown')}, "
                    f"duration_years={duration}"
                )
            text_to_embed += f"experience=[{' | '.join(experiences)}]; "
        if 'Education' in details:
            educations = []
            for edu in details['Education']:
                degree = edu.get('degree', 'unknown')
                institution = edu.get('institution', 'unknown')
                educations.append(f"degree={degree}, institution={institution}")
            text_to_embed += f"education=[{' | '.join(educations)}]; "

        embeddings[candidate_id] = embedding_model.encode(text_to_embed.strip(), convert_to_numpy=True)

    return embeddings


# Function to compute candidate scores
def compute_candidate_scores(query, candidate_data):
    """Scores candidates based on embeddings and parsed query."""
    parsed_query = parse_query(query)
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    candidate_embeddings = create_candidate_embeddings(candidate_data)

    results = []
    for candidate_id, candidate_embedding in candidate_embeddings.items():
        similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
        )

        # Adjust scores based on query-specific conditions
        candidate_details = candidate_data[candidate_id]
        bonus = 0
        for exp in candidate_details.get("Experience", []):
            duration = exp.get("duration_years", "0")
            try:
                duration = int(duration)  # Ensure `duration_years` is an integer
            except ValueError:
                duration = 0  # Default to 0 if casting fails

            condition = parsed_query.get("experience_condition")
            target_years = parsed_query.get("years_of_experience")
            if condition == "less_than" and duration < target_years:
                bonus += 0.1
            elif condition == "greater_than" and duration > target_years:
                bonus += 0.1
            elif condition == "exact" and duration == target_years:
                bonus += 0.1

        if parsed_query.get("education") and parsed_query["education"] in [
            edu.get("degree", "") for edu in candidate_details.get("Education", [])
        ]:
            bonus += 0.3  # Increased weight for education match

        results.append((candidate_id, similarity + bonus))

    # Ensure candidates with exact education matches are included
    if parsed_query.get("education"):
        for candidate_id, candidate_details in candidate_data.items():
            if parsed_query["education"] in [
                edu.get("degree", "") for edu in candidate_details.get("Education", [])
            ]:
                if candidate_id not in [c[0] for c in results]:
                    results.append((candidate_id, 1.0))  # Add with boosted score

    # Sort results by highest adjusted score
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:3]


# Function to refine the selection using LLM
def refine_selection_with_llm(query, top_candidates, candidate_data):
    """Uses LLM to refine the top candidate selection."""
    candidate_summaries = []
    for candidate_id, _ in top_candidates:
        details = candidate_data[candidate_id]
        summary = f"{candidate_id}: "
        if 'Experience' in details:
            experience = ", ".join(
                [f"{exp.get('title')} at {exp.get('company')} ({exp.get('duration_years')} years)"
                 for exp in details['Experience'] if 'title' in exp]
            )
            summary += f"{experience}; "
        if 'Skills' in details:
            summary += f"Skills: {', '.join(details['Skills'])}; "
        if 'Education' in details:
            education = ", ".join(
                [f"{edu.get('degree')} from {edu.get('institution')}" for edu in details['Education']]
            )
            summary += f"Education: {education};"
        candidate_summaries.append(summary)

    prompt = (
            f"Please select the candidate who best matches the query. "
            f"If no candidate meets the requirements, state that none match. "
            f"Explain why candidates with matching qualifications, like Candidate 4, were not selected if they appear relevant.\n\n"
            f"Query: {query}\n\n"
            f"Candidates:\n" + "\n".join(candidate_summaries) +
            "\n\nWhich candidate best matches the query?"
    )

    messages = [
        SystemMessage(content="You are an assistant that selects the best candidate for a given query."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# Main function
def main():
    # Load data
    data_file_path = "profiles_candidates.json"
    with open(data_file_path, 'r') as f:
        candidate_data = json.load(f)

    # Example query
    example_query = "Find candidates with a PhD"
    print(f"Query: {example_query}")

    # Compute top candidates
    top_candidates = compute_candidate_scores(example_query, candidate_data)
    print("Top 3 Candidates (Embedding-Based):", top_candidates)

    # Refine selection using LLM
    final_selection = refine_selection_with_llm(example_query, top_candidates, candidate_data)
    print("\nLLM Final Selection:", final_selection)


if __name__ == "__main__":
    main()
