import json
import glob
import os
from src import DATA_DIR

def load_json(file_path):
    """Loads JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_text(file_path):
    """Loads text data from a file."""
    with open(file_path, 'r') as f:
        return f.read()

def load_files_from_directory(directory, extension):
    """
    Loads all files with a given extension from a directory.
    Returns a dictionary with file names as keys and file contents as values.
    """
    file_paths = glob.glob(f"{directory}/*.{extension}")
    return {os.path.basename(path): load_text(path) for path in file_paths}

# Main function to test loading
if __name__ == "__main__":
    # Use centralized `DATA_DIR` for paths
    linkedin_path = os.path.join(DATA_DIR, 'linkedin_profiles.json')
    cv_directory = os.path.join(DATA_DIR, 'cvs')
    interview_directory = os.path.join(DATA_DIR, 'interviews')

    # Test JSON loading
    linkedin_data = load_json(linkedin_path)
    print("LinkedIn Data:")
    print(json.dumps(linkedin_data, indent=4))

    # Test text file loading
    cv_data = load_files_from_directory(cv_directory, 'txt')
    print("\nCV Data:")
    print(cv_data)

    interview_data = load_files_from_directory(interview_directory, 'txt')
    print("\nInterview Data:")
    print(interview_data)
