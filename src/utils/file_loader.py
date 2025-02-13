import json
import os

def load_json(filepath):
    """
    Load a JSON file and return its data.
    """
    with open(filepath, "r") as file:
        return json.load(file)

def export_to_json(data, filename):
    """
    Export data to a JSON file.
    """
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=2)
    print(f"Results saved to {filename}")

def load_text_files(directory):
    """
    Load all text files from a directory into a dictionary with the filename as the key.
    """
    files_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                files_data[filename] = file.read()
    return files_data
