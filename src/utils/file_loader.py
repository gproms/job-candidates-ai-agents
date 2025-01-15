import json

def load_json(filepath):
    """
    Load a JSON file and return its data.
    """
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return {}

def export_to_json(data, filename):
    """
    Export data to a JSON file.
    """
    try:
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=2)
        print(f"Results saved to {filename}")
    except IOError as e:
        print(f"Error saving JSON file: {e}")
