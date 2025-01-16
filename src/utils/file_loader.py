import json

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
