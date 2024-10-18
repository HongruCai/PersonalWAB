import json
import os
import glob

FOLDER_PATH = os.path.dirname(__file__)
def merge_json_files(file_pattern):
    merged_data = {}

    for json_file in glob.glob(file_pattern):
        with open(json_file, 'r') as f:
            data = json.load(f)
            merged_data.update(data)  
        #print(f"Loaded {json_file}")
    
    return merged_data

all_products = merge_json_files(os.path.join(FOLDER_PATH, "all_products_part_*.json"))
user_history = merge_json_files(os.path.join(FOLDER_PATH, "user_history_part_*.json"))

data = {
    "tasks": json.load(open(os.path.join(FOLDER_PATH, "user_instructions.json"))),
    "user_profile": json.load(open(os.path.join(FOLDER_PATH, "user_profiles.json"))),
    "user_history": user_history,
    "all_products": all_products
}
