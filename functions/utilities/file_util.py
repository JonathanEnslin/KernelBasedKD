'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

import json


class FileUtil:

    def load_file(self, file_path):
        # Open the file for reading
        data = None
        with open(file_path, "r") as file:
            # Load the dictionary from the file
            data = json.load(file)
        return data
    
    def save_to_file(self, history, file_path):
        with open(file_path, "w") as file:
            json.dump(history, file)
