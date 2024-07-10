import json

def load_params(file_path, param_set_name):
    with open(file_path, 'r') as file:
        params = json.load(file)
    return params.get(param_set_name)
