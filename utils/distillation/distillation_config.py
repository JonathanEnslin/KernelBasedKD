import re
import json
import os
import loss_functions as lf

def get_distillation_params(params_file: str, param_set: str, logger=print, base_path='configs') -> dict:
    params_path = os.path.join(base_path, params_file)

    # if the path does not exist return None
    if not os.path.exists(params_path):
        logger(f"File {params_path} does not exist")
        return None

    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        return params[param_set]
    except Exception as e:
        logger(f"Error loading params from {params_path}: {e}")
        return None


loss_function_map = {
    "vanilla": lf.vanilla.VanillaKDLoss,
    "at": lf.attention_transfer.ATLoss,
    "filter_at": lf.filter_at.FilterAttentionTransfer,
}




