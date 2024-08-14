import re
import json
import os
import loss_functions as lf
from utils.logger import Logger

def get_distillation_params(params_file: str, param_set: str, logger=Logger, base_path='configs') -> dict:
    params_path = os.path.join(base_path, params_file)

    # if the path does not exist return None
    if not os.path.exists(params_path):
        logger(f"File {params_path} does not exist", col='red')
        return None

    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        return params[param_set]
    except Exception as e:
        logger(f"Error loading params from {params_path}: {e}", col='red')
        return None


loss_function_map = {
    "vanilla": lf.vanilla.VanillaKDLoss,
    "at": lf.attention_transfer.ATLoss,
    "filter_at": lf.filter_at.FilterAttentionTransfer,
}

def get_kd_method(distillation_params_dict, logger: Logger):
    return distillation_params_dict.get("distillation_type", None)

def get_loss_function(distillation_params_dict, logger: Logger, preact_fmaps=None, postact_fmaps=None):
    distillation_type = distillation_params_dict.get("distillation_type", None)
    if distillation_type is None:
        logger("Distillation type not found in distillation params", col='red')
        return None
    
    loss_fn_class = loss_function_map.get(distillation_type, None)
    if loss_fn_class is None:
        logger(f"Loss function class not found for distillation type: {distillation_type}", col='red')
        return None
    
    params = distillation_params_dict.get("params", None)
    if params is None:
        logger("Params not found in distillation params", col='red')
        return None
    
    return loss_fn_class(**params)
    







