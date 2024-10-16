import re
import json
import os
from loss_functions.filter_at import FilterAttentionTransfer
from loss_functions.vanilla import VanillaKDLoss
from loss_functions.attention_transfer import ATLoss
from loss_functions.filter_magnitude import FilterAnalysis
from loss_functions.factor_transfer import FTLoss
from loss_functions.filter_factor import KFTLoss
from utils.logger import Logger
import inspect

def get_distillation_params(params_file: str, param_set: str, logger=Logger, base_path='') -> dict:
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
    "vanilla": VanillaKDLoss,
    "at": ATLoss,
    "attention_transfer": ATLoss,
    "attention": ATLoss,
    "filter_at": FilterAttentionTransfer,
    "kat": FilterAttentionTransfer,
    "Analysis": FilterAnalysis,
    "ft": FTLoss,
    "factor_transfer": FTLoss,
    "factor": FTLoss,
    "kft": KFTLoss,
    "filter_ft": KFTLoss,
}

def get_kd_method(distillation_params_dict, logger: Logger):
    return distillation_params_dict.get("distillation_type", None)

def get_loss_function(distillation_params_dict, logger: Logger, **kwargs):
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
    
    # Filter the params and kwargs to include only those accepted by the loss function class
    sig = inspect.signature(loss_fn_class)
    filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    # Combine filtered params and filtered kwargs
    combined_kwargs = {**filtered_params, **filtered_kwargs}

    return loss_fn_class(**combined_kwargs)
    







