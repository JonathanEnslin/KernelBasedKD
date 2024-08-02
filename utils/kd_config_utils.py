import json

def _parse_kd_config_string(conf: str):
    conf = conf.split(':')
    if len(conf) == 1:
        raise ValueError(f"Invalid KD configuration string: {conf}")
    
    kd_config_file = conf[0]
    kd_config_set = conf[1]
    if len(conf) > 2:
        kd_config_set += ':'.join(conf[1:])

    return kd_config_file, kd_config_set

def _load_kd_config(kd_config_file: str, kd_config_set: str):
    with open(kd_config_file, 'r') as file:
        kd_config = json.load(file)
    if kd_config_set not in kd_config:
        raise ValueError(f"Invalid KD configuration set: {kd_config_set}")
    return kd_config.get(kd_config_set)

def load_kd_config(kd_config_str: str) -> dict:
    """
    Load a knowledge distillation configuration from a string in the format of "<file>:<set>".
    """
    kd_config_file, kd_config_set = _parse_kd_config_string(kd_config_str)
    return _load_kd_config(kd_config_file, kd_config_set)
