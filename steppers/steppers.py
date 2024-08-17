import numpy as np

def parse_kwargs(kwargs_str):
    import ast
    kwargs = {}
    if kwargs_str:
        # Split the string by commas to separate the key-value pairs
        pairs = kwargs_str.split(',')
        for pair in pairs:
            # Split each pair by the first '=' to separate the key and value
            key, value = pair.split('=', 1)
            # Use ast.literal_eval to safely evaluate the value
            kwargs[key.strip()] = ast.literal_eval(value.strip())
    return kwargs

def sine_modulated_beta(loss_handler, period=1000, amplitude=0.1, vertical_shift=1.0, through_relu=False):
    loss_handler.extern_vars["sine_modulated_beta"] = {
        "original_beta": loss_handler.beta
    }
    print(f"Original beta: {loss_handler.beta}")
    print(f"Generating sine modulated beta with period={period}, amplitude={amplitude}, vertical_shift={vertical_shift}, through_relu={through_relu}")
    def _sine_modulated_beta(idx):
        # loss_handler.beta = loss_handler.extern_vars["beta_toggler"]["original_beta"] * (1.0 if batch_idx % 2 == 0 else 0.0)
        loss_handler.beta = loss_handler.extern_vars["sine_modulated_beta"]["original_beta"] * (vertical_shift + amplitude * np.sin(2 * np.pi * idx / period))
        if through_relu:
            loss_handler.beta = np.maximum(loss_handler.beta, 0.0)

    return _sine_modulated_beta



stepper_dict = {
    "sine_modulated_beta": sine_modulated_beta
}