class Args:
    def __init__(self):
        # Properties based on Logger's usage
        self.dataset = "CIFAR10"  # Example dataset name
        self.model_name = "resnet56"  # Example model name
        self.param_set = "params_v1"  # Example parameter set name
        self.run_name = None  # If a run name is not provided, Logger will generate one
        self.disable_auto_run_indexing = False  # Allows auto-indexing for run names
        self.resume = None  # Used when resuming training, path to a checkpoint file
        self.use_val = True  # Indicates if validation is used in the training process

# Example usage of the mock Args class
if __name__ == "__main__":
    # Create an instance of the Args class
    mock_args = Args()

    # # Create a logger instance with mock arguments
    # logger = Logger(args=mock_args, run_tag="experiment_1", teacher_type="resnet110", kd_set="KD_set1")

    # # Log messages
    # logger("Example log message")
