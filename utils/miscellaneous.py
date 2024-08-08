import os
from utils.logger import Logger

class AverageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, value, count=1):
        self.total += value * count
        self.count += count

    def get_average(self):
        return self.total / self.count if self.count > 0 else 0

    def __call__(self):
        return self.get_average()
    

class MovingAverageTracker:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.values = []
        self.total = 0

    def reset(self):
        self.values = []
        self.total = 0

    def update(self, value):
        self.values.append(value)
        self.total += value
        if len(self.values) > self.window_size:
            self.total -= self.values.pop(0)

    def get_std(self):
        if len(self.values) < 2:
            return 0
        avg = self.get_average()
        return (sum((value - avg) ** 2 for value in self.values) / len(self.values)) ** 0.5

    def get_average(self):
        return self.total / len(self.values) if len(self.values) > 0 else 0

    def __call__(self):
        return self.get_average()



def ensure_dir_existence(dirs, logger=print):
    """
    Creates the directories in dirs if they do not exist

    Returns:
        False if an error occurred True otherwise
    """
    for directory in dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                try:
                    logger(f"Error: {directory} could not be created. {e}")
                except Exception as ex:
                    print(f"Error: {directory} could not be created. {e}, and logger failed with {ex}")
                return False
    return True


def check_validation_args(args, logger: Logger) -> bool:
    missing_args = False
    if args.use_val:        
        if args.use_split_indices_from_file is None and args.val_split_random_state is None:
            logger("Info: --val_split_random_state is not specified. Using a random random state.", col="magenta")

        if args.val_size is None or not (0 < args.val_size < 1):
            logger("Error: --val_size must be specified and be between 0 and 1 when using --use_val.", col="red")
            missing_args = True
        
        if args.early_stopping_patience is None:
            logger("Error: --early_stopping_patience must be specified when using --use_val.", col="red")
            missing_args = True
        
        if args.early_stopping_start_epoch is None:
            logger("Error: --early_stopping_start_epoch must be specified when using --use_val.", col="red")
            missing_args = True

    return not missing_args