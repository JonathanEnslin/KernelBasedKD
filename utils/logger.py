import os

class Logger:
    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m"
    }

    def __init__(self, run_name, log_to_file=True, log_dir="./run_data/output_logs", use_color=True):
        self.run_name = run_name
        self.log_dir = log_dir
        self.use_color = use_color
        self.log_to_file = log_to_file
        if log_to_file:
            self.log_file = os.path.join(log_dir, f"{run_name}.log")
            os.makedirs(log_dir, exist_ok=True)

    def log(self, *args, **kwargs):
        col = kwargs.pop('col', None)  # Extract the 'col' keyword argument, if present
        if col is not None:
            col = col.lower()
        sep = kwargs.pop('sep', ' ')  # Extract the 'sep' keyword argument, if present
        message = sep.join(str(arg) for arg in args)
        if self.use_color and col is not None and col in Logger.COLORS:
            message = Logger.COLORS[col] + message + Logger.COLORS["reset"]
        
        print(message, **kwargs)
        if self.log_to_file:
            with open(self.log_file, "a") as f:
                print(message, file=f, **kwargs)

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

if __name__ == "__main__":
    # Example usage:
    logger = Logger(run_name="my_run")

    # Log messages with different colors using the 'col' keyword argument
    logger("This is a black message", col="black")
    logger("This is a red message", col="red")
    logger("This is a green message", col="green")
    logger("This is a yellow message", col="yellow")
    logger("This is a blue message", col="blue")
    logger("This is a magenta message", col="magenta")
    logger("This is a cyan message", col="cyan")
    logger("This is a white message", col="white")

    # Log a message without any color
    logger("This is a regular message without color")

    