import os
import csv

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

    def __init__(self, args, log_to_file=True, data_dir="run_data", run_tag=None, use_color=True, teacher_type=None, kd_set=None):
        self.args = args
        self.dataset = args.dataset
        self.model_name = args.model_name
        self.run_tag = run_tag
        self.teacher_type = teacher_type
        self.kd_set = kd_set
        self.run_lock_dir = os.path.join(data_dir, "run_name_locks")
        self.run_name = self._get_run_name(run_tag=run_tag)
        print(self.run_name)
        self.run_name_lock_file = os.path.join(self.run_lock_dir, f"{self.run_name}.lock")
        # Create a file with the run name to indiciate that the run name is in use
        Logger.create_dir_if_not_exists(os.path.dirname(self.run_name_lock_file))
        with open(self.run_name_lock_file, "w") as f:
            f.write("")

        self.log_dir = 'output_logs'
        self.use_color = use_color
        self.log_to_file = log_to_file
        self.csv_dir = 'csv_logs'
        self.fine_grained_dir = os.path.join(self.dataset, self.model_name)
        self.csv_base_dir = os.path.join(data_dir, self.csv_dir, self.fine_grained_dir)

        if log_to_file:
            self.log_file = os.path.join(data_dir, self.log_dir, self.fine_grained_dir, f"{self.run_name}.log")
            Logger.create_dir_if_not_exists(os.path.dirname(self.log_file))

        self.csv_phase_locs = {
            "test": os.path.join(self.csv_base_dir, "test", self.get_csv_name("test")),
            "val": os.path.join(self.csv_base_dir, "val", self.get_csv_name("val")),
            "train": os.path.join(self.csv_base_dir, "train", self.get_csv_name("train")),
            "NA": os.path.join(self.csv_base_dir, "NA", self.get_csv_name("NA")),
        }

    @staticmethod
    def create_dir_if_not_exists(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                return True
        except Exception as e:
            print(f"Error: {directory} could not be created. {e}")
            return False

    
    def get_run_name(self):
        return self.run_name


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


    def log_to_csv(self, data):
        phase = data.get('phase', 'NA')
        csv_file = self.csv_phase_locs[phase]
        Logger.create_dir_if_not_exists(os.path.dirname(csv_file))
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        

    def get_csv_name(self, phase):
        if phase == 'NA':
            return f"{self.run_name}.csv"
        return f"{self.run_name}.{phase}.csv"


    def _get_run_name(self):
        # Generate or use provided run name
        val_tag = ".val" if self.args.use_val else ""
        run_tag = f".{self.run_tag}" if self.run_tag is not None else ""
        teacher_tag = f"[{self.teacher_type}]" if self.teacher_type is not None else ""
        kd_tag = f"[{self.kd_set}]" if self.kd_set is not None else ""
        auto_name = f"{self.args.model_name}{teacher_tag}@{self.args.dataset}_{self.args.param_set}{kd_tag}{run_tag}"
        run_name_base = self.args.run_name or auto_name
        if not self.args.disable_auto_run_indexing:
            run_name = run_name_base + "_#1"
            run_counter = 2
            while os.path.exists(os.path.join(self.run_lock_dir, f"{run_name}.lock")):
                run_name = f"{run_name_base}_#{run_counter}"
                run_counter += 1
        else:
            run_name = run_name_base

        if self.args.run_name is not None:
            run_name  = run_name + val_tag

        # If resuming training, use the run name from the checkpoint file or the provided run name
        if self.args.resume:
            run_name = self.args.run_name or os.path.basename(self.args.resume).split('_epoch')[0]
        return run_name

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

    