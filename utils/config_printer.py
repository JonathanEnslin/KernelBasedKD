import argparse
from utils.logger import Logger

class ConfigPrinter:
    def __init__(self, args, params, logger: Logger, **kwargs):
        self.args = vars(args)
        self.params = params
        self.extra_args = kwargs
        self.logger = logger

    def print_box(self, title, content):
        lines = content.strip().split('\n')
        width = max(len(line) for line in lines)
        total_width = max(width + 4, len(title) + 6)
        self.logger(f"+{'-' * total_width}+")
        self.logger(f"|  {title.center(total_width - 4)}  |")
        self.logger(f"+{'-' * total_width}+")
        for line in lines:
            self.logger(f"|  {line.ljust(total_width - 4)}  |")
        self.logger(f"+{'-' * total_width}+")
    
    def dict_to_string(self, d, indent=0, top_level=True, sep_pars=True):
        lines = []
        first_key = True
        for key, value in d.items():
            if key == 'notes':
                continue
            if not first_key and top_level and sep_pars:
                lines.append('')
            first_key = False

            if isinstance(value, dict):
                lines.append(' ' * indent + f"{key}:")
                lines.extend(self.dict_to_string(value, indent + 2, top_level=False).split('\n'))
            elif isinstance(value, list):
                if all(isinstance(item, (int, float, str)) for item in value):
                    lines.append(' ' * indent + f"{key}: {value}")
                else:
                    lines.append(' ' * indent + f"{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            lines.extend(self.dict_to_string(item, indent + 2, top_level=False).split('\n'))
                        else:
                            lines.append(' ' * (indent + 2) + str(item))
            else:
                lines.append(' ' * indent + f"{key}: {value}")
        return '\n'.join(lines)

    def print_args(self):
        content = self.dict_to_string(self.args, indent=0, sep_pars=False)
        self.print_box("Command-Line Arguments", content)
    
    def print_params(self):
        content = self.dict_to_string(self.params, indent=0)
        self.print_box("Hyperparameters", content)
    
    def print_extra_args(self):
        content = self.dict_to_string(self.extra_args, indent=0, sep_pars=False)
        self.print_box("Other", content)
    
    def print_all(self):
        self.print_args()
        self.logger()  # Add space between boxes
        self.print_params()
        if self.extra_args:
            self.logger()  # Add space between boxes
            self.print_extra_args()


# Example usage with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example of argparse.')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    params = {
        "notes": {
            "source": "Enhancing batch normalized convolutional networks using displaced rectifier linear units: A systematic comparative study",
            "url": "http://dx.doi.org/10.1016/j.eswa.2019.01.066",
            "notes": "found to be inferior to other configurations in this file, and especially ineffective for certain kd methods"
        },
        "optimizer": {
            "type": "SGD",
            "parameters": {
                "lr": 0.1,
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 0.0005
            }
        },
        "schedulers": [
            {
                "type": "MultiStepLR",
                "parameters": {
                    "gamma": 0.2,
                    "milestones": [60, 80, 90]
                }
            },
            {
                "type": "CosineAnnealingLR",
                "parameters": {
                    "T_max": 50,
                    "eta_min": 0
                }
            }
        ],
        "training": {
            "max_epochs": 110,
            "batch_size": 128
        }
    }

    printer = ConfigPrinter(args, params, additional_param1='value1', additional_param2='value2', whatlol=4, squawk='squawk')
    printer.print_all()