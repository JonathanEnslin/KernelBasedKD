import argparse

class ParameterPrinter:
    def __init__(self, args, params, **kwargs):
        self.args = vars(args)
        self.params = params
        self.extra_args = kwargs
    
    def print_table(self, title, data):
        if not data:
            return

        # Calculate column widths
        col_widths = [max(len(str(key)), max(len(str(item[key])) for item in data)) for key in data[0]]
        total_width = sum(col_widths) + 3 * len(col_widths) + 1
        
        # Print title
        print(f"+{'-' * (total_width - 2)}+")
        print(f"| {title.center(total_width - 4)} |")
        print(f"+{'-' * (total_width - 2)}+")
        
        # Print header
        header = "| " + " | ".join(key.center(col_widths[i]) for i, key in enumerate(data[0].keys())) + " |"
        print(header)
        print(f"+{'-' * (total_width - 2)}+")
        
        # Print rows
        for item in data:
            row = "| " + " | ".join(str(item[key]).center(col_widths[i]) for i, key in enumerate(item.keys())) + " |"
            print(row)
        
        # Print footer
        print(f"+{'-' * (total_width - 2)}+")
    
    def print_args(self):
        args_data = [{'Parameter': key, 'Value': val} for key, val in self.args.items()]
        self.print_table('Program Args', args_data)
    
    def print_params(self):
        params_data = []
        
        for key, value in self.params.items():
            if key == 'optimizer':
                optimizer_type = value['type']
                optimizer_params = value['parameters']
                for param, val in optimizer_params.items():
                    params_data.append({'Category': 'Optimizer', 'Type': optimizer_type, 'Parameter': param, 'Value': val})
            elif key == 'schedulers':
                for scheduler in value:
                    scheduler_type = scheduler['type']
                    scheduler_params = scheduler['parameters']
                    for param, val in scheduler_params.items():
                        params_data.append({'Category': 'Scheduler', 'Type': scheduler_type, 'Parameter': param, 'Value': val})
            elif key == 'training':
                training_params = value
                for param, val in training_params.items():
                    params_data.append({'Category': 'Training', 'Type': 'N/A', 'Parameter': param, 'Value': val})
            elif key == 'notes':
                pass
        
        self.print_table('Hyperparameters', params_data)
    
    def print_extra_args(self):
        extra_args_data = [{'Parameter': key, 'Value': val} for key, val in self.extra_args.items()]
        self.print_table('Other Options', extra_args_data)
    
    def print_all(self):
        self.print_args()
        print()  # Add space between tables
        self.print_params()
        if self.extra_args:
            print()  # Add space between tables
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

    printer = ParameterPrinter(args, params, additional_param1='value1', additional_param2='value2')
    printer.print_all()
