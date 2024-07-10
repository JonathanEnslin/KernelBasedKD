import csv
import os
from datetime import datetime

def log_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def create_log_entry(epoch, phase, loss, accuracy, f1_score, start_time):
    current_time = datetime.now()
    elapsed_time = (current_time - start_time).total_seconds()
    return {
        'epoch': epoch,
        'phase': phase,
        'loss': loss,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'datetime': current_time.strftime("%Y-%m-%d %H:%M:%S"),
        'elapsed_time': elapsed_time
    }
