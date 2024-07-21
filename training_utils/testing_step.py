import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from utils.log_utils import create_log_entry, log_to_csv

class TestStep:
    def __init__(self, model, testloader, criterion, device, writer, csv_file, start_time):
        self.model = model
        self.testloader = testloader
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.csv_file = csv_file
        self.start_time = start_time

    def __call__(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        top5_correct = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                if torch.cuda.is_available():
                    with amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                # Calculate top-5 error
                top5 = outputs.topk(5, dim=1)[1]
                top5_correct += sum([1 if labels[i] in top5[i] else 0 for i in range(len(labels))])

        epoch_loss = running_loss / len(self.testloader)
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        top5_error = 100 * (1 - top5_correct / len(all_labels))

        self.writer.add_scalar('test_loss', epoch_loss, epoch)
        self.writer.add_scalar('test_accuracy', accuracy, epoch)
        self.writer.add_scalar('test_f1_score', f1, epoch)
        self.writer.add_scalar('test_top5_error', top5_error, epoch)

        # print the test accuracy
        print(f'--> Test accuracy: {accuracy:.2f}%')

        log_entry = create_log_entry(epoch, 'test', epoch_loss, accuracy, f1, self.start_time, self.device, top5_error)
        log_to_csv(self.csv_file, log_entry)
