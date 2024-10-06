import pyhopper
import numpy as np


def prune_if_underperforming(shared_accuracies, eval_index, folds, logger, percentile_prune1, percentile_prune2, pruner, **pruner_kwargs):
    if len(shared_accuracies) > 10:  # Allow some evaluations before pruning
        threshold = np.percentile(shared_accuracies[:-eval_index-1], percentile_prune2)
        mean_config_accs = sum(shared_accuracies[-eval_index-1:]) / (eval_index + 1)
        if mean_config_accs < threshold:
            logger(f"Pruning evaluation for Fold {eval_index + 1}, Avg. Acc of Cur Conf {mean_config_accs:.2f}% below {percentile_prune2}th percentile of {threshold}")
            # append folds - eval_index - 1 to the shared_accuracies list to keep the length consistent and skewed results because of pruning
            shared_accuracies.extend([mean_config_accs] * (len(folds) - eval_index - 1))
            return True


    if len(shared_accuracies) > 10 and eval_index != 0:  # Allow some evaluations before pruning
        threshold = np.percentile(shared_accuracies[:-eval_index-1], percentile_prune1)
        mean_config_accs = sum(shared_accuracies[-eval_index-1:]) / (eval_index + 1)
        if mean_config_accs < threshold:
            logger(f"Pruning evaluation for Fold {eval_index + 1}, Avg. Acc of Cur Conf {mean_config_accs:.2f}% below {percentile_prune1}th percentile of {threshold}")
            # append folds - eval_index - 1 to the shared_accuracies list to keep the length consistent and skewed results because of pruning
            shared_accuracies.extend([mean_config_accs] * (len(folds) - eval_index - 1))
            return True
        
    return False