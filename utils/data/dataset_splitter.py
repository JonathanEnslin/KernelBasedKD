import numpy as np
import os
import json
import utils.data.dataset_utils as dataset_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

def stratified_split(labels, test_size=0.2, random_state=None, save_to_file=None):
    """
    Perform a stratified split of the dataset.
    
    :param labels: Array of labels for stratification.
    :param test_size: Proportion or absolute number of the test set size.
    :param random_state: Seed for reproducibility. If None, a random seed is generated.
    :return: Train and test indices.
    :save_to_file: Filename to save the indices. If None, the indices are not saved.
    """
    if random_state is None:
        # generate a random seed
        random_state = np.random.randint(0, np.iinfo(np.int32).max)
    
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    if save_to_file is not None:
        save_indices(train_indices, test_indices, random_state, save_to_file)
    
    return train_indices, test_indices, random_state


def random_split(size, test_size=0.2, random_state=None, save_to_file=None):
    """
    Perform a random split of the dataset.
    
    :param size: Size of the dataset.
    :param test_size: Proportion or absolute number of the test set size.
    :param random_state: Seed for reproducibility.
    :return: Train and test indices.
    :save_to_file: Filename to save the indices. If None, the indices are not saved.
    """
    if random_state is None:
        # generate a random seed
        random_state = np.random.randint(0, np.iinfo(np.int32).max)
    
    train_indices, test_indices = train_test_split(
        np.arange(size),
        test_size=test_size,
        random_state=random_state
    )

    if save_to_file is not None:
        save_indices(train_indices, test_indices, random_state, save_to_file)
    
    return train_indices, test_indices, random_state


def save_indices(train_indices, test_indices, random_state, filename, verbose=True):
    """
    Save train and test indices to a file.
    
    :param train_indices: Array of training indices.
    :param test_indices: Array of testing indices.
    :param filename: Filename to save the indices.
    """
    indices = {
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist()
    }

    if random_state is not None:
        indices['random_state'] = random_state

    if verbose:
        print(f"Saving indices to '{filename}'")

    with open(filename, 'w') as f:
        json.dump(indices, f)

    if verbose:
        print("Indices saved successfully")


def load_indices(filename):
    """
    Load train and test indices from a file.
    
    :param filename: Filename to load the indices from.
    :return: Train and test indices.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: '{filename}'")
    
    with open(filename, 'r') as f:
        indices = json.load(f)
    
    train_indices = np.array(indices['train_indices'])
    test_indices = np.array(indices['test_indices'])
    random_state = indices.get('random_state', None)
    
    return train_indices, test_indices, random_state


def split_dataset(dataset, test_size, stratify=False, random_state=None, save_to_file=None):
    """
    Split a dataset into training and testing sets.
    
    :param dataset: Dataset to split.
    :param test_size: Proportion or absolute number of the test set size.
    :param stratify: Whether to stratify the split.
    :param random_state: Seed for reproducibility.
    :param save_to_file: Filename to save the indices.
    :return: Train and test datasets.
    """
    if stratify:
        labels = dataset_utils.get_targets(dataset)
        train_indices, test_indices, random_state = stratified_split(labels, test_size, random_state, save_to_file)
    else:
        size = len(dataset)
        train_indices, test_indices, random_state = random_split(size, test_size, random_state, save_to_file)
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset


def split_dataset_from_indices(dataset, train_indices, test_indices):
    """
    Split a dataset into training and testing sets using provided indices.
    
    :param dataset: Dataset to split.
    :param train_indices: Array of training indices.
    :param test_indices: Array of testing indices.
    :return: Train and test datasets.
    """
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset


def split_dataset_from_file(dataset, filename):
    """
    Split a dataset into training and testing sets using indices loaded from a file.
    
    :param dataset: Dataset to split.
    :param filename: Filename to load the indices from.
    :return: Train and test datasets.
    """
    train_indices, test_indices, _ = load_indices(filename)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset