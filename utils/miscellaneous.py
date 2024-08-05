import os

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
                logger(f"Error: {directory} could not be created. {e}")
                return False
    return True
