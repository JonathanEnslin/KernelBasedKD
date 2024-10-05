import os
import torch
import time
from models.base_model import BaseModel

class TeacherModelHandler:
    """
    A handler class for loading a teacher model, generating and saving logits and feature maps.
    """
    def __init__(self, teacher_model, teacher_type, teacher_file_name, device, num_classes=100, base_folder='teacher_models', printer=print):
        """
        Initializes the TeacherModelHandler.
        
        Args:
            model_class: The class of the teacher model.
            teacher_file_name: The filename of the saved teacher model.
            device: The device to run the model on (e.g., 'cpu' or 'cuda').
            num_classes: Number of classes for the model.
            printer: A function for printing messages (default is print).
            base_folder: The base folder for storing model and output files.
        """
        self.logger = printer

        self.base_folder = base_folder
        self.teacher_folder = os.path.join(self.base_folder, 'models')
        self.logits_folder = os.path.join(self.base_folder, 'cache', 'logits')
        self.feature_maps_folder = os.path.join(self.base_folder, 'cache', 'feature_maps')

        self.teacher_type = teacher_type
        self.teacher_file_name = teacher_file_name
        self.device = device
        self.num_classes = num_classes

        self._init_paths()
        self._create_directories()
        self.teacher: BaseModel = teacher_model


    def load_teacher_model(self):
        """
        Loads the teacher model from the specified path.
        
        Returns:
            The loaded teacher model, or None if loading failed.
        """
        self.logger(f"Loading teacher model from {self.teacher_path}")
        try:
            self.teacher.load(self.teacher_path, device=self.device)
            self.teacher.to(self.device)
            self.logger("Teacher model loaded")
            return self.teacher
        except Exception as e:
            self.logger(f"Failed to load teacher model: {e}")
            return None


    def generate_and_save_teacher_logits_and_feature_maps(self, trainloader, train_dataset, generate_logits=True, generate_feature_maps=True):
        """
        Generates and saves teacher logits and feature maps if they do not already exist.
        
        Args:
            trainloader: DataLoader for the training data.
            train_dataset: The training dataset.
            generate_logits: Boolean indicating whether to save logits.
            generate_feature_maps_maps: Boolean indicating whether to save feature maps.
        
        Returns:
            A tuple containing the teacher logits, pre-activation feature maps, and post-activation feature maps.
        """
        if not generate_logits and not generate_feature_maps:
            return None, None, None

        if self._check_existing_logits_and_feature_maps(generate_logits, generate_feature_maps):
            self._load_existing_logits_and_feature_maps(generate_logits, generate_feature_maps)
        else:
            self._generate_and_save_logits_and_feature_maps(trainloader, train_dataset, generate_logits, generate_feature_maps)
        
        return self.teacher_logits, self.teacher_pre_activation_fmaps, self.teacher_post_activation_fmaps


    def print_paths(self):
        """
        Prints the paths for the teacher model, logits, and feature maps.
        """
        self.logger(f"Teacher path: {self.teacher_path}")
        self.logger(f"Logits path: {self.logits_path}")
        self.logger(f"Pre-activation feature maps path: {self.pre_activation_feature_maps_path}")
        self.logger(f"Post-activation feature maps path: {self.post_activation_fmaps_path}")


    def _init_paths(self):
        """
        Initializes the file paths for the teacher model, logits, and feature maps.
        """
        self.teacher_subfolder = self.teacher_type
        self.teacher_path = os.path.join(self.teacher_folder, self.teacher_subfolder, self.teacher_file_name)

        self.logits_subfolder = self.teacher_subfolder
        self.logits_path = os.path.join(self.logits_folder, self.logits_subfolder, f'{self.teacher_file_name}.pt')

        self.feature_maps_subfolder = self.teacher_subfolder
        self.pre_activation_feature_maps_path = os.path.join(self.feature_maps_folder, self.feature_maps_subfolder, f'pre_activation_{self.teacher_file_name}.pt')
        self.post_activation_fmaps_path = os.path.join(self.feature_maps_folder, self.feature_maps_subfolder, f'post_activation_{self.teacher_file_name}.pt')


    def _create_directories(self):
        """
        Creates the necessary directories for storing logits and feature maps.
        """
        os.makedirs(os.path.join(self.logits_folder, self.logits_subfolder), exist_ok=True)
        os.makedirs(os.path.join(self.feature_maps_folder, self.feature_maps_subfolder), exist_ok=True)


    def _check_existing_logits_and_feature_maps(self, should_get_logits, should_get_fmaps):
        """
        Checks if the logits and feature maps files already exist.
        
        Returns:
            True if all files exist, False otherwise.
        """
        return (os.path.exists(self.logits_path) or not should_get_logits) and ((os.path.exists(self.pre_activation_feature_maps_path) and os.path.exists(self.post_activation_fmaps_path)) or not should_get_fmaps)


    def _load_existing_logits_and_feature_maps(self, get_logits, get_fmaps):
        """
        Loads the existing logits and feature maps from files.
        """
        self.logger("Teacher logits and feature map files already exist. Loading...")
        if get_logits:
            self.teacher_logits, self.teacher_labels = torch.load(self.logits_path)
            self.teacher_logits.to(self.device)
        else:
            self.teacher_logits = None
            self.teacher_labels = None

        if get_fmaps:
            self.teacher_pre_activation_fmaps = torch.load(self.pre_activation_feature_maps_path)
            self.teacher_post_activation_fmaps = torch.load(self.post_activation_fmaps_path)
        else:
            self.teacher_pre_activation_fmaps = None
            self.teacher_post_activation_fmaps = None
        self.logger("Teacher logits and feature maps loaded")


    def _generate_and_save_logits_and_feature_maps(self, trainloader, train_dataset, generate_logits, generate_feature_maps):
        """
        Generates and saves the teacher logits and feature maps.
        
        Args:
            trainloader: DataLoader for the training data.
            train_dataset: The training dataset.
            save_logits: Boolean indicating whether to save logits.
            save_feature_maps: Boolean indicating whether to save feature maps.
        """
        self.logger("Generating teacher logits and feature maps...")
        initial_hook_device_state = self.teacher.get_hook_device_state()
        self.teacher.set_hook_device_state("cpu")
        self.teacher.eval()
        teacher_logits, teacher_labels, teacher_pre_activation_fmaps, teacher_post_activation_fmaps = self._initialize_empty_tensors(len(train_dataset))
        
        batch_durations = []
        
        for i, (inputs, labels, indices) in enumerate(trainloader):
            if i % 10 == 0 or i == len(trainloader)-1:
                if i == 0:
                    start_time = time.time()
                    self.logger(f"Batch {i+1}/{len(trainloader)}")
                else:
                    elapsed_time = time.time() - start_time
                    remaining_time = (elapsed_time / (i + 1)) * (len(trainloader) - (i + 1))
                    self.logger(f"Batch {i+1}/{len(trainloader)}, Estimated time remaining: {remaining_time:.2f} seconds")
            
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            with torch.no_grad():
                outputs = self.teacher(inputs)
                pre_activation_fmaps_for_batch = self.teacher.get_pre_activation_fmaps(detached=True)
                post_activation_fmaps_for_batch = self.teacher.get_post_activation_fmaps(detached=True)

            self._store_batch_results(indices, outputs, labels, pre_activation_fmaps_for_batch, post_activation_fmaps_for_batch,
                                       teacher_logits, teacher_labels, teacher_pre_activation_fmaps, teacher_post_activation_fmaps,
                                       save_logits=generate_logits, save_fmaps=generate_feature_maps)

        self.teacher_logits = teacher_logits if generate_logits else None
        self.teacher_labels = teacher_labels if generate_logits else None
        self.teacher_pre_activation_fmaps = teacher_pre_activation_fmaps if generate_feature_maps else None
        self.teacher_post_activation_fmaps = teacher_post_activation_fmaps if generate_feature_maps else None
        
        # Save the results based on the provided flags
        if generate_logits:
            self._save_logits(teacher_logits, teacher_labels)
        if generate_feature_maps:
            self._save_feature_maps(teacher_pre_activation_fmaps, teacher_post_activation_fmaps)
        
        self.teacher.set_hook_device_state(initial_hook_device_state)
        self.logger("Teacher logits and feature maps generated")


    def _initialize_empty_tensors(self, length):
        """
        Initializes empty lists for storing logits and feature maps.
        
        Args:
            length: The length of the dataset.
        
        Returns:
            Four empty lists for logits, labels, pre-activation feature maps, and post-activation feature maps.
        """
        return ([None] * length, [None] * length, [None] * length, [None] * length)


    def _store_batch_results(self, indices, outputs, labels, pre_activation_fmaps_for_batch, post_activation_fmaps_for_batch,
                              teacher_logits, teacher_labels, teacher_pre_activation_fmaps, teacher_post_activation_fmaps, save_logits, save_fmaps):
        """
        Stores the results of a batch into the corresponding lists.
        
        Args:
            indices: The indices of the current batch.
            outputs: The model outputs for the current batch.
            labels: The labels for the current batch.
            pre_activation_fmaps_for_batch: The pre-activation feature maps for the current batch.
            post_activation_fmaps_for_batch: The post-activation feature maps for the current batch.
            teacher_logits: The list to store the logits.
            teacher_labels: The list to store the labels.
            teacher_pre_activation_feature_maps: The list to store the pre-activation feature maps.
            teacher_post_activation_fmaps: The list to store the post-activation feature maps.
        """
        for j, idx in enumerate(indices):
            teacher_logits[idx] = outputs[j]
            teacher_labels[idx] = labels[j]
            teacher_pre_activation_fmaps[idx] = [fmap[j] for fmap in pre_activation_fmaps_for_batch]
            teacher_post_activation_fmaps[idx] = [fmap[j] for fmap in post_activation_fmaps_for_batch]


    def _save_logits(self, teacher_logits, teacher_labels):
        """
        Saves the generated logits to a file.
        
        Args:
            teacher_logits: The logits to save.
            teacher_labels: The labels to save.
        """
        teacher_logits = torch.stack(teacher_logits)
        teacher_logits.to(self.device)
        torch.save((teacher_logits, teacher_labels), self.logits_path)


    def _save_feature_maps(self, teacher_pre_activation_fmaps, teacher_post_activation_fmaps):
        """
        Saves the generated feature maps to files.
        
        Args:
            teacher_pre_activation_fmaps: The pre-activation feature maps to save.
            teacher_post_activation_fmaps: The post-activation feature maps to save.
        """
        torch.save(teacher_pre_activation_fmaps, self.pre_activation_feature_maps_path)
        torch.save(teacher_post_activation_fmaps, self.post_activation_fmaps_path)
