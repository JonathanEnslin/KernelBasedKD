import os
import torch
import time

class TeacherModelHandler:
    """
    A handler class for loading a teacher model, generating and saving logits and feature maps.
    """
    def __init__(self, model_class, teacher_file_name, device, num_classes=100, printer=print, base_folder='teacher_models'):
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
        self.printer = printer

        self.base_folder = base_folder
        self.teacher_folder = os.path.join(self.base_folder, 'models')
        self.logits_folder = os.path.join(self.base_folder, 'logits')
        self.feature_maps_folder = os.path.join(self.base_folder, 'feature_maps')

        self.model_class = model_class
        self.teacher_file_name = teacher_file_name
        self.device = device
        self.num_classes = num_classes

        self._init_paths()
        self._create_directories()
        self.teacher = model_class(num_classes=num_classes)


    def load_teacher_model(self):
        """
        Loads the teacher model from the specified path.
        
        Returns:
            The loaded teacher model, or None if loading failed.
        """
        self.printer(f"Loading teacher model from {self.teacher_path}")
        try:
            self.teacher.load(self.teacher_path, device=self.device)
            self.teacher.to(self.device)
            self.printer("Teacher model loaded")
            return self.teacher
        except Exception as e:
            self.printer(f"Failed to load teacher model: {e}")
            return None


    def generate_and_save_teacher_logits_and_feature_maps(self, trainloader, train_dataset):
        """
        Generates and saves teacher logits and feature maps if they do not already exist.
        
        Args:
            trainloader: DataLoader for the training data.
            train_dataset: The training dataset.
        
        Returns:
            A tuple containing the teacher logits, pre-activation feature maps, and post-activation feature maps.
        """
        if self._check_existing_logits_and_feature_maps():
            self._load_existing_logits_and_feature_maps()
        else:
            self._generate_and_save_logits_and_feature_maps(trainloader, train_dataset)
        
        return self.teacher_logits, self.teacher_pre_activation_feature_maps, self.teacher_layer_group_output_feature_maps


    def print_paths(self):
        """
        Prints the paths for the teacher model, logits, and feature maps.
        """
        self.printer(f"Teacher path: {self.teacher_path}")
        self.printer(f"Logits path: {self.logits_path}")
        self.printer(f"Pre-activation feature maps path: {self.pre_activation_feature_maps_path}")
        self.printer(f"Layer group output feature maps path: {self.layer_group_output_feature_maps_path}")


    def _init_paths(self):
        """
        Initializes the file paths for the teacher model, logits, and feature maps.
        """
        self.teacher_subfolder = self.model_class.__name__
        self.teacher_path = os.path.join(self.teacher_folder, self.teacher_subfolder, self.teacher_file_name)

        self.logits_subfolder = self.teacher_subfolder
        self.logits_path = os.path.join(self.logits_folder, self.logits_subfolder, f'{self.teacher_file_name}.pt')

        self.feature_maps_subfolder = self.teacher_subfolder
        self.pre_activation_feature_maps_path = os.path.join(self.feature_maps_folder, self.feature_maps_subfolder, f'pre_activation_{self.teacher_file_name}.pt')
        self.layer_group_output_feature_maps_path = os.path.join(self.feature_maps_folder, self.feature_maps_subfolder, f'post_activation_{self.teacher_file_name}.pt')


    def _create_directories(self):
        """
        Creates the necessary directories for storing logits and feature maps.
        """
        os.makedirs(os.path.join(self.logits_folder, self.logits_subfolder), exist_ok=True)
        os.makedirs(os.path.join(self.feature_maps_folder, self.feature_maps_subfolder), exist_ok=True)


    def _check_existing_logits_and_feature_maps(self):
        """
        Checks if the logits and feature maps files already exist.
        
        Returns:
            True if all files exist, False otherwise.
        """
        return os.path.exists(self.logits_path) and os.path.exists(self.pre_activation_feature_maps_path) and os.path.exists(self.layer_group_output_feature_maps_path)


    def _load_existing_logits_and_feature_maps(self):
        """
        Loads the existing logits and feature maps from files.
        """
        self.printer("Teacher logits and feature map files already exist. Loading...")
        self.teacher_logits, self.teacher_labels = torch.load(self.logits_path)
        self.teacher_logits.to(self.device)
        self.teacher_pre_activation_feature_maps = torch.load(self.pre_activation_feature_maps_path)
        self.teacher_layer_group_output_feature_maps = torch.load(self.layer_group_output_feature_maps_path)
        self.printer("Teacher logits and feature maps loaded")


    def _generate_and_save_logits_and_feature_maps(self, trainloader, train_dataset):
        """
        Generates and saves the teacher logits and feature maps.
        
        Args:
            trainloader: DataLoader for the training data.
            train_dataset: The training dataset.
        """
        self.printer("Generating teacher logits and feature maps...")
        self.teacher.set_hook_device_state("cpu")
        self.teacher.eval()
        teacher_logits, teacher_labels, teacher_pre_activation_feature_maps, teacher_layer_group_output_feature_maps = self._initialize_empty_tensors(len(train_dataset))
        
        start_time = time.time()
        
        for i, (inputs, labels, indices) in enumerate(trainloader):
            if i % 5 == 0 or i == len(trainloader)-1:
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (i + 1)) * (len(trainloader) - (i + 1))
                self.printer(f"Batch {i+1}/{len(trainloader)}, Estimated time remaining: {remaining_time:.2f} seconds")
            
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            with torch.no_grad():
                outputs = self.teacher(inputs)
                pre_activation_fmaps_for_batch = self.teacher.get_layer_group_preactivation_feature_maps()
                layer_group_output_fmaps_for_batch = self.teacher.get_layer_group_output_feature_maps()

            self._store_batch_results(indices, outputs, labels, pre_activation_fmaps_for_batch, layer_group_output_fmaps_for_batch, teacher_logits, teacher_labels, teacher_pre_activation_feature_maps, teacher_layer_group_output_feature_maps)

        self._save_results(teacher_logits, teacher_labels, teacher_pre_activation_feature_maps, teacher_layer_group_output_feature_maps)
        self.printer("Teacher logits and feature maps generated and saved")


    def _initialize_empty_tensors(self, length):
        """
        Initializes empty lists for storing logits and feature maps.
        
        Args:
            length: The length of the dataset.
        
        Returns:
            Four empty lists for logits, labels, pre-activation feature maps, and post-activation feature maps.
        """
        return ([None] * length, [None] * length, [None] * length, [None] * length)


    def _store_batch_results(self, indices, outputs, labels, pre_activation_fmaps_for_batch, layer_group_output_fmaps_for_batch,
                              teacher_logits, teacher_labels, teacher_pre_activation_feature_maps, teacher_layer_group_output_feature_maps):
        """
        Stores the results of a batch into the corresponding lists.
        
        Args:
            indices: The indices of the current batch.
            outputs: The model outputs for the current batch.
            labels: The labels for the current batch.
            pre_activation_fmaps_for_batch: The pre-activation feature maps for the current batch.
            layer_group_output_fmaps_for_batch: The post-activation feature maps for the current batch.
            teacher_logits: The list to store the logits.
            teacher_labels: The list to store the labels.
            teacher_pre_activation_feature_maps: The list to store the pre-activation feature maps.
            teacher_layer_group_output_feature_maps: The list to store the post-activation feature maps.
        """
        for j, idx in enumerate(indices):
            teacher_logits[idx] = outputs[j]
            teacher_labels[idx] = labels[j]
            teacher_pre_activation_feature_maps[idx] = [fmap[j] for fmap in pre_activation_fmaps_for_batch]
            teacher_layer_group_output_feature_maps[idx] = [fmap[j] for fmap in layer_group_output_fmaps_for_batch]


    def _save_results(self, teacher_logits, teacher_labels, teacher_pre_activation_feature_maps, teacher_layer_group_output_feature_maps):
        """
        Saves the generated logits and feature maps to files.
        
        Args:
            teacher_logits: The logits to save.
            teacher_labels: The labels to save.
            teacher_pre_activation_feature_maps: The pre-activation feature maps to save.
            teacher_layer_group_output_feature_maps: The post-activation feature maps to save.
        """
        teacher_logits = torch.stack(teacher_logits)
        teacher_logits.to(self.device)
        torch.save((teacher_logits, teacher_labels), self.logits_path)
        torch.save(teacher_pre_activation_feature_maps, self.pre_activation_feature_maps_path)
        torch.save(teacher_layer_group_output_feature_maps, self.layer_group_output_feature_maps_path)
