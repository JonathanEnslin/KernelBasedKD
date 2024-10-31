import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
from datetime import datetime
import numpy as np
import random

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
import utils.config_printer
import utils.miscellaneous
from utils.param_utils import load_params
from utils.model_utils import initialize_model, get_optimizer, get_schedulers
from training_utils.early_stopping import EarlyStopping
from utils.best_model_tracker import BestModelTracker
from utils.data.dataset_getters import get_dataset_info
import utils.config_utils as config_utils

from training_utils.training_step import TrainStep, LossHandler
from training_utils.validation_step import ValidationStep
from training_utils.testing_step import TestStep

import loss_functions as lf
from models.resnet import resnet56
from utils.teacher.teacher_model_handler import TeacherModelHandler
from utils.amp_grad_scaling_handler import get_amp_and_grad_scaler
import utils.miscellaneous as misc
from utils.logger import Logger
from utils.distillation.distillation_config import *
import steppers.steppers as steppers
import utils
from models.FT.encoders import Paraphraser, Translator

import args as program_args

def fmt_duration(duration):
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    ms = (duration - int(duration)) * 1000
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s {int(ms)}ms"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    provisional_kd_set = args.kd_set
    provisional_teacher_type = args.teacher_type

    # Initialise logger
    logger = Logger(args,
                    log_to_file=True,
                    data_dir=args.output_data_dir,
                    run_tag=args.run_tag if args.run_tag != '' else None, 
                    teacher_type=provisional_teacher_type, 
                    kd_set=provisional_kd_set)
    
    # Get the run name
    run_name = logger.get_run_name()
    logger(f"===> Using run name: {run_name}", col='cyan')
    
    # if kd_params is provided then kd_set must be provided and vice verse, if both are provided then teacher should be provided as well
    if (args.kd_params is not None and args.kd_set is None) or (args.kd_set is not None and args.kd_params is None):
        logger("===> If kd_params is provided then kd_set must be provided and vice verse. Exiting.", col='red')
        exit(1)

    if (args.kd_params is not None and args.teacher_path is None) or (args.teacher_path is not None and args.kd_params is None):
        logger("===> If kd_params is provided then teacher_path must be provided and vice verse. Exiting.", col='red')
        exit(1)

    if (args.teacher_path is not None and args.teacher_type is None) or (args.teacher_type is not None and args.teacher_path is None):
        logger("===> If teacher_path is provided then teacher_type must be provided and vice verse. Exiting.", col='red')
        exit(1)

    # Set the seed
    # set_seed(112)
    # 
    
    # Load the training parameters
    params = load_params(args.params, args.param_set)

    # Enesure the necessary dirs exist
    if not utils.miscellaneous.ensure_dir_existence([args.model_save_dir, args.checkpoint_dir, args.output_data_dir], logger=logger):
        exit(1)

    # Set the device
    requested_device = args.device
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    logger(device)

    # Get gradscaler and autocast context class
    scaler, autocast = get_amp_and_grad_scaler(args, device, logger=logger)

    # if validation set is not used, print that track best and early stopping are disabled (if they are specified)
    if not misc.check_validation_args(args, logger=logger):
        logger("===> Validation arguments are not correctly specified. Exiting.", col='red')
        exit(1)


    config_printer = utils.config_printer.ConfigPrinter(args, params, logger, run_name=run_name, actual_device=device)
    config_printer.print_all()
    logger("")

    # Get the dataset info
    dataset_class, num_classes, transform_train, transform_test = get_dataset_info(args.dataset, logger=logger)
    if dataset_class is None:
        exit(1)

    # Load the dataset
    train_dataset = dataset_class(root=args.dataset_dir, train=True, download=True, transform=transform_train)

    trainloader, valloader, testloader, val_split_random_state \
            = config_utils.get_data_loaders(args, params, train_dataset, run_name, transform_train, transform_test, dataset_class, logger=logger)

    optimizers = []

    model_init_special_kwargs = {}
    if args.dataset == 'TinyImageNet':
        # Need to change some layers for compatibility with TinyImageNet (cifar 32x32 -> 64x64 tinyimagenet)
        model_init_special_kwargs = {
            'conv1stride': 2,
            'conv1ksize': 5,
            'conv1padding': 2
        }

    # Initialize the nn model
    model = initialize_model(args.model_name, num_classes=num_classes, device=device, logger=logger, **model_init_special_kwargs)
    if model is None:
        exit(1)

    kd_criterion = None
    vanilla_criterion = None
    gamma, alpha, beta = 1.0, 0.0, 0.0
    teacher=None
    kd_mode = 'base'
    if args.teacher_path is not None:
        teacher_model = initialize_model(args.teacher_type, num_classes=num_classes, device=device, logger=logger, **model_init_special_kwargs)
        # Load and cache (if specified) teacher model data
        logger("Setting up teacher model")
        teacher_model_handler = TeacherModelHandler(teacher_model=teacher_model,
                                                    teacher_type=args.teacher_type,
                                                    teacher_file_name=args.teacher_path,
                                                    device=device,
                                                    num_classes=num_classes,
                                                    printer=print)
        
        teacher = teacher_model_handler.load_teacher_model()
        if teacher is None:
            exit(1)

        # Default to None, will be set to the logits and feature maps if using cached teacher
        teacher_logits = None
        teacher_layer_groups_preactivation_fmaps = None
        teacher_layer_groups_post_activation_fmaps = None   
        if args.use_cached_logits or args.use_cached_feature_maps:
            teacher_logits, teacher_layer_groups_preactivation_fmaps, teacher_layer_groups_post_activation_fmaps = \
                teacher_model_handler.generate_and_save_teacher_logits_and_feature_maps(trainloader, train_dataset, args.use_cached_logits, args.use_cached_feature_maps)        
            
        del teacher_model_handler # to release references so that memory can be cleared up later
        logger("Teacher model setup completed.")
        logger("")

        # Load the distillation params
        distillation_params = get_distillation_params(args.kd_params, args.kd_set, logger=logger)
        gamma = distillation_params.get("gamma", gamma)
        alpha = distillation_params.get("alpha", alpha)
        beta = distillation_params.get("beta", beta)
        distillation_type = None
        if "distillation_type" in distillation_params:
            distillation_type = distillation_params["distillation_type"]
            if "beta" not in distillation_params:
                logger("Warning: a distillation function was specified in the kd params, but no beta was provided", col='yellow')
            
            paraphraser = None
            translator = None
            if distillation_type == "ft" or distillation_type == 'kft' or distillation_type == 'filter_ft':
                paraphraser_rate = distillation_params['k']
                paraphraser_use_bn = distillation_params['use_bn']
                # check if a paraphraser path is provided
                if args.paraphraser_path is None:
                    logger("Error: a paraphraser path must be provided when using factor transfer", col='red')
                    exit(1)
                paraphrser_path = args.paraphraser_path
                paraphraser = Paraphraser(
                    [None, teacher.all_conv_layers[-1].out_channels, None, None] if distillation_type == 'ft' else teacher.get_kernel_weights_subset([teacher.group3indices[-1]])[-1].shape, 
                    k=paraphraser_rate, 
                    use_bn=paraphraser_use_bn
                ).to(device)
                paraphraser.load(paraphrser_path, device=device.type)
                paraphraser = paraphraser.to(device)
                paraphraser.set_hook_device_state('same')
                paraphraser.eval()

                translator  = Translator(
                    [None, model.all_conv_layers[-1].out_channels, None, None] if distillation_type == 'ft' else model.get_kernel_weights_subset([model.group3indices[-1]])[-1].shape, 
                    [None, teacher.all_conv_layers[-1].out_channels, None, None] if distillation_type == 'ft' else teacher.get_kernel_weights_subset([teacher.group3indices[-1]])[-1].shape, 
                    k=paraphraser_rate, 
                    use_bn=paraphraser_use_bn
                ).to(device)
                translator.set_hook_device_state('same')
                translator.train()
                
                optimiser_factor = optim.SGD(translator.parameters(), 
                                             lr=distillation_params['params']['optimizer']['lr'],
                                             momentum=distillation_params['params']['optimizer']['momentum'], 
                                             weight_decay=distillation_params['params']['optimizer']['weight_decay'])
                
                optimizers.append(optimiser_factor)

            kd_criterion = get_loss_function(distillation_params, 
                                                logger=logger, 
                                                cached_logits=teacher_logits, 
                                                cached_pre_activation_fmaps=teacher_layer_groups_preactivation_fmaps, 
                                                cached_post_activation_fmaps=teacher_layer_groups_post_activation_fmaps,
                                                device=device,
                                                student=model,
                                                teacher=teacher,
                                                paraphraser=paraphraser,
                                                translator=translator)

        if "vanilla_temperature" in distillation_params:
            if "alpha" not in distillation_params:
                logger("Warning: a vanilla KD temperature was specified in the kd params, but no alpha value was provided", col='yellow')
            vanilla_criterion = lf.vanilla.VanillaKDLoss(temperature=distillation_params["vanilla_temperature"], cached_teacher_logits=teacher_logits)
        config_printer.print_dict(distillation_params, "KD Params")
        teacher.set_hook_device_state(args.device if torch.cuda.is_available() else "cpu")


    # Set up loss functions
    criterion = nn.CrossEntropyLoss()
    test_val_criterion = nn.CrossEntropyLoss()    

    loss_handler = LossHandler(gamma, alpha, beta,
                                criterion,
                                teacher_model=teacher,
                                vanilla_criterion=vanilla_criterion,
                                kd_criterion=kd_criterion)
    
    if args.batch_stepper is not None:
        stepper_getter = steppers.stepper_dict[args.batch_stepper]
        stepper_args = {}
        if args.batch_stepper_args is not None:
            stepper_args = steppers.parse_kwargs(args.batch_stepper_args)
        stepper = stepper_getter(loss_handler, **stepper_args)
        loss_handler.set_batch_step_fn(stepper)

    # loss_handler.add_eval_criterion(lf.attention_transfer.ATLoss(model, teacher, mode='impl'))
    if teacher is not None:
        loss_handler.add_eval_criterion("fAT", lf.filter_at.FilterAttentionTransfer(model, teacher))

    # Restore hook states to their ideal values
    model.set_hook_device_state(args.device if torch.cuda.is_available() else "cpu")

    # Define the optimizer and learning rate scheduler
    optimizer = get_optimizer(params, model)
    # prepend the optimizer to the list of optimizers
    optimizers.insert(0, optimizer)
    schedulers = get_schedulers(params, optimizer)

    # Initialize TensorBoard writer
    writer_name = config_utils.get_writer_name(kd_mode=kd_mode, args=args, run_name=run_name)
    writer = SummaryWriter(writer_name)

    # Define the start time for the logger
    start_time = datetime.now()

    # Initialize EarlyStopping if validation is used and early stopping params are passed
    early_stopping = None
    if args.use_val and (args.early_stopping_patience is not None or args.early_stopping_start_epoch is not None):
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience or 10, 
            verbose=True, 
            enabled_after_epoch=args.early_stopping_start_epoch or 10,
            monitor='loss',
            logger=logger
        )

    # Initialize BestModelTracker if validation is used
    best_model_tracker = None
    if args.use_val:
        best_model_tracker = BestModelTracker(
            verbose=True,
            delta=0,
            path=os.path.join(args.model_save_dir, f"{run_name}_best.pth"),
            monitor='loss', # Only loss is currently supported
            enabled_after_epoch=args.track_best_after_epoch or 10,
            logger=logger
        )


    num_epochs = params['training']['max_epochs']
    start_epoch = 0

    if args.resume:
        raise NotImplementedError("Resume is not implemented yet (or currently outdated and likely broken)")
        # Need to check checkpointing for all manually implemented modules and also change it to the list of optimizers
        start_epoch = load_checkpoint(model, optimizer, schedulers, scaler, filename=args.resume, logger=logger)


    train_step = TrainStep(model, trainloader, optimizers, scaler, schedulers, device, writer, start_time, autocast, loss_handler=loss_handler, logger=logger)
    if args.use_val:
        validation_step = ValidationStep(model, valloader, test_val_criterion, device, writer, start_time, autocast, early_stopping, best_model_tracker, logger=logger)
    if not args.use_val or not args.disable_test:
        test_step = TestStep(model, testloader, test_val_criterion, device, writer, start_time, autocast, logger=logger)

    logger('Starting training...', col='green')
    # Main training loop
    times_at_epoch_end = []
    start_time = datetime.now()
    prev_time = start_time
    val_accuracy = None
    test_accuracy = None
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = datetime.now()
        train_step(epoch)
        if args.use_val and (epoch >= args.disable_val_until_epoch or epoch % 10 == 0 or epoch == num_epochs - 1):
            early_stop, val_accuracy = validation_step(epoch)
            if early_stop:
                break  # Early stopping triggered, exit training loop
        if not args.use_val or not args.disable_test:
            if epoch >= args.disable_test_until_epoch or epoch % 10 == 0 or epoch == num_epochs - 1:
                test_accuracy = test_step(epoch)

        curr_time = datetime.now()
        times_at_epoch_end.append(curr_time - prev_time)
        avg_time_per_epoch = sum([dur.total_seconds() for dur in times_at_epoch_end[-12:]]) / len(times_at_epoch_end[-12:])
        prev_time = curr_time

        logger(f"Epoch {epoch} took {fmt_duration((curr_time - epoch_start_time).total_seconds())}, Total time: {fmt_duration((curr_time - start_time).total_seconds())}")
        logger(f"Average time per epoch: {fmt_duration(avg_time_per_epoch)}")
        logger(f"Estimated time remaining: {fmt_duration(avg_time_per_epoch * (num_epochs - epoch - 1))}")
        if (epoch) % args.checkpoint_freq == 0 and epoch != 0:
            checkpoint_filename = os.path.join(args.checkpoint_dir, f"{run_name}_epoch{epoch}.pth.tar")
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'schedulers': [scheduler.state_dict() for scheduler in schedulers],
                'scaler': scaler.state_dict(),
                'val_split_random_state': val_split_random_state,
            }, is_best=False, filename=checkpoint_filename, logger=logger)
        
        if epoch == start_epoch:
            # remove virst entry from epoch times as it is usually an outlier
            times_at_epoch_end = times_at_epoch_end[1:]

    model.save(f'./{args.model_save_dir}/{run_name}.pth')
    writer.close()
    return test_accuracy, val_accuracy
        

# Example usage, not necessarily implementatin that is used, please see train.py/kd_train.py (whichever one exists) for the actual usage
if __name__ == "__main__":
    parser = program_args.get_arg_parser()
    args = parser.parse_args()
    ta, va = main(args)
    if ta is not None:
        print(f"Final Test accuracy: {ta}")
    if va is not None:
        print(f"Final Validation accuracy: {va}")




