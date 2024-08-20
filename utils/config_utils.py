import os
from torch.utils.data import DataLoader, random_split
import utils.data.dataset_splitter as dataset_splitter

def get_run_name(args, kd_type=None, run_tag=None):
    # Generate or use provided run name
    val_tag = ".val" if args.use_val else ""
    kd_tag = f".{kd_type}" if kd_type is not None else ""
    auto_name = f"{args.model_name}@{args.dataset}_{args.param_set}{kd_tag}"
    run_name_base = args.run_name or auto_name
    if not args.disable_auto_run_indexing:
        run_name = run_name_base + "_#1"
        run_counter = 2
        while os.path.exists(os.path.join(args.csv_dir, f"{run_name}_metrics.csv")):
            run_name = f"{run_name_base}_#{run_counter}"
            run_counter += 1
    else:
        run_name = run_name_base

    if args.run_name is not None:
        run_name  = run_name + val_tag

    # If resuming training, use the run name from the checkpoint file or the provided run name
    if args.resume:
        run_name = args.run_name or os.path.basename(args.resume).split('_epoch')[0]
    return run_name



def print_config(params, run_name, args, device, logger=print):
    # Print out the configuration
    config = {
        "params": params,
        "run_name": run_name,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_freq": args.checkpoint_freq,
        "use_val": args.use_val,
        "val_size": args.val_size,
        "disable_test": args.disable_test,
        "csv_dir": args.csv_dir,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_start_epoch": args.early_stopping_start_epoch,
        "dataset_dir": args.dataset_dir,
        "dataset": args.dataset,
        "device": str(device),
        "model_save_dir": args.model_save_dir,
        "track_best_after_epoch": args.track_best_after_epoch,    
    }

    if not args.use_val:
        del config["val_size"]
        del config["early_stopping_patience"]
        del config["early_stopping_start_epoch"]

    logger("Configuration:")
    for key, value in config.items():
        logger(f"{key}: {value}")


def get_data_loaders(args, params, dataset, run_name, transform_train, transform_test, dataset_class, logger=print):
    trainloader = None
    valloader = None
    testloader = None

    val_split_random_state = None
    if args.use_val:
        if args.use_split_indices_from_file:
            logger(f"Using split indices from file: {args.use_split_indices_from_file}")
            train_indices, val_indices, val_split_random_state = dataset_splitter.load_indices(args.use_split_indices_from_file)
            trainset, valset = dataset_splitter.split_dataset_from_indices(dataset, train_indices, val_indices)
        else:
            logger(f"Splitting dataset with val_size={args.val_size} and stratify=True")
            if args.val_split_random_state is not None:
                logger(f"Using random state: {args.val_split_random_state}")
            indices_file_location = os.path.join(args.checkpoint_dir, f'{run_name}_indices.json')
            trainset, valset, val_split_random_state = dataset_splitter.split_dataset(
                dataset, test_size=args.val_size, stratify=True, random_state=args.val_split_random_state, save_to_file=indices_file_location
                )
    else:
        trainset = dataset

    use_persistent_workers = args.num_workers > 0 and not args.disable_persistent_workers
    trainloader = DataLoader(trainset, batch_size=params['training']['batch_size'], shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=use_persistent_workers)
    
    test_num_workers = args.num_workers // 2 if args.num_workers >= 4 else args.num_workers
    test_persistent_workers = test_num_workers > 0
    test_batch_size = params['training']['batch_size']
    if args.use_val:
        valloader = DataLoader(valset, batch_size=test_batch_size//2, shuffle=False, num_workers=test_num_workers, pin_memory=True, persistent_workers=test_persistent_workers)
    
    if not args.use_val or not args.disable_test:
        testset = dataset_class(root=args.dataset_dir, train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=test_batch_size//2, shuffle=False, num_workers=test_num_workers, pin_memory=True, persistent_workers=test_persistent_workers)
    
    return trainloader, valloader, testloader, val_split_random_state


def get_writer_name(kd_mode, args, run_name):
    return f"run_data/tensorboard/{kd_mode}/{args.model_name}/{args.dataset}/{run_name}"