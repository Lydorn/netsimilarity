import sys
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from model import Simple1DInputNet
from trainer import Trainer
import measures
import transforms

from synthetic_1d_dataset import Synthetic1DDataset

sys.path.append("../utils")
import python_utils
import print_utils
import run_utils


def train(config, run_params, dataset_params):
    # print("# --- Starting training --- #")

    run_name = run_params["run_name"]
    new_run = run_params["new_run"]
    init_run_name = run_params["init_run_name"]

    working_dir = os.path.dirname(os.path.abspath(__file__))

    # Find data_dir
    data_dirpath = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dirpath is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    # print_utils.print_info("Using data from {}".format(data_dirpath))
    root_dir = os.path.join(data_dirpath, config["data_root_partial_dirpath"])

    # setup init checkpoints directory path if one is specified:
    if init_run_name is not None:
        init_run_dirpath = run_utils.setup_run_dir(config["runs_dirpath"], init_run_name)
        _, init_checkpoints_dirpath = run_utils.setup_run_subdirs(init_run_dirpath)
    else:
        init_checkpoints_dirpath = None

    # setup run directory:
    runs_dir = os.path.join(working_dir, config["runs_dirpath"])
    run_dirpath = run_utils.setup_run_dir(runs_dir, run_name, new_run)

    # save config in logs directory
    run_utils.save_config(config, run_dirpath)

    # save args
    args_filepath = os.path.join(run_dirpath, "args.json")
    python_utils.save_json(args_filepath, {
        "run_name": run_name,
        "new_run": new_run,
        "init_run_name": init_run_name,
        "batch_size": config["batch_size"],
    })

    # Choose device
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = "cpu"  # For small networks and experiments, cpu is much faster

    # Instantiate dataset
    # sobol_generator = rand_utils.SobolGenerator()
    sobol_generator = None
    train_ds = Synthetic1DDataset(root_dir=root_dir,
                                  params=dataset_params,
                                  split_name="train",
                                  sobol_generator=sobol_generator,
                                  transform=torchvision.transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.ToDevice(device=dev)
                                  ])
                                  )
    val_ds = Synthetic1DDataset(root_dir=root_dir,
                                params=dataset_params,
                                split_name="val",
                                sobol_generator=sobol_generator,
                                transform=torchvision.transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.ToDevice(device=dev)
                                ]))

    # print(train_ds.alpha)
    # print(val_ds.alpha)
    # exit()

    # Generate test dataset here because if using Sobel numbers, all datasets should be using the same SobolGenerator
    # so that they do not generate the same samples.
    test_ds = Synthetic1DDataset(root_dir=root_dir,
                                 params=dataset_params,
                                 split_name="test",
                                 sobol_generator=sobol_generator,
                                 transform=torchvision.transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.ToDevice(device=dev)
                                 ]))
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"])

    success = False
    while not success:
        try:
            model = Simple1DInputNet(config)
            model.to(dev)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            loss_func = measures.l1_loss

            trainer = Trainer(config, model, optimizer, loss_func, init_checkpoints_dirpath, run_dirpath)
            trainer.fit(config, train_dl, val_dl)
            success = True
        except ValueError:  # Catches NaN errors
            # Try again
            run_utils.wipe_run_subdirs(run_dirpath)
            print("\nTry again\n")
            pass
