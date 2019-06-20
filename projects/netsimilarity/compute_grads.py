import sys
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from model import Simple1DInputNet
from analyzer import Analyzer
import transforms

from synthetic_1d_dataset import Synthetic1DDataset

sys.path.append("../utils")
import python_utils
import print_utils
import run_utils


def compute_grads(config, run_params, dataset_params, split_name):
    # print("# --- Compute grads --- #")

    working_dir = os.path.dirname(os.path.abspath(__file__))

    # Find data_dir
    data_dirpath = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dirpath is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    # print_utils.print_info("Using data from {}".format(data_dirpath))
    root_dir = os.path.join(data_dirpath, config["data_root_partial_dirpath"])

    # setup run directory:
    runs_dir = os.path.join(working_dir, config["runs_dirpath"])
    run_dirpath = None
    try:
        run_dirpath = run_utils.setup_run_dir(runs_dir, run_params["run_name"])
    except ValueError:
        print_utils.print_error("Run name {} was not found. Aborting...".format(run_params["run_name"]))
        exit()

    # Choose device
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate dataset
    ds = Synthetic1DDataset(root_dir=root_dir,
                            params=dataset_params,
                            split_name=split_name,
                            transform=torchvision.transforms.Compose([
                                transforms.ToTensor(),
                                transforms.ToDevice(device=dev)
                            ])
                            )
    dl = DataLoader(ds, batch_size=1)

    model = Simple1DInputNet(config, run_params["capacity"])
    model.to(dev)

    analyzer = Analyzer(config, model, run_dirpath)
    analyzer.compute_and_save_grads(dl)
