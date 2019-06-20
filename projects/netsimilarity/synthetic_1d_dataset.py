import sys
import os
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision

import synthetic_1d_generation
import transforms

sys.path.append("../utils")
import run_utils
import print_utils
import python_utils
import rand_utils

# -- Default script arguments: --- #
CONFIG = "config"


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        default=CONFIG,
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '-b', '--batch_size',
        type=int,
        help='Batch size. Generally set as large as the VRAM can handle. Default value can be set in config file.')
    argparser.add_argument(
        '-n', '--sample_count',
        default=1000,
        type=int,
        help='Total number of samples to generate.')
    argparser.add_argument(
        '-f', '--frequency',
        default=1,
        type=float,
        help='Frequency used for the sine wave.')
    argparser.add_argument(
        '-s', '--noise_std',
        default=0.2,
        type=float,
        help='Noise standard deviation')

    args = argparser.parse_args()
    return args


class Synthetic1DDataset(Dataset):

    def __init__(self, root_dir, params, split_name="train", sobol_generator=None, transform=None):
        self.root_dir = root_dir
        self.params = params
        self.split_name = split_name
        self.sobol_generator = sobol_generator
        self.transform = transform

        self.split_dir = os.path.join(self.root_dir, "raw", self.split_name)

        self.alpha = None
        self.x = None
        self.density = None
        self.gt = None
        self.noise = None
        self.curvature = None

        try:
            self._load_data()
        except FileNotFoundError:
            self._generate_data()
            self._load_data()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {
            "alpha": self.alpha[idx],
            "x": self.x[idx],
            "density": self.density[idx],
            "gt": self.gt[idx],
            "noise": self.noise[idx],
            "curvature": self.curvature[idx],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_data(self):
        params_str = python_utils.params_to_str(self.params)
        alpha_filepath = os.path.join(self.split_dir, "samples.{}.alpha.npy".format(params_str))
        x_filepath = os.path.join(self.split_dir, "samples.{}.x.npy".format(params_str))
        density_filepath = os.path.join(self.split_dir, "samples.{}.density.npy".format(params_str))
        gt_filepath = os.path.join(self.split_dir, "samples.{}.gt.npy".format(params_str))
        noise_filepath = os.path.join(self.split_dir, "samples.{}.noise.npy".format(params_str))
        curvature_filepath = os.path.join(self.split_dir, "samples.{}.curvature.npy".format(params_str))
        self.alpha = np.load(alpha_filepath)
        self.x = np.load(x_filepath)
        self.density = np.load(density_filepath)
        self.gt = np.load(gt_filepath)
        self.noise = np.load(noise_filepath)
        self.curvature = np.load(curvature_filepath)

    def _generate_data(self):
        synthetic_1d_generation.generate_data(self.root_dir, self.params,
                                              split_name=self.split_name,
                                              seed=None,
                                              sobol_generator=self.sobol_generator)


def main():
    # --- Process args --- #
    args = get_args()
    config = run_utils.load_config(args.config)
    if config is None:
        print_utils.print_error(
            "ERROR: cannot continue without a config file. Exiting now...")
        exit()
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    distribution = "uniform"
    dataset_params = {
        "n": args.sample_count,
        "f": args.frequency,
        "s": args.noise_std,
        "d": distribution,
    }

    # Find data_dir
    data_dirpath = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dirpath is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    data_dirpath = os.path.expanduser(data_dirpath)
    print_utils.print_info("Using data from {}".format(data_dirpath))
    root_dir = os.path.join(data_dirpath, config["data_root_partial_dirpath"])

    sobol_generator = rand_utils.SobolGenerator()

    train_ds = Synthetic1DDataset(root_dir=root_dir,
                                  params=dataset_params,
                                  split_name="train",
                                  sobol_generator=sobol_generator,
                                  transform=torchvision.transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.ToDevice(device="cuda")
                                  ]))
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(train_dl):
        print(i_batch,
              sample_batched['density'].max(),
              # sample_batched['gt'],
              # sample_batched['noise'],
              )


if __name__ == "__main__":
    main()
