import sys
import os
import argparse

import numpy as np

sys.path.append("../utils")
import run_utils
import print_utils
import python_utils
import rand_utils

# -- Default script arguments: --- #
CONFIG = "config"
SHAPE_CHOICES = ["rectangle", "circle", "triangle"]


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        default=CONFIG,
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '-n', '--sample_count',
        default=1000,
        type=int,
        help='Total number of samples to generate.')
    argparser.add_argument(
        '-f', '--frequency',
        default=1.0,
        type=float,
        help='Frequency used for the sine wave.')
    argparser.add_argument(
        '-s', '--noise_std',
        default=0.0,
        type=float,
        help='Noise standard deviation')

    args = argparser.parse_args()
    return args


def generate_data(root_dir, params, split_name="train", seed=None, sobol_generator=None):
    sample_count = params["n"]
    frequency = params["f"]
    noise_std = params["s"]
    distribution = params["d"]
    assert distribution in ["triangular", "uniform"]

    np.random.seed(seed)

    split_dir = os.path.join(root_dir, "raw", split_name)
    os.makedirs(split_dir, exist_ok=True)

    params_str = python_utils.params_to_str(params)
    alpha_filepath = os.path.join(split_dir, "samples.{}.alpha.npy".format(params_str))
    x_filepath = os.path.join(split_dir, "samples.{}.x.npy".format(params_str))
    density_filepath = os.path.join(split_dir, "samples.{}.density.npy".format(params_str))
    gt_filepath = os.path.join(split_dir, "samples.{}.gt.npy".format(params_str))
    noise_filepath = os.path.join(split_dir, "samples.{}.noise.npy".format(params_str))
    curvature_filepath = os.path.join(split_dir, "samples.{}.curvature.npy".format(params_str))

    if distribution == "triangular":
        alpha = np.random.triangular(0, 1, 1, sample_count)
        density = 2 * alpha
    elif distribution == "uniform":
        if sobol_generator is not None:
            alpha = sobol_generator.generate(1, sample_count)
            alpha = alpha[:, 0]
        else:
            alpha = np.random.uniform(size=sample_count)
        density = np.ones_like(alpha)
    alpha = np.sort(alpha)

    pos_x = np.cos(2 * np.pi * alpha)
    pos_y = np.sin(2 * np.pi * alpha)
    x = np.stack((pos_x, pos_y), axis=1)

    gt = np.sin(2 * np.pi * frequency * alpha)
    noise = np.random.normal(0, noise_std, sample_count)

    y_diff = 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * alpha)
    y_diff_diff = - 4 * (np.pi ** 2) * (frequency ** 2) * np.sin(2 * np.pi * frequency * alpha)
    curvature = np.abs(y_diff_diff) / np.float_power(1 + np.square(y_diff), 3 / 2)

    np.save(alpha_filepath, alpha)
    np.save(x_filepath, x)
    np.save(density_filepath, density)
    np.save(gt_filepath, gt)
    np.save(noise_filepath, noise)
    np.save(curvature_filepath, curvature)

    return alpha, x, density, gt, noise, curvature


def generate_test(config, params, split_name="train", seed=None, sobol_generator=None):
    # Find data_dir
    data_dirpath = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dirpath is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    data_dirpath = os.path.expanduser(data_dirpath)
    print_utils.print_info("Using data from {}".format(data_dirpath))
    root_dir = os.path.join(data_dirpath, config["data_root_partial_dirpath"])

    alpha, x, density, gt, noise, curvature = generate_data(root_dir, params, split_name=split_name, seed=seed, sobol_generator=sobol_generator)

    noisy_gt = gt + noise
    import matplotlib.pyplot as plt
    f = plt.figure()
    f.set_tight_layout({"pad": .0})
    ax = f.gca()
    # plt.scatter(alpha, noisy_gt, s=10)
    ax.plot(alpha, noisy_gt)
    ax.set_xlabel("alpha")
    ax.set_ylabel("y")
    # plt.title("Sinusoid, freq = {}".format(params["f"]))
    plt.show()


def main():
    # --- Process args --- #
    args = get_args()
    config = run_utils.load_config(args.config)
    if config is None:
        print_utils.print_error(
            "ERROR: cannot continue without a config file. Exiting now...")
        exit()

    distribution = "uniform"
    params = {
        "n": args.sample_count,
        "f": args.frequency,
        "s": args.noise_std,
        "d": distribution,
    }

    sobol_generator = rand_utils.SobolGenerator()
    # sobol_generator = None

    generate_test(config, params, split_name="train", seed=0, sobol_generator=sobol_generator)
    generate_test(config, params, split_name="val", seed=1, sobol_generator=sobol_generator)
    generate_test(config, params, split_name="test", seed=2, sobol_generator=sobol_generator)


if __name__ == "__main__":
    main()
