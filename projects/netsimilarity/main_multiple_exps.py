import os
import sys
import argparse
import itertools

import numpy as np
from tqdm import tqdm

import train
import compute_grads
import similarity_stats_1d

sys.path.append("../utils")
import run_utils
import print_utils
import python_utils

# -- Default script arguments: --- #
CONFIG = "config"


# --- Params: --- #

# --- --- #


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--config',
        default=CONFIG,
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '-b', '--batch_size',
        type=int,
        help='Batch size. Generally set as large as the VRAM can handle. Default value can be set in config file.')
    argparser.add_argument(
        '--exps_dirpath',
        type=str,
        help='Directory where experiments are recorded (model saves and logs).')

    argparser.add_argument(
        '--exp_name',
        type=str,
        help='Name of the experiment. This is a single word, without the timestamp.')
    argparser.add_argument(
        '--new_exp',
        action='store_true',
        help="Start experiment from scratch (when True) or finish the last (when False)")
    argparser.add_argument(
        '--recompute_stats',
        action='store_true',
        help="Recomputes only grads and stats (when True) or train and then compute grads and stats (when False)")

    argparser.add_argument(
        '--run_count',
        default=5,
        type=int,
        help='Number of runs trained for each parameter setting for statistical significance.')
    argparser.add_argument(
        '-n', '--sample_count',
        default=[2048],
        type=int,
        nargs='+',
        help='Total number of samples to generate. Is a list of values, one for each experiment.')
    argparser.add_argument(
        '-f', '--frequency',
        default=[1, 2, 4, 8],
        type=float,
        nargs='+',
        help='Frequency used for the sine wave. Is a list of values, one for each experiment.')
    argparser.add_argument(
        '-s', '--noise_std',
        default=0.0,
        type=float,
        help='Noise standard deviation')

    argparser.add_argument(
        '--neighbors_t',
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99],
        type=float,
        nargs='+',
        help='Threshold used for computing number of neighbors with the \"hard\" method.')
    argparser.add_argument(
        '--neighbors_n',
        default=[2, 3, 4],
        type=int,
        nargs='+',
        help='Exponent used for computing number of neighbors with the \"less_soft\" method.')

    args = argparser.parse_args()
    return args


def get_dataset_params(exp_param):
    dataset_params = {
        "n": exp_param["sample_count"],
        "f": exp_param["frequency"],
        "s": exp_param["noise_std"],
        "d": exp_param["distribution"],
    }
    return dataset_params


def get_exp_params(all_params):
    exp_params = {
        "n": all_params["sample_count"],
        "f": all_params["frequency"],
        "s": all_params["noise_std"],
        "d": all_params["distribution"],
    }
    return exp_params


def get_run_name(all_params):
    exp_params = get_exp_params(all_params)
    exp_params_str = python_utils.params_to_str(exp_params)

    run_name = exp_params_str + "." + python_utils.params_to_str({"run": all_params["run"]})
    return run_name


def launch_one_experiment(config, all_params, stats_params):
    run_name = get_run_name(all_params)

    run_params = {
        "run_name": run_name,
        "new_run": False,  # By default try to continue run. If run_name does not exist, it will create it.
        "init_run_name": None,
    }
    dataset_params = get_dataset_params(all_params)

    train.train(config, run_params, dataset_params)
    compute_grads.compute_grads(config, run_params, dataset_params, split_name="train")
    similarity_stats_1d.similarity_stats_1d(config, run_name, dataset_params, split_name="train", stats_params=stats_params)


def launch_experiments(config, exp_dirpath, new_exp, recompute_stats, params, stats_params):
    config = config.copy()
    config["runs_dirpath"] = exp_dirpath

    # Setup progress filepaths
    finished_exps_filepath = os.path.join(exp_dirpath, 'finished_exps.json')

    # Start a new experiment or recompute stats from an existing experiment
    remaining_exp_list = []
    for n, f in itertools.product(params["sample_count"], params["frequency"]):
        for run in range(params["run_count"]):
            all_param = {
                "run": run,
                "sample_count": n,
                "frequency": f,
                "noise_std": params["noise_std"],
                "distribution": params["distribution"],
            }
            remaining_exp_list.append(all_param)

    if new_exp or recompute_stats:
        finished_exp_list = []
        python_utils.save_json(finished_exps_filepath, finished_exp_list)
    else:
        # Continue a previous experiment. Load exp_param_lists
        finished_exp_list = python_utils.load_json(finished_exps_filepath)

    # Remove finished experiments from remaining_exp_list:
    remaining_exp_list = [item for item in remaining_exp_list if item not in finished_exp_list]

    finished_exp_count = len(finished_exp_list)
    total_exp_count = len(remaining_exp_list + finished_exp_list)

    remaining_exp_list_to_save = remaining_exp_list.copy()

    exp_pbar = tqdm(remaining_exp_list, desc="Running exps: ", initial=finished_exp_count, total=total_exp_count)
    for all_params in exp_pbar:
        launch_one_experiment(config, all_params, stats_params)
        remaining_exp_list_to_save.remove(all_params)
        finished_exp_list.append(all_params)
        python_utils.save_json(finished_exps_filepath, finished_exp_list)


def aggregate_results(exp_dirpath, params, stats_params):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    aggregated_data_names = [
        "neighbors_soft",
        "neighbors_soft_no_normalization",
        "alpha",
        "x",
        "pred",
        "gt",
        "error",
        "curvature",
        "density",
        "train_loss",
        "val_loss",
        "loss_ratio",
    ]
    for neighbors_t in stats_params["neighbors_t"]:
        aggregated_data_names.append("neighbors_hard_t_{}".format(neighbors_t))
    for neighbors_n in stats_params["neighbors_n"]:
        aggregated_data_names.append("neighbors_less_soft_n_{}".format(neighbors_n))
    for n, f in itertools.product(params["sample_count"], params["frequency"]):
        aggregated_data = {}
        for run in range(params["run_count"]):
            all_params = {
                "run": run,
                "sample_count": n,
                "frequency": f,
                "noise_std": params["noise_std"],
                "distribution": params["distribution"],
            }
            run_name = get_run_name(all_params)
            runs_dir = os.path.join(working_dir, exp_dirpath)
            run_dirpath = run_utils.setup_run_dir(runs_dir, run_name, check_exists=True)
            stats_dirpath = os.path.join(run_dirpath, "stats_1d")
            for name in aggregated_data_names:
                filepath = os.path.join(stats_dirpath, "{}.npy".format(name))
                try:
                    data = np.load(filepath)
                    if name in aggregated_data:
                        aggregated_data[name].append(data)
                    else:
                        aggregated_data[name] = [data]
                except FileNotFoundError:
                    pass
                    # print("File {} not found, skipping...".format(filepath))
        for key, value in aggregated_data.items():
            aggregated_data[key] = np.median(value, axis=0)  # Average runs with the same params

        # Save aggregated data
        all_params = {
            "sample_count": n,
            "frequency": f,
            "noise_std": params["noise_std"],
            "distribution": params["distribution"],
        }
        exp_params = get_exp_params(all_params)
        dataset_params_str = python_utils.params_to_str(exp_params)
        filepath_format = os.path.join(exp_dirpath, "{}.{{}}.npy".format(dataset_params_str))
        for key, value in aggregated_data.items():
            np.save(filepath_format.format(key), aggregated_data[key])


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
    if args.exps_dirpath is not None:
        config["exps_dirpath"] = args.exps_dirpath

    distribution = "uniform"
    params = {
        "run_count": args.run_count,
        "sample_count": args.sample_count,
        "frequency": args.frequency,
        "noise_std": args.noise_std,
        "distribution": distribution,
    }
    stats_params = {
        "neighbors_t": args.neighbors_t,
        "neighbors_n": args.neighbors_n,
    }

    working_dir = os.path.dirname(os.path.abspath(__file__))

    # Setup exp directory:
    exps_dir = os.path.join(working_dir, config["exps_dirpath"])
    exp_dirpath = run_utils.setup_run_dir(exps_dir, args.exp_name, args.new_exp)

    # Launch experiments
    launch_experiments(config, exp_dirpath, args.new_exp, args.recompute_stats, params, stats_params)

    # Aggregate results
    aggregate_results(exp_dirpath, params, stats_params)


if __name__ == '__main__':
    main()
