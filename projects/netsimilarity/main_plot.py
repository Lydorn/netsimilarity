import sys
import argparse

import plot_stats
import plot_stats_1d

sys.path.append("../utils")
import run_utils
import print_utils

# -- Default script arguments: --- #
CONFIG = "config"
SOURCE_IDX_LIST = [0, 10, 20, 30, 40]

# --- Params: --- #

# --- --- #


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        default=CONFIG,
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '-r', '--runs_dirpath',
        type=str,
        help='Directory where runs are recorded (model saves and logs).')
    argparser.add_argument(
        '--run_name',
        type=str,
        help='Continue training from run_name or if it does not exist, start training this run from scratch. '
             'This is a single word, without the timestamp.')
    argparser.add_argument(
        '--source_idx_list',
        default=SOURCE_IDX_LIST,
        type=int,
        nargs='+',
        help='List of source image index for similarity stats computation.')
    argparser.add_argument(
        '-m', '--mode',
        default="image",
        type=str,
        choices=['image', '1d'],
        help='Plot for image stats or 1D input stats')

    args = argparser.parse_args()
    return args


def main():
    # --- Process args --- #
    args = get_args()
    config = run_utils.load_config(args.config)
    if config is None:
        print_utils.print_error(
            "ERROR: cannot continue without a config file. Exiting now...")
        exit()
    if args.runs_dirpath is not None:
        config["runs_dirpath"] = args.runs_dirpath

    if args.mode == "image":
        plot_stats.plot_stats(config, args.run_name, args.source_idx_list)
    elif args.mode == "1d":
        plot_stats_1d.plot(config, args.run_name)


if __name__ == '__main__':
    main()
