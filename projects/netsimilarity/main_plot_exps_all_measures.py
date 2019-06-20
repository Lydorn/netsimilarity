import os
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--exp_dirpath',
        default="exp_results",
        type=str,
        help='Path to the experiment folder.')
    argparser.add_argument(
        '--log',
        action='store_true',
        help='If true plots values in log scale.')
    argparser.add_argument(
        '--norm',
        action='store_true',
        help='If true plots normalized values.')

    argparser.add_argument(
        '--x_name',
        default="frequency",
        type=str,
        help='Name of the value to plot on x axis. Examples: sample_count or frequency.',
    )
    argparser.add_argument(
        '--y_name',
        type=str,
        help='Name of the value to plot on y axis (if 3D plot). Examples: sample_count or frequency.',
    )

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


def get_exp_params(all_param):
    exp_params = {
        "n": all_param["sample_count"],
        "f": all_param["frequency"],
        "s": all_param["noise_std"],
        "d": all_param["distribution"],
    }
    return exp_params


def params_to_str(params):
    def to_str(value):
        if type(value) == float and value == int(value):
            return str(int(value))
        return str(value)

    return "_".join(["{}_{}".format(key, to_str(params[key])) for key in sorted(params.keys())])


def main():

    # --- Process args --- #
    args = get_args()

    # Load all data
    exp_dirpath = args.exp_dirpath

    params = {
        "sample_count": args.sample_count,
        "frequency": args.frequency,
        "noise_std": args.noise_std,
        "distribution": "uniform",
    }

    # --- Losses:
    # aggregated_names = [
    #     "neighbors_soft",
    #     "train_loss",
    #     "val_loss",
    #     "loss_ratio",
    # ]
    # line_styles = ["s:", "o-", "o-", "s:"]
    # markersizes = [12, 12, 12, 12]
    # linewidths = [2, 2, 2, 2]

    aggregated_names = [
        "neighbors_soft",
        # "neighbors_soft_no_normalization",
    ]
    line_styles = ["o-"]
    markersizes = [8]
    linewidths = [2]

    # --- Extra measures:
    stats_params = {
        "neighbors_t": args.neighbors_t,
        "neighbors_n": args.neighbors_n,
    }
    for neighbors_n in stats_params["neighbors_n"]:
        aggregated_names.append("neighbors_less_soft_n_{}".format(neighbors_n))
        line_styles.append("s:")
        markersizes.append(4)
        linewidths.append(1)
    for neighbors_t in stats_params["neighbors_t"]:
        aggregated_names.append("neighbors_hard_t_{}".format(neighbors_t))
        line_styles.append("^--")
        markersizes.append(4)
        linewidths.append(1)

    x_list = []
    y_list = []
    z_dict = {}
    for aggregated_name in aggregated_names:
        z_dict[aggregated_name] = []
    for n, f in itertools.product(params["sample_count"], params["frequency"]):
        if args.x_name == "sample_count":
            x_list.append(n)
        elif args.x_name == "frequency":
            x_list.append(f)

        if args.y_name == "sample_count":
            y_list.append(n)
        elif args.y_name == "frequency":
            y_list.append(f)

        all_params = {
            "sample_count": n,
            "frequency": f,
            "noise_std": params["noise_std"],
            "distribution": params["distribution"],
        }
        exp_params = get_exp_params(all_params)
        exp_params_str = params_to_str(exp_params)
        filepath_format = os.path.join(exp_dirpath, "{}.{{}}.npy".format(exp_params_str))
        for aggregated_name in aggregated_names:
            measure_values = np.load(filepath_format.format(aggregated_name))
            avg_measure_values = np.mean(measure_values)
            z_dict[aggregated_name].append(avg_measure_values)

    if args.log:
        x_list = np.log(x_list)
        y_list = np.log(y_list)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    if len(y_list):
        # Means we have a 3D plot
        ax = Axes3D(fig)
        for aggregated_name, line_style, markersize, linewidth in zip(aggregated_names, line_styles, markersizes, linewidths):
            z_list = z_dict[aggregated_name]
            if args.log:
                z_list = np.log(z_list)
            if args.norm:
                z_list = (z_list - z_list.min()) / (z_list.max() - z_list.min())
            ax.plot(x_list, y_list, z_list, line_style, markersize=markersize, linewidth=linewidth)
    else:
        # Means we have a 2D plot
        ax = fig.add_subplot(1, 1, 1)
        for aggregated_name, line_style, markersize, linewidth in zip(aggregated_names, line_styles, markersizes, linewidths):
            z_list = np.array(z_dict[aggregated_name])
            if args.log:
                z_list = np.log(z_list)
            if args.norm:
                z_list = (z_list - z_list.min()) / (z_list.max() - z_list.min())
            ax.plot(x_list, z_list, line_style, markersize=markersize, linewidth=linewidth)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(aggregated_names, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15)

    ax.set_title("Avg of all measures across all samples")
    if len(y_list):
        ax.set_xlabel(args.x_name)
        ax.set_ylabel(args.x_name)
        ax.set_zlabel("Neighbor counts")
    else:
        ax.set_xlabel(args.x_name.title())
        if args.log:
            ax.set_ylabel("Neighbor count (log)")
        else:
            ax.set_ylabel("Neighbor count")
    plt.savefig('{}.eps'.format("avg_of_all_measures"))
    plt.show()


if __name__ == '__main__':
    main()
