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
        '--type',
        choices={"lines", "scatter"},
        default="lines",
        type=str,
        help='Type of the plot.',)
    argparser.add_argument(
        '-x',
        type=str,
        help='Name of the data to plot on x axis. For example: density, curvature, neighbors_less_soft, etc.',
        required=True)
    argparser.add_argument(
        '--x_log',
        action='store_true',
        help='If true plots x in log scale.')
    argparser.add_argument(
        '-y',
        type=str,
        help='Name of the data to plot on y axis. For example: density, curvature, neighbors_less_soft, etc.',
        required=True)
    argparser.add_argument(
        '--y_log',
        action='store_true',
        help='If true plots y in log scale.')
    argparser.add_argument(
        '-z',
        type=str,
        help='Name of the data to plot on z axis. For example: density, curvature, neighbors_less_soft, etc.')
    argparser.add_argument(
        '--z_log',
        action='store_true',
        help='If true plots z in log scale.')

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
    args = argparser.parse_args()
    return args


def get_exp_params(all_params):
    get_exp_params = {
        "n": all_params["sample_count"],
        "f": all_params["frequency"],
        "s": all_params["noise_std"],
        "d": all_params["distribution"],
    }
    return get_exp_params


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
    axis_params_dict = {
        "x": {
            "name": args.x,
            "log": args.x_log,
        },
        "y": {
            "name": args.y,
            "log": args.y_log,
        },
    }
    if args.z is not None:
        axis_params_dict["z"] = {
            "name": args.z,
            "log": args.z_log,
        }

    params = {
        "sample_count": args.sample_count,
        "frequency": args.frequency,
        "noise_std": args.noise_std,
        "distribution": "uniform",
    }

    axis_data = {}
    for axis, axis_params in axis_params_dict.items():
        data = []
        for n, f in itertools.product(params["sample_count"], params["frequency"]):
            all_params = {
                "sample_count": n,
                "frequency": f,
                "noise_std": params["noise_std"],
                "distribution": params["distribution"],
            }
            exp_params = get_exp_params(all_params)
            exp_params_str = params_to_str(exp_params)
            filepath_format = os.path.join(exp_dirpath, "{}.{{}}.npy".format(exp_params_str))
            sample = np.load(filepath_format.format(axis_params["name"]))
            if axis_params["log"]:
                sample = np.log(1 + sample)
            data.append(sample)
        axis_data[axis] = data

    # Plot
    for axis, axis_params in axis_params_dict.items():
        label = axis_params["name"].title()
        if axis_params["log"]:
            label += " (log)"
        axis_params["label"] = label

    title = " vs ".join([axis_params_dict[axis]["label"] for axis in sorted(axis_params_dict.keys(), reverse=True)])

    fig = plt.figure()
    fig.set_tight_layout({"pad": .0})
    if "z" in axis_params_dict:
        ax = Axes3D(fig)
        for x, y, z in zip(axis_data["x"], axis_data["y"], axis_data["z"]):
            if args.type == "scatter":
                ax.scatter(x, y, z, s=3)
            else:  # args.type == "lines"
                ax.plot(x, y, z)
    else:
        ax = fig.add_subplot(1, 1, 1)
        for x, y in zip(axis_data["x"], axis_data["y"]):
            if args.type == "scatter":
                ax.scatter(x, y, s=3)
            else:
                ax.plot(x, y)
    ax.legend(["n = {}, f = {}".format(n, f)
               for n, f in itertools.product(params["sample_count"], params["frequency"])])
    ax.set_title(title)
    ax.set_xlabel(axis_params_dict["x"]["label"])
    ax.set_ylabel(axis_params_dict["y"]["label"])
    if "z" in axis_params_dict:
        ax.set_zlabel(axis_params_dict["z"]["label"])
    plt.savefig('{}.eps'.format(title))
    plt.show()


if __name__ == '__main__':
    main()
