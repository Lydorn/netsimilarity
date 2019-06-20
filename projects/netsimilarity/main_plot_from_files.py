import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-x',
        type=str,
        help='Name of the .npy file to plot on x axis, excluding the extension \".npy\"',
        required=True)
    argparser.add_argument(
        '-y',
        type=str,
        help='Name of the .npy file to plot on y axis, excluding the extension \".npy\"',
        required=True)
    argparser.add_argument(
        '-z',
        type=str,
        help='(Optional) Name of the .npy file to plot on z axis for 3D plots, excluding the extension \".npy\"')
    args = argparser.parse_args()
    return args


def main():

    # --- Process args --- #
    args = get_args()

    x = np.load("{}.npy".format(args.x))
    y = np.load("{}.npy".format(args.y))

    fig = plt.figure()
    fig.set_tight_layout({"pad": .0})
    if args.z is None:
        # 2D plot
        x_label = args.x.title()
        y_label = args.y.title()
        title = " vs ".join([y_label, x_label])
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        plt.savefig('{}.eps'.format(title))
    else:
        # 3D plot
        z = np.load("{}.npy".format(args.z))
        x_label = args.x.title()
        y_label = args.y.title()
        z_label = args.z.title()
        title = " vs ".join([z_label, y_label, x_label])

        ax = Axes3D(fig)
        ax.plot(x, y, z)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.show()
        plt.savefig('{}.eps'.format(title))


if __name__ == '__main__':
    main()
