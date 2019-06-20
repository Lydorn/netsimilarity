import sys
import os

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sys.path.append("../utils")
import print_utils
import run_utils

# --- Params --- #

# --- --- #

def load_stats_list(stats_dirpath, source_idx_list):
    stats_list = []
    for source_i in source_idx_list:
        dirpath = os.path.join(stats_dirpath, "{:05d}".format(source_i))
        image_filepath = os.path.join(dirpath, "image.png")
        k_nearest_similarity_list_filepath = os.path.join(dirpath, "k_nearest_similarity_list.npy")
        k_nearest_idx_list_filepath = os.path.join(dirpath, "k_nearest_idx_list.npy")
        hist_filepath = os.path.join(dirpath, "hist.npy")
        neighbor_count_filepath = os.path.join(dirpath, "neighbor_count.npy")

        image = skimage.io.imread(image_filepath)
        k_nearest_similarity_list = np.load(k_nearest_similarity_list_filepath)
        k_nearest_idx_list = np.load(k_nearest_idx_list_filepath)
        hist = np.load(hist_filepath)
        neighbor_count = np.load(neighbor_count_filepath)
        nearest_image_list = []
        for i, k_nearest_idx in enumerate(k_nearest_idx_list):
            nearest_filepath = os.path.join(dirpath, "nearest.{:02d}.png".format(i + 1))
            nearest_image = skimage.io.imread(nearest_filepath)
            nearest_image_list.append(nearest_image)
        bin_nearest_image_list = []
        for i in range(len(hist[1][:-1])):
            try:
                bin_nearest_filepath = os.path.join(dirpath, "bin_nearest.{:02d}.png".format(i + 1))
                bin_nearest_image = skimage.io.imread(bin_nearest_filepath)
            except FileNotFoundError:
                bin_nearest_image = None
            bin_nearest_image_list.append(bin_nearest_image)
        stats = {
            "image": image,
            "k_nearest_similarity_list": k_nearest_similarity_list,
            "hist": hist,
            "nearest_image_list": nearest_image_list,
            "bin_nearest_image_list": bin_nearest_image_list,
            "neighbor_count": neighbor_count,
        }
        stats_list.append(stats)
    return stats_list


def plot_image(ax, image, title=None):
    ax.imshow(image)
    if title is not None:
        plt.title(title)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off',
                   left='off', labelleft='off')


def plot_k_nearest(stats_list):
    fig = plt.figure(figsize=(20, 10))
    fig.set_tight_layout({"pad": .0})
    col_count = len(stats_list)
    for i, stats in enumerate(stats_list):
        row_count = 1 + len(stats["nearest_image_list"])
        # Plot image
        ax = plt.subplot(col_count, row_count, i * row_count + 1)
        plot_image(ax, stats["image"], title="Source")
        # Plot k nearest
        for j, (image, similarity) in enumerate(zip(stats["nearest_image_list"], stats["k_nearest_similarity_list"])):
            ax = plt.subplot(col_count, row_count, i * row_count + (j + 1) + 1)
            plot_image(ax, image, title="k = {0:.3g}".format(similarity))
    plt.show()


def plot_hist(stats):
    # fig = plt.figure(figsize=(20, 10))
    # fig.set_tight_layout({"pad": .0})
    # Plot image
    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(20, 10))

    plot_image(a0, stats["image"], title="Neighbor count: {:.1f}".format(stats["neighbor_count"]))

    # Plot hist
    freq_list = stats["hist"][0] / np.sum(stats["hist"][0])
    bin_pos_list = stats["hist"][1][:-1]
    a1.bar(bin_pos_list, freq_list, width=np.diff(stats["hist"][1]), align="edge")

    #
    # for bin_pos, v in zip(bin_pos_list, freq_list):
    #     ax.text(bin_pos, v + 100, str(v), color='blue', fontweight='bold')

    # Add an image for each bin
    bin_images = stats["bin_nearest_image_list"]
    bin_x = stats["hist"][1][:-1] + np.diff(stats["hist"][1]) / 2
    bin_y = freq_list
    # image_size = bin_images[0].shape[0]
    zoom = 1
    for x, y, image in zip(bin_x, bin_y, bin_images):
        if image is not None:
            im = OffsetImage(image, zoom=zoom)
            ab = AnnotationBbox(im, (x, y + 0.11), xycoords='data', frameon=False)
            a1.text(x, y + 0.025, u'\u2191', fontname='STIXGeneral', size=30, va='center', ha='center', clip_on=True)
            a1.add_artist(ab)
    a1.set_ylim([0, 1])
    # a1.autoscale()

    plt.show()


def plot_stats(config, run_name, source_idx_list):
    print("# --- Plot stats --- #")

    working_dir = os.path.dirname(os.path.abspath(__file__))

    # setup run directory:
    runs_dir = os.path.join(working_dir, config["runs_dirpath"])
    run_dirpath = None
    try:
        run_dirpath = run_utils.setup_run_dir(runs_dir, run_name)
    except ValueError:
        print_utils.print_error("Run name {} was not found. Aborting...".format(run_name))
        exit()
    stats_dirpath = os.path.join(run_dirpath, "stats")

    stats_list = load_stats_list(stats_dirpath, source_idx_list)

    plot_k_nearest(stats_list)

    for stats in stats_list:
        plot_hist(stats)



