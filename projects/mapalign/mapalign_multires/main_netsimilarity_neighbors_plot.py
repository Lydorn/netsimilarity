import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sys.path.append("../../../data/AerialImageDataset")
import read

sys.path.append("../../utils")
import run_utils
import polygon_utils
import python_utils
import netsimilarity_utils

# -- Default script arguments: --- #
CONFIG = "config"
RUNS_DIRPATH = "runs.igarss2019"
RUN_NAME = "ds_fac_2_inria_bradbury_all_2"
DS_FAC = 4

# --- Params: --- #

REFERENCE_PIXEL_SIZE = 0.3
DATASET_NAME = "AerialImageDataset"
POLYGON_DIRNAME = "gt_polygons"

OUTPUT_FILEPATH_FORMAT = "{dir}/{fold}/{out_dir}/{tile}.bbox_{b0:04d}_{b1:04d}_{b2:04d}_{b3:04d}.{out_name}.{ext}"

INDIVIDUAL_PATCH_INDEX_LIST_FILE_FORMAT = "{}.bin_index_list.npy"
INDIVIDUAL_PATCH_TILE_CITY = "bloomington"
INDIVIDUAL_PATCH_TILE_NUMBER = 22


# --- --- #

# --- Launch examples --- #

# python main_netsimilarity_neighbors_plot.py --output_dirname netsimilarity_ds_fac_4_round_0
# python main_netsimilarity_neighbors_plot.py --output_dirname netsimilarity_ds_fac_4_round_1
# python main_netsimilarity_neighbors_plot.py --output_dirname netsimilarity_ds_fac_4_round_2
# python main_netsimilarity_neighbors_plot.py --mode individual --output_dirname netsimilarity_ds_fac_4_round_0
# python main_netsimilarity_neighbors_plot.py --mode individual --output_dirname netsimilarity_ds_fac_4_round_1
# python main_netsimilarity_neighbors_plot.py --mode individual --output_dirname netsimilarity_ds_fac_4_round_2

# ---  --- #


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-m', '--mode',
        default="overall",
        type=str,
        choices=['overall', 'individual'],
        help='Mode to launch the script in:\n'
             '    - (overall) plot histogram of all neighbor soft counts. '
             '      Saves patches representative of each similarity bin.\n'
             '    - (individual) plot k closest neighbors and hist of closest neighbors\n')
    argparser.add_argument(
        '--individual_selection',
        default="tile",
        type=str,
        choices=['tile', 'index_list'],
        help='When in individual mode, this argument changes the way the individual source patch is chosen. '
             'tile: select a tile based on the global params INDIVIDUAL_PATCH_TILE_CITY and INDIVIDUAL_PATCH_TILE_NUMBER in the script.'
             'index_list: select patch based on the INDIVIDUAL_PATCH_INDEX_LIST_FILE_FORMAT global param in the script. '
             'By default it selects patches that where chosen to represent each bin in the histogram that are plotting by the overall mode. '
             'Thus it requires to launch this script with mode overall first.')
    argparser.add_argument(
        '-c', '--config',
        default=CONFIG,
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '-k',
        default=10,
        type=int,
        help='k param for the k-nearest neighbors.')
    argparser.add_argument(
        '--output_dirname',
        type=str,
        help='Name of the output directory.')

    args = argparser.parse_args()
    return args

#
# def find_closest_patch(patch_data_list, bin_edge):
#     min_i = 0
#     min_patch_data = patch_data_list[0]
#     min_dist = np.abs(min_patch_data["neighbors_soft"] - bin_edge)
#     for i, patch_data in enumerate(patch_data_list):
#         dist = np.abs(patch_data["neighbors_soft"] - bin_edge)
#         if dist < min_dist:
#             min_i = i
#             min_patch_data = patch_data
#             min_dist = dist
#     return min_patch_data, min_i


# def plot_polygons(axis, polygons):
#     for polygon in polygons:
#         axis.plot(polygon[:, 1], polygon[:, 0], color="blue", linewidth=1.0)


def get_patch_info_list(tile_info_list, output_dirname):
    patch_info_list = []
    for tile_info in tile_info_list:
        for bbox in tile_info["bbox_list"]:
            tile_name = read.IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])
            neighbors_soft_filepath = OUTPUT_FILEPATH_FORMAT.format(dir="similarity_stats", fold=tile_info["fold"],
                                                                    out_dir=output_dirname, tile=tile_name,
                                                                    b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                                    out_name="neighbors_soft", ext="npy")
            neighbors_soft = np.load(neighbors_soft_filepath)
            patch_info = {
                "city": tile_info["city"],
                "number": tile_info["number"],
                "fold": tile_info["fold"],
                "scale_factor": tile_info["scale_factor"],
                "bbox": bbox,
                "neighbors_soft": neighbors_soft,
            }
            patch_info_list.append(patch_info)
    return patch_info_list


def load_data(raw_dirpath, patch_info_list):
    image_list = []
    polygons_list = []
    for patch_info in patch_info_list:
        additional_args = {
            "overwrite_polygon_dir_name": POLYGON_DIRNAME,
        }
        image, metadata, polygons = read.load_gt_data(raw_dirpath, patch_info["city"], patch_info["number"],
                                                      additional_args=additional_args)
        scaled_bbox = np.array(patch_info["bbox"] / patch_info["scale_factor"], dtype=np.int)
        p_image = image[scaled_bbox[0]:scaled_bbox[2], scaled_bbox[1]:scaled_bbox[3], :]
        image_list.append(p_image)
        p_polygons = polygon_utils.crop_polygons_to_patch_if_touch(polygons, scaled_bbox)
        polygons_list.append(p_polygons)
    return image_list, polygons_list


def load_similarities(patch_info_list, output_dirname):
    similarities_list = []

    for i, patch_info in enumerate(patch_info_list):
        tile_name = read.IMAGE_NAME_FORMAT.format(city=patch_info["city"], number=patch_info["number"])
        bbox = patch_info["bbox"]
        similarities_filepath = OUTPUT_FILEPATH_FORMAT.format(dir="similarity_stats", fold=patch_info["fold"],
                                                              out_dir=output_dirname, tile=tile_name,
                                                              b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                              out_name="similarities", ext="npy")
        similarities = np.load(similarities_filepath)
        similarities_list.append(similarities)

    return np.stack(similarities_list, axis=0)


def plot_neighbors_soft_hist(raw_dirpath, tile_info_list, output_dirname, fig_name):
    patch_info_list = get_patch_info_list(tile_info_list, output_dirname)
    neighbors_soft_array = np.empty(len(patch_info_list))
    for i, patch_info in enumerate(tqdm(patch_info_list, desc="Reading neighbors_soft: ")):
        # additional_args = {
        #     "overwrite_polygon_dir_name": POLYGON_DIRNAME,
        # }
        # image, metadata, polygons = read.load_gt_data(raw_dirpath, tile_info["city"], tile_info["number"],
        #                                               additional_args=additional_args)
        #     scaled_bbox = np.array(bbox / scale_factor, dtype=np.int)
        #     p_image = image[scaled_bbox[0]:scaled_bbox[2], scaled_bbox[1]:scaled_bbox[3], :]
        #     p_polygons = polygon_utils.crop_polygons_to_patch_if_touch(polygons, scaled_bbox)

        tile_name = read.IMAGE_NAME_FORMAT.format(city=patch_info["city"], number=patch_info["number"])
        bbox = patch_info["bbox"]
        neighbors_soft_filepath = OUTPUT_FILEPATH_FORMAT.format(dir="similarity_stats", fold=patch_info["fold"],
                                                                out_dir=output_dirname, tile=tile_name,
                                                                b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                                out_name="neighbors_soft", ext="npy")
        neighbors_soft = np.load(neighbors_soft_filepath)
        neighbors_soft_array[i] = neighbors_soft

    # Sort patch_data_list
    # patch_data_list = sorted(patch_data_list, key=lambda patch_data: patch_data["neighbors_soft"])

    # Compute histogram:
    hist, bin_edges = np.histogram(neighbors_soft_array)
    freq = hist / np.sum(hist)
    # Find patches closest to bin_edges[1:]
    selected_index_list = [netsimilarity_utils.get_nearest_to_value(neighbors_soft_array, bin_edge) for bin_edge in bin_edges[1:]]

    # Save closest_index_list
    np.save("{}.bin_index_list.npy".format(output_dirname), selected_index_list)

    # Load images and polygons
    selected_patch_info_list = [patch_info_list[i] for i in selected_index_list]
    selected_images, selected_polygons = load_data(raw_dirpath, selected_patch_info_list)

    # Plot
    f = plt.figure(figsize=(26, 13))
    f.set_tight_layout({"pad": .0})
    ax = f.gca()
    plt.bar(bin_edges[1:], freq, width=np.diff(bin_edges), align="edge")

    bin_x_half_width = np.diff(bin_edges) / 2
    bin_x_center = bin_edges[1:] + bin_x_half_width
    bin_x_right = bin_x_center + 0.8 * bin_x_half_width
    bin_y = freq
    zoom = 0.3
    for x_center, x_right, y, image, polygons in zip(bin_x_center, bin_x_right, bin_y, selected_images, selected_polygons):
        im_polygons = polygon_utils.draw_polygons(polygons, image.shape, fill=False, edges=True, vertices=False,
                                                  line_width=5)
        im_polygons_mask = 0 < im_polygons[..., 0]
        im_display = image
        im_display[im_polygons_mask] = (255, 0, 0)
        im = OffsetImage(im_display, zoom=zoom)
        ab = AnnotationBbox(im, (x_center, y + 0.065), xycoords='data', frameon=False)
        ax.text(x_right, y + 0.01, u'\u2191', fontname='STIXGeneral', size=30, va='center', ha='center', clip_on=True)
        ax.add_artist(ab)
    ax.set_ylim([0, 0.55])
    ax.set_xlabel("Neighbors soft", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)

    plt.savefig(fig_name + ".eps", dpi=300)
    plt.show()


def draw_image_and_polygons(image, polygons):
    im_polygons = polygon_utils.draw_polygons(polygons, image.shape, fill=False, edges=True, vertices=False,
                                              line_width=5)
    im_polygons_mask = 0 < im_polygons[..., 0]
    im_display = image
    im_display[im_polygons_mask] = (255, 0, 0)
    return im_display


def plot_image_and_polygons(ax, image, polygons, title=None, xlabel=None):
    ax.imshow(draw_image_and_polygons(image, polygons))
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off',
                   left='off', labelleft='off')


def plot_k_nearest(data_list, fig_name):
    fig = plt.figure(figsize=(9, 10))
    fig.set_tight_layout({"pad": .0})
    row_count = len(data_list)
    for i, data in enumerate(data_list):
        col_count = 1 + len(data["k_nearest_image"])
        # Plot image
        ax = plt.subplot(row_count, col_count, i * col_count + 1)
        plot_image_and_polygons(ax, data["source_image"], data["source_polygons"], xlabel="Source")
        # Plot k nearest
        for j, (image, polygons, similarity) in enumerate(zip(data["k_nearest_image"], data["k_nearest_polygons"], data["k_nearest_similarities"])):
            ax = plt.subplot(row_count, col_count, i * col_count + (j + 1) + 1)
            plot_image_and_polygons(ax, image, polygons, xlabel="{0:.3g}".format(similarity))

    plt.savefig(fig_name + ".eps", dpi=300)
    plt.show()


def plot_k_nearest_hist(data, fig_name):
    # fig = plt.figure(figsize=(20, 10))
    # fig.set_tight_layout({"pad": .0})
    # Plot image
    f = plt.figure()
    f.set_tight_layout({"pad": .0})
    ax = f.gca()

    hist, bin_edges = np.histogram(data["similarities"])
    freq = hist / np.sum(hist)

    im_display = draw_image_and_polygons(data["source_image"], data["source_polygons"])
    # im_display = np.concatenate((im_display, 200*np.ones((im_display.shape[0], im_display.shape[1], 1), dtype=np.uint8)), axis=-1)
    # zoom = 0.75
    # im = OffsetImage(im_display, zoom=zoom)
    # ab = AnnotationBbox(im, (0.0, 0.8), frameon=False, box_alignment=(0, 1), bboxprops=dict(alpha=0.5))
    # ax.add_artist(ab)
    ax.imshow(im_display, zorder=0, extent=[0.0, 0.5, 0.3, 0.8])

    # Plot hist
    ax.bar(bin_edges[1:], freq, width=np.diff(bin_edges), align='center', alpha=0.8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.8])

    ax.set_xlabel("Similarity")
    ax.set_ylabel("Frequency")
    plt.title("Neighbors soft: {:.1f}".format(data["neighbors_soft"]))

    #
    # for bin_pos, v in zip(bin_pos_list, freq_list):
    #     ax.text(bin_pos, v + 100, str(v), color='blue', fontweight='bold')

    # # Add an image for each bin
    # bin_images = stats["bin_nearest_image_list"]
    # bin_x = stats["hist"][1][:-1] + np.diff(stats["hist"][1]) / 2
    # bin_y = freq_list
    # # image_size = bin_images[0].shape[0]
    # zoom = 1
    # for x, y, image in zip(bin_x, bin_y, bin_images):
    #     if image is not None:
    #         im = OffsetImage(image, zoom=zoom)
    #         ab = AnnotationBbox(im, (x, y + 0.11), xycoords='data', frameon=False)
    #         a1.text(x, y + 0.025, u'\u2191', fontname='STIXGeneral', size=30, va='center', ha='center', clip_on=True)
    #         a1.add_artist(ab)
    # a1.set_ylim([0, 1])
    # # a1.autoscale()

    plt.savefig(fig_name + ".eps", dpi=300)
    plt.show()


def plot_similarities(raw_dirpath, tile_info_list, output_dirname, individual_selection, k):
    patch_info_list = get_patch_info_list(tile_info_list, output_dirname)

    selected_index_list = None
    if individual_selection == "index_list":
        selected_index_list = np.load(INDIVIDUAL_PATCH_INDEX_LIST_FILE_FORMAT.format(output_dirname))
    elif individual_selection == "tile":
        selected_index_list = []
        i = 0
        for tile_info in tile_info_list:
            for bbox in tile_info["bbox_list"]:
                if tile_info["city"] == INDIVIDUAL_PATCH_TILE_CITY and tile_info["number"] == INDIVIDUAL_PATCH_TILE_NUMBER:
                    selected_index_list.append(i)
                i += 1

    selected_patch_info_list = [patch_info_list[i] for i in selected_index_list]
    selected_image_list, selected_polygons_list = load_data(raw_dirpath, selected_patch_info_list)
    selected_similarities_list = load_similarities(selected_patch_info_list, output_dirname)
    selected_neighbors_soft_list = [selected_patch_info["neighbors_soft"] for selected_patch_info in selected_patch_info_list]
    # Find k nearest
    k_nearest_similarities_list = []
    k_nearest_indices_list = []
    for selected_similarities in selected_similarities_list:
        k_nearest_similarities, k_nearest_indices = netsimilarity_utils.get_k_nearest(selected_similarities, k)
        k_nearest_similarities_list.append(k_nearest_similarities)
        k_nearest_indices_list.append(k_nearest_indices)

    # Load data for all k nearest
    k_nearest_image_list = []
    k_nearest_polygons_list = []
    for k_nearest_indices in k_nearest_indices_list:
        k_nearest_patch_info_list = [patch_info_list[i] for i in k_nearest_indices]
        k_nearest_image, k_nearest_polygons = load_data(raw_dirpath, k_nearest_patch_info_list)
        k_nearest_image_list.append(k_nearest_image)
        k_nearest_polygons_list.append(k_nearest_polygons)

    # Merge data
    individual_data_list = []
    for selected_image, selected_polygons, selected_similarities, selected_neighbors_soft, k_nearest_similarities, k_nearest_image, k_nearest_polygons in \
            zip(selected_image_list, selected_polygons_list, selected_similarities_list, selected_neighbors_soft_list,
                k_nearest_similarities_list, k_nearest_image_list, k_nearest_polygons_list):
        individual_data_list.append({
            "source_image": selected_image,
            "source_polygons": selected_polygons,
            "similarities": selected_similarities,
            "neighbors_soft": selected_neighbors_soft,
            "k_nearest_similarities": k_nearest_similarities,
            "k_nearest_image": k_nearest_image,
            "k_nearest_polygons": k_nearest_polygons,
        })

    # Plot
    fig_base_name = output_dirname
    if individual_selection == "tile":
        fig_base_name += "." + read.IMAGE_NAME_FORMAT.format(city=INDIVIDUAL_PATCH_TILE_CITY,
                                                        number=INDIVIDUAL_PATCH_TILE_NUMBER)
    elif individual_selection == "index_list":
        fig_base_name += ".from_overall_hist"

    fig_name = fig_base_name + ".k_nearest"
    plot_k_nearest(individual_data_list, fig_name)

    for i, individual_data in enumerate(individual_data_list):
        fig_name = fig_base_name + ".individual_hist.{:02d}".format(i)
        plot_k_nearest_hist(individual_data, fig_name)


def main():
    # TODO: pick center pixel when computing gradients
    # TODO: solve bug.= (look at output)
    # TODO: display input polygons as well in final plot
    # TODO: find theta (rotation) that minimizes k(.,.) in closed form
    # TODO: measure k(., .) with different models trained at different rounds
    args = get_args()

    # load overwrite_config file
    overwrite_config = run_utils.load_config(args.config)

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(overwrite_config["data_dir_candidates"])
    if data_dir is None:
        print("ERROR: Data directory not found!")
        exit()
    else:
        print("Using data from {}".format(data_dir))
    raw_dirpath = os.path.join(data_dir, DATASET_NAME, "raw")

    # Get all tiles
    print("Loading tile_info_list from disk...")
    tile_info_list_filepath = "{}.tile_info_list.npy".format(args.output_dirname)
    tile_info_list = np.load(tile_info_list_filepath)

    # tile_info_list = tile_info_list[-60:-50]  # TODO: remove to take all tiles

    if args.mode == "overall":
        print("Plot overall histogram of neighbors_soft:")
        fig_name = args.output_dirname + ".overall_hist"
        plot_neighbors_soft_hist(raw_dirpath, tile_info_list, args.output_dirname, fig_name)
    elif args.mode == "individual":
        plot_similarities(raw_dirpath, tile_info_list, args.output_dirname, args.individual_selection, args.k)


if __name__ == '__main__':
    main()
