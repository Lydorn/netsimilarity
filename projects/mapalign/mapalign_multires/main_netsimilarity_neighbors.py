import sys
import os
import argparse
import random
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

import compute_grads
import similarity_stats

sys.path.append("../../../data/AerialImageDataset")
import read

sys.path.append("../../utils")
import run_utils
import polygon_utils
import python_utils
import image_utils

# -- Default script arguments: --- #
CONFIG = "config"
RUNS_DIRPATH = "runs.igarss2019"

# --- Params: --- #
SEED = 0

REFERENCE_PIXEL_SIZE = 0.3
DATASET_NAME = "AerialImageDataset"
POLYGON_DIRNAME = "gt_polygons"
PATCH_PER_TILE = 10  # 360 tiles maximum, 10 patches per tile means 3600 patches whose neighbor count we will compute
PATCH_RES = 124  # Minimum patch size = 124

OUTPUT_FILEPATH_FORMAT = "{dir}/{fold}/{out_dir}/{tile}.bbox_{b0:04d}_{b1:04d}_{b2:04d}_{b3:04d}.{out_name}.{ext}"


INDIVIDUAL_PATCH_INDEX_LIST_FILE_FORMAT = "{}.bin_index_list.npy"
INDIVIDUAL_PATCH_TILE_CITY = "bloomington"
INDIVIDUAL_PATCH_TILE_NUMBER = 22

# --- --- #

# --- Launch examples --- #

# python main_netsimilarity_neighbors.py --run_name ds_fac_4_inria_bradbury_all   --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_0
# python main_netsimilarity_neighbors.py --run_name ds_fac_4_inria_bradbury_all_1 --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_1
# python main_netsimilarity_neighbors.py --run_name ds_fac_4_inria_bradbury_all_2 --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_2

# python main_netsimilarity_neighbors.py --mode individual --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_0
# python main_netsimilarity_neighbors.py --mode individual --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_1
# python main_netsimilarity_neighbors.py --mode individual --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_2

# ---  --- #


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-m', '--mode',
        default="compute",
        type=str,
        choices=['compute', 'individual'],
        help='Mode to launch the script in:\n'
             '    - (compute) gradients for all patches and then compute neighbor count\n'
             '    - (individual) compute all similarity measures for a few individual patches. See global params in the .py to change the selected individual patches.\n')
    argparser.add_argument(
        '--individual_selection',
        default="tile",
        type=str,
        choices=['tile', 'index_list'])
    argparser.add_argument(
        '-c', '--config',
        default=CONFIG,
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '--runs_dirpath',
        default=RUNS_DIRPATH,
        type=str,
        help='Name of directory where the model run can be found.')
    argparser.add_argument(
        '--run_name',
        type=str,
        help='Name of run.')
    argparser.add_argument(
        '-d', '--ds_fac',
        type=int,
        help='Downscaling factor. Should be an integer and it is used to retrieve the run name.')
    argparser.add_argument(
        '--output_dirname',
        type=str,
        help='Name of the output directory.')

    args = argparser.parse_args()
    return args


def sample_patches(params):
    raw_dirpath, tile_info, ds_fac, size, count, seed = params

    im_filepath = read.get_image_filepath(raw_dirpath, tile_info["city"], tile_info["number"])
    im_size = image_utils.get_image_size(im_filepath)
    polygon_list = read.load_polygons(raw_dirpath, read.POLYGON_DIRNAME, tile_info["city"], tile_info["number"])

    # Rescale data
    corrected_factor = ds_fac * REFERENCE_PIXEL_SIZE / tile_info["pixelsize"]
    scale_factor = 1 / corrected_factor
    im_size = (int(np.round(im_size[0] * scale_factor)), int(np.round(im_size[1] * scale_factor)))
    ds_polygon_list = polygon_utils.rescale_polygon(polygon_list, 1 / corrected_factor)

    bbox_list = image_utils.compute_patch_boundingboxes(im_size, size, size)

    random.seed(seed)
    random.shuffle(bbox_list)

    # Sample <count> patches in tile, making sure there is at least a polygon inside
    sampled_bbox_list = []
    for bbox in bbox_list:
        bbox_polygon_list = polygon_utils.filter_polygons_in_bounding_box(ds_polygon_list, bbox)
        if 1 <= len(bbox_polygon_list):
            sampled_bbox_list.append(bbox)
        if count <= len(sampled_bbox_list):
            break

    tile_info["bbox_list"] = sampled_bbox_list
    tile_info["scale_factor"] = scale_factor

    return tile_info


def filter_already_computed_grads(raw_dirpath, tile_info_list, output_dirname):
    new_tile_info_list = []
    for tile_info in tile_info_list:
        tile_name = read.IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])
        # Are all grads already computed for that tile?
        new_bbox_list = []
        for bbox in tile_info["bbox_list"]:
            grads_filepath = OUTPUT_FILEPATH_FORMAT.format(dir=raw_dirpath, fold=tile_info["fold"],
                                                           out_dir=output_dirname, tile=tile_name,
                                                           b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                           out_name="grads", ext="npy")
            if not os.path.exists(grads_filepath):
                new_bbox_list.append(bbox)
        if len(new_bbox_list):
            tile_info["bbox_list"] = new_bbox_list
            new_tile_info_list.append(tile_info)

    return new_tile_info_list


def compute(args, raw_dirpath, overwrite_config, tile_info_list):
    # Filter out patches whose grads have already been computed:
    compute_grads_tile_info_list = filter_already_computed_grads(raw_dirpath, tile_info_list,
                                                                 args.output_dirname)

    if len(compute_grads_tile_info_list):
        print("Compute gradients:")
        compute_grads.compute_grads(raw_dirpath, args.runs_dirpath, args.run_name, args.ds_fac,
                                    overwrite_config, compute_grads_tile_info_list, POLYGON_DIRNAME,
                                    args.output_dirname, OUTPUT_FILEPATH_FORMAT)
    else:
        print("Gradients are all already computed, skipping...")

    print("Compute neighbor soft count:")
    similarity_stats.compute_neighbor_count(raw_dirpath, tile_info_list, args.output_dirname, OUTPUT_FILEPATH_FORMAT)


def individual(args, raw_dirpath, tile_info_list):
    individual_patch_index_list = None
    if args.individual_selection == "index_list":
        individual_patch_index_list = np.load(INDIVIDUAL_PATCH_INDEX_LIST_FILE_FORMAT.format(args.output_dirname))
    elif args.individual_selection == "tile":
        individual_patch_index_list = []
        i = 0
        for tile_info in tile_info_list:
            for bbox in tile_info["bbox_list"]:
                if tile_info["city"] == INDIVIDUAL_PATCH_TILE_CITY and tile_info["number"] == INDIVIDUAL_PATCH_TILE_NUMBER:
                    individual_patch_index_list.append(i)
                i += 1
    similarity_stats.compute_similarities_list(raw_dirpath, tile_info_list, individual_patch_index_list, args.output_dirname, OUTPUT_FILEPATH_FORMAT)


def main():
    # TODO: pick center pixel when computing gradients
    # TODO: solve bug.= (look at output)
    # TODO: display input polygons as well in final plot
    # TODO: find theta (rotation) that minimizes k(.,.) in closed form
    # TODO: measure k(., .) with different models trained at different rounds
    args = get_args()

    # load overwrite_config file
    overwrite_config = run_utils.load_config(args.config)
    if args.runs_dirpath is not None:
        overwrite_config["runs_dirpath"] = args.runs_dirpath
    overwrite_config["input_res"] = PATCH_RES
    overwrite_config["batch_size"] = 1

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(overwrite_config["data_dir_candidates"])
    if data_dir is None:
        print("ERROR: Data directory not found!")
        exit()
    else:
        print("Using data from {}".format(data_dir))

    raw_dirpath = os.path.join(data_dir, DATASET_NAME, "raw")

    # Get all tiles
    tile_info_list_filepath = "{}.tile_info_list.npy".format(args.output_dirname)
    try:
        print("Loading tile_info_list from disk...")
        tile_info_list = np.load(tile_info_list_filepath)
    except FileNotFoundError:
        tile_info_list = read.get_tile_info_list(raw_dirpath=raw_dirpath)

        # Sample patches in each tile
        pool_size = 4
        with Pool(pool_size) as p:
            params_list = [(raw_dirpath, tile_info, args.ds_fac, PATCH_RES, PATCH_PER_TILE, SEED) for tile_info in tile_info_list]
            tile_info_list = list(tqdm(p.imap(sample_patches, params_list), total=len(params_list), desc="Sample patches: "))
        np.save(tile_info_list_filepath, tile_info_list)

    # tile_info_list = tile_info_list[-60:-50]  # TODO: remove to take all tiles

    if args.mode == "compute":
        compute(args, raw_dirpath, overwrite_config, tile_info_list)
    elif args.mode == "individual":
        individual(args, raw_dirpath, tile_info_list)


if __name__ == '__main__':
    main()
