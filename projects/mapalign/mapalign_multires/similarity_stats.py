import sys
import os
from multiprocessing import Pool

import numpy as np

from tqdm import tqdm

sys.path.append("../../utils")
import netsimilarity_utils

# --- Params --- #
IMAGE_NAME_FORMAT = "{city}{number}"


# --- --- #


def read_grads(grads_filepath):
    grads = np.load(grads_filepath)
    grads = grads.flatten()
    return grads


def read_all_grads(grads_filepath_list):
    with Pool(4) as p:
        r = list(tqdm(p.imap(read_grads, grads_filepath_list), total=len(grads_filepath_list), desc="Read all grads: "))
    grads_matrix = np.stack(r, axis=0)
    return grads_matrix


def compute_neighbor_count(raw_dirpath, tile_info_list, output_dirname, output_filepath_format):
    # Get all grads filepaths:
    grads_filepath_list = []
    neighbors_soft_filepath_list = []
    for tile_info in tile_info_list:
        tile_name = IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])

        for bbox in tile_info["bbox_list"]:
            grads_filepath = output_filepath_format.format(dir=raw_dirpath, fold=tile_info["fold"],
                                                           out_dir=output_dirname, tile=tile_name,
                                                           b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                           out_name="grads", ext="npy")
            grads_filepath_list.append(grads_filepath)
            neighbors_soft_filepath = output_filepath_format.format(dir="similarity_stats", fold=tile_info["fold"],
                                                                    out_dir=output_dirname, tile=tile_name,
                                                                    b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                                    out_name="neighbors_soft", ext="npy")
            neighbors_soft_filepath_list.append(neighbors_soft_filepath)

    neighbors_soft_count_array = netsimilarity_utils.compute_soft_neighbor_count_multidim_on_disk(grads_filepath_list)

    # Save
    for neighbors_soft_filepath, neighbors_soft_count in zip(neighbors_soft_filepath_list, neighbors_soft_count_array):
        os.makedirs(os.path.dirname(neighbors_soft_filepath), exist_ok=True)
        np.save(neighbors_soft_filepath, neighbors_soft_count)


def compute_similarities_list(raw_dirpath, tile_info_list, individual_patch_index_list, output_dirname,
                              output_filepath_format):
    # Get all grads filepaths:
    grads_filepath_list = []
    similarities_filepath_list = []
    for tile_info in tile_info_list:
        tile_name = IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])

        for bbox in tile_info["bbox_list"]:
            grads_filepath = output_filepath_format.format(dir=raw_dirpath, fold=tile_info["fold"],
                                                           out_dir=output_dirname, tile=tile_name,
                                                           b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                           out_name="grads", ext="npy")
            grads_filepath_list.append(grads_filepath)
            similarities_filepath = output_filepath_format.format(dir="similarity_stats", fold=tile_info["fold"],
                                                                  out_dir=output_dirname, tile=tile_name,
                                                                  b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                                  out_name="similarities", ext="npy")
            similarities_filepath_list.append(similarities_filepath)

    similarities_mat = netsimilarity_utils.compute_similarities_multidim_on_disk(grads_filepath_list,
                                                                                 individual_patch_index_list)

    # Save
    for i, individual_patch_index in enumerate(individual_patch_index_list):
        similarities_filepath = similarities_filepath_list[individual_patch_index]
        os.makedirs(os.path.dirname(similarities_filepath), exist_ok=True)
        np.save(similarities_filepath, similarities_mat[i])
