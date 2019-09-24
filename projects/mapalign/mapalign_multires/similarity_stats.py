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
                              output_filepath_format, normalized=True, scalar=True):
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
            out_name = "similarities"
            if not normalized:
                out_name += ".not_normalized"
            if not scalar:
                out_name += ".not_scalar"
            similarities_filepath = output_filepath_format.format(dir="similarity_stats", fold=tile_info["fold"],
                                                                  out_dir=output_dirname, tile=tile_name,
                                                                  b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                                  out_name=out_name, ext="npy")
            similarities_filepath_list.append(similarities_filepath)

    # Check if all of similarities_mat was already computed:
    individual_similarities_filepath_list = [similarities_filepath_list[i] for i in individual_patch_index_list]
    already_computed = True
    for individual_similarities_filepath in individual_similarities_filepath_list:
        if not os.path.exists(individual_similarities_filepath):
            already_computed = False
            break

    if already_computed:
        print("Similarities were already computed, load them instead of re-computing them (delete at least one to recompute):")
        if scalar:
            similarities_mat = np.empty((len(individual_patch_index_list), len(grads_filepath_list)))
        else:
            source_grads_0 = np.load(grads_filepath_list[0])
            d = source_grads_0.shape[1]
            similarities_mat = np.empty((len(individual_patch_index_list), len(grads_filepath_list), d, d))
        for i, individual_patch_index in enumerate(individual_patch_index_list):
            similarities_filepath = similarities_filepath_list[individual_patch_index]
            similarities_mat[i] = np.load(similarities_filepath)
    else:
        similarities_mat = netsimilarity_utils.compute_similarities_multidim_on_disk(grads_filepath_list,
                                                                                     individual_patch_index_list,
                                                                                     normalized=normalized, scalar=scalar)
        # Save
        for i, individual_patch_index in enumerate(individual_patch_index_list):
            similarities_filepath = similarities_filepath_list[individual_patch_index]
            os.makedirs(os.path.dirname(similarities_filepath), exist_ok=True)
            np.save(similarities_filepath, similarities_mat[i])

    return similarities_mat


def compute_denoising_factors_list(tile_info_list, individual_patch_index_list, output_dirname,
                              output_filepath_format, similarities_mat):
    # Get all grads filepaths:
    denoising_factors_filepath_list = []
    for tile_info in tile_info_list:
        tile_name = IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])

        for bbox in tile_info["bbox_list"]:
            denoising_factors_filepath = output_filepath_format.format(dir="similarity_stats", fold=tile_info["fold"],
                                                                  out_dir=output_dirname, tile=tile_name,
                                                                  b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                                  out_name="denoising_factor", ext="npy")
            denoising_factors_filepath_list.append(denoising_factors_filepath)

    denoising_factors_vector = netsimilarity_utils.compute_denoising_factors(similarities_mat)
    print("denoising_factors_vector:")
    print(denoising_factors_vector)

    # Save
    for i, individual_patch_index in enumerate(individual_patch_index_list):
        denoising_factors_filepath = denoising_factors_filepath_list[individual_patch_index]
        os.makedirs(os.path.dirname(denoising_factors_filepath), exist_ok=True)
        np.save(denoising_factors_filepath, denoising_factors_vector[i])


def compute_neighbor_consistency_list(raw_dirpath, tile_info_list, individual_patch_index_list, output_dirname,
                              output_filepath_format, similarities_mat):
    # Get all disp pred filepaths:
    disp_pred_list = []
    neighbor_consistency_filepath_list = []
    for tile_info in tile_info_list:
        tile_name = IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])

        for bbox in tile_info["bbox_list"]:
            disp_pred_filepath = output_filepath_format.format(dir=raw_dirpath,
                                                                          fold=tile_info["fold"],
                                                                          out_dir=output_dirname, tile=tile_name,
                                                                          b0=bbox[0], b1=bbox[1], b2=bbox[2],
                                                                          b3=bbox[3],
                                                                          out_name="disp_pred", ext="npy")
            disp_pred = np.load(disp_pred_filepath)
            # Here pick disp_pred has shape HxWh2, pick middle 2D vector
            disp_pred = disp_pred[disp_pred.shape[0]//2, disp_pred.shape[1]//2, :]
            disp_pred_list.append(disp_pred)
            neighbor_consistency_filepath = output_filepath_format.format(dir="similarity_stats", fold=tile_info["fold"],
                                                                  out_dir=output_dirname, tile=tile_name,
                                                                  b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                                  out_name="neighbor_consistency", ext="npy")
            neighbor_consistency_filepath_list.append(neighbor_consistency_filepath)
    disp_pred_mat = np.stack(disp_pred_list)

    neighbor_consistency_vector = netsimilarity_utils.compute_neighbor_consistency(disp_pred_mat, similarities_mat, individual_patch_index_list)
    print("neighbor_consistency_vector:")
    print(neighbor_consistency_vector)

    # Save
    for i, individual_patch_index in enumerate(individual_patch_index_list):
        neighbor_consistency_filepath = neighbor_consistency_filepath_list[individual_patch_index]
        os.makedirs(os.path.dirname(neighbor_consistency_filepath), exist_ok=True)
        np.save(neighbor_consistency_filepath, neighbor_consistency_vector[i])

