import sys
import time
import os.path
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import model

sys.path.append("../../../data/AerialImageDataset")
import read

# Import visualization first as it might change the matplotlib backend
sys.path.append("../utils")
import process_utils

sys.path.append("../../utils")
import polygon_utils
import run_utils


def get_flattened_gradients(grads):
    flattened_gradients = []
    for gradient in grads:
        flattened_gradients.append(gradient.flatten())
    flattened_gradients = np.concatenate(flattened_gradients, axis=0)
    return flattened_gradients


def compute_grads(raw_dirpath, runs_dirpath, run_name, ds_fac, overwrite_config, tile_info_list, polygon_dirname,
                  output_dirname, output_filepath_format):
    # -- Params:

    # Setup run dir and load config file
    run_dir = run_utils.setup_run_dir(runs_dirpath, run_name)
    _, checkpoints_dir = run_utils.setup_run_subdirs(run_dir)

    config = run_utils.load_config(config_dirpath=run_dir)

    # --- Instantiate model
    output_res = model.MapAlignModel.get_output_res(overwrite_config["input_res"], config["pool_count"])
    map_align_model = model.MapAlignModel(config["model_name"], overwrite_config["input_res"],

                                          config["add_image_input"], config["image_channel_count"],
                                          config["image_feature_base_count"],

                                          config["add_poly_map_input"], config["poly_map_channel_count"],
                                          config["poly_map_feature_base_count"],

                                          config["common_feature_base_count"], config["pool_count"],

                                          config["add_disp_output"], config["disp_channel_count"],

                                          config["add_seg_output"], config["seg_channel_count"],

                                          output_res,
                                          overwrite_config["batch_size"],

                                          config["loss_params"],
                                          config["level_loss_coefs_params"],

                                          config["learning_rate_params"],
                                          config["weight_decay"],

                                          config["image_dynamic_range"], config["disp_map_dynamic_range_fac"],
                                          config["disp_max_abs_value"])
    map_align_model.setup_compute_grads()  # Add ops to compute gradients

    saver = tf.train.Saver(save_relative_paths=True)
    with tf.Session() as sess:
        # Restore checkpoint
        restore_checkpoint_success = map_align_model.restore_checkpoint(sess, saver, checkpoints_dir)
        if not restore_checkpoint_success:
            sys.exit('No checkpoint found in {}'.format(checkpoints_dir))

        # Compute patch count
        patch_total_count = 0
        for tile_info in tile_info_list:
            patch_total_count += len(tile_info["bbox_list"])

        pbar = tqdm(total=patch_total_count, desc="Computing patch gradients: ")
        for tile_info in tile_info_list:
            # --- Path setup:
            unused_filepath = output_filepath_format.format(dir=raw_dirpath, fold=tile_info["fold"],
                                                            out_dir=output_dirname, tile="",
                                                            b0=0, b1=0, b2=0, b3=0,
                                                            out_name="", ext="")
            os.makedirs(os.path.dirname(unused_filepath), exist_ok=True)
            tile_name = read.IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])

            # Compute grads for that image
            additional_args = {
                "overwrite_polygon_dir_name": polygon_dirname,
            }
            # t = time.clock()
            image, metadata, polygons = read.load_gt_data(raw_dirpath, tile_info["city"], tile_info["number"],
                                                          additional_args=additional_args)
            # t_read = time.clock() - t
            # Downsample
            image, polygons = process_utils.downsample_data(image, metadata, polygons, ds_fac,
                                                            config["reference_pixel_size"])
            spatial_shape = image.shape[:2]

            # Draw polygon map
            # t = time.clock()
            polygon_map = polygon_utils.draw_polygon_map(polygons, spatial_shape, fill=True, edges=True, vertices=True)
            # t_draw = time.clock() - t

            t_grads = 0
            t_save = 0
            for bbox in tile_info["bbox_list"]:
                p_im = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
                p_polygon_map = polygon_map[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
                # p_polygons = polygon_utils.crop_polygons_to_patch_if_touch(polygons, bbox)

                # Grad compute
                t = time.clock()
                grads = map_align_model.compute_grads(sess, p_im, p_polygon_map)
                t_grads += time.clock() - t

                # Saving
                t = time.clock()
                flattened_grads_x = get_flattened_gradients(grads["x"])
                flattened_grads_y = get_flattened_gradients(grads["y"])
                flattened_grads = np.stack([flattened_grads_x, flattened_grads_y], axis=-1)

                # # Save patch for later visualization
                # im_filepath = output_filepath_format.format(dir=raw_dirpath, fold=tile_info["fold"],
                #                                             out_dir=output_dirname, tile=tile_name,
                #                                             b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                #                                             out_name="image", ext="png")
                # skimage.io.imsave(im_filepath, p_im)
                # # Save polygons as well
                # polygons_filepath = output_filepath_format.format(dir=raw_dirpath, fold=tile_info["fold"],
                #                                                   out_dir=output_dirname, tile=tile_name,
                #                                                   b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                #                                                   out_name="polygons", ext="npy")
                # np.save(polygons_filepath, p_polygons)
                # Save grads
                grads_filepath = output_filepath_format.format(dir=raw_dirpath, fold=tile_info["fold"],
                                                               out_dir=output_dirname, tile=tile_name,
                                                               b0=bbox[0], b1=bbox[1], b2=bbox[2], b3=bbox[3],
                                                               out_name="grads", ext="npy")
                np.save(grads_filepath, flattened_grads)
                t_save += time.clock() - t

            pbar.update(len(tile_info["bbox_list"]))
            pbar.set_postfix(t_grads=t_grads, t_save=t_save)
        pbar.close()
