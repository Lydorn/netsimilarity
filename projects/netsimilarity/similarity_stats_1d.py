import sys
import os

import numpy as np
# from sklearn.cluster import DBSCAN
from tqdm import tqdm

from synthetic_1d_dataset import Synthetic1DDataset

sys.path.append("../utils")
import python_utils
import print_utils
import run_utils
import netsimilarity_utils


# --- Params --- #
COMPUTE_ONLY_NEIGHBORS_SOFT = True  # Allows for more optimized code if True as this estimator is a lot faster to compute.

# --- --- #


def similarity_stats_1d(config, run_name, dataset_params, split_name, stats_params):
    # print("# --- Similarity Stats --- #")

    working_dir = os.path.dirname(os.path.abspath(__file__))

    # Find data_dir
    data_dirpath = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dirpath is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    # print_utils.print_info("Using data from {}".format(data_dirpath))
    root_dir = os.path.join(data_dirpath, config["data_root_partial_dirpath"])

    # setup run directory:
    runs_dir = os.path.join(working_dir, config["runs_dirpath"])
    run_dirpath = None
    try:
        run_dirpath = run_utils.setup_run_dir(runs_dir, run_name)
    except ValueError:
        print_utils.print_error("Run name {} was not found. Aborting...".format(run_name))
        exit()

    # Instantiate dataset
    # ds = Synthetic1DDataset(root_dir=root_dir, params=dataset_params, split_name="test",
    #                         distribution="triangular"
    #                         )
    ds = Synthetic1DDataset(root_dir=root_dir,
                            params=dataset_params,
                            split_name=split_name,
                            transform=None)
    sample_count = len(ds)

    # Load grads and pred
    grads_dirpath = os.path.join(run_dirpath, "grads")
    grads_filepath_list = python_utils.get_filepaths(grads_dirpath, endswith_str=".npy", startswith_str="grads.")
    grads_list = [np.load(grads_filepath) for grads_filepath in tqdm(grads_filepath_list, desc="Loading grads")]
    # print("Grads shape: {}".format(grads_list[0].shape))
    pred_filepath_list = python_utils.get_filepaths(grads_dirpath, endswith_str=".npy", startswith_str="pred.")
    pred_list = [np.load(pred_filepath) for pred_filepath in tqdm(pred_filepath_list, desc="Loading pred")]

    # Create stats dir
    stats_dirpath = os.path.join(run_dirpath, "stats_1d")
    os.makedirs(stats_dirpath, exist_ok=True)

    # import time
    # t1 = time.clock()

    neighbor_count, neighbor_count_no_normalization = netsimilarity_utils.compute_soft_neighbor_count(grads_list)
    neighbors_filepath = os.path.join(stats_dirpath, "neighbors_soft.npy")
    np.save(neighbors_filepath, neighbor_count)
    neighbors_filepath = os.path.join(stats_dirpath, "neighbors_soft_no_normalization.npy")
    np.save(neighbors_filepath, neighbor_count_no_normalization)

    if not COMPUTE_ONLY_NEIGHBORS_SOFT:
        # Compute similarity matrix
        similarity_mat = netsimilarity_utils.compute_similarity_mat_1d(grads_list)

        # Compute number of neighbors
        # Hard-thresholding:
        for t in stats_params["neighbors_t"]:
            neighbor_count = netsimilarity_utils.compute_neighbor_count(similarity_mat, "hard", t=t)
            neighbors_filepath = os.path.join(stats_dirpath, "neighbors_hard_t_{}.npy".format(t))
            np.save(neighbors_filepath, neighbor_count)

        # # Soft estimate
        # neighbor_count = netsimilarity_utils.compute_neighbor_count(similarity_mat, "soft")
        # neighbors_filepath = os.path.join(stats_dirpath, "neighbors_soft.npy")
        # np.save(neighbors_filepath, neighbor_count)

        # Mix
        for n in stats_params["neighbors_n"]:
            neighbor_count = netsimilarity_utils.compute_neighbor_count(similarity_mat, "less_soft",
                                                                        n=n)
            neighbors_filepath = os.path.join(stats_dirpath, "neighbors_less_soft_n_{}.npy".format(n))
            np.save(neighbors_filepath, neighbor_count)

    # print("Time to compute number of neighbors:")
    # print(time.clock() - t1)

    # Save inputs
    for key in ["alpha", "x", "density", "gt", "noise", "curvature"]:
        filepath = os.path.join(stats_dirpath, "{}.npy".format(key))
        values = [sample[key] for sample in ds]
        np.save(filepath, values)

    # Save outputs
    pred_filepath = os.path.join(stats_dirpath, "pred.npy")
    pred = [pred[0] for pred in pred_list]
    np.save(pred_filepath, pred)

    # Error
    error_filepath = os.path.join(stats_dirpath, "error.npy")
    error = [np.abs(sample["gt"] - pred[0]) for sample, pred in zip(ds, pred_list)]
    np.save(error_filepath, error)

    # Losses
    logs_dirpath = os.path.join(run_dirpath, "logs")
    final_losses = python_utils.load_json(os.path.join(logs_dirpath, "final_losses.json"))
    train_loss_filepath = os.path.join(stats_dirpath, "train_loss.npy")
    np.save(train_loss_filepath, final_losses["train_loss"])
    val_loss_filepath = os.path.join(stats_dirpath, "val_loss.npy")
    np.save(val_loss_filepath, final_losses["val_loss"])
    loss_ratio_filepath = os.path.join(stats_dirpath, "loss_ratio.npy")
    np.save(loss_ratio_filepath, final_losses["val_loss"] / final_losses["train_loss"])

    # # Cluster points based on their similarity
    # threshold = 0.8
    # distance_matrix = 1 - (similarity_mat - threshold)/(1 - threshold)
    # db = DBSCAN(eps=0.15, metric="precomputed").fit(distance_matrix)
    # # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # # core_samples_mask[db.core_sample_indices_] = True
    # clusters = db.labels_
    # clusters_filepath = os.path.join(stats_dirpath, "clusters.npy")
    # np.save(clusters_filepath, clusters)
