import numpy as np
import scipy
from tqdm import tqdm
import multiprocessing as mp

import python_utils


def compute_kernel_matrix(x, y):
    return x.transpose() @ y


def compute_normalized_kernel_matrix(x, y):
    assert x.shape == y.shape
    c_x = compute_kernel_matrix(x, x)
    c_y = compute_kernel_matrix(y, y)
    k = compute_kernel_matrix(x, y)
    sqrt_c_x = scipy.linalg.sqrtm(c_x)
    sqrt_c_y = scipy.linalg.sqrtm(c_y)
    inv_sqrt_c_x = np.linalg.inv(sqrt_c_x)
    inv_sqrt_c_y = np.linalg.inv(sqrt_c_y)
    normed_k = inv_sqrt_c_x @ k @ inv_sqrt_c_y
    return normed_k


def compute_similarity(x, y):
    assert x.shape == y.shape
    d = x.shape[-1]
    normed_k = compute_normalized_kernel_matrix(x, y)
    k = np.trace(normed_k) / d
    return k


def _compute_similarities_multidim_on_disk(params):
    grads_filepath_list, source_grads_list = params
    similarities_mat = np.empty((len(source_grads_list), len(grads_filepath_list)))
    for j, grads_filepath in enumerate(grads_filepath_list):
        target_grads = np.load(grads_filepath)
        for i, source_grads in enumerate(source_grads_list):
            similarity = compute_similarity(source_grads, target_grads)
            similarities_mat[i, j] = similarity
    return similarities_mat


def compute_similarities_multidim_on_disk(grads_filepath_list, source_index_list):
    process_count = 4
    # Load grad from index_list
    source_grads_list = [np.load(grads_filepath_list[i]) for i in source_index_list]
    grads_filepath_list_chuncks = python_utils.split_list_into_chunks(grads_filepath_list, min(10, len(
        grads_filepath_list) // process_count))
    params_list = [(grads_filepath_list_chunck, source_grads_list) for grads_filepath_list_chunck in
                   grads_filepath_list_chuncks]

    with mp.Pool(process_count) as p:
        similarities_mat_list = list(
            tqdm(p.imap(_compute_similarities_multidim_on_disk, params_list), total=len(params_list),
                 desc="Similarities: "))
    similarities_mat = np.concatenate(similarities_mat_list, axis=-1)

    # for j, grads_filepath in enumerate(tqdm(grads_filepath_list, desc="Compute similarities: ")):
    #     target_grads = np.load(grads_filepath)
    #     for i, source_grads in enumerate(source_grads_list):
    #         similarity = compute_similarity(source_grads, target_grads)
    #         similarities_mat[i, j] = similarity

    return similarities_mat


def compute_similarity_1d(x, y):
    assert x.shape == y.shape
    d = x.shape[-1]
    assert d == 1

    c_x = np.dot(x[:, 0], x[:, 0])
    c_y = np.dot(y[:, 0], y[:, 0])
    k = np.dot(x[:, 0], y[:, 0])
    sqrt_c_x = np.sqrt(c_x)
    sqrt_c_y = np.sqrt(c_y)
    inv_sqrt_c_x = 1 / sqrt_c_x
    inv_sqrt_c_y = 1 / sqrt_c_y
    normed_k = inv_sqrt_c_x * k * inv_sqrt_c_y

    return normed_k


# def compute_similarity_mat_1d(grads_list):
#     grads_mat = np.concatenate(grads_list, axis=-1)
#     sample_count = grads_mat.shape[-1]
#
#     # Compute inv of sqrt of self norm
#     inv_sqrt_c = 1 / np.sqrt(np.sum(np.square(grads_mat), axis=0))
#
#     # Compute pair norms
#
#
#     similarity_mat = np.empty((sample_count, sample_count))
#     for source_i in tqdm(range(sample_count), desc="Compute similarity_mat"):
#         x = grads_list[source_i]
#         similarity_list = []
#         for j in range(len(grads_list)):
#             y = grads_list[j]
#             similarity_list.append(compute_similarity_1d(x, y))
#         similarity_mat[source_i, :] = similarity_list
#
#     return similarity_mat


def compute_similarity_mat_1d(grads_list):
    grads_mat = np.transpose(np.concatenate(grads_list, axis=-1))
    similarity_mat = 1 - scipy.spatial.distance.pdist(grads_mat, "cosine")
    similarity_mat = scipy.spatial.distance.squareform(similarity_mat)

    return similarity_mat


def get_k_nearest(similarity_list, k):
    similarity_vector = np.array(similarity_list)
    k_nearest_indices = np.argpartition(similarity_vector, -k)[-k:]
    k_nearest_indices = k_nearest_indices[np.argsort(similarity_vector[k_nearest_indices])[::-1]]
    k_nearest_similarities = similarity_vector[k_nearest_indices]
    return k_nearest_similarities, k_nearest_indices


def get_nearest_to_value(similarity_list, value):
    similarity_vector = np.array(similarity_list)
    abs_diff_vector = np.abs(similarity_vector - value)
    index = np.argmin(abs_diff_vector)
    return index


# def compute_neighbor_count(similarity_list):
#     similarity_vector = np.array(similarity_list)
#     pos_similarity_vector = np.maximum(0, similarity_vector)
#     neighbor_count = pos_similarity_vector.sum()
#     return neighbor_count


def compute_neighbor_count(similarity_mat, method, **kwargs):
    assert method in ["hard", "soft", "less_soft"]
    if method == "hard":
        threshold = kwargs["t"]
        above_mask = threshold < similarity_mat
        neighbor_count = above_mask.sum(axis=-1)
    elif method == "soft":
        neighbor_count = similarity_mat.sum(axis=-1)
    elif method == "less_soft":
        threshold = 0.0
        n = kwargs["n"]
        zero_indices = np.where(similarity_mat <= threshold)
        zeroed_similarity_mat = similarity_mat.copy()
        zeroed_similarity_mat[zero_indices[0], zero_indices[1]] = 0
        zeroed_similarity_mat_power_n = np.power(zeroed_similarity_mat, n)
        neighbor_count = zeroed_similarity_mat_power_n.sum(axis=-1)
    return neighbor_count


def compute_soft_neighbor_count(grads_input):
    if type(grads_input) == list:
        grads_mat = np.concatenate(grads_input, axis=-1)
        grads_mat = np.transpose(grads_mat)
    else:
        grads_mat = grads_input
    norms_vector = np.sqrt(np.sum(np.square(grads_mat), axis=-1, keepdims=True))  # TODO: use np.linalg.norm and test it
    normed_grads_mat = grads_mat / norms_vector
    sum_normed_grad = np.sum(normed_grads_mat, axis=0)
    normed_similarity_vector = np.sum(normed_grads_mat * sum_normed_grad, axis=-1)
    sum_grad = np.sum(grads_mat, axis=0)
    similarity_vector = np.sum(grads_mat * sum_grad, axis=-1)
    return normed_similarity_vector, similarity_vector


def compute_soft_neighbor_count_on_disk(grads_filepath_list):
    # Compute sum of grads
    grads_0 = np.load(grads_filepath_list[0]).flatten()
    sum_normed_grads = np.zeros_like(grads_0)
    for grads_filepath in tqdm(grads_filepath_list, desc="Sum grads: "):
        grads = np.load(grads_filepath).flatten()
        normed_grads = grads / np.linalg.norm(grads)
        sum_normed_grads += normed_grads

    # Compute scalar product with sum_grads
    normed_count_list = []
    for grads_filepath in tqdm(grads_filepath_list, desc="Similarity: "):
        grads = np.load(grads_filepath).flatten()
        normed_grads = grads / np.linalg.norm(grads)
        normed__count = np.dot(normed_grads, sum_normed_grads)
        normed_count_list.append(normed__count)
    normed_count_array = np.array(normed_count_list)
    return normed_count_array


def compute_kernel_corrected_grads(grads, mat_k):
    return grads @ np.linalg.inv(scipy.linalg.sqrtm(mat_k))


def compute_soft_neighbor_count_multidim_on_disk(grads_filepath_list):
    # Compute sum of matrices a_j, shape [N, d]
    grads_0 = np.load(grads_filepath_list[0])
    n = grads_0.shape[0]
    d = grads_0.shape[1]
    del grads_0
    sum_a_j = np.zeros((n, d))
    for grads_filepath in tqdm(grads_filepath_list, desc="Sum a_j: "):
        grads = np.load(grads_filepath)
        mat_k = compute_kernel_matrix(grads, grads)
        a_j = compute_kernel_corrected_grads(grads, mat_k)
        sum_a_j += a_j

    # Compute scalar product with sum_grads
    sample_count = len(grads_filepath_list)
    normed_count_array = np.empty(sample_count)
    for i, grads_filepath in enumerate(tqdm(grads_filepath_list, desc="Neighbors soft: ")):
        grads = np.load(grads_filepath)
        mat_k = compute_kernel_matrix(grads, grads)
        a_i = compute_kernel_corrected_grads(grads, mat_k)
        mat = a_i.transpose() @ sum_a_j
        n_s = np.trace(mat)
        normed_count_array[i] = n_s
    normed_count_array /= d
    return normed_count_array

# --- Multi-processes:
#
# def pbar_listener(q, total, desc):
#     pbar = tqdm(total=total, desc=desc)
#     for item in iter(q.get, None):
#         pbar.update()
#
#
# def compute_sum_of_grads(_grads_filepath_list):
#     print("process")
#     grads_0 = np.load(_grads_filepath_list[0]).flatten()
#     sum_normed_grads = np.zeros_like(grads_0)
#     for grads_filepath in _grads_filepath_list:
#         grads = np.load(grads_filepath).flatten()
#         normed_grads = grads / np.linalg.norm(grads)
#         sum_normed_grads += normed_grads
#         # q.put(1)  # Does not matter what we put here
#     return sum_normed_grads
#
#
# def compute_soft_neighbor_count_on_disk(grads_filepath_list):
#     process_count = 4
#     # Compute sum of grads
#     grads_filepath_list_chunks = python_utils.split_list_into_chunks(grads_filepath_list, len(grads_filepath_list) // process_count)
#
#     # q = mp.Queue()
#     # proc = mp.Process(target=pbar_listener, args=(q, len(grads_filepath_list, "Sum grads: ")))
#     # proc.start()
#     # workers = [mp.Process(target=compute_sum_of_grads, args=(q, grads_filepath_list_chunk))
#     #            for grads_filepath_list_chunk in grads_filepath_list_chunks]
#     # for worker in workers:
#     #     worker.start()
#     # for worker in workers:
#     #     worker.join()
#     # q.put(None)
#     # proc.join()
#
#     with mp.Pool(process_count) as p:
#         r = list(tqdm(p.imap(compute_sum_of_grads, grads_filepath_list_chunks), total=len(grads_filepath_list), desc="Sum grads: "))
#     print(len(r))
#     exit()
#
#     # Compute scalar product with sum_grads
#     normed_count_list = []
#     for grads_filepath in tqdm(grads_filepath_list, desc="Similarity: "):
#         grads = np.load(grads_filepath).flatten()
#         normed_grads = grads / np.linalg.norm(grads)
#         normed__count = np.dot(normed_grads, sum_normed_grads)
#         normed_count_list.append(normed__count)
#     normed_count_array = np.array(normed_count_list)
#     return normed_count_array
