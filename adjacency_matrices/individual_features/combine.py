import numpy as np
import networkx as nx
import pickle
import community as community_louvain  # This is the python-louvain library
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import json
from scipy.sparse import save_npz
import matplotlib.pyplot as plt
import os
np.set_printoptions(suppress=True, precision=3)

FEATURE_SETS = ['HISTO', 'VIT', 'BERT', 'PHASH', 'CLIP_TEXT', 'CLIP_IMAGE', 'SURF', 'FACES', 'SURF_NO_TEXT', 'BERT_NGRAMS']

# Create a name that combines all feature sets with '-'
combined_name = '-'.join(FEATURE_SETS)
print(f"Combined name: {combined_name}")

# Load adjacency matrices
adjacency_matrices = {}
file_path = f'adjacency_matrices/individual_features/{combined_name}.pkl'

largest_matrix = ''
largest_size = 0
for feature_set in FEATURE_SETS:
    adjacency_matrices[feature_set] = sp.load_npz(f'adjacency_matrices/individual_features/{feature_set}.npz')
    size = adjacency_matrices[feature_set].shape[0]
    if size > largest_size:
        largest_matrix = feature_set
        largest_size = size

combined_matrix = adjacency_matrices[largest_matrix]

mappings = {}
id_to_url_features = {}
url_to_id_features = {}

urls_in_largest = []
with open(f'adjacency_matrices/individual_features/matrix_info/{largest_matrix}.json') as f:
    mapping = json.load(f)
mappings[largest_matrix] = mapping
id_to_url_features[largest_matrix] = {}
url_to_id_features[largest_matrix] = {}
for entry in mapping:
    index_id = entry.get('index_id', entry.get('id', ''))
    url = entry.get('url', entry.get('image_url', ''))
    if url:
        urls_in_largest.append(url)
    id_to_url_features[largest_matrix][index_id] = url
    url_to_id_features[largest_matrix][url] = index_id

for feature_set in FEATURE_SETS:
    if feature_set == largest_matrix:
        continue 
    with open(f'adjacency_matrices/individual_features/matrix_info/{feature_set}.json') as f:
        mapping = json.load(f)
    mappings[feature_set] = mapping
    id_to_url_features[feature_set] = {}
    url_to_id_features[feature_set] = {}
    for entry in mapping:
        index_id = entry.get('index_id', entry.get('id', ''))
        url = entry.get('url', entry.get('image_url', ''))
        id_to_url_features[feature_set][index_id] = url
        url_to_id_features[feature_set][url] = index_id


ids_list = np.arange(largest_size)

ids_across_sets = {}
for id in ids_list:
    ids_across_sets[id] = {}
    url = id_to_url_features[largest_matrix][id]
    for feature_set in FEATURE_SETS:
        if feature_set == largest_matrix:
            continue
        if url in url_to_id_features[feature_set]:
            id_in_set = url_to_id_features[feature_set][url]
            ids_across_sets[id][feature_set] = id_in_set

for feature_set in FEATURE_SETS:
    if feature_set == largest_matrix:
        continue

    add_row = []
    for id in ids_list:
        if feature_set not in ids_across_sets[id]:
            add_row.append(id)

    current_matrix = adjacency_matrices[feature_set]
    num_new_nodes = len(add_row)

    if num_new_nodes > 0:  
        # Convert the sparse matrix to a dense matrix
        dense_matrix = current_matrix.toarray()

        # Create a list of zero rows and columns to be inserted
        num_rows, num_cols = dense_matrix.shape
        new_rows = np.zeros((len(add_row), num_cols))
        new_cols = np.zeros((num_rows + len(add_row), len(add_row)))

        # Sort indices to maintain order during insertion
        add_row_sorted = sorted(add_row)
        add_col_sorted = add_row_sorted

        # Insert new rows at specific indices in a batch
        insert_indices_rows = [i for i in range(num_rows + len(add_row)) if i not in add_row_sorted]
        dense_matrix_with_rows = np.zeros((num_rows + len(add_row), num_cols))
        dense_matrix_with_rows[insert_indices_rows, :] = dense_matrix
        for idx, row in enumerate(add_row_sorted):
            dense_matrix_with_rows[row, :] = new_rows[idx]

        # Insert new columns at specific indices in a batch
        insert_indices_cols = [i for i in range(num_cols + len(add_row)) if i not in add_col_sorted]
        dense_matrix_with_cols = np.zeros((num_rows + len(add_row), num_cols + len(add_row)))
        dense_matrix_with_cols[:, insert_indices_cols] = dense_matrix_with_rows
        for idx, col in enumerate(add_col_sorted):
            dense_matrix_with_cols[:, col] = new_cols[:, idx]
    

        current_matrix = sp.csr_matrix(dense_matrix_with_cols)    

        adjacency_matrices[feature_set] = current_matrix    

# Save adjacency matrices to pickle
with open(f'adjacency_matrices/individual_features/{combined_name}.pkl', 'wb') as f:
    pickle.dump(adjacency_matrices, f)