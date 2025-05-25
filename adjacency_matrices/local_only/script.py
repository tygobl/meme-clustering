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

# Load adjacency matrices
adjacency_matrices = {}
file_path = f'adjacency_matrices/individual_features/HISTO-VIT-BERT-PHASH-CLIP_TEXT-CLIP_IMAGE-SURF-FACES-SURF_NO_TEXT-BERT_NGRAMS.pkl'

with open(file_path, 'rb') as f:
    adjacency_matrices = pickle.load(f)

for feature_set in FEATURE_SETS:
    matrix = adjacency_matrices[feature_set]

    # Set diagnoal to zero
    matrix.setdiag(0)

    # Set negative values to zero
    matrix.data[matrix.data < 0] = 0

    # Save the filtered matrix
    adjacency_matrices[feature_set] = matrix

def normalize_matrix_shifted(matrix, target_mean=2):
    """Normalize non-zero values of a sparse matrix to follow a standard normal distribution, centered around target mean."""
    # Get the non-zero values
    non_zero_values = matrix.data
    
    # Calculate mean and standard deviation of non-zero values
    mean = np.mean(non_zero_values)
    std = np.std(non_zero_values)
    
    # Avoid division by zero
    if std > 0:
        # Apply Z-score normalization
        normalized_values = (non_zero_values - mean) / std
    else:
        # If std is 0, set normalized values to 0
        normalized_values = non_zero_values - mean
    
    # Shift the distribution to be centered around the target mean (1)
    shifted_values = normalized_values + target_mean
    
    # Replace the matrix's data with the normalized and shifted values
    normalized_shifted_matrix = csr_matrix((shifted_values, matrix.indices, matrix.indptr), shape=matrix.shape)
    
    return normalized_shifted_matrix

# Apply normalization to each feature-specific adjacency matrix
normalized_matrices = {}

for feature_set, matrix in adjacency_matrices.items():
    print("Normalize")
    normalized_matrices[feature_set] = normalize_matrix_shifted(matrix)


for feature_set in FEATURE_SETS:
    matrix = normalized_matrices[feature_set]

    # Set diagnoal to zero
    matrix.setdiag(0)

    # Set negative values to zero
    matrix.data[matrix.data < 0] = 0

    # Save the filtered matrix
    normalized_matrices[feature_set] = matrix


combined_matrix =  normalized_matrices['FACES'] + normalized_matrices['SURF_NO_TEXT']  

save_npz(f'adjacency_matrices/local_only/adjacency_matrix.npz', combined_matrix)