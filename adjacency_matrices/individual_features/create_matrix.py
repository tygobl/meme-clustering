import faiss
import pickle
import numpy as np
import scipy.sparse as sp

# Change this to the feature set you want to create the matrix for
FEATURE_SET = 'BERT_NGRAMS'
INDEX_FILE = f'faiss/indices/{FEATURE_SET}.idx'
STARTING_POINTS_PERC = 1
NUM_RETRIEVE = 100
THRESHOLD = 0.001
DEBUG_MODE = True

# Load the index
index = faiss.read_index(INDEX_FILE)

# Load the data
with open('kym_bert_ngrams.pkl', 'rb') as f:
    data_kym = pickle.load(f)

with open('reddit_bert_ngrams.pkl', 'rb') as f:
    data_reddit = pickle.load(f)

features = np.concatenate((np.array(list(data_kym.values())), np.array(list(data_reddit.values()))), axis=0)

# Initialize adjacency matrix
num_images = index.ntotal
adjacency_matrix = np.zeros((num_images, num_images), dtype=np.float32)

# Select a random subset of images as starting points
ids_list = np.arange(num_images)
initial_selection_size = int(STARTING_POINTS_PERC * num_images)
starting_point_list  = np.random.choice(ids_list, size=initial_selection_size, replace=False)

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def compute_similarity(distance):
    return 1 - np.tanh(distance)

def update_adjacency_matrix(adjacency_matrix, index, ind):
    start_feature = np.expand_dims(features[int(ind)], axis=0)
    start_feature = normalize_vectors(start_feature)

    # Obtain distances and indices of neighbors
    distances, indices = index.search(start_feature, NUM_RETRIEVE)
    distances = np.abs(distances)  # Make distances absolute

    # Update adjacency matrix
    for i_dist, i_img in enumerate(indices[0]):
        distance = distances[0][i_dist]
        similarity = compute_similarity(distance)
        if similarity > THRESHOLD:
            adjacency_matrix[int(ind)][int(i_img)] = similarity
            adjacency_matrix[int(i_img)][int(ind)] = similarity

for i, ind in enumerate(starting_point_list):
    update_adjacency_matrix(adjacency_matrix, index, ind)

# Save the sparse matrix
adjacency_matrix_sparse = sp.csr_matrix(adjacency_matrix)
sp.save_npz(f'adjacency_matrices/individual_features/{FEATURE_SET}.npz', adjacency_matrix_sparse)