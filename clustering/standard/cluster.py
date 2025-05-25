import pickle
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
import numpy as np
import pickle
import json
import numpy as np
import networkx as nx
import pickle
import community as community_louvain  # This is the python-louvain library
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from collections import defaultdict

set_titles = ['combined', 'local_only', 'global_only', 'vit_only']
image_numbers = [5000, 8500, 11000]

for set_title in set_titles:
    for number_of_images in image_numbers:
        print(f"\n{'='*50}")
        print(f"Processing set_title: {set_title}, number_of_images: {number_of_images}")
        print(f"{'='*50}\n")

        FILTER = True
        ONLY_KEEP = 1 # %

        # Load sparse matrix
        templates_adj_matrix = f'adjacency_matrices/{set_title}/adjacency_matrix.npz'
        adjacency_matrix_sparse = sp.load_npz(templates_adj_matrix)

        # Print the number of non-zero cells in the modified matrix
        num_non_zero_cells = adjacency_matrix_sparse.nnz
        print(f"Number of non-zero cells in the filtered matrix: {num_non_zero_cells/100000}")

        def get_total_members(threshold, adjacency_matrix_sparse):
            # Convert sparse matrix to COO format
            adjacency_matrix_coo = adjacency_matrix_sparse.tocoo()
            
            # Create mask for elements above threshold
            mask = adjacency_matrix_coo.data >= threshold
            
            # Filter the matrix
            filtered_row = adjacency_matrix_coo.row[mask]
            filtered_col = adjacency_matrix_coo.col[mask]
            filtered_data = adjacency_matrix_coo.data[mask]
            
            # Create filtered sparse matrix
            filtered_matrix = csr_matrix((filtered_data, (filtered_row, filtered_col)), 
                                       shape=adjacency_matrix_sparse.shape)
            
            # Create graph and find communities
            graph = nx.from_scipy_sparse_array(filtered_matrix)
            communities = community_louvain.best_partition(graph, resolution=1)
            
            # Count members in valid clusters (size > 1)
            cluster_dict = defaultdict(list)
            for image_index, cluster_id in communities.items():
                cluster_dict[cluster_id].append(image_index)
            
            valid_clusters = [members for members in cluster_dict.values() if len(members) > 1]
            total = sum(len(members) for members in valid_clusters)
            
            return total

        if FILTER:
            # Initialize binary search parameters
            min_threshold = float(np.min(adjacency_matrix_sparse.data))
            max_threshold = float(np.max(adjacency_matrix_sparse.data))
            target_members = number_of_images
            tolerance = 1  # Allow difference of ±50 members
            
            # Binary search for the right threshold
            while True:
                threshold = (min_threshold + max_threshold) / 2
                current_total = get_total_members(threshold, adjacency_matrix_sparse)
                
                print(f"Threshold: {threshold:.2f}, Members: {current_total}")
                
                if abs(current_total - target_members) <= tolerance:
                    break
                elif current_total > target_members:
                    min_threshold = threshold
                else:
                    max_threshold = threshold
            
            # Apply the found threshold
            adjacency_matrix_coo = adjacency_matrix_sparse.tocoo()
            mask = adjacency_matrix_coo.data >= threshold
            filtered_row = adjacency_matrix_coo.row[mask]
            filtered_col = adjacency_matrix_coo.col[mask]
            filtered_data = adjacency_matrix_coo.data[mask]
            adjacency_matrix_sparse = csr_matrix((filtered_data, (filtered_row, filtered_col)), 
                                               shape=adjacency_matrix_sparse.shape)

            print(f"Final threshold: {threshold}")
            print(f"Number of non-zero cells in the filtered matrix: {adjacency_matrix_sparse.nnz/100000}")

        # Create a graph from the filtered sparse matrix
        graph = nx.from_scipy_sparse_array(adjacency_matrix_sparse)

        # Perform Louvain community detection
        louvain_communities = community_louvain.best_partition(graph, resolution=1)

        IMAGE_INFO = f'adjacency_matrices/individual_features/matrix_info/VIT.json'
        # Load image info
        with open(IMAGE_INFO, 'r') as file:
            image_info = json.load(file)

        clusters = louvain_communities

        # Create a default dictionary to group members by cluster_id
        cluster_dict = defaultdict(list)

        # Populate the dictionary with image indices grouped by cluster_id
        for image_index, cluster_id in clusters.items():
            cluster_dict[cluster_id].append(image_index)

        # Convert the dictionary to the desired list of dictionaries
        clusters_list = [
            {'cluster_id': cluster_id, 'members': members}
            for cluster_id, members in cluster_dict.items()
        ]

        # Remove clusters with 1 member and clusters with more than 100 members
        clusters_list = [cluster for cluster in clusters_list if len(cluster['members']) > 1]

        # Calculate the total number of members across all clusters
        total_members = sum(len(cluster['members']) for cluster in clusters_list)
        print(f"Total number of members across all clusters: {total_members}")

        # Prune clusters with the lowest 5% average_edge_weight
        pruned_clusters = clusters_list

        def find_image_by_id(image_info, id):
            for image in image_info:
                if image['index_id'] == id:
                    return image
            return None


        for cluster in pruned_clusters: 
            members = cluster['members']
            platforms = {'reddit': 0, 'kym': 0}
            templates = {}
            urls = []
            for member in members:
                image = find_image_by_id(image_info, member)
                template = image.get('template', '')
                url = image.get('url', '')
                if url:
                    urls.append(url)
                if template:
                    if template in templates:
                        templates[template] += 1
                    else:
                        templates[template] = 1
                if image['set'] == 'kym':
                    platforms['kym'] += 1
                elif image['set'] == 'reddit':
                    platforms['reddit'] += 1
            cluster['templates'] = templates
            if platforms != {}:
                cluster['platforms'] = platforms
            if urls != []:
                cluster['urls'] = urls

        partition_dict = {}
        for i, cluster in enumerate(pruned_clusters): 
            members = cluster['members']
            for member in members:
                partition_dict[member] = i

        total_members = sum(len(cluster['members']) for cluster in pruned_clusters)
        print(f"Total number of images across pruned clusters: {total_members}")
        print(f"Number of clusters after pruning: {len(pruned_clusters)}")

        # Save pruned_clusters to JSON
        with open(f'clustering/standard/{set_title}_{number_of_images}.json', 'w') as file:
            json.dump(pruned_clusters, file, indent=4)