import json
import scipy.sparse as sp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import copy
import pickle
import time

set_titles = ['combined', 'global_only', 'local_only', 'vit_only']
image_numbers = [5000, 8500, 11000]

for set_title in set_titles:
    for image_number in image_numbers:
        print(f"\n{'='*50}")
        print(f"Processing {set_title} with {image_number} images")
        print(f"{'='*50}\n")
        start_time = time.time()

        IMAGE_INFO = f'adjacency_matrices/individual_features/matrix_info/VIT.json'
        with open(IMAGE_INFO, 'r') as file:
            image_info = json.load(file)

        # Load sparse matrix
        print(f"Loading adjacency matrix for {set_title}...")
        adjacency_matrix_sparse = sp.load_npz(f'adjacency_matrices/{set_title}/adjacency_matrix.npz')

        # Load the JSON data
        print(f"Loading template data...")
        with open(f'clustering/standard/{set_title}_5000.json') as file:
            templates = json.load(file)

        def get_total_members(threshold, templates, cluster_sums, image_info, all_template_members_set):
            temp_templates = copy.deepcopy(templates)
            for i, entry in enumerate(image_info):
                if i in all_template_members_set:
                    continue
                
                max_sim = 0
                max_template_id = None
                for template_id, cluster_sum in enumerate(cluster_sums):
                    similarity_to_template = cluster_sum[i]
                    if similarity_to_template > max_sim:
                        max_sim = similarity_to_template
                        max_template_id = template_id
                
                if max_template_id is not None and max_sim > threshold:
                    temp_templates[max_template_id]['members'].append(i)
            
            all_members = []
            for template in temp_templates:
                all_members.extend(template['members'])
            return len(all_members)

        all_template_members = []
        for template in templates:
            all_template_members.extend(template['members'])

        # Convert to set for faster lookup
        all_template_members_set = set(all_template_members)

        # Precompute the cluster sums for all templates
        cluster_sums = []
        for template in templates:
            member_indices = template['members']
            cluster_sum = adjacency_matrix_sparse[:, member_indices].sum(axis=1).A.flatten()
            cluster_sum /= len(member_indices)
            cluster_sums.append(cluster_sum)

        # Initialize binary search parameters
        min_threshold = float(np.min(adjacency_matrix_sparse.data))
        max_threshold = float(np.max(adjacency_matrix_sparse.data))
        tolerance = 1  # Allow difference of ±1 members

        # Binary search for the right threshold
        while True:
            THRESHOLD = (min_threshold + max_threshold) / 2
            current_total = get_total_members(THRESHOLD, templates, cluster_sums, image_info, all_template_members_set)
            
            print(f"Threshold: {THRESHOLD:.3f}, Members: {current_total}")
            
            if abs(current_total - image_number) <= tolerance:
                break
            elif current_total > image_number:
                min_threshold = THRESHOLD
            else:
                max_threshold = THRESHOLD

        print(f"Final threshold: {THRESHOLD}")

        def process_image(i, entry):
            if i in all_template_members_set:
                return None
            
            max_sim = 0
            max_template_id = None
            for template_id, cluster_sum in enumerate(cluster_sums):
                similarity_to_template = cluster_sum[i]
                if similarity_to_template > max_sim:
                    max_sim = similarity_to_template
                    max_template_id = template_id
            
            if max_template_id is not None:
                if max_sim > THRESHOLD:
                    if "new_members" in templates[max_template_id]:
                        templates[max_template_id]['new_members'].append(i)
                        templates[max_template_id]['new_urls'].append(entry['url'])
                    else:
                        templates[max_template_id]['new_urls'] = []
                        templates[max_template_id]['new_members'] = []
                        templates[max_template_id]['new_members'].append(i)
                        templates[max_template_id]['new_urls'].append(entry['url'])

                    templates[max_template_id]['members'].append(i)
                    templates[max_template_id]['urls'].append(entry['url'])
                    if entry['set'] == 'kym':
                        templates[max_template_id]['platforms']['kym'] += 1
                    else:
                        templates[max_template_id]['platforms']['reddit'] += 1
                    
                    if 'template' in entry:
                        if entry['template'] in templates[max_template_id]:
                            templates[max_template_id][entry['template']] += 1
                        else:
                            templates[max_template_id][entry['template']] = 1

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, i, entry) for i, entry in enumerate(image_info)]
            for future in futures:
                future.result()  # Ensure all tasks complete

        all_members = []
        for template in templates:
            members = template['members']
            all_members.extend(members)

        print(f"Number of images clustered: {len(all_members)}")

        # Save updated templates to JSON file
        print(f"\nSaving results for {set_title}_{image_number}...")
        with open(f'clustering/templatic/{set_title}_{image_number}.json', 'w') as file:
            json.dump(templates, file, indent=4)
            
        elapsed_time = time.time() - start_time
        print(f"Completed {set_title}_{image_number} in {elapsed_time:.2f} seconds\n")