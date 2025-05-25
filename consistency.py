import json
import matplotlib.pyplot as plt

# Set the font to a serif one
plt.rcParams["font.family"] = "Arial"

# Function to calculate the consistency metric for a cluster
def calculate_consistency_metric(cluster):
    clustering = cluster.get("templates", {})
    total_images = sum(clustering.values())
    
    if total_images == 0:
        return 0  # No KYM images, so consistency is 0
    
    if total_images == 1:
        return 0  # No KYM images, so consistency is 0
    
    max_template_count = max(clustering.values())

    return max_template_count / total_images

# Load and process clusters from a given file
def process_clusters(file_path):
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    
    consistency_metrics = []
    kym_images_list = []
    for cluster in clusters:
        total_images = sum(cluster.get("platforms", {}).values())
        kym_images = cluster.get("platforms", {}).get("kym", 0)
        
        if kym_images > 2:  # Only consider clusters with more than 4 KYM images
            metric = calculate_consistency_metric(cluster)
            consistency_metrics.append(metric)
            kym_images_list.append(kym_images)
    
    return consistency_metrics, kym_images_list

# Process cluster files for 5000, 8000, and 11000 clusters
consistency_metrics_combined_std_5000, kym_images_combined__std_5000 = process_clusters('clustering/standard/combined_5000.json')
consistency_metrics_combined_std_8500, kym_images_combined__std_8500 = process_clusters('clustering/standard/combined_8500.json')
consistency_metrics_combined_std_11000, kym_images_combined__std_11000 = process_clusters('clustering/standard/combined_11000.json')

consistency_metrics_global_std_5000, kym_images_global__std_5000 = process_clusters('clustering/standard/global_only_5000.json')
consistency_metrics_global_std_8500, kym_images_global__std_8500 = process_clusters('clustering/standard/global_only_8500.json')
consistency_metrics_global_std_11000, kym_images_global__std_11000 = process_clusters('clustering/standard/global_only_11000.json')

consistency_metrics_local_std_5000, kym_images_local__std_5000 = process_clusters('clustering/standard/local_only_5000.json')
consistency_metrics_local_std_8500, kym_images_local__std_8500 = process_clusters('clustering/standard/local_only_8500.json')
consistency_metrics_local_std_11000, kym_images_local__std_11000 = process_clusters('clustering/standard/local_only_11000.json')

consistency_metrics_vit_std_5000, kym_images_vit__std_5000 = process_clusters('clustering/standard/vit_only_5000.json')
consistency_metrics_vit_std_8500, kym_images_vit__std_8500 = process_clusters('clustering/standard/vit_only_8500.json')
consistency_metrics_vit_std_11000, kym_images_vit__std_11000 = process_clusters('clustering/standard/vit_only_11000.json')


# NON STANDARD

consistency_metrics_local_nstd_5000, kym_images_local__nstd_5000 = process_clusters('clustering/standard/local_only_5000.json')
consistency_metrics_local_nstd_8500, kym_images_local__nstd_8500 = process_clusters('clustering/templatic/local_only_8500.json')
consistency_metrics_local_nstd_11000, kym_images_local__nstd_11000 = process_clusters('clustering/templatic/local_only_11000.json')

consistency_metrics_global_nstd_5000, kym_images_global__nstd_5000 = process_clusters('clustering/standard/global_only_5000.json')
consistency_metrics_global_nstd_8500, kym_images_global__nstd_8500 = process_clusters('clustering/templatic/global_only_8500.json')
consistency_metrics_global_nstd_11000, kym_images_global__nstd_11000 = process_clusters('clustering/templatic/global_only_11000.json')

consistency_metrics_combined_nstd_5000, kym_images_combined__nstd_5000 = process_clusters('clustering/standard/combined_5000.json')
consistency_metrics_combined_nstd_11000, kym_images_combined__nstd_11000 = process_clusters('clustering/templatic/combined_11000.json')
consistency_metrics_combined_nstd_8500, kym_images_combined__nstd_8500 = process_clusters('clustering/templatic/combined_8500.json')


consistency_metrics_vit_nstd_5000, kym_images_vit__nstd_5000 = process_clusters('clustering/standard/vit_only_5000.json')
consistency_metrics_vit_nstd_8500, kym_images_vit__nstd_8500 = process_clusters('clustering/templatic/vit_only_8500.json')
consistency_metrics_vit_nstd_11000, kym_images_vit__nstd_11000 = process_clusters('clustering/templatic/vit_only_11000.json')

# Function to calculate the average consistency metric
def average_consistency(metrics):
    if not metrics:
        return 0
    return sum(metrics) / len(metrics)

# Function to calculate the weighted average consistency metric
def weighted_average_consistency(metrics, weights):
    if not metrics or not weights or sum(weights) == 0:
        return 0
    weighted_sum = sum(m * w for m, w in zip(metrics, weights))
    return weighted_sum / sum(weights)

# # Function to calculate the weighted average consistency metric
# def weighted_average_consistency(metrics, weights):
#     if not metrics:
#         return 0
#     return sum(metrics) / len(metrics)

# Calculate weighted average consistency metrics
weighted_avg_consistency_combined_std_5000 = weighted_average_consistency(consistency_metrics_combined_std_5000, kym_images_combined__std_5000)
weighted_avg_consistency_combined_std_8500 = weighted_average_consistency(consistency_metrics_combined_std_8500, kym_images_combined__std_8500)
weighted_avg_consistency_combined_std_11000 = weighted_average_consistency(consistency_metrics_combined_std_11000, kym_images_combined__std_11000)

weighted_avg_consistency_global_std_5000 = weighted_average_consistency(consistency_metrics_global_std_5000, kym_images_global__std_5000)
weighted_avg_consistency_global_std_8500 = weighted_average_consistency(consistency_metrics_global_std_8500, kym_images_global__std_8500)
weighted_avg_consistency_global_std_11000 = weighted_average_consistency(consistency_metrics_global_std_11000, kym_images_global__std_11000)

weighted_avg_consistency_local_std_5000 = weighted_average_consistency(consistency_metrics_local_std_5000, kym_images_local__std_5000)
weighted_avg_consistency_local_std_8500 = weighted_average_consistency(consistency_metrics_local_std_8500, kym_images_local__std_8500)
weighted_avg_consistency_local_std_11000 = weighted_average_consistency(consistency_metrics_local_std_11000, kym_images_local__std_11000)


weighted_avg_consistency_vit_std_5000 = weighted_average_consistency(consistency_metrics_vit_std_5000, kym_images_vit__std_5000)
weighted_avg_consistency_vit_std_8500 = weighted_average_consistency(consistency_metrics_vit_std_8500, kym_images_vit__std_8500)
weighted_avg_consistency_vit_std_11000 = weighted_average_consistency(consistency_metrics_vit_std_11000, kym_images_vit__std_11000)



# NON STANDARD

weighted_avg_consistency_local_nstd_5000 = weighted_average_consistency(consistency_metrics_local_nstd_5000, kym_images_local__nstd_5000)
weighted_avg_consistency_local_nstd_8500 = weighted_average_consistency(consistency_metrics_local_nstd_8500, kym_images_local__nstd_8500)
weighted_avg_consistency_local_nstd_11000 = weighted_average_consistency(consistency_metrics_local_nstd_11000, kym_images_local__nstd_11000)



weighted_avg_consistency_global_nstd_5000 = weighted_average_consistency(consistency_metrics_global_nstd_5000, kym_images_global__nstd_5000)
weighted_avg_consistency_global_nstd_8500 = weighted_average_consistency(consistency_metrics_global_nstd_8500, kym_images_global__nstd_8500)
weighted_avg_consistency_global_nstd_11000 = weighted_average_consistency(consistency_metrics_global_nstd_11000, kym_images_global__nstd_11000)



weighted_avg_consistency_combined_nstd_5000 = weighted_average_consistency(consistency_metrics_combined_nstd_5000, kym_images_combined__nstd_5000)
weighted_avg_consistency_combined_nstd_8500 = weighted_average_consistency(consistency_metrics_combined_nstd_8500, kym_images_combined__nstd_8500)

weighted_avg_consistency_combined_nstd_11000 = weighted_average_consistency(consistency_metrics_combined_nstd_11000, kym_images_combined__nstd_11000)


weighted_avg_consistency_vit_nstd_5000 = weighted_average_consistency(consistency_metrics_vit_nstd_5000, kym_images_vit__nstd_5000)
weighted_avg_consistency_vit_nstd_8500 = weighted_average_consistency(consistency_metrics_vit_nstd_8500, kym_images_vit__nstd_8500)
weighted_avg_consistency_vit_nstd_11000 = weighted_average_consistency(consistency_metrics_vit_nstd_11000, kym_images_vit__nstd_11000)

# Create the table
table = [
    ["Dataset", "5000 images", "8000 images", "11000 images"],
    ["Combined (standard)", weighted_avg_consistency_combined_std_5000, weighted_avg_consistency_combined_std_8500, weighted_avg_consistency_combined_std_11000],
    ["Global (standard)", weighted_avg_consistency_global_std_5000, weighted_avg_consistency_global_std_8500, weighted_avg_consistency_global_std_11000],
    ["Local (standard)", weighted_avg_consistency_local_std_5000, weighted_avg_consistency_local_std_8500, weighted_avg_consistency_local_std_11000],
    ["VIT (standard)", weighted_avg_consistency_vit_std_5000, weighted_avg_consistency_vit_std_8500, weighted_avg_consistency_vit_std_11000],
    ["Global (templatic)", weighted_avg_consistency_global_nstd_5000, weighted_avg_consistency_global_nstd_8500, weighted_avg_consistency_global_nstd_11000],
    ["Local (templatic)", weighted_avg_consistency_local_nstd_5000, weighted_avg_consistency_local_nstd_8500, weighted_avg_consistency_local_nstd_11000],
    ["Combined (templatic)", weighted_avg_consistency_combined_nstd_5000, weighted_avg_consistency_combined_nstd_8500, weighted_avg_consistency_combined_nstd_11000],
    ["VIT (templatic)", weighted_avg_consistency_vit_nstd_5000, weighted_avg_consistency_vit_nstd_8500, weighted_avg_consistency_vit_nstd_11000]
]

# Print the table
for row in table:
    if row[0] == "Dataset":
        print("{:<20}\t{:<20}\t{:<20}\t{:<20}".format(row[0], row[1], row[2], row[3]))    
    else:
        print("{:<20}\t{:<.2f}\t{:<.2f}\t{:<.2f}".format(row[0], row[1], row[2], row[3]))
