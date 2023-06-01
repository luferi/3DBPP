import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community as community_louvain
import random


def load_dataset(file_path):
    df = pd.read_csv(file_path, sep=',', dtype=str,
                     usecols=['Order', 'Product', 'Quantity', 'Length', 'Width',
                              'Height', 'Weight'])
    df[['Quantity', 'Length', 'Width', 'Height']] = df[
        ['Quantity', 'Length', 'Width', 'Height']].astype(int)
    df[['Weight']] = df[['Weight']].astype(float)
    return df


def get_b_dict(a_list, freq_table):
    b_dict = {}
    for a in a_list:
        b_dict[a] = [(b, freq_table.loc[a][('Quantity', b)]) for b in freq_table.columns.levels[1] if
                     freq_table.loc[a][('Quantity', b)] > 0]
    return b_dict


def cosine_similarity_community_clustering(freq_table, threshold=100):
    # calculate cosine similarity matrix
    cos_sim = cosine_similarity(freq_table)

    cos_sim_df = pd.DataFrame(cos_sim, index=freq_table.index, columns=freq_table.index)

    # convert the adjacency matrix to a NetworkX graph
    np.fill_diagonal(cos_sim_df.values, 0)

    g = nx.from_pandas_adjacency(cos_sim_df)

    # run the Louvain algorithm to detect communities
    partition = community_louvain.best_partition(g, resolution=1)

    community_dict = {}
    for node, community in partition.items():
        if community in community_dict:
            community_dict[community].append(node)
        else:
            community_dict[community] = [node]

    final_community_dict = {}
    for community, nodes in community_dict.items():
        if len(nodes) <= threshold:
            final_community_dict[community] = nodes
        else:
            sub_community_dict = cosine_similarity_community_clustering(freq_table.loc[nodes], threshold)
            for sub_community, sub_nodes in sub_community_dict.items():
                final_community_dict[f"{community}.{sub_community}"] = sub_nodes

    return final_community_dict


def cluster_distribution(communities_flat, total_data_points):
    cluster_dist = {}
    for cluster_num, cluster_data in communities_flat.items():
        num_points = len(cluster_data)
        percent_points = (num_points / total_data_points) * 100
        cluster_dist[cluster_num] = percent_points
    return cluster_dist


def sample_cluster(cluster_dist):
    total = sum(cluster_dist.values())
    rand_val = random.uniform(0, total)
    curr_total = 0
    for cluster, percent in cluster_dist.items():
        curr_total += percent
        if curr_total >= rand_val:
            return cluster


def generate_representative_data_point(b_dict):
    b_occurrences = {}
    c_distribution = {}

    for a, b_list in b_dict.items():
        for b, c in b_list:
            if b in b_occurrences:
                b_occurrences[b] += 1
            else:
                b_occurrences[b] = 1

            if b not in c_distribution:
                c_distribution[b] = []

            c_distribution[b].append(c)

    selected_bs = random.choices(list(b_occurrences.keys()), list(b_occurrences.values()),
                                 k=random.randint(1, len(b_occurrences)))

    representative_data_point = {}
    for b in selected_bs:
        c_values = c_distribution[b]
        random_c = random.choice(c_values)
        representative_data_point[b] = random_c

    return representative_data_point


def generate_random_non_repeating(a, b):
    numbers = list(range(a, b + 1))  # Create a list of numbers from a to b
    random.shuffle(numbers)  # Shuffle the list randomly
    return numbers.pop()  # Remove and return the last number from the shuffled list


def create_new_dataset(new_df, original_df, values_dict):
    document = generate_random_non_repeating(100000, 999999)

    for product, src_qty in values_dict.items():
        # Retrieve the matching row from the original_df for the product
        matching_row = original_df.loc[original_df['Product'] == product].iloc[0]

        # Copy the values from the matching row to create a new row
        new_row = {
            'Order': document,
            'Product': product,
            'Quantity': src_qty,
            'Length': matching_row['Length'],
            'Width': matching_row['Width'],
            'Height': matching_row['Height'],
            'Weight': matching_row['Weight']
        }

        new_row_df = pd.DataFrame([new_row])
        new_df = pd.concat([new_df, new_row_df], ignore_index=True)

    return new_df


def main_workflow(input_file_path, num_samples, output_file_path):
    print("Loading the dataset...")
    df = load_dataset(input_file_path)

    print("Preparing main frequency table...")
    freq_table = df.groupby(['Order', 'Product']).agg({'Quantity': 'sum'}).unstack(fill_value=0)
    print(freq_table)

    print("Calculating cosine sim for main frequency table...")
    communities_flat = cosine_similarity_community_clustering(freq_table, threshold=100)

    print("Calculating cluster distribution...")
    cluster_dist = cluster_distribution(communities_flat, len(df))

    new_df = pd.DataFrame(columns=df.columns)  # Empty DataFrame with the same columns as the original df

    for i in range(num_samples):
        sampled_cluster = sample_cluster(cluster_dist)
        b_dict = get_b_dict(communities_flat[sampled_cluster], freq_table)
        data_point = generate_representative_data_point(b_dict)
        new_df = create_new_dataset(new_df, df, data_point)

    new_df.to_csv(output_file_path, index=False)


def sample_generator(freq_table, communities_flat, cluster_dist):
    while True:
        sampled_cluster = sample_cluster(cluster_dist)
        b_dict = get_b_dict(communities_flat[sampled_cluster], freq_table)
        data_point = generate_representative_data_point(b_dict)
        yield data_point


if __name__ == '__main__':

    main_workflow(r"C:\LDRstuff\Private\MyDevel\FLAP\DataSetPublic\Dataset1000.csv", 100, r"C:\LDRstuff\Private\MyDevel\FLAP\DataSetPublic\NewDS.csv")

    df = load_dataset(r"C:\LDRstuff\Private\MyDevel\FLAP\DataSetPublic\Dataset1000.csv")
    freq_table = df.groupby(['Order', 'Product']).agg({'Quantity': 'sum'}).unstack(fill_value=0)
    communities_flat = cosine_similarity_community_clustering(freq_table, threshold=100)
    cluster_dist = cluster_distribution(communities_flat, len(df))

    generator = sample_generator(freq_table, communities_flat, cluster_dist)

    # Generate the first 100 samples
    for _ in range(100):
        data_point = next(generator)
        # Process the data point here
        print(data_point)
