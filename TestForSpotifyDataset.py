import pandas as pd
from patterns import apriori, association_rules, find_support
import numpy as np
from itertools import chain
# Load the dataset
combined_dfForFPA = pd.read_csv('combined_tracks_info_ArijitSingh_KK.csv')


# Load the Spotify dataset (assuming 'energy', 'valence', and 'danceability' columns exist)
spotify_data = combined_dfForFPA

# 1. Discretize 'energy', 'valence', and 'danceability' into 'Low', 'Medium', 'High'
def categorize_feature(feature_column, num_bins=3):
    categories = ['Low', 'Medium', 'High']
    return pd.cut(feature_column, bins=num_bins, labels=categories)

# Apply the discretization function to the relevant features
spotify_data['energy_category'] = categorize_feature(spotify_data['energy'])
spotify_data['valence_category'] = categorize_feature(spotify_data['valence'])
spotify_data['danceability_category'] = categorize_feature(spotify_data['danceability'])

# 2. Convert the categorized features into a transactional format (list of itemsets)
# Combine the categories into a single list of itemsets
itemsets = spotify_data.apply(
    lambda row: [
        f"energy_{row['energy_category']}",
        f"valence_{row['valence_category']}",
        f"danceability_{row['danceability_category']}"
    ], axis=1
).tolist()

# 3. Apply the Apriori algorithm to discover frequent itemsets
# min_support defines the minimum threshold for an itemset to be considered frequent
min_support = 0.1
frequent_itemsets = apriori(itemsets, threshold=min_support)

# 4. Generate association rules from the discovered frequent itemsets
# Use confidence as the metric and a minimum threshold for filtering rules
metric = "confidence"
metric_threshold = 0.5
association_rules_list = association_rules(itemsets, frequent_itemsets, metric, metric_threshold)

# 5. Sort the rules by 'lift' to find the most interesting or impactful ones
# Convert the rules to a DataFrame for easier sorting and display
rules_df = pd.DataFrame(
    association_rules_list,
    columns=['antecedents', 'consequents', 'metric_value']
)

# Calculate additional metrics: lift, confidence, and support
rules_df['support'] = rules_df['antecedents'].apply(lambda x: find_support(x, itemsets))
rules_df['confidence'] = rules_df.apply(
    lambda row: row['metric_value'] if metric == "confidence" else None, axis=1
)
rules_df['lift'] = rules_df.apply(
    lambda row: row['confidence'] / find_support(row['consequents'], itemsets) if row['confidence'] else None, axis=1
)

# Sort by lift
sorted_rules = rules_df.sort_values(by='lift', ascending=False)

# 6. Display the most relevant columns of the association rules DataFrame
print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


#Function to see the Radial Visualization  for it 
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Filter rules for the top 10 based on lift (you can adjust this)
top_rules = sorted_rules.sort_values(by='lift', ascending=False).head(10)

# Create a directed graph
G = nx.DiGraph()

# Add edges with antecedents as start node and consequents as end node
for i, rule in top_rules.iterrows():
    G.add_edge(', '.join(list(rule['antecedents'])),
               ', '.join(list(rule['consequents'])),
               weight=rule['lift'])

# Radial layout (circular node positioning)
pos = nx.spring_layout(G, k=1, seed=42)

# Calculate node sizes based on support, but ensure they match the number of nodes
# Get a list of all nodes in the graph
all_nodes = list(G.nodes())

# Create a dictionary mapping node names to their sizes
node_sizes = {}
for _, rule in top_rules.iterrows():
    antecedent = ', '.join(list(rule['antecedents']))
    consequent = ', '.join(list(rule['consequents']))
    node_sizes[antecedent] = 5000 * rule['support']  # Size for antecedent node
    node_sizes[consequent] = 5000 * rule['support']  # Size for consequent node

# Get the node sizes in the correct order for drawing
node_sizes_list = [node_sizes.get(node, 1000) for node in all_nodes]  # Default size if not in top_rules

# Customize node colors based on confidence
node_color = sns.color_palette("coolwarm", len(G))

# Draw the nodes with proportional sizes
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_sizes_list, alpha=0.85)

# Draw the edges with varying thickness based on lift
edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20,
                               edge_color='grey', width=[G[u][v]['weight'] for u,v in G.edges])

# Draw labels for nodes
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='black')

# Add edge labels (showing the lift of each rule)
edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Title and visual adjustments
plt.title('Top 10 Association Rules (Radial Visualization)', fontsize=14)
plt.axis('off')  # Turn off the axis
plt.show()