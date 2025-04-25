import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import re

# Directory where LightRAG data is stored
WORKING_DIR = "./my_lightrag_project"

def create_embedding_visualization():
    """Create visualization using only the text and embeddings"""
    # Load story
    try:
        with open("story.txt", 'r') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading story file: {e}")
        return None
    
    # Extract meaningful terms from the text
    important_terms = []
    
    # Extract named entities (capitalized words)
    entities = re.findall(r'\b[A-Z][a-z]+\b', text)
    entity_counts = {}
    for entity in entities:
        if len(entity) > 2:  # Skip short terms
            if entity not in entity_counts:
                entity_counts[entity] = 0
            entity_counts[entity] += 1
    
    # Get terms that appear multiple times
    important_terms = [e for e, c in entity_counts.items() if c >= 1]
    
    # Add key terms from the story
    key_terms = ["library", "books", "magical", "village", "knowledge", "discovery", "Lily", "tree", "children"]
    for term in key_terms:
        if term not in important_terms and term.capitalize() not in important_terms:
            important_terms.append(term)
    
    # Use all extracted terms
    terms = important_terms
    
    # Limit to a reasonable number
    if len(terms) > 25:
        # Sort by frequency if we have counts
        terms = sorted(terms, key=lambda t: entity_counts.get(t, 0), reverse=True)[:25]
    
    print(f"Using {len(terms)} terms for visualization")
    
    # Create embeddings using the same model as LightRAG
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(terms)
    
    # Create graph from embeddings
    G = nx.Graph()
    
    # Add nodes
    for term in terms:
        G.add_node(term)
    
    # Add edges based on embedding similarity
    threshold = 0.45  # Lower threshold to create more connections
    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            if sim > threshold:
                G.add_edge(terms[i], terms[j], weight=sim)
    
    # Create visualization
    plt.figure(figsize=(14, 12))
    
    # Compute communities for coloring
    communities = nx.community.greedy_modularity_communities(G)
    
    # Create a dictionary mapping each node to its community
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
            
    # Set node colors based on communities
    community_colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
    node_colors = [community_colors[community_map[node]] for node in G.nodes()]
    
    # Set node sizes based on connectivity (degree centrality)
    centrality = nx.degree_centrality(G)
    node_sizes = [2000 * centrality[node] + 500 for node in G.nodes()]
    
    # Create layout
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    
    # Draw edges with width based on similarity
    for u, v, d in G.edges(data=True):
        weight = d.get('weight', 0.5)
        width = weight * 4  # Scale for visibility
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Semantic Embedding Network from Story Text", fontsize=16)
    plt.figtext(0.5, 0.01, "Node size indicates importance in the network | Edge thickness shows semantic similarity", 
            ha="center", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('story_semantic_network_v3.png', dpi=300, bbox_inches='tight')
    
    print("Created visualization as 'story_semantic_network_v3.png'")
    return G

if __name__ == "__main__":
    G = create_embedding_visualization()
    if G:
        print(f"Network created with {len(G.nodes())} nodes and {len(G.edges())} edges")