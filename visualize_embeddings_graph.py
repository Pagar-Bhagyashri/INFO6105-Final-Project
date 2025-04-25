import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import re

# Directory where LightRAG data is stored
WORKING_DIR = "./my_lightrag_project"

def extract_key_entities(text):
    """Extract potential key entities from text using a simpler approach"""
    # Simple sentence splitting by punctuation
    sentences = re.split(r'[.!?]+', text)
    
    # Extract potential named entities (simplified approach)
    entities = []
    for sentence in sentences:
        # Look for capitalized words that might be names or important concepts
        potential_entities = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        entities.extend(potential_entities)
    
    # Count frequencies and get most common
    entity_counts = {}
    for entity in entities:
        if entity not in entity_counts:
            entity_counts[entity] = 0
        entity_counts[entity] += 1
    
    # Filter to entities that appear more than once
    key_entities = [e for e, c in entity_counts.items() if c >= 1]
    
    return key_entities

def create_embedding_graph():
    """Create a graph of word embeddings based on the story text"""
    # Load the story text
    try:
        with open("story.txt", 'r') as f:
            story_text = f.read()
    except Exception as e:
        print(f"Error reading story.txt: {e}")
        return
    
    # Extract key entities from the text
    entities = extract_key_entities(story_text)
    print(f"Found {len(entities)} potential entities in the text")
    
    # Limit to top 15 entities if there are too many
    if len(entities) > 15:
        entities = entities[:15]
    
    # Add key terms from the story that might not be detected as entities
    key_terms = ["library", "books", "magical", "village", "knowledge", "discovery"]
    for term in key_terms:
        if term.capitalize() not in entities and term not in entities:
            entities.append(term)
    
    # Get embeddings for entities
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(entities)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for entity in entities:
        G.add_node(entity)
    
    # Add edges based on embedding similarity
    threshold = 0.7  # Similarity threshold
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            if similarity > threshold:
                G.add_edge(entities[i], entities[j], weight=similarity)
    
    # Create a visualization
    plt.figure(figsize=(14, 10))
    
    # Set positions and node colors
    pos = nx.spring_layout(G, seed=42)
    
    # Create color map
    node_colors = plt.cm.tab20(np.linspace(0, 1, len(entities)))
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edges with varying thickness based on weight
    for u, v, d in G.edges(data=True):
        width = d.get('weight', 1) * 3  # Scale the width
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7)
    
    plt.title('Semantic Relationships Between Key Terms in the Story')
    plt.axis('off')
    plt.savefig('embeddings_network.png', dpi=300)
    plt.close()
    print("Created embeddings network visualization as 'embeddings_network.png'")
    
    # Also create a more sophisticated visualization with different colored communities
    try:
        communities = nx.community.greedy_modularity_communities(G)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
        
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes colored by community
        node_colors = {}
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = colors[i]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=[node_colors.get(node, 'gray') for node in G.nodes()],
                              node_size=700)
        
        # Draw edges with varying thickness based on weight
        for u, v, d in G.edges(data=True):
            width = d.get('weight', 1) * 3  # Scale the width
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        plt.title('Semantic Communities in Story Terms')
        plt.axis('off')
        plt.savefig('semantic_communities.png', dpi=300)
        plt.close()
        print("Created community detection visualization as 'semantic_communities.png'")
    except Exception as e:
        print(f"Could not create community visualization: {e}")

if __name__ == "__main__":
    create_embedding_graph()
    print("Done!")