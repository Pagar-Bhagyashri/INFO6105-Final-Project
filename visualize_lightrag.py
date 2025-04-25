import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sentence_transformers import SentenceTransformer

# Directory where LightRAG data is stored
WORKING_DIR = "./my_lightrag_project"

def debug_json_structure(file_path):
    """Print the structure of a JSON file to understand its format"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"File: {file_path}")
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for key in list(data.keys())[:3]:  # Show first 3 keys
                print(f"Sample value for key '{key}': {type(data[key])}")
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())}")
        
        print("\n")
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

def create_simple_visualization():
    """Create a simple visualization to show system is working"""
    # Create a sample graph
    G = nx.DiGraph()
    G.add_node("Document", color="lightblue")
    G.add_node("RAG System", color="lightgreen")
    G.add_node("Query", color="salmon")
    G.add_node("Response", color="gold")
    
    G.add_edge("Document", "RAG System", label="Input")
    G.add_edge("Query", "RAG System", label="Question")
    G.add_edge("RAG System", "Response", label="Answer")
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes with different colors
    node_colors = [G.nodes[n].get('color', 'skyblue') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Add edge labels
    edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title('LightRAG System Architecture Visualization')
    plt.axis('off')
    plt.savefig('lightrag_architecture.png')
    plt.close()
    print("Created simple architecture visualization as 'lightrag_architecture.png'")

def main():
    print("Analyzing LightRAG data structures...")
    
    # Debug JSON files
    debug_json_structure(f"{WORKING_DIR}/vdb_chunks.json")
    debug_json_structure(f"{WORKING_DIR}/vdb_entities.json")
    debug_json_structure(f"{WORKING_DIR}/vdb_relationships.json")
    
    # Create a simple visualization that doesn't depend on the exact data structure
    create_simple_visualization()
    
    print("Visualization generated successfully!")

if __name__ == "__main__":
    main()