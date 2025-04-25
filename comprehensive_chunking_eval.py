import os
import re
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm

def download_test_data():
    """Download various test documents to evaluate chunking strategies"""
    test_data = {
        "scientific_paper": "https://raw.githubusercontent.com/karpathy/arxiv-sanity-preserver/master/data/txt/1506.02640v5.txt",
        "news_article": "https://raw.githubusercontent.com/LIAAD/KeywordExtractor-Datasets/master/datasets/cacm/docsutf8/00.txt",
        "wikipedia": "https://raw.githubusercontent.com/attardi/wikiextractor/master/doc/sample-enwiki-20171201.xml"
    }
    
    os.makedirs("test_data", exist_ok=True)
    
    for name, url in test_data.items():
        try:
            print(f"Downloading {name}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"test_data/{name}.txt", "w", encoding="utf-8") as f:
                    f.write(response.text)
            else:
                print(f"Failed to download {name}: status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    # Also add the original story.txt
    if os.path.exists("story.txt"):
        with open("story.txt", "r", encoding="utf-8") as f:
            content = f.read()
        with open("test_data/story.txt", "w", encoding="utf-8") as f:
            f.write(content)

def baseline_chunking(text, chunk_size=500):
    """Baseline chunking: simple fixed-size chunks without respecting content structure"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def improved_chunking(text, chunk_size=500):
    """Improved chunking: respects paragraphs and sentences"""
    # Split by paragraphs
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed the chunk size and we already have content
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            # Find a good breaking point - end of a sentence if possible
            sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
            if len(sentences) > 1:
                # Keep most sentences in the current chunk
                break_point = len(current_chunk) - len(sentences[-1]) - 1
                chunks.append(current_chunk[:break_point])
                current_chunk = current_chunk[break_point:] + "\n\n" + para
            else:
                # No good breaking point, just add the chunk
                chunks.append(current_chunk)
                current_chunk = para
        else:
            # Add separator if needed
            if current_chunk and not current_chunk.endswith("\n"):
                current_chunk += "\n\n"
            current_chunk += para
    
    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def count_broken_sentences(chunks):
    """Count sentences broken across chunk boundaries"""
    broken_count = 0
    for i in range(len(chunks) - 1):
        # Check if chunk ends with incomplete sentence
        if not re.search(r'[.!?]\s*$', chunks[i]):
            broken_count += 1
    return broken_count

def calculate_length_variation(chunks):
    """Calculate variation in chunk lengths (lower is better)"""
    lengths = [len(chunk) for chunk in chunks]
    mean_length = sum(lengths) / len(lengths)
    variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
    return variance

def semantic_coherence_score(chunks):
    """Estimate semantic coherence by checking for related terms within chunks"""
    # This is a simplified approximation - in a full implementation, 
    # we would use embeddings to measure true semantic coherence
    score = 0
    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        if len(sentences) > 1:
            # Calculate term overlap between adjacent sentences
            terms_by_sentence = []
            for sentence in sentences:
                terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower()))
                terms_by_sentence.append(terms)
            
            # Calculate average overlap
            overlaps = []
            for i in range(len(terms_by_sentence) - 1):
                if terms_by_sentence[i] and terms_by_sentence[i+1]:
                    overlap = len(terms_by_sentence[i] & terms_by_sentence[i+1]) / len(terms_by_sentence[i] | terms_by_sentence[i+1])
                    overlaps.append(overlap)
            
            if overlaps:
                score += sum(overlaps) / len(overlaps)
    
    return score / len(chunks) if chunks else 0

def comprehensive_chunking_evaluation():
    """Evaluate chunking strategies across multiple datasets"""
    # Make sure we have test data
    if not os.path.exists("test_data"):
        download_test_data()
    
    # Get all text files in the test_data directory
    test_files = [f for f in os.listdir("test_data") if f.endswith(".txt")]
    
    if not test_files:
        print("No test files found. Please make sure the test_data directory contains .txt files.")
        return
    
    # Prepare results storage
    results = {
        "broken_sentences": {"baseline": [], "improved": []},
        "length_variation": {"baseline": [], "improved": []},
        "semantic_coherence": {"baseline": [], "improved": []}
    }
    
    # Process each test file
    for file_name in tqdm(test_files, desc="Evaluating files"):
        file_path = os.path.join("test_data", file_name)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Skip very small files
            if len(text) < 1000:
                continue
                
            # Apply chunking strategies
            baseline_chunks = baseline_chunking(text)
            improved_chunks = improved_chunking(text)
            
            # Calculate metrics
            results["broken_sentences"]["baseline"].append(count_broken_sentences(baseline_chunks))
            results["broken_sentences"]["improved"].append(count_broken_sentences(improved_chunks))
            
            results["length_variation"]["baseline"].append(calculate_length_variation(baseline_chunks))
            results["length_variation"]["improved"].append(calculate_length_variation(improved_chunks))
            
            results["semantic_coherence"]["baseline"].append(semantic_coherence_score(baseline_chunks))
            results["semantic_coherence"]["improved"].append(semantic_coherence_score(improved_chunks))
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Calculate average improvements
    improvements = {}
    for metric in results:
        baseline_avg = np.mean(results[metric]["baseline"]) if results[metric]["baseline"] else 0
        improved_avg = np.mean(results[metric]["improved"]) if results[metric]["improved"] else 0
        
        if metric == "semantic_coherence":
            # For coherence, higher is better
            if baseline_avg > 0:
                improvements[metric] = (improved_avg / baseline_avg - 1) * 100
            else:
                improvements[metric] = 0
        else:
            # For broken sentences and length variation, lower is better
            if baseline_avg > 0:
                improvements[metric] = (baseline_avg - improved_avg) / baseline_avg * 100
            else:
                improvements[metric] = 0
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot broken sentences comparison
    axes[0].bar(["Baseline", "Improved"], 
               [np.mean(results["broken_sentences"]["baseline"]), 
                np.mean(results["broken_sentences"]["improved"])],
               color=["#3498db", "#2ecc71"])
    axes[0].set_title("Broken Sentences\n(lower is better)")
    axes[0].annotate(f"{improvements['broken_sentences']:.2f}% better", 
                   xy=(1, np.mean(results["broken_sentences"]["improved"])),
                   xytext=(0, 10),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   color='green', fontweight='bold')
    
    # Plot length variation comparison
    axes[1].bar(["Baseline", "Improved"], 
               [np.mean(results["length_variation"]["baseline"]), 
                np.mean(results["length_variation"]["improved"])],
               color=["#3498db", "#2ecc71"])
    axes[1].set_title("Length Consistency\n(lower variation is better)")
    axes[1].annotate(f"{improvements['length_variation']:.2f}% better", 
                   xy=(1, np.mean(results["length_variation"]["improved"])),
                   xytext=(0, 10),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   color='green', fontweight='bold')
    
    # Plot semantic coherence comparison
    axes[2].bar(["Baseline", "Improved"], 
               [np.mean(results["semantic_coherence"]["baseline"]), 
                np.mean(results["semantic_coherence"]["improved"])],
               color=["#3498db", "#2ecc71"])
    axes[2].set_title("Semantic Coherence\n(higher is better)")
    axes[2].annotate(f"{improvements['semantic_coherence']:.2f}% better", 
                   xy=(1, np.mean(results["semantic_coherence"]["improved"])),
                   xytext=(0, 10),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   color='green', fontweight='bold')
    
    plt.suptitle("Chunking Strategy Comparison Across Multiple Datasets", fontsize=16)
    plt.tight_layout()
    plt.savefig('comprehensive_chunking_evaluation.png', dpi=300)
    
    print("Comprehensive evaluation complete.")
    print("\nAverage improvements across all test datasets:")
    for metric, value in improvements.items():
        metric_name = metric.replace("_", " ").title()
        print(f"  {metric_name}: {value:.2f}% better")
    
    print("\nVisualization saved as 'comprehensive_chunking_evaluation.png'")

if __name__ == "__main__":
    comprehensive_chunking_evaluation()