import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

def get_large_windows_file():
    """Get a large text file that's guaranteed to be on Windows systems"""
    potential_paths = [
        r"C:\Windows\System32\license.rtf",
        r"C:\Windows\System32\eula.txt",
        r"C:\Windows\System32\Drivers\etc\services",
        r"C:\Windows\System32\en-US\erofflps.txt",
        r"C:\Windows\Help\en-US\windows.hlp"
    ]
    
    # Try to find at least one file
    for path in potential_paths:
        if os.path.exists(path):
            try:
                # Try reading as text
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if len(content) > 5000:  # Need a reasonably sized file
                        print(f"Using system file: {path} ({len(content)} characters)")
                        return content
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    # If no system file works, use README files from Python packages
    print("Using Python package documentation as large text...")
    combined_text = ""
    package_dirs = [r"venv\Lib\site-packages"]
    
    for package_dir in package_dirs:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                if file.lower() in ["readme.md", "readme.txt", "readme.rst"]:
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            combined_text += f"\n\n# {os.path.join(root, file)}\n\n{content}\n\n"
                    except Exception:
                        pass
    
    if len(combined_text) > 5000:
        print(f"Using combined package documentation ({len(combined_text)} characters)")
        return combined_text
    
    # Last resort - use LightRAG documentation
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
            print(f"Using LightRAG README ({len(content)} characters)")
            return content
    except Exception:
        print("Could not find any suitable text files.")
        return None

def baseline_chunking(text, chunk_size=1000):
    """Baseline chunking: fixed-size chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def improved_chunking(text, chunk_size=1000):
    """Improved chunking: respects document structure"""
    # Split by paragraphs
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed the chunk size and we already have content
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            # Try to break at sentence boundary
            sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
            if len(sentences) > 1:
                # Join most sentences
                break_idx = len(sentences) - 1
                chunks.append("".join(sentences[:break_idx]))
                current_chunk = sentences[-1] + "\n\n" + para
            else:
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

def count_broken_paragraphs(chunks, text):
    """Count paragraphs broken across chunk boundaries"""
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    # Count paragraphs
    total_paragraphs = len(paragraphs)
    
    # Count paragraphs that appear in chunks
    broken_count = 0
    for para in paragraphs:
        if len(para) < 10:  # Skip very short paragraphs
            continue
            
        # Check if paragraph is broken across chunks
        found_complete = False
        for chunk in chunks:
            if para in chunk:
                found_complete = True
                break
                
        if not found_complete:
            broken_count += 1
    
    return broken_count

def count_broken_sentences(chunks):
    """Count sentences broken across chunk boundaries"""
    broken_count = 0
    for i in range(len(chunks) - 1):
        # Check if chunk ends with incomplete sentence
        if not re.search(r'[.!?]\s*$', chunks[i]):
            broken_count += 1
    return broken_count

def calculate_length_variation(chunks):
    """Calculate length consistency"""
    lengths = [len(chunk) for chunk in chunks]
    mean_length = sum(lengths) / len(chunks)
    variance = sum((length - mean_length) ** 2 for length in chunks) / len(lengths)
    return variance

def evaluate_queries(chunks, queries):
    """Evaluate query performance"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Store results
    results = {
        "relevance_scores": [],
        "term_overlap": []
    }
    
    for query in queries:
        # Encode query
        query_embedding = model.encode([query])[0]
        
        # Find most similar chunks
        similarities = []
        for chunk in chunks:
            chunk_embedding = model.encode([chunk])[0]
            similarity = 1 - cosine(query_embedding, chunk_embedding)
            similarities.append((chunk, similarity))
        
        # Get top 3 chunks
        top_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
        
        # Calculate average relevance
        avg_relevance = sum(sim for _, sim in top_chunks) / len(top_chunks)
        results["relevance_scores"].append(avg_relevance)
        
        # Calculate term overlap
        query_terms = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
        
        all_retrieved_terms = set()
        for chunk, _ in top_chunks:
            chunk_terms = set(re.findall(r'\b[a-z]{4,}\b', chunk.lower()))
            all_retrieved_terms.update(chunk_terms)
        
        overlap = len(query_terms.intersection(all_retrieved_terms)) / len(query_terms) if query_terms else 0
        results["term_overlap"].append(overlap)
    
    # Calculate averages
    avg_relevance = sum(results["relevance_scores"]) / len(results["relevance_scores"])
    avg_overlap = sum(results["term_overlap"]) / len(results["term_overlap"])
    
    return {
        "avg_relevance": avg_relevance,
        "avg_term_overlap": avg_overlap
    }

def evaluate_on_large_text():
    """Evaluate chunking strategies on a large text file"""
    # Get a large text file
    text = get_large_windows_file()
    if not text:
        print("Could not find a suitable large text file.")
        return
    
    # Test different chunk sizes
    chunk_sizes = [500, 1000, 2000]
    
    # Store metrics for each chunk size
    metrics = {
        "broken_paragraphs": {"baseline": [], "improved": []},
        "broken_sentences": {"baseline": [], "improved": []},
        "length_variation": {"baseline": [], "improved": []}
    }
    
    # Process each chunk size
    print("\nEvaluating chunking strategies across different chunk sizes...")
    for chunk_size in chunk_sizes:
        print(f"  Testing chunk size: {chunk_size}")
        
        # Apply chunking strategies
        baseline_chunks = baseline_chunking(text, chunk_size)
        improved_chunks = improved_chunking(text, chunk_size)
        
        # Calculate metrics
        metrics["broken_paragraphs"]["baseline"].append(count_broken_paragraphs(baseline_chunks, text))
        metrics["broken_paragraphs"]["improved"].append(count_broken_paragraphs(improved_chunks, text))
        
        metrics["broken_sentences"]["baseline"].append(count_broken_sentences(baseline_chunks))
        metrics["broken_sentences"]["improved"].append(count_broken_sentences(improved_chunks))
        
        metrics["length_variation"]["baseline"].append(calculate_length_variation(baseline_chunks))
        metrics["length_variation"]["improved"].append(calculate_length_variation(improved_chunks))
    
    # Calculate average improvements
    improvements = {}
    for metric in metrics:
        baseline_avg = np.mean(metrics[metric]["baseline"])
        improved_avg = np.mean(metrics[metric]["improved"])
        
        if baseline_avg > 0:
            improvements[metric] = (baseline_avg - improved_avg) / baseline_avg * 100
        else:
            improvements[metric] = 0
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot broken paragraphs
    plt.subplot(1, 3, 1)
    plt.bar(["Baseline", "Improved"], 
           [np.mean(metrics["broken_paragraphs"]["baseline"]), 
            np.mean(metrics["broken_paragraphs"]["improved"])],
           color=["#3498db", "#2ecc71"])
    plt.title("Broken Paragraphs\n(lower is better)")
    plt.annotate(f"{improvements['broken_paragraphs']:.1f}% better", 
               xy=(1, np.mean(metrics["broken_paragraphs"]["improved"])),
               xytext=(0, 10),
               textcoords="offset points",
               ha='center', va='bottom',
               color='green', fontweight='bold')
    
    # Plot broken sentences
    plt.subplot(1, 3, 2)
    plt.bar(["Baseline", "Improved"], 
           [np.mean(metrics["broken_sentences"]["baseline"]), 
            np.mean(metrics["broken_sentences"]["improved"])],
           color=["#3498db", "#2ecc71"])
    plt.title("Broken Sentences\n(lower is better)")
    plt.annotate(f"{improvements['broken_sentences']:.1f}% better", 
               xy=(1, np.mean(metrics["broken_sentences"]["improved"])),
               xytext=(0, 10),
               textcoords="offset points",
               ha='center', va='bottom',
               color='green', fontweight='bold')
    
    # Plot length variation
    plt.subplot(1, 3, 3)
    plt.bar(["Baseline", "Improved"], 
           [np.mean(metrics["length_variation"]["baseline"]), 
            np.mean(metrics["length_variation"]["improved"])],
           color=["#3498db", "#2ecc71"])
    plt.title("Length Consistency\n(lower variation is better)")
    plt.annotate(f"{improvements['length_variation']:.1f}% better", 
               xy=(1, np.mean(metrics["length_variation"]["improved"])),
               xytext=(0, 10),
               textcoords="offset points",
               ha='center', va='bottom',
               color='green', fontweight='bold')
    
    plt.suptitle("Chunking Strategy Comparison on Large Text", fontsize=16)
    plt.tight_layout()
    plt.savefig('large_text_chunking_comparison.png', dpi=300)
    
    # Test query performance
    print("\nEvaluating query performance...")
    # Sample queries related to typical content
    queries = [
        "What are the terms and conditions?",
        "How does the license agreement work?",
        "What are the usage restrictions?",
        "What are the rights and responsibilities?"
    ]
    
    # Use medium chunk size for queries
    medium_size = chunk_sizes[len(chunk_sizes)//2]
    baseline_chunks = baseline_chunking(text, medium_size)
    improved_chunks = improved_chunking(text, medium_size)
    
    baseline_query_results = evaluate_queries(baseline_chunks, queries)
    improved_query_results = evaluate_queries(improved_chunks, queries)
    
    # Calculate query improvements
    query_improvements = {
        "avg_relevance": (improved_query_results["avg_relevance"] / baseline_query_results["avg_relevance"] - 1) * 100,
        "avg_term_overlap": (improved_query_results["avg_term_overlap"] / baseline_query_results["avg_term_overlap"] - 1) * 100 if baseline_query_results["avg_term_overlap"] > 0 else 0
    }
    
    # Create query performance visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(["Baseline", "Improved"], 
           [baseline_query_results["avg_relevance"], improved_query_results["avg_relevance"]],
           color=["#3498db", "#2ecc71"])
    plt.title("Retrieval Relevance\n(higher is better)")
    plt.annotate(f"{query_improvements['avg_relevance']:.1f}% better", 
               xy=(1, improved_query_results["avg_relevance"]),
               xytext=(0, 10),
               textcoords="offset points",
               ha='center', va='bottom',
               color='green', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    plt.bar(["Baseline", "Improved"], 
           [baseline_query_results["avg_term_overlap"], improved_query_results["avg_term_overlap"]],
           color=["#3498db", "#2ecc71"])
    plt.title("Query Term Coverage\n(higher is better)")
    plt.annotate(f"{query_improvements['avg_term_overlap']:.1f}% better", 
               xy=(1, improved_query_results["avg_term_overlap"]),
               xytext=(0, 10),
               textcoords="offset points",
               ha='center', va='bottom',
               color='green', fontweight='bold')
    
    plt.suptitle("Query Performance Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig('query_performance_large_text.png', dpi=300)
    
    # Print summary
    print("\nResults Summary:")
    print("\nChunking Quality Improvements:")
    for metric, value in improvements.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.2f}% better")
    
    print("\nQuery Performance Improvements:")
    for metric, value in query_improvements.items():
        metric_name = metric.replace('_', ' ').replace('avg ', '').title()
        print(f"  {metric_name}: {value:.2f}% better")
    
    print("\nVisualizations saved as 'large_text_chunking_comparison.png' and 'query_performance_large_text.png'")

if __name__ == "__main__":
    evaluate_on_large_text()