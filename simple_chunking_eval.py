import os
import re
import matplotlib.pyplot as plt
import numpy as np

def get_large_windows_file():
    """Get a large text file that's guaranteed to be on Windows systems"""
    potential_paths = [
        r"C:\Windows\System32\license.rtf",
        r"C:\Windows\System32\eula.txt",
        r"C:\Windows\System32\Drivers\etc\services"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if len(content) > 5000:
                        print(f"Using system file: {path} ({len(content)} characters)")
                        return content
            except Exception as e:
                print(f"Error reading {path}: {e}")
    
    # Use README as fallback
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
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            if current_chunk and not current_chunk.endswith("\n"):
                current_chunk += "\n\n"
            current_chunk += para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def count_broken_sentences(chunks):
    """Count sentences broken across chunk boundaries"""
    broken_count = 0
    for i, chunk in enumerate(chunks[:-1]):
        if not chunk.rstrip().endswith(('.', '!', '?')):
            broken_count += 1
    return broken_count

def simple_evaluation():
    """Perform a simple evaluation of chunking strategies"""
    # Get a large text file
    text = get_large_windows_file()
    if not text:
        return
    
    # Test different chunk sizes
    chunk_sizes = [500, 1000, 2000]
    
    # Store results
    results = {
        "broken_sentences": {"baseline": [], "improved": []},
        "chunk_count": {"baseline": [], "improved": []}
    }
    
    print("\nEvaluating chunking strategies...")
    for chunk_size in chunk_sizes:
        print(f"Testing chunk size: {chunk_size}")
        
        # Create chunks
        baseline_chunks = baseline_chunking(text, chunk_size)
        improved_chunks = improved_chunking(text, chunk_size)
        
        # Count broken sentences
        baseline_broken = count_broken_sentences(baseline_chunks)
        improved_broken = count_broken_sentences(improved_chunks)
        
        # Store results
        results["broken_sentences"]["baseline"].append(baseline_broken)
        results["broken_sentences"]["improved"].append(improved_broken)
        
        results["chunk_count"]["baseline"].append(len(baseline_chunks))
        results["chunk_count"]["improved"].append(len(improved_chunks))
        
        print(f"  Baseline: {len(baseline_chunks)} chunks, {baseline_broken} broken sentences")
        print(f"  Improved: {len(improved_chunks)} chunks, {improved_broken} broken sentences")
    
    # Calculate average improvements
    avg_baseline_broken = np.mean(results["broken_sentences"]["baseline"])
    avg_improved_broken = np.mean(results["broken_sentences"]["improved"])
    
    broken_improvement = ((avg_baseline_broken - avg_improved_broken) / 
                         max(1, avg_baseline_broken)) * 100
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(["Baseline", "Improved"], 
           [avg_baseline_broken, avg_improved_broken],
           color=["#3498db", "#2ecc71"])
    plt.title("Broken Sentences\n(lower is better)")
    plt.xlabel("Chunking Strategy")
    plt.ylabel("Average Broken Sentences")
    plt.annotate(f"{broken_improvement:.1f}% better", 
               xy=(1, avg_improved_broken),
               xytext=(0, 10),
               textcoords="offset points",
               ha='center', va='bottom',
               color='green', fontweight='bold')
    
    # Plot performance across chunk sizes
    plt.subplot(1, 2, 2)
    
    x = np.arange(len(chunk_sizes))
    width = 0.35
    
    plt.bar(x - width/2, results["broken_sentences"]["baseline"], width, 
           label='Baseline', color='#3498db')
    plt.bar(x + width/2, results["broken_sentences"]["improved"], width,
           label='Improved', color='#2ecc71')
    
    plt.xlabel('Chunk Size')
    plt.ylabel('Broken Sentences')
    plt.title('Performance Across Chunk Sizes')
    plt.xticks(x, chunk_sizes)
    plt.legend()
    
    plt.suptitle("LightRAG Chunking Strategy Improvement", fontsize=16)
    plt.tight_layout()
    plt.savefig('chunking_evaluation_results.png', dpi=300)
    
    print("\nResults Summary:")
    print(f"Broken Sentences: {broken_improvement:.2f}% improvement")
    print("\nVisualization saved as 'chunking_evaluation_results.png'")
    
    # Show sample chunks
    print("\nSample Chunks:")
    print("\nBaseline chunk example:")
    print(baseline_chunks[1][:200] + "..." if len(baseline_chunks[1]) > 200 else baseline_chunks[1])
    
    print("\nImproved chunk example:")
    print(improved_chunks[1][:200] + "..." if len(improved_chunks[1]) > 200 else improved_chunks[1])

if __name__ == "__main__":
    simple_evaluation()