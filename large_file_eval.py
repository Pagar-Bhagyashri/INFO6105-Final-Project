import os
import re
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def create_large_test_file():
    """Create a large test file from Python files in the current directory"""
    print("Creating large test file from local Python code...")
    
    # Find all Python files in the current directory and subdirectories
    python_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    # Combine content into one large file
    combined_text = ""
    for py_file in tqdm(python_files, desc="Processing Python files"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                combined_text += f"\n\n# File: {py_file}\n\n{content}\n\n"
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    # Write to a test file
    with open("large_test_file.txt", "w", encoding="utf-8") as f:
        f.write(combined_text)
    
    print(f"Created large test file ({len(combined_text)} characters, {len(python_files)} Python files)")
    return combined_text

def baseline_chunking(text, chunk_size=800):
    """Baseline chunking: simple fixed-size chunks without respecting content structure"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def improved_chunking(text, chunk_size=800):
    """Improved chunking: respects paragraphs and logical code blocks"""
    # Split by logical blocks (paragraphs for text, function blocks for code)
    blocks = re.split(r'\n\s*\n', text)
    blocks = [b for b in blocks if b.strip()]
    
    chunks = []
    current_chunk = ""
    
    for block in blocks:
        # If adding this block would exceed the chunk size and we already have content
        if len(current_chunk) + len(block) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = block
        else:
            # Add separator if needed
            if current_chunk and not current_chunk.endswith("\n"):
                current_chunk += "\n\n"
            current_chunk += block
    
    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def calculate_metrics(chunks, is_code=True):
    """Calculate various metrics for chunk quality"""
    metrics = {}
    
    # Count broken logical blocks (code functions/classes or paragraphs)
    broken_blocks = 0
    for i, chunk in enumerate(chunks[:-1]):
        # For code, check for broken function/class definitions
        if is_code:
            if (chunk.count('def ') > chunk.count('\n\ndef ') or
                chunk.count('class ') > chunk.count('\n\nclass ')):
                broken_blocks += 1
        # For text, check for incomplete sentences
        else:
            if not re.search(r'[.!?]\s*$', chunk):
                broken_blocks += 1
    
    metrics["broken_blocks"] = broken_blocks
    
    # Calculate length variation (consistency)
    lengths = [len(chunk) for chunk in chunks]
    mean_length = sum(lengths) / len(chunks)
    variance = sum((length - mean_length) ** 2 for length in chunks) / len(chunks)
    metrics["length_variation"] = variance
    
    # Count code-specific metrics if it's code
    if is_code:
        # Count broken import blocks
        broken_imports = 0
        for i, chunk in enumerate(chunks[:-1]):
            if 'import ' in chunk and 'import ' in chunks[i+1]:
                # Check if import block is split
                if not re.search(r'\n\n[^i]', chunk.split('import ')[-1]):
                    broken_imports += 1
        
        metrics["broken_imports"] = broken_imports
    
    return metrics

def evaluate_on_large_file():
    """Evaluate chunking strategies on a large code-based file"""
    # Create large test file if it doesn't exist
    if not os.path.exists("large_test_file.txt"):
        text = create_large_test_file()
    else:
        with open("large_test_file.txt", "r", encoding="utf-8") as f:
            text = f.read()
    
    print(f"Evaluating chunking strategies on file of size {len(text)} characters")
    
    # Test multiple chunk sizes
    chunk_sizes = [500, 1000, 2000, 4000]
    results = {
        "broken_blocks": {"baseline": [], "improved": []},
        "length_variation": {"baseline": [], "improved": []},
        "broken_imports": {"baseline": [], "improved": []}
    }
    
    for chunk_size in tqdm(chunk_sizes, desc="Testing chunk sizes"):
        # Apply chunking strategies
        baseline_chunks = baseline_chunking(text, chunk_size)
        improved_chunks = improved_chunking(text, chunk_size)
        
        # Calculate metrics
        baseline_metrics = calculate_metrics(baseline_chunks)
        improved_metrics = calculate_metrics(improved_chunks)
        
        # Store results
        for metric in ["broken_blocks", "length_variation"]:
            results[metric]["baseline"].append(baseline_metrics[metric])
            results[metric]["improved"].append(improved_metrics[metric])
        
        if "broken_imports" in baseline_metrics:
            results["broken_imports"]["baseline"].append(baseline_metrics["broken_imports"])
            results["broken_imports"]["improved"].append(improved_metrics["broken_imports"])
    
    # Calculate average improvements
    improvements = {}
    for metric in results:
        baseline_avg = np.mean(results[metric]["baseline"])
        improved_avg = np.mean(results[metric]["improved"])
        
        # All metrics are "lower is better"
        if baseline_avg > 0:
            improvements[metric] = (baseline_avg - improved_avg) / baseline_avg * 100
        else:
            improvements[metric] = 0
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot broken blocks comparison
    axes[0].bar(["Baseline", "Improved"], 
               [np.mean(results["broken_blocks"]["baseline"]), 
                np.mean(results["broken_blocks"]["improved"])],
               color=["#3498db", "#2ecc71"])
    axes[0].set_title("Broken Logical Blocks\n(lower is better)")
    axes[0].annotate(f"{improvements['broken_blocks']:.1f}% better", 
                    xy=(1, np.mean(results["broken_blocks"]["improved"])),
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
    axes[1].annotate(f"{improvements['length_variation']:.1f}% better", 
                    xy=(1, np.mean(results["length_variation"]["improved"])),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='green', fontweight='bold')
    
    # Plot broken imports comparison if available
    if results["broken_imports"]["baseline"]:
        axes[2].bar(["Baseline", "Improved"], 
                   [np.mean(results["broken_imports"]["baseline"]), 
                    np.mean(results["broken_imports"]["improved"])],
                   color=["#3498db", "#2ecc71"])
        axes[2].set_title("Broken Import Blocks\n(lower is better)")
        axes[2].annotate(f"{improvements['broken_imports']:.1f}% better", 
                       xy=(1, np.mean(results["broken_imports"]["improved"])),
                       xytext=(0, 10),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       color='green', fontweight='bold')
    
    plt.suptitle("Chunking Strategy Comparison on Large Codebase", fontsize=16)
    plt.tight_layout()
    plt.savefig('large_file_chunking_evaluation.png', dpi=300)
    
    # Create secondary visualization showing performance across chunk sizes
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(chunk_sizes, results["broken_blocks"]["baseline"], 'o-', color='#3498db', label='Baseline')
    plt.plot(chunk_sizes, results["broken_blocks"]["improved"], 'o-', color='#2ecc71', label='Improved')
    plt.xlabel('Chunk Size')
    plt.ylabel('Broken Blocks Count')
    plt.title('Broken Blocks vs. Chunk Size')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(chunk_sizes, results["length_variation"]["baseline"], 'o-', color='#3498db', label='Baseline')
    plt.plot(chunk_sizes, results["length_variation"]["improved"], 'o-', color='#2ecc71', label='Improved')
    plt.xlabel('Chunk Size')
    plt.ylabel('Length Variation')
    plt.title('Length Variation vs. Chunk Size')
    plt.legend()
    
    plt.suptitle("Performance Across Different Chunk Sizes", fontsize=14)
    plt.tight_layout()
    plt.savefig('chunking_size_comparison.png', dpi=300)
    
    # Report results
    print("\nResults overview:")
    print(f"Tested on file size: {len(text)} characters")
    print(f"Chunk sizes tested: {chunk_sizes}")
    
    print("\nAverage improvements across all chunk sizes:")
    for metric, value in improvements.items():
        metric_name = metric.replace("_", " ").title()
        print(f"  {metric_name}: {value:.2f}% better")
    
    # Print a sample chunk comparison
    medium_size = chunk_sizes[len(chunk_sizes)//2]
    baseline_sample = baseline_chunking(text, medium_size)[5]  # Get a middle chunk
    improved_sample = improved_chunking(text, medium_size)[5]  # Get a middle chunk
    
    print("\nSample chunk from baseline strategy:")
    print(baseline_sample[:200] + "..." if len(baseline_sample) > 200 else baseline_sample)
    
    print("\nSample chunk from improved strategy:")
    print(improved_sample[:200] + "..." if len(improved_sample) > 200 else improved_sample)
    
    print("\nVisualizations saved as 'large_file_chunking_evaluation.png' and 'chunking_size_comparison.png'")

if __name__ == "__main__":
    evaluate_on_large_file()