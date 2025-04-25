import os
import re
import matplotlib.pyplot as plt

def baseline_chunking(text, chunk_size=500):
    """Baseline chunking: simple fixed-size chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def improved_chunking(text, chunk_size=500):
    """Improved chunking: respects paragraphs and sentences"""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def compare_chunking_strategies():
    # Load text
    with open("story.txt", "r") as f:
        text = f.read()
    
    # Apply chunking strategies
    baseline_chunks = baseline_chunking(text)
    improved_chunks = improved_chunking(text)
    
    # Compare results
    print(f"Baseline chunks: {len(baseline_chunks)}")
    print(f"Improved chunks: {len(improved_chunks)}")
    
    # Measure metrics
    baseline_broken_sentences = count_broken_sentences(baseline_chunks)
    improved_broken_sentences = count_broken_sentences(improved_chunks)
    
    baseline_avg_length = sum(len(c) for c in baseline_chunks) / len(baseline_chunks)
    improved_avg_length = sum(len(c) for c in improved_chunks) / len(improved_chunks)
    
    # Calculate improvements
    broken_sent_improvement = ((baseline_broken_sentences - improved_broken_sentences) / 
                              max(1, baseline_broken_sentences)) * 100
    
    length_variation_baseline = calculate_length_variation(baseline_chunks)
    length_variation_improved = calculate_length_variation(improved_chunks)
    variation_improvement = ((length_variation_baseline - length_variation_improved) / 
                            max(0.001, length_variation_baseline)) * 100
    
    # Create visualization
    metrics = ['Broken Sentences', 'Length Consistency']
    baseline_values = [baseline_broken_sentences, length_variation_baseline]
    improved_values = [improved_broken_sentences, length_variation_improved]
    
    plt.figure(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], baseline_values, width, label='Baseline Chunking', color='#3498db')
    plt.bar([i + width/2 for i in x], improved_values, width, label='Improved Chunking', color='#2ecc71')
    
    plt.ylabel('Value (lower is better)')
    plt.title('Chunking Strategy Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Add improvements as percentages
    improvements = [broken_sent_improvement, variation_improvement]
    for i, imp in enumerate(improvements):
        plt.annotate(f"{imp:.2f}% better", 
                   xy=(i + width/2, improved_values[i]),
                   xytext=(0, -20),
                   textcoords="offset points",
                   ha='center', va='top', 
                   color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('chunking_improvements.png', dpi=300)
    print("Visualization saved as 'chunking_improvements.png'")
    
    # Print summary
    print("\nImprovement Summary:")
    print(f"Broken sentences: {broken_sent_improvement:.2f}% improvement")
    print(f"Length consistency: {variation_improvement:.2f}% improvement")
    
    # Show example chunks
    print("\nExample chunks from baseline strategy:")
    print_sample_chunk(baseline_chunks[0])
    
    print("\nExample chunks from improved strategy:")
    print_sample_chunk(improved_chunks[0])

def count_broken_sentences(chunks):
    """Count sentences broken across chunk boundaries"""
    broken_count = 0
    for i in range(len(chunks) - 1):
        # Check if chunk ends with incomplete sentence
        if not chunks[i].endswith('.') and not chunks[i].endswith('!') and not chunks[i].endswith('?'):
            broken_count += 1
    return broken_count

def calculate_length_variation(chunks):
    """Calculate variation in chunk lengths"""
    lengths = [len(chunk) for chunk in chunks]
    mean_length = sum(lengths) / len(lengths)
    variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
    return variance

def print_sample_chunk(chunk):
    """Print a shortened sample of a chunk"""
    print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

if __name__ == "__main__":
    compare_chunking_strategies()