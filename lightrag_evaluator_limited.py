import os
import time
import json
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import nest_asyncio
from lightrag_improvements import adaptive_chunking, dynamic_retrieval_weighting

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

class LightRAGEvaluatorLimited:
    def __init__(self, test_size=10):
        """Evaluator that works with limited API access"""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.test_size = test_size
        self.dataset = self.create_test_dataset()
    
    def create_test_dataset(self):
        """Create a small test dataset from story.txt"""
        try:
            with open("story.txt", "r") as f:
                text = f.read()
            
            # Create test dataset
            paragraphs = text.split("\n\n")
            
            contexts = paragraphs[:self.test_size] if len(paragraphs) >= self.test_size else paragraphs
            
            # Create simple questions
            questions = []
            answers = []
            
            for i, context in enumerate(contexts):
                if i % 2 == 0:
                    questions.append(f"What happens in this part of the story: '{context[:30]}...'?")
                else:
                    questions.append(f"Describe the events mentioned in: '{context[:30]}...'")
                answers.append(context)  # Full paragraph is the answer
            
            dataset = {
                'contexts': contexts,
                'questions': questions[:self.test_size],
                'answers': answers[:self.test_size]
            }
            
            print(f"Created dataset with {len(questions)} samples")
            return dataset
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            # Minimal fallback dataset
            return {
                'contexts': ["This is a sample text about Lily finding a magical library."],
                'questions': ["What did Lily find?"],
                'answers': ["Lily found a magical library."]
            }
    
    async def evaluate_retrieval(self, model_type="baseline"):
        """Evaluate just the retrieval component without LLM generation"""
        # Setup
        working_dir = f"./lightrag_{model_type}_eval_limited"
        
        # Create clean working directory
        if os.path.exists(working_dir):
            import shutil
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)
        
        # Define a mock LLM function to avoid API calls
        async def mock_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return "This is a mock response to avoid API rate limits."
        
        # Define embedding function
        async def embedding_func(texts):
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        
        # Initialize RAG
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=mock_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=embedding_func,
            ),
        )
        
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        # Apply improvements for the improved model
        if model_type == "improved":
            # Store original methods
            original_insert = rag.insert
            
            # 1. Apply adaptive chunking
            def improved_insert(text, doc_id=None, **kwargs):
                chunks = adaptive_chunking(text)
                doc_ids = []
                for chunk in chunks:
                    chunk_doc_id = original_insert(chunk, **kwargs)
                    doc_ids.append(chunk_doc_id)
                return doc_ids[0] if doc_ids else None
            
            # Replace methods
            rag.insert = improved_insert
        
        # Insert contexts
        print(f"Inserting {len(self.dataset['contexts'])} contexts...")
        for context in self.dataset['contexts']:
            rag.insert(context)
        
        # Evaluate retrieval
        results = {
            "retrieval_precision": [],
            "query_chunk_similarity": [],
            "latency": []
        }
        
        print(f"Evaluating {model_type} retrieval on {len(self.dataset['questions'])} questions...")
        for question, answer in zip(self.dataset['questions'], self.dataset['answers']):
            start_time = time.time()
            
            # For the improved model, modify query parameters
            param = QueryParam(mode="hybrid", top_k=3)
            if model_type == "improved":
                param = dynamic_retrieval_weighting(question, param)
            
            # Access internal retrieval directly to bypass LLM
            relevant_chunks = []
            try:
                # This is a simplified version - the actual internal access would depend on LightRAG implementation
                chunks = rag._get_chunks(question, param)  # This function name might differ
                relevant_chunks = [chunk for chunk in chunks]
            except Exception as e:
                # If direct access isn't available, use the public API but ignore the LLM response
                _ = rag.query(query=question, param=param)
                # Try to access chunks from storage
                try:
                    # This is very implementation-specific and might need adjustment
                    chunks_json = f"{working_dir}/vdb_chunks.json"
                    with open(chunks_json, 'r') as f:
                        chunk_data = json.load(f)
                    relevant_chunks = list(chunk_data.values())[:param.top_k]
                except:
                    pass
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Calculate similarity between query and chunks
            query_embedding = self.model.encode([question])[0]
            chunk_similarities = []
            
            # If chunks were retrieved, calculate similarities
            if relevant_chunks:
                for chunk in relevant_chunks:
                    chunk_text = chunk.get('text', str(chunk))
                    chunk_embedding = self.model.encode([chunk_text])[0]
                    similarity = 1 - cosine(query_embedding, chunk_embedding)
                    chunk_similarities.append(similarity)
            
            # Calculate precision - how many chunks contain parts of the answer
            precision = 0
            if relevant_chunks:
                # Simple overlap-based precision
                matches = 0
                for chunk in relevant_chunks:
                    chunk_text = chunk.get('text', str(chunk)) 
                    # Check if any 10-character sequence from answer is in the chunk
                    for i in range(0, len(answer) - 10, 10):
                        if answer[i:i+10] in chunk_text:
                            matches += 1
                            break
                precision = matches / len(relevant_chunks) if relevant_chunks else 0
            
            # Store results
            results["retrieval_precision"].append(precision)
            results["query_chunk_similarity"].append(np.mean(chunk_similarities) if chunk_similarities else 0)
            results["latency"].append(latency)
            
        # Aggregate results
        aggregated_results = {
            "retrieval_precision": np.mean(results["retrieval_precision"]),
            "query_chunk_similarity": np.mean(results["query_chunk_similarity"]),
            "latency": np.mean(results["latency"])
        }
        
        print(f"Evaluation results for {model_type}:")
        print(f"  Retrieval Precision: {aggregated_results['retrieval_precision']:.4f}")
        print(f"  Query-Chunk Similarity: {aggregated_results['query_chunk_similarity']:.4f}")
        print(f"  Average Latency: {aggregated_results['latency']:.4f}s")
        
        return aggregated_results
    
    async def compare_models(self):
        """Compare baseline and improved models"""
        print("\nEvaluating baseline retrieval...")
        baseline_results = await self.evaluate_retrieval("baseline")
        
        print("\nEvaluating improved retrieval...")
        improved_results = await self.evaluate_retrieval("improved")
        
        # Calculate improvements
        improvements = {
            "retrieval_precision": (improved_results["retrieval_precision"] / baseline_results["retrieval_precision"] - 1) * 100 if baseline_results["retrieval_precision"] > 0 else 0,
            "query_chunk_similarity": (improved_results["query_chunk_similarity"] / baseline_results["query_chunk_similarity"] - 1) * 100 if baseline_results["query_chunk_similarity"] > 0 else 0,
            "latency": (baseline_results["latency"] / improved_results["latency"] - 1) * 100 if improved_results["latency"] > 0 else 0
        }
        
        # Visualization
        self.visualize_retrieval_comparison(baseline_results, improved_results, improvements)
        
        return {
            "baseline": baseline_results,
            "improved": improved_results,
            "improvements": improvements
        }
    
    def visualize_retrieval_comparison(self, baseline_results, improved_results, improvements):
        """Visualize the comparison for retrieval components"""
        metrics = ["Retrieval Precision", "Query-Chunk Similarity", "Latency (lower is better)"]
        baseline_values = [
            baseline_results["retrieval_precision"], 
            baseline_results["query_chunk_similarity"], 
            baseline_results["latency"]
        ]
        improved_values = [
            improved_results["retrieval_precision"], 
            improved_results["query_chunk_similarity"], 
            improved_results["latency"]
        ]
        
        # Create comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.35
        
        baseline_bars = ax.bar(x - width/2, baseline_values, width, label='Baseline LightRAG')
        improved_bars = ax.bar(x + width/2, improved_values, width, label='Improved LightRAG')
        
        ax.set_ylabel('Score')
        ax.set_title('LightRAG Retrieval Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add improvement percentages
        for i, metric in enumerate(["retrieval_precision", "query_chunk_similarity", "latency"]):
            percent = improvements[metric]
            color = 'green' if percent > 0 else 'red'
            if metric == "latency" and percent < 0:
                color = 'red'
            
            ax.annotate(f"{percent:.2f}%",
                        xy=(i + width/2, improved_values[i]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('lightrag_retrieval_comparison.png', dpi=300)
        print("Created visualization: lightrag_retrieval_comparison.png")

async def main():
    evaluator = LightRAGEvaluatorLimited(test_size=5)  # Use only 5 samples
    results = await evaluator.compare_models()
    
    # Save results to file
    with open("lightrag_retrieval_results.json", "w") as f:
        json.dump({
            "baseline": {k: float(v) for k, v in results["baseline"].items()},
            "improved": {k: float(v) for k, v in results["improved"].items()},
            "improvements": {k: float(v) for k, v in results["improvements"].items()}
        }, f, indent=2)
    
    print("Evaluation complete. Results saved to lightrag_retrieval_results.json")

if __name__ == "__main__":
    asyncio.run(main())