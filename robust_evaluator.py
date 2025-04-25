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
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Apply nest_asyncio
nest_asyncio.apply()

class RobustLightRAGEvaluator:
    """Evaluator for LightRAG that works without relying on external APIs"""
    
    def __init__(self, data_source="story.txt", test_size=10):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.test_size = test_size
        self.data_source = data_source
        self.dataset = self.create_evaluation_dataset()
    
    def create_evaluation_dataset(self):
        """Create dataset from text file or downloadable corpus"""
        try:
            if self.data_source.endswith(".txt"):
                # Load from file
                with open(self.data_source, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                # Use default story
                with open("story.txt", "r", encoding="utf-8") as f:
                    text = f.read()
            
            # Create test dataset
            paragraphs = []
            for para in text.split("\n\n"):
                if para.strip():
                    paragraphs.append(para.strip())
            
            if len(paragraphs) > self.test_size * 2:
                selected_paragraphs = paragraphs[:self.test_size * 2]
            else:
                selected_paragraphs = paragraphs
            
            # Split into context and test sets
            contexts = selected_paragraphs[:len(selected_paragraphs)//2]
            test_paras = selected_paragraphs[len(selected_paragraphs)//2:]
            
            # Create questions and answers
            questions = []
            answers = []
            
            # First set - direct questions about entities in paragraphs
            entities = re.findall(r'\b[A-Z][a-z]+\b', text)
            entity_counts = defaultdict(int)
            for entity in entities:
                entity_counts[entity] += 1
            
            # Get top entities
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for entity, _ in top_entities:
                questions.append(f"What do we know about {entity}?")
                # Find relevant paragraphs mentioning this entity
                relevant_text = []
                for para in test_paras:
                    if entity in para:
                        relevant_text.append(para)
                
                if relevant_text:
                    answers.append("\n".join(relevant_text))
                else:
                    answers.append(f"No specific information about {entity}.")
            
            # Second set - conceptual questions
            concept_questions = [
                "What are the main themes in this text?",
                "How does the story begin?",
                "What key events happen in the narrative?",
                "How does the story conclude?",
                "What are the important relationships described?"
            ]
            
            # Create simple conceptual answers
            for q in concept_questions:
                questions.append(q)
                # Use TF-IDF to find most relevant paragraph for this question
                tfidf = TfidfVectorizer().fit_transform([q] + test_paras)
                cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
                
                if len(cosine_similarities) > 0:
                    most_similar_idx = cosine_similarities.argmax()
                    answers.append(test_paras[most_similar_idx])
                else:
                    answers.append("No relevant information found.")
            
            # Create dataset
            dataset = {
                'contexts': contexts,
                'questions': questions[:self.test_size],
                'answers': answers[:self.test_size]
            }
            
            print(f"Created dataset with {len(dataset['questions'])} samples")
            return dataset
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            # Minimal fallback dataset
            return {
                'contexts': ["This is a sample text about Lily finding a magical library."],
                'questions': ["What did Lily find?"],
                'answers': ["Lily found a magical library."]
            }

    async def setup_baseline_rag(self, working_dir):
        """Set up the baseline RAG model"""
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
        
        return rag
    
    async def setup_improved_rag(self, working_dir):
        """Set up the improved RAG model with enhancements"""
        baseline_rag = await self.setup_baseline_rag(working_dir)
        
        # Store original methods
        original_insert = baseline_rag.insert
        original_query = baseline_rag.query
        
        # 1. Improved chunking
        def adaptive_chunking(text, min_chunk_size=100, max_chunk_size=500):
            """Chunk text respecting sentence boundaries"""
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed the max size
                if current_length + sentence_length > max_chunk_size and len(current_chunk) > 0:
                    # Add the current chunk to results
                    chunks.append(" ".join(current_chunk))
                    
                    # Start a new chunk with 30% overlap
                    overlap_point = max(0, len(current_chunk) - 2)
                    current_chunk = current_chunk[overlap_point:]
                    current_length = sum(len(s) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return chunks
        
        # 2. Enhanced query processing
        def enhance_query(query):
            """Enhance query with related terms"""
            # Extract key terms
            key_terms = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
            expanded_terms = set()
            
            # Simple term expansion (in a real system this would use wordnet or word embeddings)
            expansions = {
                "find": ["discover", "locate", "identify"],
                "story": ["narrative", "tale", "account"],
                "book": ["text", "volume", "publication"],
                "character": ["person", "figure", "individual"],
                "magic": ["supernatural", "enchanted", "mystical"],
                "forest": ["woods", "woodland", "grove"]
            }
            
            for term in key_terms:
                if term in expansions:
                    expanded_terms.update(expansions[term])
            
            # Add expanded terms to query
            if expanded_terms:
                enhanced = f"{query} (Related terms: {', '.join(expanded_terms)})"
                return enhanced
            return query
        
        # 3. Contextual query weighting
        def contextual_weighting(query, param):
            """Adjust retrieval parameters based on query characteristics"""
            # Check if query is factual (who, what, when, where, why, how)
            is_factual = bool(re.search(r'\b(who|what|when|where|why|how)\b', query.lower()))
            
            # Check if query contains named entities
            has_entities = bool(re.findall(r'\b[A-Z][a-z]+\b', query))
            
            # Adjust parameters based on query type
            if is_factual and has_entities:
                param.top_k = max(param.top_k, 4)  # More chunks for specific factual queries
                param.mode = "hybrid"  # Use hybrid mode for entity queries
            elif is_factual:
                param.top_k = max(param.top_k, 3)  # Standard for general factual queries
                param.mode = "hybrid"  # Use hybrid for factual queries
            else:
                # For conceptual queries, focus more on semantic similarity
                param.mode = "semantic"
            
            return param
        
        # Override methods with improved versions
        def improved_insert(text, doc_id=None, **kwargs):
            chunks = adaptive_chunking(text)
            doc_ids = []
            for chunk in chunks:
                chunk_doc_id = original_insert(chunk, **kwargs)
                doc_ids.append(chunk_doc_id)
            return doc_ids[0] if doc_ids else None
        
        def improved_query(query, param=None, **kwargs):
            # Apply enhancements
            enhanced_query = enhance_query(query)
            
            # Apply context-aware parameter adjustment
            if param:
                param = contextual_weighting(query, param)
            
            # Call original query with enhancements
            return original_query(enhanced_query, param, **kwargs)
        
        # Replace methods
        baseline_rag.insert = improved_insert
        baseline_rag.query = improved_query
        
        return baseline_rag
    
    async def evaluate_model(self, model_type="baseline"):
        """Evaluate a model variant"""
        working_dir = f"./lightrag_{model_type}_robust_eval"
        
        if model_type == "baseline":
            rag = await self.setup_baseline_rag(working_dir)
        else:
            rag = await self.setup_improved_rag(working_dir)
        
        # Insert contexts
        print(f"Inserting {len(self.dataset['contexts'])} contexts...")
        for context in self.dataset['contexts']:
            rag.insert(context)
        
        # Evaluate on test queries
        results = {
            "retrieval_relevance": [],
            "chunk_coverage": [],
            "latency": []
        }
        
        print(f"Evaluating {model_type} retrieval on {len(self.dataset['questions'])} questions...")
        for i, (question, answer) in enumerate(zip(self.dataset['questions'], self.dataset['answers'])):
            print(f"Processing query {i+1}/{len(self.dataset['questions'])}: {question[:40]}...")
            
            # Time the retrieval
            start_time = time.time()
            
            # Get query results
            param = QueryParam(mode="hybrid", top_k=3)
            response = rag.query(question, param)
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract chunks from the response (implementation-specific)
            # For demonstration, we'll use the response text directly
            retrieved_text = response
            
            # Calculate metrics
            # 1. Semantic similarity between response and answer
            response_embedding = self.model.encode([retrieved_text])[0]
            answer_embedding = self.model.encode([answer])[0]
            semantic_relevance = 1 - cosine(response_embedding, answer_embedding)
            
            # 2. Coverage of key terms from the answer
            answer_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower()))
            response_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', retrieved_text.lower()))
            
            coverage = len(answer_terms.intersection(response_terms)) / len(answer_terms) if answer_terms else 0
            
            # Store results
            results["retrieval_relevance"].append(semantic_relevance)
            results["chunk_coverage"].append(coverage)
            results["latency"].append(latency)
        
        # Aggregate results
        aggregated = {
            "retrieval_relevance": float(np.mean(results["retrieval_relevance"])),
            "chunk_coverage": float(np.mean(results["chunk_coverage"])),
            "latency": float(np.mean(results["latency"]))
        }
        
        print(f"Evaluation results for {model_type}:")
        print(f"  Retrieval Relevance: {aggregated['retrieval_relevance']:.4f}")
        print(f"  Chunk Coverage: {aggregated['chunk_coverage']:.4f}")
        print(f"  Average Latency: {aggregated['latency']:.4f}s")
        
        return aggregated
    
    async def compare_models(self):
        """Compare baseline and improved implementations"""
        print("\nEvaluating baseline model...")
        baseline_results = await self.evaluate_model("baseline")
        
        print("\nEvaluating improved model...")
        improved_results = await self.evaluate_model("improved")
        
        # Calculate improvements
        improvements = {
            "retrieval_relevance": (improved_results["retrieval_relevance"] / baseline_results["retrieval_relevance"] - 1) * 100 if baseline_results["retrieval_relevance"] > 0 else 0,
            "chunk_coverage": (improved_results["chunk_coverage"] / baseline_results["chunk_coverage"] - 1) * 100 if baseline_results["chunk_coverage"] > 0 else 0,
            "latency": (baseline_results["latency"] / improved_results["latency"] - 1) * 100 if improved_results["latency"] > 0 else 0
        }
        
        # Create visualization
        self.visualize_comparison(baseline_results, improved_results, improvements)
        
        # Save numerical results
        results = {
            "baseline": baseline_results,
            "improved": improved_results,
            "improvements": improvements
        }
        
        with open("lightrag_performance_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def visualize_comparison(self, baseline_results, improved_results, improvements):
        """Create comparison visualization"""
        metrics = ["Retrieval Relevance", "Chunk Coverage", "Latency (lower is better)"]
        baseline_values = [
            baseline_results["retrieval_relevance"],
            baseline_results["chunk_coverage"],
            baseline_results["latency"]
        ]
        improved_values = [
            improved_results["retrieval_relevance"],
            improved_results["chunk_coverage"],
            improved_results["latency"]
        ]
        
        # Create comparison chart
        plt.figure(figsize=(14, 10))
        
        # Create bar chart
        ax = plt.subplot(2, 1, 1)
        x = np.arange(len(metrics))
        width = 0.35
        
        baseline_bars = ax.bar(x - width/2, baseline_values, width, label='Baseline LightRAG', color='#3498db')
        improved_bars = ax.bar(x + width/2, improved_values, width, label='Improved LightRAG', color='#2ecc71')
        
        ax.set_ylabel('Score')
        ax.set_title('LightRAG Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add improvement percentages
        for i, metric in enumerate(["retrieval_relevance", "chunk_coverage", "latency"]):
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
        
        # Add improvement radar chart
        ax2 = plt.subplot(2, 1, 2, polar=True)
        
        # Convert improvements to positive scale for radar (0-100)
        radar_metrics = ["Retrieval Relevance", "Chunk Coverage", "Latency Improvement"]
        radar_baseline = [50, 50, 50]  # Baseline centered at 50%
        
        # Scale improvements to radar chart (keep between 0-100)
        radar_improved = []
        for metric, value in improvements.items():
            scaled_value = min(max(50 + value/2, 0), 100)
            radar_improved.append(scaled_value)
        
        # Close the loop
        radar_metrics = np.concatenate((radar_metrics, [radar_metrics[0]]))
        radar_baseline = np.concatenate((radar_baseline, [radar_baseline[0]]))
        radar_improved = np.concatenate((radar_improved, [radar_improved[0]]))
        
        # Convert to radians
        theta = np.linspace(0, 2*np.pi, len(radar_metrics))
        
        # Plot radar
        ax2.plot(theta, radar_baseline, color='#3498db', linewidth=2)
        ax2.plot(theta, radar_improved, color='#2ecc71', linewidth=2)
        ax2.fill(theta, radar_improved, color='#2ecc71', alpha=0.25)
        ax2.fill(theta, radar_baseline, color='#3498db', alpha=0.25)
        
        # Add metric labels
        ax2.set_xticks(theta[:-1])
        ax2.set_xticklabels(radar_metrics[:-1])
        ax2.set_title('Improvement Radar')
        
        plt.tight_layout()
        plt.savefig('lightrag_enhanced_comparison.png', dpi=300, bbox_inches='tight')
        print("Created enhanced visualization: lightrag_enhanced_comparison.png")
        
        # Create a second visualization focusing on improvement percentages
        plt.figure(figsize=(10, 6))
        improvement_names = ["Retrieval Relevance", "Chunk Coverage", "Latency"]
        improvement_values = [
            improvements["retrieval_relevance"],
            improvements["chunk_coverage"],
            improvements["latency"]
        ]
        
        colors = ['#27ae60' if v > 0 else '#c0392b' for v in improvement_values]
        if improvement_values[2] < 0:  # For latency, negative is bad
            colors[2] = '#c0392b'
        
        # Create horizontal bar chart of improvements
        plt.barh(improvement_names, improvement_values, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Percentage Improvement')
        plt.title('LightRAG Improvements')
        
        # Add value labels
        for i, v in enumerate(improvement_values):
            plt.text(v + np.sign(v)*2, i, f"{v:.2f}%", va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('lightrag_improvement_percentages.png', dpi=300)
        print("Created improvement visualization: lightrag_improvement_percentages.png")

async def main():
    # Create evaluator with larger test size
    evaluator = RobustLightRAGEvaluator(test_size=10)
    
    # Run evaluation
    results = await evaluator.compare_models()
    
    print("\nEvaluation complete. Improvement summary:")
    for metric, value in results["improvements"].items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())