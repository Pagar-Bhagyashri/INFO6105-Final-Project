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
import datasets

# Apply nest_asyncio
nest_asyncio.apply()

class SimpleEvaluator:
    def __init__(self, sample_size=10):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sample_size = sample_size
        self.dataset = self.load_dataset()
    
    def load_dataset(self):
        """Load a small dataset from SQuAD"""
        try:
            # Load SQuAD dataset
            data = datasets.load_dataset('squad', split='validation[:100]')
            
            # Format for our use
            contexts = []
            questions = []
            answers = []
            
            # Get unique contexts
            unique_contexts = {}
            for example in data:
                if example['context'] not in unique_contexts:
                    unique_contexts[example['context']] = []
                unique_contexts[example['context']].append({
                    'question': example['question'],
                    'answer': example['answers']['text'][0]
                })
            
            # Select a sample
            count = 0
            for context, qas in unique_contexts.items():
                if count >= self.sample_size:
                    break
                
                contexts.append(context)
                if qas:
                    qa = qas[0]
                    questions.append(qa['question'])
                    answers.append(qa['answer'])
                    count += 1
            
            print(f"Loaded {len(questions)} samples from SQuAD")
            return {
                'contexts': contexts[:self.sample_size],
                'questions': questions[:self.sample_size],
                'answers': answers[:self.sample_size]
            }
        except Exception as e:
            print(f"Error loading SQuAD: {e}")
            return self.create_fallback_dataset()
    
    def create_fallback_dataset(self):
        """Create a dataset from story.txt"""
        try:
            with open("story.txt", "r") as f:
                text = f.read()
            
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            # Create questions
            questions = []
            answers = []
            
            for i, para in enumerate(paragraphs[:self.sample_size]):
                # Simple question
                questions.append(f"What is described in this text: '{para[:30]}...'?")
                answers.append(para)
            
            return {
                'contexts': paragraphs[:self.sample_size],
                'questions': questions,
                'answers': answers
            }
        except Exception as e:
            print(f"Error creating fallback dataset: {e}")
            return {
                'contexts': ["This is sample text."],
                'questions': ["What is in the text?"],
                'answers': ["This is sample text."]
            }

    async def setup_baseline_rag(self, working_dir):
        """Setup baseline LightRAG"""
        if os.path.exists(working_dir):
            import shutil
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)
        
        # Define mock LLM function
        async def mock_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return f"Response to: {prompt[:50]}..."
        
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
        """Setup improved LightRAG with known compatible enhancements"""
        base_rag = await self.setup_baseline_rag(working_dir)
        
        # Store original methods
        original_insert = base_rag.insert
        original_query = base_rag.query
        
        # 1. Better chunking
        def improved_chunking(text, max_size=500):
            """Improved chunking that respects paragraphs and sentences"""
            # Split by paragraphs first
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                # If adding this paragraph exceeds max size
                if len(current_chunk) + len(para) > max_size and current_chunk:
                    chunks.append(current_chunk)
                    
                    # Create overlap with last sentence
                    sentences = current_chunk.split('.')
                    overlap = sentences[-1] if sentences else ""
                    current_chunk = overlap + " " + para
                else:
                    # Add separator if needed
                    if current_chunk and not current_chunk.endswith("\n"):
                        current_chunk += "\n\n"
                    current_chunk += para
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
        
        # 2. Improved query with term expansion
        def enhance_query(query):
            """Simple query enhancement with related terms"""
            expansions = {
                "who": ["person", "individual", "character"],
                "what": ["thing", "object", "concept"],
                "where": ["location", "place", "site"],
                "when": ["time", "date", "period"],
                "how": ["method", "way", "process"],
                "why": ["reason", "cause", "purpose"]
            }
            
            query_words = query.lower().split()
            expanded_terms = []
            
            for word in query_words:
                if word in expansions:
                    expanded_terms.extend(expansions[word])
            
            if expanded_terms:
                return f"{query} (Related: {', '.join(expanded_terms[:3])})"
            return query
            
        # 3. Query parameter optimization
        def optimize_params(query, param):
            """Optimize retrieval parameters based on query type"""
            # Default to hybrid mode for safety
            param.mode = "hybrid"
            
            # Check for entity questions
            if re.search(r'\b(who|what|where|when)\b', query.lower()):
                param.top_k = 4  # More chunks for factual queries
            else:
                param.top_k = 3  # Fewer for others
                
            return param
        
        # Override methods
        def improved_insert(text, doc_id=None, **kwargs):
            chunks = improved_chunking(text)
            doc_ids = []
            for chunk in chunks:
                chunk_id = original_insert(chunk, **kwargs)
                doc_ids.append(chunk_id)
            return doc_ids[0] if doc_ids else None
        
        def improved_query(query, param=None, **kwargs):
            # Provide default param if none provided
            if param is None:
                param = QueryParam(mode="hybrid", top_k=3)
                
            # Apply enhancements
            enhanced_query = enhance_query(query)
            optimized_param = optimize_params(query, param)
            
            # Call original query with improvements
            return original_query(enhanced_query, optimized_param, **kwargs)
        
        # Apply improvements
        base_rag.insert = improved_insert
        base_rag.query = improved_query
        
        return base_rag
    
    async def evaluate_model(self, model_type="baseline"):
        """Evaluate RAG model"""
        working_dir = f"./lightrag_{model_type}_simple_eval"
        
        if model_type == "baseline":
            rag = await self.setup_baseline_rag(working_dir)
        else:
            rag = await self.setup_improved_rag(working_dir)
        
        # Insert contexts
        print(f"Inserting {len(self.dataset['contexts'])} contexts...")
        for i, context in enumerate(self.dataset['contexts']):
            rag.insert(context)
            if (i+1) % 5 == 0:
                print(f"  Inserted {i+1}/{len(self.dataset['contexts'])} contexts")
        
        # Initialize metrics
        metrics = {
            "relevance_score": [],
            "term_coverage": [],
            "latency": []
        }
        
        # Run evaluation
        print(f"\nEvaluating {model_type} model on {len(self.dataset['questions'])} questions...")
        
        for i, (question, answer) in enumerate(zip(self.dataset['questions'], self.dataset['answers'])):
            if i % 5 == 0:
                print(f"  Processing query {i+1}/{len(self.dataset['questions'])}")
            
            # Time the query
            start_time = time.time()
            
            # Run query with safe parameters
            result = rag.query(question, QueryParam(mode="hybrid", top_k=3))
            
            # Measure latency
            latency = time.time() - start_time
            
            # Calculate relevance
            result_embedding = self.model.encode([result])[0]
            answer_embedding = self.model.encode([answer])[0]
            relevance = 1 - cosine(result_embedding, answer_embedding)
            
            # Calculate term coverage
            answer_terms = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
            result_terms = set(re.findall(r'\b[a-z]{4,}\b', result.lower()))
            coverage = len(answer_terms.intersection(result_terms)) / len(answer_terms) if answer_terms else 0
            
            # Store metrics
            metrics["relevance_score"].append(relevance)
            metrics["term_coverage"].append(coverage)
            metrics["latency"].append(latency)
        
        # Calculate aggregate metrics
        results = {}
        for key, values in metrics.items():
            results[key] = float(np.mean(values))
        
        # Report results
        print(f"\nResults for {model_type} model:")
        for metric, value in results.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        return results
    
    async def compare_models(self):
        """Compare both models"""
        print("\n=== Evaluating Baseline Model ===")
        baseline_results = await self.evaluate_model("baseline")
        
        print("\n=== Evaluating Improved Model ===")
        improved_results = await self.evaluate_model("improved")
        
        # Calculate improvements
        improvements = {}
        for metric in baseline_results:
            if metric == "latency":
                # For latency, lower is better
                change = (baseline_results[metric] / improved_results[metric] - 1) * 100
            else:
                # For other metrics, higher is better
                change = (improved_results[metric] / baseline_results[metric] - 1) * 100
            
            improvements[metric] = change
        
        # Create visualization
        self.create_visualization(baseline_results, improved_results, improvements)
        
        # Save results
        results = {
            "baseline": baseline_results,
            "improved": improved_results,
            "improvements": improvements
        }
        
        with open("lightrag_comparison_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n=== Improvement Summary ===")
        for metric, change in improvements.items():
            print(f"  {metric.replace('_', ' ').title()}: {change:.2f}%")
        
        return results
    
    def create_visualization(self, baseline, improved, improvements):
        """Create comparison visualization"""
        # Prepare data
        metrics = ["Relevance Score", "Term Coverage", "Latency (lower is better)"]
        baseline_values = [
            baseline["relevance_score"],
            baseline["term_coverage"],
            baseline["latency"]
        ]
        improved_values = [
            improved["relevance_score"],
            improved["term_coverage"],
            improved["latency"]
        ]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, baseline_values, width, label='Baseline LightRAG', color='#3498db')
        plt.bar(x + width/2, improved_values, width, label='Improved LightRAG', color='#2ecc71')
        
        plt.ylabel('Score')
        plt.title('LightRAG Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        
        # Add percentage labels
        for i, metric in enumerate(["relevance_score", "term_coverage", "latency"]):
            pct = improvements[metric]
            color = 'green' if pct > 0 else 'red'
            if metric == "latency" and pct < 0:
                color = 'red'
            
            plt.annotate(f"{pct:.2f}%",
                       xy=(i + width/2, improved_values[i]),
                       xytext=(0, 10),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('lightrag_comparison.png', dpi=300)
        print("Created visualization: lightrag_comparison.png")

async def main():
    # Create evaluator with smaller sample size for faster processing
    evaluator = SimpleEvaluator(sample_size=8)
    
    # Run evaluation
    results = await evaluator.compare_models()

if __name__ == "__main__":
    asyncio.run(main())