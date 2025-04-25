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
from sklearn.feature_extraction.text import TfidfVectorizer

# Apply nest_asyncio
nest_asyncio.apply()

class ImprovedEvaluator:
    def __init__(self, dataset_name="squad", sample_size=20):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sample_size = sample_size
        self.dataset = self.load_dataset(dataset_name)
    
    def load_dataset(self, dataset_name):
        """Load data from Hugging Face datasets"""
        print(f"Loading {dataset_name} dataset...")
        
        try:
            # Load dataset from Hugging Face
            if dataset_name == "squad":
                data = datasets.load_dataset('squad', split='validation')
                
                # Format for our use
                contexts = []
                questions = []
                answers = []
                
                # Get unique contexts to avoid duplicates
                unique_contexts = {}
                for example in data:
                    if example['context'] not in unique_contexts:
                        unique_contexts[example['context']] = []
                    unique_contexts[example['context']].append({
                        'question': example['question'],
                        'answer': example['answers']['text'][0]
                    })
                
                # Select a diverse set of examples
                count = 0
                for context, qas in unique_contexts.items():
                    if count >= self.sample_size:
                        break
                    
                    contexts.append(context)
                    # Select one Q&A pair for this context
                    if qas:
                        qa = qas[0]
                        questions.append(qa['question'])
                        answers.append(qa['answer'])
                        count += 1
                
                print(f"Loaded {len(questions)} samples from {dataset_name}")
                return {
                    'contexts': contexts[:self.sample_size],
                    'questions': questions[:self.sample_size],
                    'answers': answers[:self.sample_size]
                }
            else:
                # Fallback to story.txt
                return self.create_fallback_dataset()
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self.create_fallback_dataset()
    
    def create_fallback_dataset(self):
        """Create a dataset from story.txt if HF datasets are unavailable"""
        try:
            with open("story.txt", "r") as f:
                text = f.read()
            
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            
            # Generate questions automatically
            questions = []
            answers = []
            
            for i, para in enumerate(paragraphs[:self.sample_size]):
                # Create a question about each paragraph
                first_sentence = para.split('.')[0]
                entity_match = re.search(r'\b[A-Z][a-z]+\b', first_sentence)
                
                if entity_match:
                    entity = entity_match.group(0)
                    questions.append(f"What does {entity} do in the story?")
                else:
                    questions.append(f"What happens in this part of the story?")
                
                answers.append(para)
            
            return {
                'contexts': paragraphs[:self.sample_size],
                'questions': questions,
                'answers': answers
            }
        except Exception as e:
            print(f"Error creating fallback dataset: {e}")
            return {
                'contexts': ["Sample text for testing."],
                'questions': ["What is in the text?"],
                'answers': ["Sample text for testing."]
            }

    async def setup_baseline_rag(self, working_dir):
        """Set up baseline LightRAG"""
        if os.path.exists(working_dir):
            import shutil
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)
        
        # Define mock LLM function
        async def mock_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # For testing, return a simple response based on the prompt
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
        """Set up improved LightRAG with substantial enhancements"""
        base_rag = await self.setup_baseline_rag(working_dir)
        
        # Store original methods
        original_insert = base_rag.insert
        original_query = base_rag.query
        
        # 1. Semantic-aware document chunking
        def semantic_chunking(text, min_size=100, max_size=500):
            """Chunk text based on semantic coherence and natural boundaries"""
            # First split by paragraphs
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            
            # Initialize chunks
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                # If adding this paragraph would exceed max size
                if len(current_chunk) + len(para) > max_size and len(current_chunk) >= min_size:
                    # Add current chunk
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap with previous chunk
                    sentences = current_chunk.split('.')
                    last_sentences = '.'.join(sentences[-3:]) if len(sentences) > 3 else sentences[-1]
                    current_chunk = last_sentences + " " + para
                else:
                    # Add separator if needed
                    if current_chunk and not current_chunk.endswith("\n"):
                        current_chunk += "\n\n"
                    current_chunk += para
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Further split any chunks that are still too large
            final_chunks = []
            for chunk in chunks:
                if len(chunk) <= max_size:
                    final_chunks.append(chunk)
                else:
                    # Split by sentences with overlap
                    sentences = chunk.split('.')
                    current = ""
                    
                    for i, sentence in enumerate(sentences):
                        if len(current) + len(sentence) > max_size and current:
                            final_chunks.append(current.strip())
                            
                            # Create 2-sentence overlap
                            overlap_start = max(0, i-2)
                            current = '.'.join(sentences[overlap_start:i]) + '.'
                        else:
                            current += sentence + '.'
                    
                    if current:
                        final_chunks.append(current.strip())
            
            return final_chunks
        
        # 2. Query expansion with semantic analysis
        def expand_query(query):
            """Expand query with semantically related terms"""
            # Extract key terms from query
            query_terms = re.findall(r'\b[a-z]{4,}\b', query.lower())
            
            # Use our embedding model to find related terms
            if query_terms:
                # Get embeddings for query terms
                term_embeddings = self.model.encode(query_terms)
                
                # Define candidate expansion terms
                expansion_candidates = [
                    "find", "discover", "locate", "identify", "search",
                    "describe", "explain", "elaborate", "detail", "clarify",
                    "analyze", "examine", "investigate", "explore", "study",
                    "understand", "comprehend", "grasp", "interpret",
                    "story", "narrative", "account", "tale", "text",
                    "character", "person", "figure", "individual", "entity",
                    "event", "occurrence", "incident", "happening", "situation",
                    "location", "place", "setting", "area", "region",
                    "time", "period", "duration", "interval", "moment",
                    "relationship", "connection", "association", "correlation"
                ]
                
                # Get embeddings for candidates
                candidate_embeddings = self.model.encode(expansion_candidates)
                
                # Find related terms for each query term
                expansions = []
                for i, term in enumerate(query_terms):
                    # Calculate similarity with all candidates
                    similarities = []
                    for j, candidate in enumerate(expansion_candidates):
                        if candidate != term:  # Skip exact matches
                            sim = 1 - cosine(term_embeddings[i], candidate_embeddings[j])
                            similarities.append((candidate, sim))
                    
                    # Get top 2 most similar terms for each query term
                    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:2]
                    expansions.extend([term for term, _ in top_similar])
                
                # Add unique expansions to query
                unique_expansions = list(set(expansions))
                if unique_expansions:
                    expanded_query = f"{query} (Relevant concepts: {', '.join(unique_expansions)})"
                    return expanded_query
            
            return query
        
        # 3. Dynamic retrieval optimization
        def optimize_retrieval(query, param):
            """Dynamically adjust retrieval parameters based on query analysis"""
            # Analyze query complexity
            word_count = len(query.split())
            has_entities = bool(re.findall(r'\b[A-Z][a-z]+\b', query))
            is_question = bool(re.search(r'\b(who|what|when|where|why|how)\b', query.lower()))
            
            # Simple question about named entity
            if is_question and has_entities and word_count < 8:
                param.mode = "hybrid"
                param.top_k = 3
            # Complex conceptual question
            elif word_count > 10:
                param.mode = "semantic"
                param.top_k = 5
            # Default approach for other queries
            else:
                param.mode = "hybrid"
                param.top_k = 4
            
            return param
        
        # 4. Context-aware chunk ranking
        def context_aware_ranking(chunks, query):
            """Re-rank chunks based on semantic and contextual relevance"""
            if not chunks:
                return chunks
                
            query_embedding = self.model.encode([query])[0]
            
            # Calculate relevance scores that combine similarity and additional factors
            ranked_chunks = []
            for chunk in chunks:
                # Get chunk text
                chunk_text = chunk.get('text', str(chunk))
                
                # Calculate base semantic similarity
                chunk_embedding = self.model.encode([chunk_text])[0]
                semantic_sim = 1 - cosine(query_embedding, chunk_embedding)
                
                # Calculate term overlap importance
                query_terms = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
                chunk_terms = set(re.findall(r'\b[a-z]{4,}\b', chunk_text.lower()))
                term_overlap = len(query_terms.intersection(chunk_terms)) / len(query_terms) if query_terms else 0
                
                # Check for entity match importance
                query_entities = set(re.findall(r'\b[A-Z][a-z]+\b', query))
                chunk_entities = set(re.findall(r'\b[A-Z][a-z]+\b', chunk_text))
                entity_match = len(query_entities.intersection(chunk_entities)) / len(query_entities) if query_entities else 0
                
                # Combined score with weightings
                combined_score = (semantic_sim * 0.5) + (term_overlap * 0.3) + (entity_match * 0.2)
                
                # Store score with chunk
                ranked_chunks.append((chunk, combined_score))
            
            # Sort by combined score
            ranked_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Return ranked chunks
            return [chunk for chunk, _ in ranked_chunks]
        
        # Override the methods with improved versions
        def improved_insert(text, doc_id=None, **kwargs):
            """Enhanced document insertion with semantic chunking"""
            chunks = semantic_chunking(text)
            
            # Insert each chunk
            chunk_ids = []
            for chunk in chunks:
                chunk_id = original_insert(chunk, **kwargs)
                chunk_ids.append(chunk_id)
            
            return chunk_ids[0] if chunk_ids else None
        
        def improved_query(query, param=None, **kwargs):
            """Enhanced query processing with expansions and dynamic optimization"""
            # Add fallback if param is None
            if not param:
                param = QueryParam(mode="hybrid", top_k=3)
            
            # Apply query expansion
            expanded_query = expand_query(query)
            
            # Optimize retrieval parameters
            optimized_param = optimize_retrieval(query, param)
            
            # Get initial results
            result = original_query(expanded_query, optimized_param, **kwargs)
            
            # Note: In a full implementation, we would also apply context-aware ranking here
            # But this requires more direct access to the internal chunks than the public API provides
            
            return result
        
        # Replace the methods
        base_rag.insert = improved_insert
        base_rag.query = improved_query
        
        return base_rag
    
    async def evaluate_model(self, model_type="baseline"):
        """Evaluate a model with comprehensive metrics"""
        working_dir = f"./lightrag_{model_type}_eval"
        
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
            "semantic_relevance": [],
            "answer_coverage": [],
            "entity_recall": [],
            "latency": []
        }
        
        # Run queries and measure performance
        print(f"\nEvaluating {model_type} model on {len(self.dataset['questions'])} questions...")
        
        for i, (question, answer) in enumerate(zip(self.dataset['questions'], self.dataset['answers'])):
            if i % 5 == 0:
                print(f"  Processing query {i+1}/{len(self.dataset['questions'])}")
            
            # Time the retrieval
            start_time = time.time()
            
            # Run query
            result = rag.query(question, QueryParam(mode="hybrid", top_k=3))
            
            # Measure latency
            latency = time.time() - start_time
            
            # Calculate semantic relevance
            result_embedding = self.model.encode([result])[0]
            answer_embedding = self.model.encode([answer])[0]
            semantic_relevance = 1 - cosine(result_embedding, answer_embedding)
            
            # Calculate answer coverage
            answer_terms = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
            result_terms = set(re.findall(r'\b[a-z]{4,}\b', result.lower()))
            coverage = len(answer_terms.intersection(result_terms)) / len(answer_terms) if answer_terms else 0
            
            # Calculate entity recall
            answer_entities = set(re.findall(r'\b[A-Z][a-z]+\b', answer))
            result_entities = set(re.findall(r'\b[A-Z][a-z]+\b', result))
            entity_recall = len(answer_entities.intersection(result_entities)) / len(answer_entities) if answer_entities else 0
            
            # Store metrics
            metrics["semantic_relevance"].append(semantic_relevance)
            metrics["answer_coverage"].append(coverage)
            metrics["entity_recall"].append(entity_recall)
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
        """Compare baseline and improved models"""
        # Evaluate baseline
        print("\n=== Evaluating Baseline Model ===")
        baseline_results = await self.evaluate_model("baseline")
        
        # Evaluate improved model
        print("\n=== Evaluating Improved Model ===")
        improved_results = await self.evaluate_model("improved")
        
        # Calculate percentage improvements
        improvements = {}
        for metric in baseline_results:
            if metric == "latency":
                # For latency, lower is better
                change = (baseline_results[metric] / improved_results[metric] - 1) * 100
            else:
                # For other metrics, higher is better
                change = (improved_results[metric] / baseline_results[metric] - 1) * 100
            
            improvements[metric] = change
        
        # Create visualizations
        self.create_visualizations(baseline_results, improved_results, improvements)
        
        # Save all results
        all_results = {
            "baseline": baseline_results,
            "improved": improved_results,
            "improvements": improvements
        }
        
        with open("lightrag_comprehensive_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("\n=== Improvement Summary ===")
        for metric, change in improvements.items():
            print(f"  {metric.replace('_', ' ').title()}: {change:.2f}%")
        
        return all_results
    
    def create_visualizations(self, baseline, improved, improvements):
        """Create comprehensive visualizations"""
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # 1. Bar chart comparison
        plt.subplot(2, 1, 1)
        
        metrics = ["Semantic Relevance", "Answer Coverage", "Entity Recall", "Latency (lower is better)"]
        baseline_values = [
            baseline["semantic_relevance"],
            baseline["answer_coverage"],
            baseline["entity_recall"],
            baseline["latency"]
        ]
        improved_values = [
            improved["semantic_relevance"],
            improved["answer_coverage"],
            improved["entity_recall"],
            improved["latency"]
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, baseline_values, width, label='Baseline LightRAG', color='#3498db')
        plt.bar(x + width/2, improved_values, width, label='Improved LightRAG', color='#2ecc71')
        
        plt.ylabel('Score')
        plt.title('LightRAG Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        
        # Add percentage labels
        for i, metric in enumerate(["semantic_relevance", "answer_coverage", "entity_recall", "latency"]):
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
        
        # 2. Improvement percentage bars
        plt.subplot(2, 1, 2)
        
        improvement_metrics = ["Semantic Relevance", "Answer Coverage", "Entity Recall", "Latency"]
        improvement_values = [
            improvements["semantic_relevance"],
            improvements["answer_coverage"],
            improvements["entity_recall"],
            improvements["latency"]
        ]
        
        colors = ['#27ae60' if v > 0 else '#c0392b' for v in improvement_values]
        if improvement_values[3] < 0:  # For latency, negative is bad
            colors[3] = '#c0392b'
        
        plt.barh(improvement_metrics, improvement_values, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Percentage Improvement')
        plt.title('LightRAG Performance Improvements')
        
        # Add value labels
        for i, v in enumerate(improvement_values):
            plt.text(v + np.sign(v)*2, i, f"{v:.2f}%", va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('lightrag_comprehensive_comparison.png', dpi=300)
        print("Created visualization: lightrag_comprehensive_comparison.png")

async def main():
    # Create evaluator with SQuAD dataset and reasonable sample size
    evaluator = ImprovedEvaluator(dataset_name="squad", sample_size=20)
    
    # Run comprehensive evaluation
    results = await evaluator.compare_models()

if __name__ == "__main__":
    asyncio.run(main())