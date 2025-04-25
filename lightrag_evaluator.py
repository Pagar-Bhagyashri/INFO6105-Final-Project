import os
import time
import json
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.metrics import precision_score, recall_score, f1_score
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import google.generativeai as genai
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()
load_dotenv()

# Get API key from environment
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

class LightRAGEvaluator:
    def __init__(self, dataset_path=None):
        """
        Initialize the evaluator with optional dataset path
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dataset_path = dataset_path
        self.dataset = None
        self.load_dataset()
    
    def load_dataset(self):
        """
        Load evaluation dataset from SQuAD or create from documents
        """
        try:
            import datasets
            print("Loading SQuAD dataset...")
            squad = datasets.load_dataset('squad', split='validation[:100]')
            
            # Format dataset for our use
            self.dataset = {
                'contexts': [example['context'] for example in squad],
                'questions': [example['question'] for example in squad],
                'answers': [example['answers']['text'][0] for example in squad]
            }
            print(f"Loaded {len(self.dataset['questions'])} samples from SQuAD")
            
        except Exception as e:
            print(f"Could not load SQuAD dataset: {e}")
            print("Using fallback dataset generation...")
            self.generate_fallback_dataset()
    
    def generate_fallback_dataset(self):
        """
        Generate a fallback dataset if SQuAD loading fails
        """
        # Try to use a sample text file
        try:
            with open("story.txt", "r") as f:
                text = f.read()
            
            # Generate questions automatically
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            contexts = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3) if i+3 <= len(sentences)]
            
            # Generate simple questions from contexts
            questions = []
            answers = []
            
            for context in contexts[:5]:  # Limit to 5 samples
                # Extract entities
                entities = re.findall(r'\b[A-Z][a-z]+\b', context)
                if entities:
                    entity = entities[0]
                    questions.append(f"What is {entity} doing in the text?")
                    answers.append(context)  # Use full context as answer for evaluation
                else:
                    questions.append("What is described in this text?")
                    answers.append(context)
            
            self.dataset = {
                'contexts': contexts[:5],
                'questions': questions,
                'answers': answers
            }
            print(f"Generated fallback dataset with {len(questions)} samples")
            
        except Exception as e:
            print(f"Could not generate fallback dataset: {e}")
            # Create minimal dataset
            self.dataset = {
                'contexts': ["This is a sample text."],
                'questions': ["What is this?"],
                'answers': ["This is a sample text."]
            }
    
    async def setup_rag(self, working_dir, model_type="baseline"):
        """
        Set up a LightRAG instance with specified parameters
        """
        # Create clean working directory
        if os.path.exists(working_dir):
            import shutil
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)
        
        # Define embedding function
        async def embedding_func(texts):
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        
        # Define LLM function
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            try:
                model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
                
                combined_prompt = ""
                if system_prompt:
                    combined_prompt += f"{system_prompt}\n"
                
                if history_messages:
                    for msg in history_messages:
                        combined_prompt += f"{msg['role']}: {msg['content']}\n"
                
                combined_prompt += f"user: {prompt}"
                
                response = model.generate_content(combined_prompt)
                return response.text
            
            except Exception as e:
                print(f"Error generating with Gemini: {e}")
                return f"Error generating response: {str(e)}"
        
        # Initialize RAG
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=embedding_func,
            ),
        )
        
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        # Apply specific configurations based on model type
        if model_type == "improved":
            # Custom configurations for improved version
            # These configurations will be defined in the lightrag_improvements.py file
            from lightrag_improvements import apply_improvements
            rag = apply_improvements(rag)
        
        return rag
    
    async def evaluate_model(self, model_type="baseline"):
        """
        Evaluate a LightRAG model on the dataset
        """
        working_dir = f"./lightrag_{model_type}_eval"
        rag = await self.setup_rag(working_dir, model_type)
        
        # Insert contexts into RAG
        print(f"Inserting {len(self.dataset['contexts'])} contexts into {model_type} model...")
        for i, context in enumerate(self.dataset['contexts']):
            rag.insert(context)
            if i % 10 == 0:
                print(f"Inserted {i+1}/{len(self.dataset['contexts'])} contexts")
        
        # Measure performance
        results = {
            "retrieval_precision": [],
            "answer_relevance": [],
            "latency": []
        }
        
        print(f"Evaluating {model_type} model on {len(self.dataset['questions'])} questions...")
        for i, (question, answer) in enumerate(zip(self.dataset['questions'], self.dataset['answers'])):
            # Time the retrieval process
            start_time = time.time()
            
            # Query the system
            response = rag.query(
                query=question,
                param=QueryParam(mode="hybrid", top_k=3),
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Calculate answer relevance using embedding similarity
            answer_embedding = self.model.encode([answer])[0]
            response_embedding = self.model.encode([response])[0]
            similarity = 1 - cosine(answer_embedding, response_embedding)
            
            # Store results
            results["retrieval_precision"].append(1.0 if similarity > 0.5 else 0.0)  # Simple threshold-based precision
            results["answer_relevance"].append(similarity)
            results["latency"].append(latency)
            
            # Show progress
            if i % 5 == 0:
                print(f"Processed {i+1}/{len(self.dataset['questions'])} questions")
                print(f"  Sample Q: {question}")
                print(f"  Response: {response[:100]}...")
                print(f"  Similarity: {similarity:.4f}, Latency: {latency:.4f}s")
        
        # Aggregate results
        aggregated_results = {
            "retrieval_precision": np.mean(results["retrieval_precision"]),
            "answer_relevance": np.mean(results["answer_relevance"]),
            "latency": np.mean(results["latency"])
        }
        
        print(f"Evaluation results for {model_type}:")
        print(f"  Retrieval Precision: {aggregated_results['retrieval_precision']:.4f}")
        print(f"  Answer Relevance: {aggregated_results['answer_relevance']:.4f}")
        print(f"  Average Latency: {aggregated_results['latency']:.4f}s")
        
        return aggregated_results
    
    async def compare_models(self):
        """
        Compare baseline and improved models
        """
        print("Evaluating baseline model...")
        baseline_results = await self.evaluate_model("baseline")
        
        print("\nEvaluating improved model...")
        improved_results = await self.evaluate_model("improved")
        
        # Calculate improvements
        improvements = {
            "retrieval_precision": (improved_results["retrieval_precision"] / baseline_results["retrieval_precision"] - 1) * 100,
            "answer_relevance": (improved_results["answer_relevance"] / baseline_results["answer_relevance"] - 1) * 100,
            "latency": (baseline_results["latency"] / improved_results["latency"] - 1) * 100  # Latency improvement is inverse
        }
        
        # Visualization
        self.visualize_comparison(baseline_results, improved_results, improvements)
        
        return {
            "baseline": baseline_results,
            "improved": improved_results,
            "improvements": improvements
        }
    
    def visualize_comparison(self, baseline_results, improved_results, improvements):
        """
        Visualize the comparison between baseline and improved models
        """
        metrics = ["Retrieval Precision", "Answer Relevance", "Latency (lower is better)"]
        baseline_values = [
            baseline_results["retrieval_precision"], 
            baseline_results["answer_relevance"], 
            baseline_results["latency"]
        ]
        improved_values = [
            improved_results["retrieval_precision"], 
            improved_results["answer_relevance"], 
            improved_results["latency"]
        ]
        
        # Create a comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.35
        
        baseline_bars = ax.bar(x - width/2, baseline_values, width, label='Baseline LightRAG')
        improved_bars = ax.bar(x + width/2, improved_values, width, label='Improved LightRAG')
        
        ax.set_ylabel('Score')
        ax.set_title('LightRAG Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add improvement percentages
        for i, metric in enumerate(["retrieval_precision", "answer_relevance", "latency"]):
            percent = improvements[metric]
            color = 'green' if percent > 0 else 'red'
            if metric == "latency" and percent < 0:
                color = 'red'  # Lower latency is better
            
            ax.annotate(f"{percent:.2f}%",
                        xy=(i + width/2, improved_values[i]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('lightrag_performance_comparison.png', dpi=300)
        print("Created visualization: lightrag_performance_comparison.png")

async def main():
    evaluator = LightRAGEvaluator()
    results = await evaluator.compare_models()
    
    # Save results to file
    with open("lightrag_evaluation_results.json", "w") as f:
        json.dump({
            "baseline": {k: float(v) for k, v in results["baseline"].items()},
            "improved": {k: float(v) for k, v in results["improved"].items()},
            "improvements": {k: float(v) for k, v in results["improvements"].items()}
        }, f, indent=2)
    
    print("Evaluation complete. Results saved to lightrag_evaluation_results.json")

if __name__ == "__main__":
    asyncio.run(main())