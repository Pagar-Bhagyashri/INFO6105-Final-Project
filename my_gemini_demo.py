import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
import nest_asyncio
import time

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()
load_dotenv()

# Get your API key from the environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Working directory for LightRAG
WORKING_DIR = "./my_lightrag_project"
if os.path.exists(WORKING_DIR):
    import shutil
    shutil.rmtree(WORKING_DIR)
os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    try:
        # Create the model with flash instead of pro to avoid rate limits
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        
        # Combine prompts: system prompt, history, and user prompt
        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{system_prompt}\n"
        
        if history_messages:
            for msg in history_messages:
                combined_prompt += f"{msg['role']}: {msg['content']}\n"
        
        # Add the new user prompt
        combined_prompt += f"user: {prompt}"
        
        # Generate the response
        response = model.generate_content(combined_prompt)
        return response.text
    
    except Exception as e:
        print(f"Error generating with Gemini: {e}")
        return f"Error generating response: {str(e)}"

async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

def main():
    print("Initializing RAG instance...")
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    
    print("Loading story text...")
    # Load the story text
    file_path = "story.txt"
    with open(file_path, "r") as file:
        text = file.read()
    
    print("Inserting text into LightRAG...")
    # Insert the text into LightRAG
    rag.insert(text)
    
    print("Running first query...")
    # Query the system
    response = rag.query(
        query="What is the main theme of the story?",
        param=QueryParam(mode="hybrid", top_k=5, response_type="single line"),
    )
    print("\nQuery: What is the main theme of the story?")
    print("Response:", response)
    
    print("Waiting before second query...")
    # Add a delay to avoid rate limits before the second query
    time.sleep(5)
    
    print("Running second query...")
    # Try another query
    response2 = rag.query(
        query="Who is Lily and what did she discover?",
        param=QueryParam(mode="hybrid", top_k=5),
    )
    print("\nQuery: Who is Lily and what did she discover?")
    print("Response:", response2)
    
    print("Script completed successfully!")

if __name__ == "__main__":
    main()