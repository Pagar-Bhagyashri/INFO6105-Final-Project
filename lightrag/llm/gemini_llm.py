import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio
from typing import Any, Dict, List, Optional

load_dotenv()

# Configure the Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

async def gemini_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, Any]] = [],
    model: str = "models/gemini-1.5-pro",
    temperature: float = 0.7,
    **kwargs: Any,
) -> str:
    """Generate a response using Google's Gemini model."""
    try:
        # Make sure model name has the proper format
        if not model.startswith("models/"):
            model = f"models/{model}"
            
        # Create the generative model
        genai_model = genai.GenerativeModel(model_name=model, 
                                         generation_config={"temperature": temperature})
        
        # Build the prompt with system instructions if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
        # Generate response
        response = genai_model.generate_content(full_prompt)
        return response.text
            
    except Exception as e:
        print(f"Error generating with Gemini: {e}")
        return f"Error generating response: {str(e)}"

# Example usage
if __name__ == "__main__":
    async def main():
        result = await gemini_complete("What is machine learning?")
        print(result)
    
    asyncio.run(main())