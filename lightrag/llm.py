import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiLLM:
    def __init__(self, model_name="gemini-pro", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.model = genai.GenerativeModel(model_name=model_name, 
                                          generation_config={"temperature": temperature})
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using Gemini model."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return "Error generating response."
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate multiple responses for a batch of prompts."""
        responses = []
        for prompt in prompts:
            responses.append(self.generate(prompt, **kwargs))
        return responses