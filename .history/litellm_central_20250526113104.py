from litellm import completion, ModelResponse
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelManager:
    def __init__(self):
        # Initialize API keys from environment variables
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "nebius": os.getenv("NEBIUS_API_KEY"),
        }
        
        # Define available models for each provider
        self.models = {
            "openai": [
                "gpt-4",
                "gpt-4-turbo-preview",
                "gpt-3.5-turbo",
            ],
            "anthropic": [
                "claude-4",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            "google": [
                "gemini-pro",
                "gemini-pro-vision",
            ],
            "openrouter": [
                "openai/gpt-4",
                "anthropic/claude-4",
                "anthropic/claude-3-opus",
                "google/gemini-pro",
            ],
            "ollama": [
                "llama2",
                "mistral",
                "codellama",
            ],
        }

    def get_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ModelResponse:
        """
        Get completion from any supported model.
        
        Args:
            model (str): Model identifier
            messages (List[Dict[str, str]]): List of message dictionaries
            temperature (float): Sampling temperature
            max_tokens (Optional[int]): Maximum tokens to generate
            stream (bool): Whether to stream the response
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse: The model's response
        """
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"Error getting completion from {model}: {str(e)}")
            raise

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models grouped by provider."""
        return self.models

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model (str): Model identifier
            
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model": model,
            "provider": self._get_provider_from_model(model),
            "available": self._is_model_available(model),
        }

    def _get_provider_from_model(self, model: str) -> str:
        """Get the provider name for a given model."""
        for provider, models in self.models.items():
            if model in models:
                return provider
        return "unknown"

    def _is_model_available(self, model: str) -> bool:
        """Check if a model is available in our configuration."""
        return any(model in models for models in self.models.values())

# Example usage
if __name__ == "__main__":
    # Initialize the model manager
    manager = ModelManager()
    
    # Example: List all available models
    print("Available Models:")
    for provider, models in manager.list_available_models().items():
        print(f"\n{provider.upper()}:")
        for model in models:
            print(f"  - {model}")
    
    # Example: Get completion from a model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ]
    
    try:
        # Try with GPT-3.5
        response = manager.get_completion(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        print("\nGPT-3.5 Response:", response.choices[0].message.content)
        
        # Try with Claude
        response = manager.get_completion(
            model="claude-3-haiku-20240307",
            messages=messages,
            temperature=0.7
        )
        print("\nClaude Response:", response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {str(e)}") 