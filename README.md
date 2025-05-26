# LiteLLM Central

A centralized interface for working with multiple AI language models using LiteLLM.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```bash
cp .env.example .env
```

4. Edit the `.env` file and add your API keys for the services you want to use.

## Usage

The script provides a unified interface for working with multiple AI models:

```python
from litellm_central import ModelManager

# intialize the manager
manager = ModelManager()

# list available models
models = manager.list_available_models()

# get a completion
response = manager.get_completion(
    model="gpt-3.5-turbo",  # or any other supported model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
```

## Supported Models

- OpenAI: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- Anthropic: Claude 4, Claude 3 Opus, Sonnet, and Haiku
- Google: Gemini Pro and Gemini Pro Vision
- OpenRouter: Various models from different providers
- Ollama: Local models like Llama2, Mistral, and CodeLlama

## Features

- Unified interface for multiple AI providers
- Environment variable management for API keys
- Comprehensive model listing
- Error handling and model availability checking
- Support for streaming responses
- Flexible parameter passing 