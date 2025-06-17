"""
Assets Package for Philosophy Extraction System

This package provides utility modules for API clients, CSV processing,
and other supporting functionality for the philosophy extraction system.
"""

# API Clients
from .openai_chat_client import OpenAIChatClient
from .ollama_api_client import (
    OllamaAPIClient,
    simple_api_call,
)

# CSV Processing
from .csv_handler import CSVHandler
from .csv_processor import CSVProcessor
from .csv_convenience_functions import (
    process_csv_with_llm,
    read_and_validate_csv,
    process_csv_simple,
)

# Version info
__version__ = "1.0.0"
__author__ = "Philosophy Extraction System Assets"

# Main exports
__all__ = [
    # API Clients
    "OpenAIChatClient",
    "OllamaAPIClient",
    "simple_api_call",
    # CSV Processing
    "CSVHandler",
    "CSVProcessor",
    "process_csv_with_llm",
    "read_and_validate_csv",
    "process_csv_simple",
]


# Convenience functions for quick access
def create_openai_client(
    api_key: str = None,
    model: str = "deepseek-ai/DeepSeek-V3",
    system_prompt: str = "You are a helpful assistant.",
    **kwargs,
) -> OpenAIChatClient:
    """Create an OpenAI chat client with philosophy-optimized defaults"""
    return OpenAIChatClient(
        api_key=api_key, model=model, system_prompt=system_prompt, **kwargs
    )


def create_philosophy_openai_client(api_key: str = None) -> OpenAIChatClient:
    """Create an OpenAI client optimized for philosophy tasks"""
    philosophy_prompt = """You are a sophisticated philosophical analysis assistant. 
    You have deep knowledge of philosophy across all major traditions and periods. 
    Provide clear, nuanced, and academically rigorous responses while remaining accessible."""

    return create_openai_client(
        api_key=api_key,
        system_prompt=philosophy_prompt,
        temperature=0.3,  # Lower temperature for more consistent philosophical analysis
        max_tokens=1000,  # More tokens for detailed philosophical responses
    )


def create_ollama_client(
    base_url: str = "http://127.0.0.1:11434", timeout: int = 300
) -> OllamaAPIClient:
    """Create an Ollama API client with sensible defaults"""
    return OllamaAPIClient(base_url=base_url, timeout=timeout)


def test_api_connections(
    ollama_base_url: str = "http://127.0.0.1:11434", openai_api_key: str = None
) -> dict:
    """Test connections to available API services"""
    results = {"ollama": False, "openai": False, "errors": []}

    # Test Ollama connection
    try:
        ollama_client = create_ollama_client(ollama_base_url)
        results["ollama"] = ollama_client.test_connection()
        if not results["ollama"]:
            results["errors"].append("Ollama connection failed")
    except Exception as e:
        results["errors"].append(f"Ollama error: {str(e)}")

    # Test OpenAI connection
    if openai_api_key:
        try:
            openai_client = create_openai_client(api_key=openai_api_key)
            # Simple test call
            response = openai_client.ask("Test connection")
            results["openai"] = not response.startswith("Error:")
            if not results["openai"]:
                results["errors"].append("OpenAI connection failed")
        except Exception as e:
            results["errors"].append(f"OpenAI error: {str(e)}")

    return results


def create_csv_processor(ollama_client: OllamaAPIClient = None) -> CSVProcessor:
    """Create a CSV processor with optional Ollama client"""
    if ollama_client is None:
        ollama_client = create_ollama_client()
    return CSVProcessor(ollama_client)


async def quick_csv_analysis(
    csv_path: str,
    text_column: int,
    analysis_prompt: str = "Analyze this text and extract key themes: {text}",
    model_name: str = "deepseek-r1:7b",
    output_path: str = None,
    **kwargs,
) -> "pd.DataFrame":
    """Quick CSV text analysis using default settings"""
    return await process_csv_with_llm(
        csv_path=csv_path,
        text_column=text_column,
        model_name=model_name,
        prompt_template=analysis_prompt,
        output_path=output_path,
        **kwargs,
    )


async def philosophy_csv_analysis(
    csv_path: str,
    text_column: int,
    philosophy_focus: str = "general",
    model_name: str = "deepseek-r1:7b",
    output_path: str = None,
    **kwargs,
) -> "pd.DataFrame":
    """Analyze CSV texts for philosophical content"""

    # Philosophy-specific prompts
    philosophy_prompts = {
        "general": "Analyze this text for philosophical themes, concepts, and arguments. Extract key philosophical ideas: {text}",
        "ethics": "Analyze this text for ethical principles, moral arguments, and value judgments: {text}",
        "metaphysics": "Analyze this text for metaphysical claims about reality, existence, and the nature of being: {text}",
        "epistemology": "Analyze this text for claims about knowledge, truth, belief, and justification: {text}",
        "logic": "Analyze this text for logical structure, arguments, reasoning patterns, and validity: {text}",
        "aesthetics": "Analyze this text for aesthetic judgments, theories of beauty, and philosophy of art: {text}",
    }

    prompt = philosophy_prompts.get(philosophy_focus, philosophy_prompts["general"])

    return await process_csv_with_llm(
        csv_path=csv_path,
        text_column=text_column,
        model_name=model_name,
        prompt_template=prompt,
        output_path=output_path,
        **kwargs,
    )


# Add convenience functions to exports
__all__.extend(
    [
        "create_openai_client",
        "create_philosophy_openai_client",
        "create_ollama_client",
        "test_api_connections",
        "create_csv_processor",
        "quick_csv_analysis",
        "philosophy_csv_analysis",
    ]
)

# Package-level documentation
__doc__ = """
Assets Package for Philosophy Extraction System

This package provides essential utility modules and API clients for the
philosophy extraction system, including:

1. API Clients:
   - OpenAIChatClient: Client for OpenAI-compatible APIs (SiliconFlow)
   - OllamaAPIClient: Client for local Ollama API server
   - Support for async operations and error handling

2. CSV Processing:
   - CSVHandler: Pure CSV operations (read, write, validate)
   - CSVProcessor: CSV processing with LLM integration
   - Batch processing and concurrent request handling

3. Convenience Functions:
   - High-level functions for common tasks
   - Philosophy-specific processing pipelines
   - Connection testing and validation

Usage Examples:

    # API Clients
    from assets import create_philosophy_openai_client, create_ollama_client
    
    # OpenAI client for philosophy
    openai_client = create_philosophy_openai_client(api_key="your-key")
    response = openai_client.ask("What is the nature of consciousness?")
    
    # Ollama client
    ollama_client = create_ollama_client()
    
    # Test connections
    from assets import test_api_connections
    status = test_api_connections(openai_api_key="your-key")
    
    # CSV Processing
    from assets import philosophy_csv_analysis
    import asyncio
    
    # Analyze CSV for philosophical content
    df = asyncio.run(philosophy_csv_analysis(
        csv_path="texts.csv",
        text_column=1,
        philosophy_focus="ethics",
        output_path="analyzed.csv"
    ))
    
    # Simple CSV operations
    from assets import CSVHandler, read_and_validate_csv
    
    df = read_and_validate_csv("data.csv", expected_columns=["text", "author"])
    CSVHandler.save_csv(df, "output.csv")
    
    # Quick text analysis
    from assets import quick_csv_analysis
    
    df = await quick_csv_analysis(
        csv_path="documents.csv",
        text_column=0,
        analysis_prompt="Extract main themes from: {text}",
        model_name="deepseek-r1:7b"
    )

Features:

- **Async Support**: All LLM operations support async/await
- **Error Handling**: Comprehensive error handling and logging
- **Batch Processing**: Efficient batch processing for large datasets
- **Connection Testing**: Built-in connection validation
- **Philosophy Integration**: Pre-configured for philosophy-specific tasks
- **Flexible Configuration**: Customizable models, prompts, and parameters
"""

# Initialize package-level logging
import logging

logger = logging.getLogger(__name__)
logger.info(f"Assets package loaded (version {__version__})")

# Validate dependencies
try:
    import pandas as pd
    import aiohttp
    import requests

    logger.info("All required dependencies available")
except ImportError as e:
    logger.warning(f"Missing optional dependency: {e}")

# Check for environment variables
import os

if not os.getenv("SILICONFLOW_API_KEY"):
    logger.info(
        "SILICONFLOW_API_KEY not found in environment - OpenAI client will require explicit API key"
    )

# Connection status cache
_connection_status = {}


def get_connection_status(force_refresh: bool = False) -> dict:
    """Get cached connection status or refresh if needed"""
    global _connection_status

    if not _connection_status or force_refresh:
        try:
            _connection_status = test_api_connections()
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            _connection_status = {"ollama": False, "openai": False, "errors": [str(e)]}

    return _connection_status


# Add to exports
__all__.append("get_connection_status")
