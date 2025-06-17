# ollama_client.py
"""
Ollama API Client Module
Handles communication with Ollama API server
"""

import json
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaAPIClient:
    """
    Core Ollama API client for making requests to the API
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 300):
        """
        Initialize Ollama API client
        
        Args:
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        self.timeout = timeout
    
    async def call_api(self, model_name: str, prompt: str, 
                      system_prompt: Optional[str] = None) -> str:
        """
        Make API call to Ollama and return response content
        
        Args:
            model_name: Name of the model to use
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Full response content as string
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        request_body = {
            "model": model_name,
            "messages": messages,
            "stream": True
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        full_content = ''
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, json=request_body, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"API returned status {response.status}")
                        error_text = await response.text()
                        raise Exception(f"API Error {response.status}: {error_text}")
                    
                    async for chunk in response.content.iter_chunked(1024):
                        lines = chunk.decode('utf-8').split('\n')
                        lines = [line.strip() for line in lines if line.strip()]
                        
                        for line in lines:
                            try:
                                parsed = json.loads(line)
                                if parsed.get('message', {}).get('content'):
                                    full_content += parsed['message']['content']
                                
                                # Check if response is done
                                if parsed.get('done', False):
                                    break
                                    
                            except json.JSONDecodeError as err:
                                logger.debug(f"JSON parsing warning: {err}")
                                continue
                                
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self.timeout} seconds")
            raise Exception(f"Request timed out after {self.timeout} seconds")
        except Exception as error:
            logger.error(f"API Error: {error}")
            raise
        
        return full_content
    
    async def call_api_with_json_response(self, model_name: str, prompt: str, 
                                        system_prompt: Optional[str] = None) -> Dict[Any, Any]:
        """
        Make API call and attempt to parse response as JSON
        
        Args:
            model_name: Name of the model to use
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response or error dict
        """
        try:
            response = await self.call_api(model_name, prompt, system_prompt)
            
            # Try to extract JSON from response
            response = response.strip()
            
            # Look for JSON block if response contains extra text
            if response.startswith('```json'):
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != 0:
                    response = response[start:end]
            
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON response: {e}")
            return {
                "error": "json_parse_error",
                "raw_response": response,
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {
                "error": "api_call_error",
                "message": str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

# Convenience function for simple usage
async def simple_api_call(model_name: str, prompt: str, 
                         base_url: str = "http://127.0.0.1:11434",
                         system_prompt: Optional[str] = None) -> str:
    """
    Simple function to make a single API call
    
    Args:
        model_name: Ollama model name
        prompt: User prompt
        base_url: Ollama API base URL
        system_prompt: Optional system prompt
        
    Returns:
        Response content
    """
    client = OllamaAPIClient(base_url)
    return await client.call_api(model_name, prompt, system_prompt)

# Example usage
if __name__ == "__main__":
    async def test_client():
        client = OllamaAPIClient()
        
        # Test connection
        if not client.test_connection():
            print("Cannot connect to Ollama API")
            return
        
        # Simple call
        response = await client.call_api(
            model_name="deepseek-r1:7b",
            prompt="Write hello world in Python"
        )
        print("Response:", response)
        
        # JSON call
        json_response = await client.call_api_with_json_response(
            model_name="deepseek-r1:7b",
            prompt="Return this as JSON: {'message': 'hello world', 'language': 'python'}"
        )
        print("JSON Response:", json_response)
    
    asyncio.run(test_client())