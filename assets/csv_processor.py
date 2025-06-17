"""
CSV Processing with LLM Integration
"""

import asyncio
from typing import Optional, List, Dict, Any
from csv_handler import CSVHandler
from assets.ollama_api_client import OllamaAPIClient
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class CSVProcessor:
    """
    CSV processor with LLM integration
    """
    
    def __init__(self, ollama_client: Optional[OllamaAPIClient] = None):
        self.ollama_client = ollama_client or OllamaAPIClient()
        self.csv_handler = CSVHandler()
    
    async def process_texts_with_llm(self, texts: List[Optional[str]], 
                                   model_name: str, prompt_template: str,
                                   batch_size: int = 1, max_concurrent: int = 3) -> List[Dict[Any, Any]]:
        """
        Process list of texts with LLM
        
        Args:
            texts: List of text strings to process
            model_name: LLM model name
            prompt_template: Prompt template with {text} placeholder
            batch_size: Batch size for processing
            max_concurrent: Max concurrent requests
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing {len(texts)} texts with model '{model_name}'")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_text(idx: int, text: Optional[str]) -> Dict[Any, Any]:
            async with semaphore:
                if text is None:
                    return {"row_index": idx, "error": "empty_text"}
                
                try:
                    full_prompt = prompt_template.format(text=text)
                    response = await self.ollama_client.call_api_with_json_response(
                        model_name, full_prompt
                    )
                    response["row_index"] = idx
                    return response
                    
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    return {"row_index": idx, "error": str(e)}
        
        # Create and execute tasks in batches
        tasks = [process_single_text(idx, text) for idx, text in enumerate(texts)]
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(tasks) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch error: {result}")
                    results.append({"error": str(result)})
                else:
                    results.append(result)
        
        return results
    
    async def process_csv_column(self, df: pd.DataFrame, text_column: int,
                               model_name: str, prompt_template: str,
                               batch_size: int = 1, max_concurrent: int = 3) -> pd.DataFrame:
        """
        Process CSV column with LLM and return merged DataFrame
        """
        # Extract texts from column
        texts = self.csv_handler.extract_column_data(df, text_column)
        
        # Process with LLM
        results = await self.process_texts_with_llm(
            texts, model_name, prompt_template, batch_size, max_concurrent
        )
        
        # Merge results
        return self.csv_handler.merge_results(df, results)