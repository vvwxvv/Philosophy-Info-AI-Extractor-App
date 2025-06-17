"""
High-level convenience functions
"""

import pandas as pd
import logging
from typing import Optional, List
from assets.ollama_api_client import OllamaAPIClient
from assets.csv_handler import CSVHandler
from assets.csv_processor import CSVProcessor

logger = logging.getLogger(__name__)


async def process_csv_with_llm(
    csv_path: str,
    text_column: int,
    model_name: str,
    prompt_template: str,
    output_path: Optional[str] = None,
    batch_size: int = 1,
    max_concurrent: int = 3,
    ollama_base_url: str = "http://127.0.0.1:11434",
) -> pd.DataFrame:
    """
    Complete CSV processing pipeline with LLM
    """
    # Initialize components
    ollama_client = OllamaAPIClient(ollama_base_url)
    processor = CSVProcessor(ollama_client)

    # Read CSV
    df = CSVHandler.read_csv(csv_path)

    # Process with LLM
    result_df = await processor.process_csv_column(
        df, text_column, model_name, prompt_template, batch_size, max_concurrent
    )

    # Save if requested
    if output_path:
        CSVHandler.save_csv(result_df, output_path)

    return result_df


def read_and_validate_csv(
    csv_path: str, expected_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Read and validate CSV structure"""
    df = CSVHandler.read_csv(csv_path)

    if expected_columns:
        CSVHandler.validate_columns(df, expected_columns)

    return df


def process_csv_simple(
    csv_path: str,
    output_path: Optional[str] = None,
    transformations: Optional[List[callable]] = None,
) -> pd.DataFrame:
    """
    Simple CSV processing without LLM

    Args:
        csv_path: Input CSV path
        output_path: Optional output path
        transformations: List of transformation functions to apply

    Returns:
        Processed DataFrame
    """
    df = CSVHandler.read_csv(csv_path)

    if transformations:
        for transform_func in transformations:
            df = transform_func(df)

    if output_path:
        CSVHandler.save_csv(df, output_path)

    return df
