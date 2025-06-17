"""
CSV Handling Module
Pure CSV operations without LLM dependencies
"""

import pandas as pd
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class CSVHandler:
    """
    Pure CSV operations - reading, writing, validation
    """
    
    @staticmethod
    def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """Read CSV file with validation"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
            
            if df.empty:
                logger.warning("CSV file is empty")
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
    
    @staticmethod
    def save_csv(df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Save DataFrame to CSV"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, **kwargs)
            logger.info(f"CSV saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            raise
    
    @staticmethod
    def validate_column_index(df: pd.DataFrame, column_index: int) -> str:
        """Validate column index and return column name"""
        if column_index >= len(df.columns) or column_index < 0:
            raise ValueError(
                f"Column index {column_index} out of range. "
                f"CSV has {len(df.columns)} columns (0-{len(df.columns)-1})"
            )
        return df.columns[column_index]
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, expected_columns: List[str]) -> None:
        """Validate expected columns exist"""
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
    
    @staticmethod
    def extract_column_data(df: pd.DataFrame, column_index: int) -> List[Optional[str]]:
        """Extract text data from specified column"""
        col_name = CSVHandler.validate_column_index(df, column_index)
        
        texts = []
        for _, row in df.iterrows():
            text_content = str(row[col_name])
            if pd.isna(text_content) or text_content.strip() == '':
                texts.append(None)
            else:
                texts.append(text_content.strip())
        
        return texts
    
    @staticmethod
    def merge_results(df: pd.DataFrame, results: List[Dict[Any, Any]]) -> pd.DataFrame:
        """Merge processing results with original DataFrame"""
        results_df = pd.json_normalize(results)
        
        if len(results_df) != len(df):
            logger.warning(
                f"Results count ({len(results_df)}) doesn't match "
                f"DataFrame count ({len(df)})"
            )
        
        return pd.concat([df, results_df], axis=1)
