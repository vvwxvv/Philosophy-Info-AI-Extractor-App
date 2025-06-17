
from dataclasses import dataclass
from typing import Dict, List, Any,Tuple
from datetime import datetime
import json
from dataclasses import dataclass, field
import re

@dataclass
class ValidationRule:
    """Validation rule for field extraction"""
    name: str
    rule_type: str  # "pattern", "range", "enum", "custom"
    value: Any
    error_message: str = ""
    
    def validate(self, data: Any) -> Tuple[bool, str]:
        """Validate data against this rule"""
        try:
            if self.rule_type == "pattern":
                return bool(re.match(self.value, str(data))), self.error_message
            elif self.rule_type == "range":
                min_val, max_val = self.value
                return min_val <= data <= max_val, self.error_message
            elif self.rule_type == "enum":
                return data in self.value, self.error_message
            elif self.rule_type == "custom":
                return self.value(data), self.error_message
        except Exception as e:
            return False, f"Validation error: {str(e)}"
        return True, ""

@dataclass
class ExtractionResult:
    """Result of an art information extraction"""
    extracted_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    errors: List[str] = None
    raw_text: str = None
    metadata: Dict[str, Any] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        result = {
            "data": self.extracted_data,
            "confidence": self.confidence_scores,
            "errors": self.errors or [],
            "metadata": {
                **(self.metadata or {}),
                "timestamp": self.timestamp
            }
        }
        
        if self.raw_text:
            result["raw_text"] = self.raw_text
            
        return result

    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save_to_file(self, filepath: str) -> None:
        """Save result to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json()) 