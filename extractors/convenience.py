"""
Convenience functions for the extractors package
"""

from typing import List, Dict, Any, Optional
from .api import PhilosophyExtractorAPI
from .types import PhilosophicalCategory, PhilosophySourceType


def get_available_templates() -> List[Dict[str, str]]:
    """Get list of all available extraction templates"""
    extractor = PhilosophyExtractorAPI()
    return extractor.get_templates()


def get_available_fields(category: str = None) -> List[Dict[str, Any]]:
    """Get list of all available extraction fields"""
    extractor = PhilosophyExtractorAPI()
    return extractor.get_fields(category)


def get_philosophy_categories() -> List[Dict[str, str]]:
    """Get available philosophy categories"""
    extractor = PhilosophyExtractorAPI()
    return extractor.get_philosophy_categories()


def get_extraction_modes() -> List[Dict[str, str]]:
    """Get available extraction modes"""
    extractor = PhilosophyExtractorAPI()
    return extractor.get_extraction_modes()


def analyze_text_for_templates(text: str) -> List[str]:
    """Analyze text and recommend appropriate templates"""
    extractor = PhilosophyExtractorAPI()
    word_count = len(text.split())
    return extractor.get_template_recommendations(
        text_length=word_count, use_case="analysis"
    )


def validate_category(category: str) -> bool:
    """Validate if a category name is valid"""
    try:
        PhilosophicalCategory(category)
        return True
    except ValueError:
        return False


def validate_source_type(source_type: str) -> bool:
    """Validate if a source type is valid"""
    try:
        PhilosophySourceType(source_type)
        return True
    except ValueError:
        return False


def get_category_info(category: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific category"""
    try:
        cat_enum = PhilosophicalCategory(category)
        return {
            "name": cat_enum.value,
            "description": cat_enum.name.lower().replace("_", " ").title(),
            "subcategories": cat_enum.subcategories,
        }
    except ValueError:
        return None


def recommend_extraction_config(
    text_length: int, target_audience: str = "academic", use_case: str = "analysis"
) -> Dict[str, Any]:
    """Recommend extraction configuration based on input parameters"""

    # Determine depth based on use case and audience
    depth_map = {
        ("academic", "research"): "expert",
        ("academic", "analysis"): "detailed",
        ("educational", "analysis"): "intermediate",
        ("general", "analysis"): "basic",
        ("professional", "analysis"): "detailed",
    }

    depth = depth_map.get((target_audience, use_case), "detailed")

    # Determine mode based on use case
    mode_map = {
        "research": "comprehensive",
        "analysis": "focused",
        "comparison": "comparative",
        "exploration": "exploratory",
        "education": "focused",
    }

    mode = mode_map.get(use_case, "comprehensive")

    # Determine source type from text length
    source_type = PhilosophySourceType.from_text_length(text_length)

    return {
        "extraction_depth": depth,
        "extraction_mode": mode,
        "target_audience": target_audience,
        "source_type": source_type.value,
        "recommended_template": _recommend_template_for_config(
            source_type, use_case, target_audience
        ),
    }


def _recommend_template_for_config(
    source_type: PhilosophySourceType, use_case: str, audience: str
) -> str:
    """Internal function to recommend template based on configuration"""

    # Template recommendations based on source type and use case
    recommendations = {
        (PhilosophySourceType.ESSAY, "research"): "comprehensive",
        (PhilosophySourceType.ESSAY, "analysis"): "philosophy_basic",
        (PhilosophySourceType.TREATISE, "research"): "comprehensive",
        (PhilosophySourceType.TREATISE, "analysis"): "philosophical_argument",
        (PhilosophySourceType.DIALOGUE, "analysis"): "philosophical_argument",
        (PhilosophySourceType.FRAGMENT, "analysis"): "philosophy_basic",
        (PhilosophySourceType.LECTURE, "education"): "philosophical_concept",
        (PhilosophySourceType.COMMENTARY, "analysis"): "critical_analysis",
    }

    return recommendations.get((source_type, use_case), "philosophy_basic")


# Export all convenience functions
__all__ = [
    "get_available_templates",
    "get_available_fields",
    "get_philosophy_categories",
    "get_extraction_modes",
    "analyze_text_for_templates",
    "validate_category",
    "validate_source_type",
    "get_category_info",
    "recommend_extraction_config",
]
