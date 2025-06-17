"""
Extractors Package for Philosophy Extraction System

This package provides comprehensive philosophical text extraction capabilities
including field definitions, templates, configuration, and advanced extraction methods.
"""

from typing import List, Dict, Any
import logging

# Initialize logging first
logger = logging.getLogger(__name__)

# Core types and enums
try:
    from .types import (
        DataType,
        PhilosophySourceType,
        ExtractionDepth,
        TargetAudience,
        ExtractionMode,
        PhilosophicalCategory,
        ValidationResult,
        ExtractionMetadata,
        PhilosophyExtractionConfig,
        ExtractionItem,
        PhilosophyExtractionResult,
    )

    TYPES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import types: {e}")
    TYPES_AVAILABLE = False

# Field system
try:
    from .fields import (
        PhilosophyExtractionField,
        PhilosophyFieldSet,
        PhilosophyFieldRegistry,
        philosophy_field_registry,
        ValidationRule,
        MinMaxRule,
        PatternRule,
        LengthRule,
        OptionsRule,
        RequiredFieldsRule,
        PhilosophyFieldProcessor,
    )

    FIELDS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import fields: {e}")
    FIELDS_AVAILABLE = False

# Enhanced field library (with encoding error handling)
try:
    from .field_library import (
        PhilosophyFieldLibrary,
        philosophy_field_library,
        FieldCategory,
        PhilosophyField,
        create_philosophy_field,
        field_registry,
        CORE_FIELDS,
        ANALYSIS_FIELDS,
        CONTEXT_FIELDS,
        METADATA_FIELDS,
    )

    FIELD_LIBRARY_AVAILABLE = True
except (ImportError, UnicodeDecodeError) as e:
    logger.error(
        f"Failed to import field_library (possibly due to encoding issue): {e}"
    )
    FIELD_LIBRARY_AVAILABLE = False
    # Create dummy objects to prevent import errors
    PhilosophyFieldLibrary = None
    philosophy_field_library = None
    FieldCategory = None
    PhilosophyField = None
    create_philosophy_field = None
    field_registry = None
    CORE_FIELDS = []
    ANALYSIS_FIELDS = []
    CONTEXT_FIELDS = []
    METADATA_FIELDS = []

# Configuration system
try:
    from .config import PhilosophyExtractorConfig as ConfigPhilosophyExtractorConfig

    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import config: {e}")
    CONFIG_AVAILABLE = False
    ConfigPhilosophyExtractorConfig = None

# Advanced extractor
try:
    from .advanced_extractor import (
        AdvancedPhilosophyExtractor,
        PhilosophyPromptBuilder,
        ExtractionContext,
        PhilosophyCategory,
        ExtractionMode as AdvancedExtractionMode,
        PhilosophyPromptConfig,
        extract_philosophy,
        extract_ethical_philosophy,
        discover_philosophical_themes,
        compare_philosophical_texts,
    )

    ADVANCED_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import advanced_extractor: {e}")
    ADVANCED_EXTRACTOR_AVAILABLE = False
    AdvancedPhilosophyExtractor = None
    PhilosophyPromptBuilder = None
    ExtractionContext = None
    PhilosophyCategory = None
    AdvancedExtractionMode = None
    PhilosophyPromptConfig = None
    extract_philosophy = None
    extract_ethical_philosophy = None
    discover_philosophical_themes = None
    compare_philosophical_texts = None

# Prompt generation
try:
    from .generator import (
        PhilosophyExtractionPromptGenerator,
        PhilosophyTemplateMatcher,
        AdvancedPhilosophyPromptBuilder,
        PromptConfig,
        IPromptGenerator,
        philosophy_prompt_generator,
        philosophy_prompt_builder,
    )

    GENERATOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import generator: {e}")
    GENERATOR_AVAILABLE = False
    PhilosophyExtractionPromptGenerator = None
    PhilosophyTemplateMatcher = None
    AdvancedPhilosophyPromptBuilder = None
    PromptConfig = None
    IPromptGenerator = None
    philosophy_prompt_generator = None
    philosophy_prompt_builder = None

# API interface (depends on other modules)
try:
    from .api import (
        PhilosophyExtractorAPI,
        ExtractionRequest,
        ExtractionResult,
        ExtractionStrategy,
    )

    API_AVAILABLE = True

    # Try to import strategy classes (these might fail if dependencies are missing)
    try:
        from .api import (
            LegacyExtractionStrategy,
            AdvancedExtractionStrategy,
            OllamaExtractionStrategy,
        )

        STRATEGIES_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        logger.warning(f"Some extraction strategies not available: {e}")
        STRATEGIES_AVAILABLE = False
        LegacyExtractionStrategy = None
        AdvancedExtractionStrategy = None
        OllamaExtractionStrategy = None

except ImportError as e:
    logger.error(f"Failed to import api: {e}")
    API_AVAILABLE = False
    PhilosophyExtractorAPI = None
    ExtractionRequest = None
    ExtractionResult = None
    ExtractionStrategy = None
    LegacyExtractionStrategy = None
    AdvancedExtractionStrategy = None
    OllamaExtractionStrategy = None

# System integration
try:
    from .system import (
        PhilosophySystemIntegration,
        validate_field_compatibility,
        validate_extraction_data,
        map_fields_to_library,
        map_categories_to_fields,
        create_extraction_result,
        create_extraction_config,
        recommend_field_set,
        get_fields_for_category,
        analyze_field_coverage,
    )

    SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import system: {e}")
    SYSTEM_AVAILABLE = False
    PhilosophySystemIntegration = None
    validate_field_compatibility = None
    validate_extraction_data = None
    map_fields_to_library = None
    map_categories_to_fields = None
    create_extraction_result = None
    create_extraction_config = None
    recommend_field_set = None
    get_fields_for_category = None
    analyze_field_coverage = None

# Elements and structures
try:
    from .elements import (
        BaseElement,
        ArgumentElement,
        ConceptElement,
        TraditionElement,
        EthicalElement,
        PhilosophicalAnalysis,
        ArgumentType,
        ArgumentStrength,
        ValidityStatus,
        ElementFactory,
    )

    ELEMENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import elements: {e}")
    ELEMENTS_AVAILABLE = False
    BaseElement = None
    ArgumentElement = None
    ConceptElement = None
    TraditionElement = None
    EthicalElement = None
    PhilosophicalAnalysis = None
    ArgumentType = None
    ArgumentStrength = None
    ValidityStatus = None
    ElementFactory = None

# Validation rules
try:
    from .rules import (
        ValidationRule as RuleValidationRule,
        ExtractionResult as RuleExtractionResult,
    )

    RULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import rules: {e}")
    RULES_AVAILABLE = False
    RuleValidationRule = None
    RuleExtractionResult = None

# Ollama integration (if available)
try:
    from .ollama_extractor import (
        PhilosophyOllamaExtractor,
        extract_philosophical_text,
    )

    OLLAMA_AVAILABLE = True
except ImportError as e:
    logger.info(f"Ollama extractor not available: {e}")
    PhilosophyOllamaExtractor = None
    extract_philosophical_text = None
    OLLAMA_AVAILABLE = False

# Version info
__version__ = "2.0.0"
__author__ = "Philosophy Extraction System"

# Build dynamic exports based on what's available
__all__ = []

# Core types and configuration
if TYPES_AVAILABLE:
    __all__.extend(
        [
            "DataType",
            "PhilosophySourceType",
            "ExtractionDepth",
            "TargetAudience",
            "ExtractionMode",
            "PhilosophicalCategory",
            "ValidationResult",
            "ExtractionMetadata",
            "PhilosophyExtractionConfig",
            "ExtractionItem",
            "PhilosophyExtractionResult",
        ]
    )

# Field system
if FIELDS_AVAILABLE:
    __all__.extend(
        [
            "PhilosophyExtractionField",
            "PhilosophyFieldSet",
            "PhilosophyFieldRegistry",
            "philosophy_field_registry",
            "ValidationRule",
            "MinMaxRule",
            "PatternRule",
            "LengthRule",
            "OptionsRule",
            "RequiredFieldsRule",
            "PhilosophyFieldProcessor",
        ]
    )

# Enhanced field library
if FIELD_LIBRARY_AVAILABLE:
    __all__.extend(
        [
            "PhilosophyFieldLibrary",
            "philosophy_field_library",
            "FieldCategory",
            "PhilosophyField",
            "create_philosophy_field",
            "field_registry",
            "CORE_FIELDS",
            "ANALYSIS_FIELDS",
            "CONTEXT_FIELDS",
            "METADATA_FIELDS",
        ]
    )

# Configuration
if CONFIG_AVAILABLE:
    __all__.append("ConfigPhilosophyExtractorConfig")

# API interface
if API_AVAILABLE:
    __all__.extend(
        [
            "PhilosophyExtractorAPI",
            "ExtractionRequest",
            "ExtractionResult",
            "ExtractionStrategy",
        ]
    )

    if STRATEGIES_AVAILABLE:
        __all__.extend(
            [
                "LegacyExtractionStrategy",
                "AdvancedExtractionStrategy",
                "OllamaExtractionStrategy",
            ]
        )

# Advanced extractor
if ADVANCED_EXTRACTOR_AVAILABLE:
    __all__.extend(
        [
            "AdvancedPhilosophyExtractor",
            "PhilosophyPromptBuilder",
            "ExtractionContext",
            "PhilosophyCategory",
            "AdvancedExtractionMode",
            "PhilosophyPromptConfig",
            "extract_philosophy",
            "extract_ethical_philosophy",
            "discover_philosophical_themes",
            "compare_philosophical_texts",
        ]
    )

# Prompt generation
if GENERATOR_AVAILABLE:
    __all__.extend(
        [
            "PhilosophyExtractionPromptGenerator",
            "PhilosophyTemplateMatcher",
            "AdvancedPhilosophyPromptBuilder",
            "PromptConfig",
            "IPromptGenerator",
            "philosophy_prompt_generator",
            "philosophy_prompt_builder",
        ]
    )

# System integration
if SYSTEM_AVAILABLE:
    __all__.extend(
        [
            "PhilosophySystemIntegration",
            "validate_field_compatibility",
            "validate_extraction_data",
            "map_fields_to_library",
            "map_categories_to_fields",
            "create_extraction_result",
            "create_extraction_config",
            "recommend_field_set",
            "get_fields_for_category",
            "analyze_field_coverage",
        ]
    )

# Elements and structures
if ELEMENTS_AVAILABLE:
    __all__.extend(
        [
            "BaseElement",
            "ArgumentElement",
            "ConceptElement",
            "TraditionElement",
            "EthicalElement",
            "PhilosophicalAnalysis",
            "ArgumentType",
            "ArgumentStrength",
            "ValidityStatus",
            "ElementFactory",
        ]
    )

# Validation rules
if RULES_AVAILABLE:
    __all__.extend(
        [
            "RuleValidationRule",
            "RuleExtractionResult",
        ]
    )

# Ollama integration (conditional)
__all__.extend(
    [
        "PhilosophyOllamaExtractor",
        "extract_philosophical_text",
        "OLLAMA_AVAILABLE",
    ]
)

# Add availability flags
__all__.extend(
    [
        "TYPES_AVAILABLE",
        "FIELDS_AVAILABLE",
        "FIELD_LIBRARY_AVAILABLE",
        "CONFIG_AVAILABLE",
        "API_AVAILABLE",
        "STRATEGIES_AVAILABLE",
        "ADVANCED_EXTRACTOR_AVAILABLE",
        "GENERATOR_AVAILABLE",
        "SYSTEM_AVAILABLE",
        "ELEMENTS_AVAILABLE",
        "RULES_AVAILABLE",
    ]
)


# Replace the problematic convenience functions with these fixed versions:


def create_extractor(
    use_ollama: bool = False, ollama_model: str = "deepseek-r1:7b", **kwargs
):
    """Create a philosophy extractor with the specified configuration"""
    if not API_AVAILABLE:
        raise ImportError("PhilosophyExtractorAPI not available due to import errors")

    return PhilosophyExtractorAPI(
        use_ollama=use_ollama, ollama_model=ollama_model, **kwargs
    )


def extract_text(
    text: str,
    template: str = "philosophy_basic",
    language: str = "mixed",
    mode: str = "comprehensive",
    **kwargs,
):
    """Quick text extraction with default settings"""
    extractor = create_extractor(**kwargs)
    return extractor.extract(
        text=text, template_name=template, language=language, extraction_mode=mode
    )


def get_available_templates() -> List[Dict[str, str]]:
    """Get list of all available extraction templates"""
    if not API_AVAILABLE:
        logger.warning("API not available, returning empty template list")
        return []

    extractor = PhilosophyExtractorAPI()
    return extractor.get_templates()


def get_available_fields(category: str = None) -> List[Dict[str, Any]]:
    """Get list of all available extraction fields"""
    if not API_AVAILABLE:
        logger.warning("API not available, returning empty field list")
        return []

    extractor = PhilosophyExtractorAPI()
    return extractor.get_fields(category)


def validate_configuration(config: Dict[str, Any]):
    """Validate an extraction configuration"""
    if not TYPES_AVAILABLE:
        # Create a simple validation result
        return {"is_valid": False, "errors": ["ValidationResult not available"]}

    try:
        if hasattr(PhilosophyExtractionConfig, "from_dict"):
            PhilosophyExtractionConfig.from_dict(config)
        return ValidationResult(is_valid=True)
    except Exception as e:
        result = ValidationResult(is_valid=False)
        if hasattr(result, "add_error"):
            result.add_error(str(e))
        return result


def create_field(
    name: str,
    description: str,
    category: str = "core",
    required: bool = False,
    **kwargs,
):
    """Create a custom extraction field"""
    if not FIELD_LIBRARY_AVAILABLE:
        raise ImportError("Field library not available due to import errors")

    return create_philosophy_field(
        name=name,
        description=description,
        category=FieldCategory(category),
        required=required,
        **kwargs,
    )


def analyze_text_for_templates(text: str) -> List[str]:
    """Analyze text and recommend appropriate templates"""
    if not API_AVAILABLE:
        logger.warning("API not available, returning basic template recommendation")
        return ["philosophy_basic"]

    try:
        extractor = PhilosophyExtractorAPI()
        word_count = len(text.split())
        return extractor.get_template_recommendations(
            text_length=word_count, use_case="analysis"
        )
    except Exception as e:
        logger.error(f"Error analyzing text for templates: {e}")
        return ["philosophy_basic"]


def get_package_status() -> Dict[str, bool]:
    """Get the availability status of all package components"""
    return {
        "types": TYPES_AVAILABLE,
        "fields": FIELDS_AVAILABLE,
        "field_library": FIELD_LIBRARY_AVAILABLE,
        "config": CONFIG_AVAILABLE,
        "api": API_AVAILABLE,
        "strategies": STRATEGIES_AVAILABLE,
        "advanced_extractor": ADVANCED_EXTRACTOR_AVAILABLE,
        "generator": GENERATOR_AVAILABLE,
        "system": SYSTEM_AVAILABLE,
        "elements": ELEMENTS_AVAILABLE,
        "rules": RULES_AVAILABLE,
        "ollama": OLLAMA_AVAILABLE,
    }


# Add convenience functions to exports
__all__.extend(
    [
        "create_extractor",
        "extract_text",
        "get_available_templates",
        "get_available_fields",
        "validate_configuration",
        "create_field",
        "analyze_text_for_templates",
        "get_package_status",
    ]
)

# Package-level documentation
__doc__ = """
Philosophy Extractors Package

This package provides a comprehensive system for extracting philosophical content
from texts using various methods and configurations.

Package Status:
- Types System: Available if TYPES_AVAILABLE is True
- Field System: Available if FIELDS_AVAILABLE is True
- Field Library: Available if FIELD_LIBRARY_AVAILABLE is True
- Configuration: Available if CONFIG_AVAILABLE is True
- API Interface: Available if API_AVAILABLE is True
- Advanced Extractor: Available if ADVANCED_EXTRACTOR_AVAILABLE is True
- Prompt Generation: Available if GENERATOR_AVAILABLE is True
- System Integration: Available if SYSTEM_AVAILABLE is True
- Ollama Integration: Available if OLLAMA_AVAILABLE is True

Use get_package_status() to check component availability.

Basic Usage (if components are available):

    # Check what's available
    from extractors import get_package_status
    status = get_package_status()
    print(status)

    # Quick extraction (if API available)
    from extractors import extract_text
    result = extract_text("Your philosophical text here")

    # Advanced extraction (if API available)
    from extractors import create_extractor
    extractor = create_extractor(use_ollama=True)
    result = extractor.extract(
        text="Text to analyze",
        template_name="comprehensive",
        extraction_mode="focused",
        categories=["ethics", "epistemology"]
    )
"""

# Initialize package-level logging
logger.info(f"Philosophy Extractors package loaded (version {__version__})")

# Log component availability
status = get_package_status()
available_components = [k for k, v in status.items() if v]
unavailable_components = [k for k, v in status.items() if not v]

if available_components:
    logger.info(f"Available components: {', '.join(available_components)}")
if unavailable_components:
    logger.warning(f"Unavailable components: {', '.join(unavailable_components)}")

# Validate package integrity on import (only for available components)
try:
    if API_AVAILABLE:
        _ = PhilosophyExtractorAPI()
        logger.info("API component validation passed")

    if FIELDS_AVAILABLE:
        _ = philosophy_field_registry.get_required_fields()
        logger.info("Fields component validation passed")

    logger.info("Available package components validated successfully")
except Exception as e:
    logger.warning(f"Package component validation failed: {e}")

# Provide helpful guidance
if not API_AVAILABLE:
    logger.error("Core API not available - check for import errors in dependencies")
if not FIELD_LIBRARY_AVAILABLE:
    logger.error(
        "Field library not available - possibly due to Unicode encoding issues in field_library.py"
    )
if not OLLAMA_AVAILABLE:
    logger.info(
        "Ollama integration not available - install ollama dependencies for LLM support"
    )
