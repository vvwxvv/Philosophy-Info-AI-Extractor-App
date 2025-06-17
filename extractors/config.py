"""Enhanced configuration for philosophical extraction"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging

# Define logger first
logger = logging.getLogger(__name__)

# Import types
from .types import (
    PhilosophySourceType,
    ExtractionDepth,
    TargetAudience,
    ExtractionMode,
    PhilosophicalCategory,
    ValidationResult,
)

# Import template and field systems
try:
    from prompts.templates import (
        PhilosophyExtractionTemplate,
        philosophy_template_library,
    )

    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False
    logger.warning("Template library not available")
    # Create a fallback type for type hints
    PhilosophyExtractionTemplate = type("PhilosophyExtractionTemplate", (), {})
    philosophy_template_library = None

try:
    from .fields import philosophy_field_registry

    FIELDS_AVAILABLE = True
except ImportError:
    FIELDS_AVAILABLE = False
    logger.warning("Field registry not available")


@dataclass
class PhilosophyExtractorConfig:
    """Enhanced configuration for philosophy extractors"""

    # Core configuration
    template: Union[str, PhilosophyExtractionTemplate, None] = None
    source_type: PhilosophySourceType = PhilosophySourceType.ESSAY
    language: str = "mixed"

    # Extraction parameters
    extraction_depth: ExtractionDepth = ExtractionDepth.DETAILED
    extraction_mode: ExtractionMode = ExtractionMode.COMPREHENSIVE
    target_audience: TargetAudience = TargetAudience.ACADEMIC
    categories_focus: List[PhilosophicalCategory] = field(default_factory=list)

    # Field configuration
    additional_fields: List[str] = field(default_factory=list)
    exclude_fields: List[str] = field(default_factory=list)
    required_fields_override: List[str] = field(default_factory=list)

    # Feature flags
    include_examples: bool = True
    include_references: bool = True
    include_historical_context: bool = True
    include_influences: bool = True
    include_criticisms: bool = True
    include_applications: bool = True
    extract_implicit_content: bool = True
    preserve_original_language: bool = True

    # Processing options
    confidence_threshold: float = 0.7
    max_extraction_time: int = 300
    enable_post_processing: bool = True
    enable_validation: bool = True
    enable_knowledge_enhancement: bool = True

    # Output configuration
    output_format: str = "json"
    include_metadata: bool = True
    include_confidence_scores: bool = True
    include_field_coverage: bool = True

    # Custom parameters
    custom_guidelines: List[str] = field(default_factory=list)
    custom_output_format: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Advanced options
    parallel_processing: bool = False
    batch_size: int = 1
    retry_attempts: int = 2
    fallback_template: Optional[str] = None

    def __post_init__(self):
        """Validate and process configuration"""
        self._validate()
        self._process_template()
        self._set_defaults()

    def _validate(self):
        """Validate configuration parameters"""
        # Validate confidence threshold
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        # Validate max extraction time
        if self.max_extraction_time < 10:
            raise ValueError("Max extraction time must be at least 10 seconds")

        # Validate language
        valid_languages = ["CN", "EN", "mixed", "auto"]
        if self.language not in valid_languages:
            raise ValueError(f"Language must be one of: {valid_languages}")

        # Validate output format
        valid_formats = ["json", "dict", "structured"]
        if self.output_format not in valid_formats:
            raise ValueError(f"Output format must be one of: {valid_formats}")

        # Validate batch size
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")

        # Validate retry attempts
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts must be non-negative")

    def _process_template(self):
        """Process and validate template reference"""
        if not self.template:
            # Use default template based on extraction mode
            self.template = self._get_default_template()
            return

        if isinstance(self.template, str):
            if TEMPLATES_AVAILABLE:
                template_obj = philosophy_template_library.get_template(self.template)
                if not template_obj:
                    # Try fallback template
                    if self.fallback_template:
                        template_obj = philosophy_template_library.get_template(
                            self.fallback_template
                        )
                        if template_obj:
                            logger.warning(
                                f"Using fallback template '{self.fallback_template}' instead of '{self.template}'"
                            )
                            self.template = template_obj
                            return

                    raise ValueError(f"Template '{self.template}' not found")
                self.template = template_obj
            else:
                # Keep as string if templates not available
                logger.warning(
                    f"Template library not available, keeping template as string: {self.template}"
                )

    def _get_default_template(self) -> str:
        """Get default template based on configuration"""
        if self.extraction_mode == ExtractionMode.FOCUSED and self.categories_focus:
            category = self.categories_focus[0]
            category_templates = {
                PhilosophicalCategory.ETHICS: "philosophy_ethical",
                PhilosophicalCategory.METAPHYSICS: "philosophy_metaphysical",
                PhilosophicalCategory.EPISTEMOLOGY: "philosophy_epistemological",
                PhilosophicalCategory.LOGIC: "philosophical_argument",
            }
            return category_templates.get(category, "philosophy_basic")

        # Default templates by extraction depth
        depth_templates = {
            ExtractionDepth.BASIC: "philosophy_basic",
            ExtractionDepth.INTERMEDIATE: "philosophical_concept",
            ExtractionDepth.DETAILED: "comprehensive",
            ExtractionDepth.EXPERT: "comprehensive",
        }

        return depth_templates.get(self.extraction_depth, "philosophy_basic")

    def _set_defaults(self):
        """Set configuration defaults based on other parameters"""
        # Adjust features based on extraction depth
        if self.extraction_depth == ExtractionDepth.BASIC:
            self.include_criticisms = False
            self.include_influences = False
            self.extract_implicit_content = False

        # Adjust features based on target audience
        if self.target_audience == TargetAudience.GENERAL:
            self.include_examples = True
            self.preserve_original_language = False

        # Adjust processing based on source type
        if self.source_type in [
            PhilosophySourceType.FRAGMENT,
            PhilosophySourceType.APHORISM,
        ]:
            self.max_extraction_time = min(self.max_extraction_time, 120)
            self.include_historical_context = False

        # Set metadata defaults
        if "created_at" not in self.metadata:
            from datetime import datetime

            self.metadata["created_at"] = datetime.utcnow().isoformat()

        self.metadata.update(
            {
                "config_version": "2.0",
                "auto_configured": True,
            }
        )

    def get_fields(self) -> Dict[str, Any]:
        """Get all fields for this configuration"""
        fields = {}

        # Get template fields if available
        if TEMPLATES_AVAILABLE and isinstance(
            self.template, PhilosophyExtractionTemplate
        ):
            try:
                fields.update(self.template.get_fields())
            except AttributeError:
                # Template might not have get_fields method
                if hasattr(self.template, "fields"):
                    # Convert field names to field objects
                    if FIELDS_AVAILABLE:
                        for field_name in self.template.fields:
                            field = philosophy_field_registry.get_field(field_name)
                            if field:
                                fields[field_name] = field

        # Add additional fields
        if FIELDS_AVAILABLE:
            for field_name in self.additional_fields:
                if field_name not in fields:
                    field = philosophy_field_registry.get_field(field_name)
                    if field:
                        fields[field_name] = field

        # Remove excluded fields
        for field_name in self.exclude_fields:
            fields.pop(field_name, None)

        # Override required status
        for field_name in self.required_fields_override:
            if field_name in fields and hasattr(fields[field_name], "required"):
                fields[field_name].required = True

        return fields

    def get_extraction_guidelines(self) -> List[str]:
        """Get all extraction guidelines"""
        guidelines = []

        # Add template guidelines if available
        if TEMPLATES_AVAILABLE and isinstance(
            self.template, PhilosophyExtractionTemplate
        ):
            try:
                guidelines.extend(self.template.get_guidelines())
            except AttributeError:
                # Template might not have get_guidelines method
                pass

        # Add mode-specific guidelines
        mode_guidelines = {
            ExtractionMode.COMPREHENSIVE: "Extract all philosophical aspects comprehensively",
            ExtractionMode.FOCUSED: "Focus on specific categories with deep analysis",
            ExtractionMode.EXPLORATORY: "Discover and explore philosophical themes",
            ExtractionMode.COMPARATIVE: "Compare and contrast philosophical positions",
            ExtractionMode.THEMATIC: "Organize analysis around central themes",
            ExtractionMode.HISTORICAL: "Emphasize historical context and development",
            ExtractionMode.CRITICAL: "Provide critical analysis and evaluation",
        }

        if self.extraction_mode in mode_guidelines:
            guidelines.append(mode_guidelines[self.extraction_mode])

        # Add audience-specific guidelines
        audience_guidelines = {
            TargetAudience.ACADEMIC: "Use scholarly terminology and cite philosophical traditions",
            TargetAudience.GENERAL: "Explain concepts clearly without excessive jargon",
            TargetAudience.PROFESSIONAL: "Focus on practical applications and decision-making",
            TargetAudience.EDUCATIONAL: "Structure for learning with clear examples",
        }

        if self.target_audience in audience_guidelines:
            guidelines.append(audience_guidelines[self.target_audience])

        # Add category focus guidelines
        if self.categories_focus:
            categories_str = ", ".join(cat.value for cat in self.categories_focus)
            guidelines.append(f"Focus primarily on: {categories_str}")

        # Add feature-based guidelines
        feature_guidelines = []

        if self.extract_implicit_content:
            feature_guidelines.append(
                "Extract both explicit and implicit philosophical content"
            )

        if self.preserve_original_language:
            feature_guidelines.append(
                "Preserve original language terms where philosophically significant"
            )

        if self.include_examples:
            feature_guidelines.append("Include relevant examples and illustrations")

        if self.include_historical_context:
            feature_guidelines.append(
                "Provide historical context and philosophical development"
            )

        if self.include_criticisms:
            feature_guidelines.append("Include critical perspectives and objections")

        guidelines.extend(feature_guidelines)

        # Add confidence and quality guidelines
        guidelines.append(
            f"Maintain confidence threshold of {self.confidence_threshold}"
        )

        if self.include_confidence_scores:
            guidelines.append("Provide confidence scores for all extracted information")

        # Add custom guidelines
        guidelines.extend(self.custom_guidelines)

        return guidelines

    def get_output_format(self) -> Dict[str, Any]:
        """Get output format specification"""
        if self.custom_output_format:
            return self.custom_output_format

        # Build output format from fields
        fields = self.get_fields()
        output_format = {}

        for field_name, field in fields.items():
            if hasattr(field, "data_type"):
                if field.data_type.value == "array":
                    output_format[field_name] = [f"{field.description}"]
                else:
                    output_format[field_name] = field.description
            else:
                # Fallback if field doesn't have data_type
                output_format[field_name] = f"Information about {field_name}"

        # Add metadata fields if enabled
        if self.include_metadata:
            output_format["metadata"] = {
                "extraction_id": "unique_identifier",
                "processing_time": "seconds",
                "template_used": "template_name",
                "configuration": "config_summary",
            }

        # Add confidence scores if enabled
        if self.include_confidence_scores:
            output_format["confidence_scores"] = {
                "field_name": "confidence_value_0_to_1"
            }

        # Add field coverage if enabled
        if self.include_field_coverage:
            output_format["field_coverage"] = {
                "total_fields": "number",
                "extracted_fields": "number",
                "coverage_percentage": "percentage",
            }

        return output_format

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {
            "template": (
                self.template.name
                if isinstance(self.template, PhilosophyExtractionTemplate)
                else str(self.template) if self.template else None
            ),
            "source_type": self.source_type.value,
            "language": self.language,
            "extraction_depth": self.extraction_depth.value,
            "extraction_mode": self.extraction_mode.value,
            "target_audience": self.target_audience.value,
            "categories_focus": [cat.value for cat in self.categories_focus],
            "fields": {
                "additional": self.additional_fields,
                "exclude": self.exclude_fields,
                "required_override": self.required_fields_override,
                "total_available": len(self.get_fields()) if FIELDS_AVAILABLE else 0,
            },
            "features": {
                "include_examples": self.include_examples,
                "include_references": self.include_references,
                "include_historical_context": self.include_historical_context,
                "include_influences": self.include_influences,
                "include_criticisms": self.include_criticisms,
                "include_applications": self.include_applications,
                "extract_implicit_content": self.extract_implicit_content,
                "preserve_original_language": self.preserve_original_language,
            },
            "processing": {
                "confidence_threshold": self.confidence_threshold,
                "max_extraction_time": self.max_extraction_time,
                "enable_post_processing": self.enable_post_processing,
                "enable_validation": self.enable_validation,
                "enable_knowledge_enhancement": self.enable_knowledge_enhancement,
                "parallel_processing": self.parallel_processing,
                "batch_size": self.batch_size,
                "retry_attempts": self.retry_attempts,
            },
            "output": {
                "format": self.output_format,
                "include_metadata": self.include_metadata,
                "include_confidence_scores": self.include_confidence_scores,
                "include_field_coverage": self.include_field_coverage,
            },
            "custom": {
                "guidelines": self.custom_guidelines,
                "output_format": self.custom_output_format,
                "fallback_template": self.fallback_template,
            },
            "metadata": self.metadata,
            "system_info": {
                "templates_available": TEMPLATES_AVAILABLE,
                "fields_available": FIELDS_AVAILABLE,
            },
        }

        return config_dict

    def validate_compatibility(self) -> ValidationResult:
        """Validate configuration compatibility"""
        result = ValidationResult(is_valid=True)

        # Check template compatibility
        if not TEMPLATES_AVAILABLE and isinstance(self.template, str):
            result.add_warning("Template library not available, using basic processing")

        # Check field registry compatibility
        if not FIELDS_AVAILABLE and (self.additional_fields or self.exclude_fields):
            result.add_warning(
                "Field registry not available, field configuration may be ignored"
            )

        # Check feature compatibility
        if self.parallel_processing and self.batch_size == 1:
            result.add_warning("Parallel processing enabled but batch size is 1")

        # Check language-feature compatibility
        if self.language == "CN" and not self.preserve_original_language:
            result.add_warning(
                "Chinese language detected but original language preservation disabled"
            )

        # Check audience-depth compatibility
        if (
            self.target_audience == TargetAudience.GENERAL
            and self.extraction_depth == ExtractionDepth.EXPERT
        ):
            result.add_warning("Expert depth may not be suitable for general audience")

        return result

    @classmethod
    def create_for_use_case(
        cls, use_case: str, **kwargs
    ) -> "PhilosophyExtractorConfig":
        """Create configuration for specific use case"""
        use_case_configs = {
            "quick_analysis": {
                "template": "philosophy_basic",
                "extraction_depth": ExtractionDepth.BASIC,
                "extraction_mode": ExtractionMode.FOCUSED,
                "max_extraction_time": 60,
                "include_examples": False,
                "include_criticisms": False,
                "include_historical_context": False,
                "target_audience": TargetAudience.GENERAL,
            },
            "academic_research": {
                "template": "comprehensive",
                "extraction_depth": ExtractionDepth.EXPERT,
                "extraction_mode": ExtractionMode.COMPREHENSIVE,
                "target_audience": TargetAudience.ACADEMIC,
                "include_references": True,
                "include_historical_context": True,
                "include_influences": True,
                "include_criticisms": True,
                "extract_implicit_content": True,
                "confidence_threshold": 0.8,
                "max_extraction_time": 600,
            },
            "concept_analysis": {
                "template": "philosophical_concept",
                "extraction_depth": ExtractionDepth.DETAILED,
                "extraction_mode": ExtractionMode.FOCUSED,
                "categories_focus": [
                    PhilosophicalCategory.METAPHYSICS,
                    PhilosophicalCategory.EPISTEMOLOGY,
                ],
                "include_examples": True,
                "extract_implicit_content": True,
            },
            "ethical_analysis": {
                "template": "philosophy_ethical",
                "extraction_depth": ExtractionDepth.DETAILED,
                "extraction_mode": ExtractionMode.FOCUSED,
                "categories_focus": [PhilosophicalCategory.ETHICS],
                "include_applications": True,
                "include_examples": True,
            },
            "historical_study": {
                "template": "historical_philosophy",
                "extraction_depth": ExtractionDepth.DETAILED,
                "extraction_mode": ExtractionMode.HISTORICAL,
                "include_historical_context": True,
                "include_influences": True,
                "include_references": True,
                "preserve_original_language": True,
            },
            "comparative_analysis": {
                "template": "comparative_philosophy",
                "extraction_depth": ExtractionDepth.DETAILED,
                "extraction_mode": ExtractionMode.COMPARATIVE,
                "include_criticisms": True,
                "include_examples": True,
                "parallel_processing": True,
                "batch_size": 2,
            },
            "educational": {
                "template": "philosophy_basic",
                "extraction_depth": ExtractionDepth.INTERMEDIATE,
                "extraction_mode": ExtractionMode.FOCUSED,
                "target_audience": TargetAudience.EDUCATIONAL,
                "include_examples": True,
                "include_applications": True,
                "include_criticisms": False,
                "preserve_original_language": False,
            },
        }

        if use_case not in use_case_configs:
            available_cases = list(use_case_configs.keys())
            raise ValueError(
                f"Unknown use case: {use_case}. Available: {available_cases}"
            )

        # Get base configuration
        config_params = use_case_configs[use_case].copy()

        # Override with kwargs
        config_params.update(kwargs)

        # Set source type if not provided
        if "source_type" not in config_params:
            config_params["source_type"] = PhilosophySourceType.ESSAY

        # Add use case to metadata
        if "metadata" not in config_params:
            config_params["metadata"] = {}
        config_params["metadata"]["use_case"] = use_case

        return cls(**config_params)

    def clone(self, **overrides) -> "PhilosophyExtractorConfig":
        """Create a copy of this configuration with optional overrides"""
        # Convert to dict
        config_dict = self.to_dict()

        # Remove system info and non-constructor fields
        config_dict.pop("system_info", None)

        # Flatten nested structures for constructor
        flat_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, dict) and key in ["features", "processing", "output"]:
                flat_dict.update(value)
            elif key == "fields":
                if isinstance(value, dict):
                    flat_dict.update(
                        {
                            "additional_fields": value.get("additional", []),
                            "exclude_fields": value.get("exclude", []),
                            "required_fields_override": value.get(
                                "required_override", []
                            ),
                        }
                    )
            elif key not in ["custom", "system_info"]:
                flat_dict[key] = value

        # Handle custom fields
        if "custom" in config_dict:
            custom = config_dict["custom"]
            flat_dict.update(
                {
                    "custom_guidelines": custom.get("guidelines", []),
                    "custom_output_format": custom.get("output_format"),
                    "fallback_template": custom.get("fallback_template"),
                }
            )

        # Apply overrides
        flat_dict.update(overrides)

        # Reconstruct enum values
        if "source_type" in flat_dict and isinstance(flat_dict["source_type"], str):
            flat_dict["source_type"] = PhilosophySourceType(flat_dict["source_type"])
        if "extraction_depth" in flat_dict and isinstance(
            flat_dict["extraction_depth"], str
        ):
            flat_dict["extraction_depth"] = ExtractionDepth(
                flat_dict["extraction_depth"]
            )
        if "extraction_mode" in flat_dict and isinstance(
            flat_dict["extraction_mode"], str
        ):
            flat_dict["extraction_mode"] = ExtractionMode(flat_dict["extraction_mode"])
        if "target_audience" in flat_dict and isinstance(
            flat_dict["target_audience"], str
        ):
            flat_dict["target_audience"] = TargetAudience(flat_dict["target_audience"])
        if "categories_focus" in flat_dict:
            flat_dict["categories_focus"] = [
                PhilosophicalCategory(cat) if isinstance(cat, str) else cat
                for cat in flat_dict["categories_focus"]
            ]

        return self.__class__(**flat_dict)

    def optimize_for_performance(self) -> "PhilosophyExtractorConfig":
        """Create an optimized version for better performance"""
        return self.clone(
            extraction_depth=ExtractionDepth.INTERMEDIATE,
            include_criticisms=False,
            include_influences=False,
            extract_implicit_content=False,
            max_extraction_time=120,
            enable_post_processing=False,
            parallel_processing=True,
            batch_size=3,
        )

    def optimize_for_quality(self) -> "PhilosophyExtractorConfig":
        """Create an optimized version for better quality"""
        return self.clone(
            extraction_depth=ExtractionDepth.EXPERT,
            confidence_threshold=0.8,
            include_criticisms=True,
            include_influences=True,
            include_historical_context=True,
            extract_implicit_content=True,
            enable_validation=True,
            enable_post_processing=True,
            max_extraction_time=600,
            retry_attempts=3,
        )


# Convenience functions
def create_quick_config(**kwargs) -> PhilosophyExtractorConfig:
    """Create a quick analysis configuration"""
    return PhilosophyExtractorConfig.create_for_use_case("quick_analysis", **kwargs)


def create_academic_config(**kwargs) -> PhilosophyExtractorConfig:
    """Create an academic research configuration"""
    return PhilosophyExtractorConfig.create_for_use_case("academic_research", **kwargs)


def create_educational_config(**kwargs) -> PhilosophyExtractorConfig:
    """Create an educational configuration"""
    return PhilosophyExtractorConfig.create_for_use_case("educational", **kwargs)
