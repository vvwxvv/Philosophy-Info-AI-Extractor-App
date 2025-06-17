"""Integration module to ensure all components work together"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from extractors.types import (
    PhilosophySourceType,
    PhilosophyExtractionConfig,
    PhilosophyExtractionResult,
    ExtractionDepth,
    TargetAudience,
    ExtractionMode,
    PhilosophicalCategory,
    ExtractionItem,
    ExtractionMetadata,
    ValidationResult,
)
from extractors.fields import (
    PhilosophyFieldSet,
    philosophy_field_registry,
)
from extractors.field_library import (
    PhilosophyFieldLibrary,
    FieldCategory,
    PhilosophyField,
)


logger = logging.getLogger(__name__)


class PhilosophySystemIntegration:
    """Ensures all philosophy extraction components work together"""

    @staticmethod
    def validate_field_compatibility(
        field_set: str,
        source_type: PhilosophySourceType,
        extraction_depth: ExtractionDepth,
    ) -> Tuple[bool, ValidationResult]:
        """Check if a field set is compatible with source type and depth"""
        validation_result = ValidationResult(is_valid=True)

        # Get field set definition
        field_set_enum = PhilosophyFieldSet(field_set)
        if not field_set_enum:
            validation_result.add_error(f"Field set '{field_set}' not found")
            return False, validation_result

        # Check complexity compatibility
        # For now, use a simple complexity check based on field count
        field_count = len(philosophy_field_registry.get_field_set(field_set_enum))
        complexity_score = min(5, max(1, field_count // 5))

        if complexity_score > extraction_depth.complexity_score:
            validation_result.add_warning(
                f"Field set '{field_set}' complexity ({complexity_score}) "
                f"exceeds extraction depth ({extraction_depth.complexity_score})"
            )
            validation_result.add_suggestion(
                f"Consider using {extraction_depth.name} depth or simpler field set"
            )

        # Check source type compatibility
        source_compatibility = {
            PhilosophySourceType.ESSAY: [
                PhilosophyFieldSet.BASIC,
                PhilosophyFieldSet.CONCEPT,
                PhilosophyFieldSet.ARGUMENT,
            ],
            PhilosophySourceType.TREATISE: [
                PhilosophyFieldSet.COMPREHENSIVE,
                PhilosophyFieldSet.CRITICAL,
            ],
            PhilosophySourceType.DIALOGUE: [
                PhilosophyFieldSet.ARGUMENT,
                PhilosophyFieldSet.BASIC,
            ],
            PhilosophySourceType.FRAGMENT: [PhilosophyFieldSet.BASIC],
        }

        compatible_sets = source_compatibility.get(
            source_type, list(PhilosophyFieldSet)
        )
        if field_set_enum not in compatible_sets:
            validation_result.add_warning(
                f"Field set '{field_set}' may not be optimal for source type '{source_type.value}'"
            )
            validation_result.add_suggestion(
                f"Consider using: {', '.join(fs.value for fs in compatible_sets)}"
            )

        return validation_result.is_valid, validation_result

    @staticmethod
    def validate_extraction_data(
        data: Dict[str, Any], field_set: str
    ) -> ValidationResult:
        """Validate extracted data against field definitions"""
        validation_result = ValidationResult(is_valid=True)

        # Get field set definition
        field_set_enum = PhilosophyFieldSet(field_set)
        if not field_set_enum:
            validation_result.add_error(f"Field set '{field_set}' not found")
            return validation_result

        # Get fields for this field set
        field_names = philosophy_field_registry.get_field_set(field_set_enum)

        # Check required fields
        for field_name in field_names:
            field = philosophy_field_registry.get_field(field_name)
            if field and field.required:
                if field_name not in data or data[field_name] is None:
                    validation_result.add_error(
                        f"Required field '{field_name}' is missing"
                    )

        # Validate field values
        for field_name, value in data.items():
            field = philosophy_field_registry.get_field(field_name)
            if field:
                field_validation = field.validate_value(value)
                if not field_validation.is_valid:
                    validation_result.errors.extend(field_validation.errors)

        return validation_result

    @staticmethod
    def map_fields_to_library(field_set: str) -> Dict[str, PhilosophyField]:
        """Map field set fields to PhilosophyFieldLibrary fields"""
        field_set_enum = PhilosophyFieldSet(field_set)
        if not field_set_enum:
            return {}

        field_names = philosophy_field_registry.get_field_set(field_set_enum)
        mapped_fields = {}

        for field_name in field_names:
            # Try to get field from registry
            field = philosophy_field_registry.get_field(field_name)
            if field:
                # Convert to PhilosophyField if needed
                if isinstance(field, PhilosophyField):
                    mapped_fields[field_name] = field
                else:
                    # Create PhilosophyField from PhilosophyExtractionField
                    mapped_fields[field_name] = PhilosophyField(
                        name=field.name,
                        description=field.description,
                        data_type=field.data_type,
                        required=field.required,
                        category=field.category,
                        priority=field.priority,
                    )

        return mapped_fields

    @staticmethod
    def map_categories_to_fields(
        categories: List[PhilosophicalCategory],
    ) -> Dict[FieldCategory, List[PhilosophyField]]:
        """Map philosophical categories to field categories and get relevant fields"""
        category_mapping = {
            PhilosophicalCategory.ETHICS: FieldCategory.PHILOSOPHICAL,
            PhilosophicalCategory.METAPHYSICS: FieldCategory.PHILOSOPHICAL,
            PhilosophicalCategory.EPISTEMOLOGY: FieldCategory.PHILOSOPHICAL,
            PhilosophicalCategory.LOGIC: FieldCategory.ANALYSIS,
            PhilosophicalCategory.AESTHETICS: FieldCategory.PHILOSOPHICAL,
            PhilosophicalCategory.POLITICAL: FieldCategory.CONTEXT,
            PhilosophicalCategory.PHILOSOPHY_OF_MIND: FieldCategory.PHILOSOPHICAL,
            PhilosophicalCategory.PHILOSOPHY_OF_LANGUAGE: FieldCategory.LINGUISTIC,
            PhilosophicalCategory.CONTINENTAL: FieldCategory.HISTORICAL,
            PhilosophicalCategory.ANALYTIC: FieldCategory.ANALYSIS,
            PhilosophicalCategory.EASTERN: FieldCategory.HISTORICAL,
        }

        field_map = {}
        for phil_category in categories:
            field_category = category_mapping.get(phil_category, FieldCategory.CORE)
            if field_category not in field_map:
                field_map[field_category] = []

            # Get fields for this category
            fields = PhilosophyFieldLibrary.get_fields_by_category(field_category)
            field_map[field_category].extend(fields)

        # Remove duplicates while preserving order
        for category in field_map:
            seen = set()
            unique_fields = []
            for field in field_map[category]:
                if field.name not in seen:
                    seen.add(field.name)
                    unique_fields.append(field)
            field_map[category] = unique_fields

        return field_map

    @staticmethod
    def create_extraction_result(
        extracted_data: Dict[str, Any],
        config: PhilosophyExtractionConfig,
        extraction_time: float,
    ) -> PhilosophyExtractionResult:
        """Create a properly structured extraction result from raw data"""
        # Create metadata
        metadata = ExtractionMetadata(
            extraction_id=f"phil_{datetime.utcnow().timestamp()}",
            duration_seconds=extraction_time,
            parameters=config.to_dict(),
            statistics={},
        )

        def create_items(
            values: List[Any], confidence: float = 0.9
        ) -> List[ExtractionItem]:
            """Helper to create extraction items"""
            if not isinstance(values, list):
                values = [values] if values else []

            items = []
            for value in values:
                if isinstance(value, dict):
                    # Handle structured data
                    item = ExtractionItem(
                        value=value.get("value", value),
                        confidence=value.get("confidence", confidence),
                        context=value.get("context"),
                        metadata=value.get("metadata", {}),
                    )
                else:
                    # Handle simple values
                    item = ExtractionItem(
                        value=value,
                        confidence=confidence,
                    )
                items.append(item)
            return items

        # Create result with extracted data
        result = PhilosophyExtractionResult(
            metadata=metadata,
        )

        # Map extracted data to result fields
        if "key_concepts" in extracted_data:
            result.key_concepts = create_items(extracted_data["key_concepts"])

        if "arguments" in extracted_data:
            result.arguments = create_items(extracted_data["arguments"])

        if "philosophers" in extracted_data:
            result.philosophers = create_items(extracted_data["philosophers"])

        if "ethical_principles" in extracted_data:
            result.ethical_principles = create_items(
                extracted_data["ethical_principles"]
            )

        if "epistemological_claims" in extracted_data:
            result.epistemological_claims = create_items(
                extracted_data["epistemological_claims"]
            )

        if "metaphysical_positions" in extracted_data:
            result.metaphysical_positions = create_items(
                extracted_data["metaphysical_positions"]
            )

        if "logical_structures" in extracted_data:
            result.logical_structures = create_items(
                extracted_data["logical_structures"]
            )

        if "aesthetic_theories" in extracted_data:
            result.aesthetic_theories = create_items(
                extracted_data["aesthetic_theories"]
            )

        if "political_theories" in extracted_data:
            result.political_theories = create_items(
                extracted_data["political_theories"]
            )

        if "historical_context" in extracted_data:
            result.historical_context = create_items(
                extracted_data["historical_context"]
            )

        if "influences" in extracted_data:
            result.influences = create_items(extracted_data["influences"])

        if "criticisms" in extracted_data:
            result.criticisms = create_items(extracted_data["criticisms"])

        if "applications" in extracted_data:
            result.applications = create_items(extracted_data["applications"])

        # Set analysis fields
        if "philosophical_tradition" in extracted_data:
            result.philosophical_tradition = extracted_data["philosophical_tradition"]

        if "argument_type" in extracted_data:
            result.argument_type = extracted_data["argument_type"]

        if "methodology" in extracted_data:
            result.methodology = extracted_data["methodology"]

        if "main_thesis" in extracted_data:
            result.main_thesis = extracted_data["main_thesis"]

        return result

    @staticmethod
    def create_extraction_config(
        source_type: str, field_set: str, extraction_params: Dict[str, Any]
    ) -> PhilosophyExtractionConfig:
        """Create extraction configuration from parameters"""
        # Convert source_type string to enum
        try:
            source_type_enum = PhilosophySourceType(source_type)
        except ValueError:
            source_type_enum = PhilosophySourceType.ESSAY

        # Extract parameters with defaults
        config = PhilosophyExtractionConfig(
            source_type=source_type_enum,
            language=extraction_params.get("language", "mixed"),
            extraction_depth=ExtractionDepth(
                extraction_params.get("depth_level", "detailed")
            ),
            target_audience=TargetAudience(
                extraction_params.get("target_audience", "academic")
            ),
            extraction_mode=ExtractionMode(
                extraction_params.get("extraction_mode", "comprehensive")
            ),
            categories_focus=[
                PhilosophicalCategory(cat)
                for cat in extraction_params.get("categories", [])
            ],
            include_examples=extraction_params.get("include_examples", True),
            include_references=extraction_params.get("include_references", True),
            include_historical_context=extraction_params.get(
                "include_historical_context", True
            ),
            include_influences=extraction_params.get("include_influences", True),
            include_criticisms=extraction_params.get("include_criticisms", True),
            include_applications=extraction_params.get("include_applications", True),
            include_cross_references=extraction_params.get(
                "include_cross_references", True
            ),
            confidence_threshold=extraction_params.get("confidence_threshold", 0.7),
            max_extraction_time=extraction_params.get("max_extraction_time", 300),
            preserve_original_language=extraction_params.get(
                "preserve_original_language", True
            ),
            extract_implicit_content=extraction_params.get(
                "extract_implicit_content", True
            ),
            custom_parameters=extraction_params.get("custom_parameters", {}),
        )

        return config

    @staticmethod
    def recommend_field_set(
        source_type: PhilosophySourceType,
        use_case: str,
        extraction_depth: ExtractionDepth,
        categories: Optional[List[PhilosophicalCategory]] = None,
    ) -> str:
        """Recommend appropriate field set based on parameters"""
        # Base recommendations by source type
        base_recommendations = {
            PhilosophySourceType.ESSAY: PhilosophyFieldSet.BASIC,
            PhilosophySourceType.TREATISE: PhilosophyFieldSet.COMPREHENSIVE,
            PhilosophySourceType.DIALOGUE: PhilosophyFieldSet.ARGUMENT,
            PhilosophySourceType.FRAGMENT: PhilosophyFieldSet.BASIC,
            PhilosophySourceType.LECTURE: PhilosophyFieldSet.CONCEPT,
            PhilosophySourceType.COMMENTARY: PhilosophyFieldSet.CRITICAL,
        }

        # Use case specific recommendations
        use_case_recommendations = {
            "quick_analysis": PhilosophyFieldSet.BASIC,
            "detailed_study": PhilosophyFieldSet.COMPREHENSIVE,
            "argument_analysis": PhilosophyFieldSet.ARGUMENT,
            "concept_exploration": PhilosophyFieldSet.CONCEPT,
            "critical_review": PhilosophyFieldSet.CRITICAL,
            "historical_study": PhilosophyFieldSet.HISTORICAL,
            "comparative_analysis": PhilosophyFieldSet.COMPARATIVE,
        }

        # Start with base recommendation
        recommended = base_recommendations.get(source_type, PhilosophyFieldSet.BASIC)

        # Adjust based on use case
        if use_case in use_case_recommendations:
            recommended = use_case_recommendations[use_case]

        # Adjust based on depth
        if extraction_depth == ExtractionDepth.BASIC:
            recommended = PhilosophyFieldSet.BASIC
        elif extraction_depth == ExtractionDepth.EXPERT:
            recommended = PhilosophyFieldSet.COMPREHENSIVE

        # Adjust based on categories
        if categories:
            if PhilosophicalCategory.ETHICS in categories:
                recommended = PhilosophyFieldSet.ETHICAL
            elif PhilosophicalCategory.LOGIC in categories:
                recommended = PhilosophyFieldSet.ARGUMENT

        return recommended.value

    @staticmethod
    def get_fields_for_category(
        category: PhilosophicalCategory, include_core: bool = True
    ) -> List[str]:
        """Get field names for a specific philosophical category"""
        field_names = []

        # Get fields from registry by category
        fields = philosophy_field_registry.get_fields_by_category(category)
        field_names.extend([field.name for field in fields.values()])

        # Add core fields if requested
        if include_core:
            core_fields = philosophy_field_registry.get_required_fields()
            field_names.extend([field.name for field in core_fields.values()])

        # Remove duplicates while preserving order
        seen = set()
        unique_fields = []
        for field_name in field_names:
            if field_name not in seen:
                seen.add(field_name)
                unique_fields.append(field_name)

        return unique_fields

    @staticmethod
    def analyze_field_coverage(
        extracted_data: Dict[str, Any], expected_fields: List[str]
    ) -> Dict[str, float]:
        """Analyze how well the extraction covered expected fields"""
        coverage = {}

        for field_name in expected_fields:
            if field_name in extracted_data:
                value = extracted_data[field_name]
                if value is not None and value != "":
                    if isinstance(value, list):
                        coverage[field_name] = len(value) / 10.0  # Normalize to 0-1
                    else:
                        coverage[field_name] = 1.0
                else:
                    coverage[field_name] = 0.0
            else:
                coverage[field_name] = 0.0

        return coverage


# Export all integration utilities
__all__ = [
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

# Convenience functions
validate_field_compatibility = PhilosophySystemIntegration.validate_field_compatibility
validate_extraction_data = PhilosophySystemIntegration.validate_extraction_data
map_fields_to_library = PhilosophySystemIntegration.map_fields_to_library
map_categories_to_fields = PhilosophySystemIntegration.map_categories_to_fields
create_extraction_result = PhilosophySystemIntegration.create_extraction_result
create_extraction_config = PhilosophySystemIntegration.create_extraction_config
recommend_field_set = PhilosophySystemIntegration.recommend_field_set
get_fields_for_category = PhilosophySystemIntegration.get_fields_for_category
analyze_field_coverage = PhilosophySystemIntegration.analyze_field_coverage
