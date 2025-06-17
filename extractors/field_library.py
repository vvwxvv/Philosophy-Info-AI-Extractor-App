"""
Enhanced Field Library for Philosophy Extraction System
Provides backward compatibility while using the new field system
"""

from typing import List, Optional, Dict, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import from the new enhanced modules
from extractors.types import DataType, PhilosophicalCategory
from extractors.fields import (
    PhilosophyExtractionField,
    PhilosophyFieldSet,
    philosophy_field_registry,
    MinMaxRule,
    LengthRule,
    OptionsRule,
)

logger = logging.getLogger(__name__)


class FieldCategory(str, Enum):
    """Categories for field organization (maps to PhilosophicalCategory)"""

    CORE = "core"
    CONTENT = "content"
    CONTEXT = "context"
    ANALYSIS = "analysis"
    METADATA = "metadata"
    PHILOSOPHICAL = "philosophical"
    HISTORICAL = "historical"
    LINGUISTIC = "linguistic"

    def to_philosophical_category(self) -> Optional[PhilosophicalCategory]:
        """Convert to PhilosophicalCategory enum"""
        mapping = {
            "PHILOSOPHICAL": PhilosophicalCategory.METAPHYSICS,
            "HISTORICAL": PhilosophicalCategory.POLITICAL,
            "ANALYSIS": PhilosophicalCategory.LOGIC,
            "LINGUISTIC": PhilosophicalCategory.PHILOSOPHY_OF_LANGUAGE,
            "CORE": PhilosophicalCategory.METAPHYSICS,
            "CONTENT": PhilosophicalCategory.EPISTEMOLOGY,
            "CONTEXT": PhilosophicalCategory.POLITICAL,
            "METADATA": None,  # No philosophical category for metadata
        }
        return mapping.get(self.value.upper())


@dataclass
class PhilosophyField(PhilosophyExtractionField):
    """Extended field class for philosophical content (wrapper for compatibility)"""

    # Additional legacy properties
    extraction_complexity: int = 1  # 1-5 scale

    def __post_init__(self):
        """Additional initialization"""
        try:
            super().__post_init__()
        except Exception as e:
            logger.warning(f"Error in parent post_init: {e}")

        # Map FieldCategory to PhilosophicalCategory if needed
        if hasattr(self, "_field_category") and not self.category:
            fc = getattr(self, "_field_category", None)
            if fc and isinstance(fc, FieldCategory):
                self.category = fc.to_philosophical_category()

    def is_compatible_with(self, other_field: "PhilosophyField") -> bool:
        """Check if this field is compatible with another"""
        # Fields in the same category are generally compatible
        if hasattr(self, "related_fields") and hasattr(other_field, "related_fields"):
            return (
                self.category == other_field.category
                or self.name in other_field.related_fields
                or other_field.name in self.related_fields
            )
        else:
            return self.category == other_field.category


class PhilosophyFieldRegistry:
    """Registry pattern wrapper around the new field registry"""

    def __init__(self):
        # Use the global registry from the new system
        self._registry = philosophy_field_registry
        self._field_categories: Dict[FieldCategory, Set[str]] = {
            cat: set() for cat in FieldCategory
        }

    def register(
        self, field: Union[PhilosophyField, PhilosophyExtractionField]
    ) -> None:
        """Register a field in the registry"""
        try:
            # Register in the new system
            self._registry.register_field(field)

            # Track by legacy category if it's a PhilosophyField
            if hasattr(field, "_field_category"):
                self._field_categories[field._field_category].add(field.name)
        except Exception as e:
            logger.error(f"Failed to register field {field.name}: {e}")

    def get(self, field_name: str) -> Optional[PhilosophyExtractionField]:
        """Get a field by name"""
        return self._registry.get_field(field_name)

    def get_by_category(
        self, category: Union[FieldCategory, PhilosophicalCategory]
    ) -> List[PhilosophyExtractionField]:
        """Get all fields in a category"""
        try:
            if isinstance(category, FieldCategory):
                # Map to PhilosophicalCategory
                phil_cat = category.to_philosophical_category()
                if phil_cat:
                    return list(
                        self._registry.get_fields_by_category(phil_cat).values()
                    )
                else:
                    # Return fields tracked by legacy category
                    field_names = self._field_categories.get(category, set())
                    return [
                        self._registry.get_field(name)
                        for name in field_names
                        if self._registry.get_field(name)
                    ]
            else:
                return list(self._registry.get_fields_by_category(category).values())
        except Exception as e:
            logger.error(f"Error getting fields by category {category}: {e}")
            return []

    def search_by_keyword(self, keyword: str) -> List[PhilosophyExtractionField]:
        """Search fields by keyword"""
        try:
            return self._registry.search_fields(keyword)
        except Exception as e:
            logger.error(f"Error searching fields by keyword {keyword}: {e}")
            return []

    def get_required_fields(self) -> List[PhilosophyExtractionField]:
        """Get all required fields"""
        try:
            return list(self._registry.get_required_fields().values())
        except Exception as e:
            logger.error(f"Error getting required fields: {e}")
            return []

    def get_by_complexity(self, max_complexity: int) -> List[PhilosophyExtractionField]:
        """Get fields up to a certain complexity level"""
        all_fields = []
        try:
            for field_name, field in self._registry._fields.items():
                if (
                    hasattr(field, "extraction_complexity")
                    and field.extraction_complexity <= max_complexity
                ):
                    all_fields.append(field)
                elif field.priority >= (
                    6 - max_complexity
                ):  # Map priority to complexity
                    all_fields.append(field)
        except Exception as e:
            logger.error(f"Error getting fields by complexity: {e}")
        return all_fields

    def validate_field_set(self, field_names: List[str]) -> Dict[str, Any]:
        """Validate a set of fields for compatibility and completeness"""
        validation_result = {
            "valid": True,
            "missing_required": [],
            "unknown_fields": [],
            "compatibility_issues": [],
        }

        try:
            # Check for unknown fields
            for name in field_names:
                if not self._registry.get_field(name):
                    validation_result["unknown_fields"].append(name)
                    validation_result["valid"] = False

            # Check for missing required fields
            required_fields = {f.name for f in self.get_required_fields()}
            provided_fields = set(field_names)
            missing = required_fields - provided_fields
            if missing:
                validation_result["missing_required"] = list(missing)
                validation_result["valid"] = False
        except Exception as e:
            logger.error(f"Error validating field set: {e}")
            validation_result["valid"] = False

        return validation_result


# Initialize global field registry wrapper
field_registry = PhilosophyFieldRegistry()


# Helper function to create philosophy fields
def create_philosophy_field(
    name: str,
    description: str,
    category: FieldCategory = FieldCategory.CORE,
    priority: int = 5,
    required: bool = False,
    data_type: DataType = DataType.STRING,
    extraction_complexity: int = 1,
    **kwargs,
) -> PhilosophyField:
    """Helper function to create philosophy fields with legacy support"""

    try:
        # Map FieldCategory to PhilosophicalCategory
        phil_category = category.to_philosophical_category()

        # Extract legacy-specific kwargs
        legacy_kwargs = {
            "extraction_complexity": extraction_complexity,
            "_field_category": category,  # Store original category
        }

        # Create field with new system parameters
        field = PhilosophyField(
            name=name,
            description=description,
            category=phil_category,
            priority=priority,
            required=required,
            data_type=data_type,
            **kwargs,
            **legacy_kwargs,
        )

        return field
    except Exception as e:
        logger.error(f"Error creating philosophy field {name}: {e}")
        # Return a basic field as fallback
        return PhilosophyField(
            name=name,
            description=description,
            category=None,
            priority=priority,
            required=required,
            data_type=data_type,
            extraction_complexity=extraction_complexity,
            _field_category=category,
        )


# Core philosophical fields (using fields already defined in fields.py)
def create_core_fields():
    """Create core fields with error handling"""
    fields = []

    try:
        fields.append(
            create_philosophy_field(
                name="main_argument",
                description="The main philosophical argument or thesis being presented",
                category=FieldCategory.CORE,
                priority=1,
                required=True,
                extraction_hints=[
                    "Look for main claims and supporting premises",
                    "Identify the logical structure of the argument",
                    "Note conclusion indicators like 'therefore', 'thus', 'hence'",
                ],
                contextual_keywords=[
                    "argument",
                    "thesis",
                    "claim",
                    "premise",
                    "conclusion",
                    "reasoning",
                ],
                examples={
                    "en": [
                        "All humans are mortal; Socrates is human; Therefore, Socrates is mortal"
                    ],
                    "zh": ["所有人都是有限的；苏格拉底是人；因此，苏格拉底是有限的"],
                },
                extraction_complexity=3,
                validation_rules=[LengthRule(10, 2000)],
            )
        )
    except Exception as e:
        logger.error(f"Error creating main_argument field: {e}")

    try:
        fields.append(
            create_philosophy_field(
                name="philosophical_methodology",
                description="The philosophical method or approach used",
                category=FieldCategory.PHILOSOPHICAL,
                priority=2,
                extraction_hints=[
                    "Look for methodological approaches",
                    "Identify analytical techniques",
                    "Note philosophical tools used",
                ],
                contextual_keywords=[
                    "method",
                    "approach",
                    "analysis",
                    "methodology",
                    "technique",
                ],
                examples={
                    "en": [
                        "phenomenological",
                        "analytical",
                        "dialectical",
                        "hermeneutical",
                    ],
                    "zh": ["现象学的", "分析的", "辩证的", "解释学的"],
                },
                extraction_complexity=3,
                validation_rules=[
                    OptionsRule(
                        [
                            "phenomenological",
                            "analytical",
                            "dialectical",
                            "hermeneutical",
                            "deconstructive",
                            "pragmatic",
                            "empirical",
                            "conceptual",
                            "other",
                        ]
                    )
                ],
            )
        )
    except Exception as e:
        logger.error(f"Error creating philosophical_methodology field: {e}")

    try:
        fields.append(
            create_philosophy_field(
                name="philosophical_implications",
                description="Implications and consequences of the philosophical position",
                category=FieldCategory.ANALYSIS,
                priority=3,
                data_type=DataType.ARRAY,
                extraction_hints=[
                    "Look for consequence indicators",
                    "Identify practical implications",
                    "Note theoretical ramifications",
                ],
                contextual_keywords=[
                    "implies",
                    "consequences",
                    "implications",
                    "leads to",
                    "results in",
                ],
                extraction_complexity=4,
            )
        )
    except Exception as e:
        logger.error(f"Error creating philosophical_implications field: {e}")

    return [f for f in fields if f]  # Filter out None values


def create_analysis_fields():
    """Create analysis fields with error handling"""
    fields = []

    try:
        fields.append(
            create_philosophy_field(
                name="logical_structure",
                description="The logical structure of the argument",
                category=FieldCategory.ANALYSIS,
                priority=2,
                extraction_hints=[
                    "Identify premise-conclusion relationships",
                    "Map logical dependencies",
                    "Note inference patterns",
                ],
                contextual_keywords=[
                    "structure",
                    "logic",
                    "inference",
                    "deduction",
                    "induction",
                ],
                extraction_complexity=4,
                data_type=DataType.OBJECT,
            )
        )
    except Exception as e:
        logger.error(f"Error creating logical_structure field: {e}")

    try:
        fields.append(
            create_philosophy_field(
                name="philosophical_assumptions",
                description="Underlying assumptions in the philosophical argument",
                category=FieldCategory.ANALYSIS,
                priority=3,
                data_type=DataType.ARRAY,
                extraction_hints=[
                    "Identify unstated premises",
                    "Look for background assumptions",
                    "Note conceptual presuppositions",
                ],
                contextual_keywords=[
                    "assumes",
                    "presupposes",
                    "takes for granted",
                    "implicit",
                    "underlying",
                ],
                extraction_complexity=5,
            )
        )
    except Exception as e:
        logger.error(f"Error creating philosophical_assumptions field: {e}")

    try:
        fields.append(
            create_philosophy_field(
                name="dialectical_tensions",
                description="Tensions or contradictions explored in the work",
                category=FieldCategory.ANALYSIS,
                priority=4,
                data_type=DataType.ARRAY,
                extraction_hints=[
                    "Identify opposing viewpoints",
                    "Note paradoxes or contradictions",
                    "Look for dialectical movements",
                ],
                contextual_keywords=[
                    "tension",
                    "contradiction",
                    "paradox",
                    "opposition",
                    "dialectic",
                ],
                extraction_complexity=5,
            )
        )
    except Exception as e:
        logger.error(f"Error creating dialectical_tensions field: {e}")

    return [f for f in fields if f]


def create_context_fields():
    """Create context fields with error handling"""
    fields = []

    try:
        fields.append(
            create_philosophy_field(
                name="intellectual_influences",
                description="Intellectual influences and predecessors",
                category=FieldCategory.HISTORICAL,
                priority=3,
                data_type=DataType.ARRAY,
                extraction_hints=[
                    "Look for references to other philosophers",
                    "Identify intellectual traditions",
                    "Note scholarly influences",
                ],
                contextual_keywords=[
                    "influenced by",
                    "following",
                    "building on",
                    "tradition of",
                    "school of",
                ],
                extraction_complexity=3,
                post_processors=["normalize_philosopher"],
            )
        )
    except Exception as e:
        logger.error(f"Error creating intellectual_influences field: {e}")

    try:
        fields.append(
            create_philosophy_field(
                name="philosophical_innovations",
                description="Novel contributions or innovations in the work",
                category=FieldCategory.CONTENT,
                priority=2,
                data_type=DataType.ARRAY,
                extraction_hints=[
                    "Identify new concepts introduced",
                    "Look for original arguments",
                    "Note methodological innovations",
                ],
                contextual_keywords=[
                    "novel",
                    "new",
                    "innovative",
                    "original",
                    "introduces",
                    "proposes",
                ],
                extraction_complexity=4,
            )
        )
    except Exception as e:
        logger.error(f"Error creating philosophical_innovations field: {e}")

    try:
        fields.append(
            create_philosophy_field(
                name="cross_cultural_elements",
                description="Cross-cultural philosophical elements or comparisons",
                category=FieldCategory.CONTEXT,
                priority=5,
                extraction_hints=[
                    "Identify cross-cultural references",
                    "Note comparative philosophy",
                    "Look for cultural contexts",
                ],
                contextual_keywords=[
                    "culture",
                    "tradition",
                    "Eastern",
                    "Western",
                    "comparative",
                ],
                extraction_complexity=4,
            )
        )
    except Exception as e:
        logger.error(f"Error creating cross_cultural_elements field: {e}")

    return [f for f in fields if f]


def create_metadata_fields():
    """Create metadata fields with error handling"""
    fields = []

    try:
        fields.append(
            create_philosophy_field(
                name="extraction_quality_score",
                description="Quality score of the extraction",
                category=FieldCategory.METADATA,
                priority=10,
                data_type=DataType.NUMBER,
                validation_rules=[MinMaxRule(0.0, 1.0)],
                extraction_complexity=1,
                default_value=0.0,
            )
        )
    except Exception as e:
        logger.error(f"Error creating extraction_quality_score field: {e}")

    try:
        fields.append(
            create_philosophy_field(
                name="philosophical_genre",
                description="Genre or type of philosophical work",
                category=FieldCategory.METADATA,
                priority=7,
                extraction_hints=[
                    "Identify the type of philosophical text",
                    "Note the genre conventions",
                    "Look for format indicators",
                ],
                examples={
                    "en": ["treatise", "dialogue", "essay", "aphorism", "commentary"],
                    "zh": ["论文", "对话", "散文", "格言", "注释"],
                },
                extraction_complexity=2,
                validation_rules=[
                    OptionsRule(
                        [
                            "treatise",
                            "dialogue",
                            "essay",
                            "aphorism",
                            "commentary",
                            "lecture",
                            "letter",
                            "meditation",
                            "critique",
                            "manifesto",
                        ]
                    )
                ],
            )
        )
    except Exception as e:
        logger.error(f"Error creating philosophical_genre field: {e}")

    return [f for f in fields if f]


# Create all field collections
CORE_FIELDS = create_core_fields()
ANALYSIS_FIELDS = create_analysis_fields()
CONTEXT_FIELDS = create_context_fields()
METADATA_FIELDS = create_metadata_fields()

# Register all fields
try:
    for field in CORE_FIELDS + ANALYSIS_FIELDS + CONTEXT_FIELDS + METADATA_FIELDS:
        if field:  # Only register non-None fields
            field_registry.register(field)
except Exception as e:
    logger.error(f"Error registering fields: {e}")


class PhilosophyFieldLibrary:
    """Enhanced field library with registry pattern (backward compatible)"""

    # Class-level registry
    _registry = field_registry

    @classmethod
    def get_field(cls, field_name: str) -> Optional[PhilosophyExtractionField]:
        """Get a specific field by name"""
        try:
            return cls._registry.get(field_name)
        except Exception as e:
            logger.error(f"Error getting field {field_name}: {e}")
            return None

    @classmethod
    def get_fields_by_category(
        cls, category: Union[str, FieldCategory, PhilosophicalCategory]
    ) -> List[PhilosophyExtractionField]:
        """Get all fields in a specific category"""
        try:
            if isinstance(category, str):
                # Try FieldCategory first
                try:
                    category = FieldCategory(category)
                except ValueError:
                    # Try PhilosophicalCategory
                    try:
                        category = PhilosophicalCategory(category)
                    except ValueError:
                        return []

            return cls._registry.get_by_category(category)
        except Exception as e:
            logger.error(f"Error getting fields by category {category}: {e}")
            return []

    @classmethod
    def get_required_fields(cls) -> List[PhilosophyExtractionField]:
        """Get all required fields"""
        try:
            return cls._registry.get_required_fields()
        except Exception as e:
            logger.error(f"Error getting required fields: {e}")
            return []

    @classmethod
    def search_fields(cls, keyword: str) -> List[PhilosophyExtractionField]:
        """Search fields by keyword"""
        try:
            return cls._registry.search_by_keyword(keyword)
        except Exception as e:
            logger.error(f"Error searching fields by keyword {keyword}: {e}")
            return []

    @classmethod
    def get_fields_for_complexity_level(
        cls, level: str
    ) -> List[PhilosophyExtractionField]:
        """Get fields appropriate for a complexity level"""
        try:
            complexity_map = {"basic": 2, "intermediate": 3, "detailed": 4, "expert": 5}
            max_complexity = complexity_map.get(level, 3)
            return cls._registry.get_by_complexity(max_complexity)
        except Exception as e:
            logger.error(f"Error getting fields for complexity level {level}: {e}")
            return []

    @classmethod
    def validate_field_selection(cls, field_names: List[str]) -> Dict[str, Any]:
        """Validate a selection of fields"""
        try:
            return cls._registry.validate_field_set(field_names)
        except Exception as e:
            logger.error(f"Error validating field selection: {e}")
            return {"valid": False, "error": str(e)}

    @classmethod
    def get_field_set(cls, set_name: Union[str, PhilosophyFieldSet]) -> List[str]:
        """Get predefined field set"""
        try:
            if isinstance(set_name, str):
                try:
                    set_name = PhilosophyFieldSet(set_name)
                except ValueError:
                    return []

            return philosophy_field_registry.get_field_set(set_name)
        except Exception as e:
            logger.error(f"Error getting field set {set_name}: {e}")
            return []

    @classmethod
    def get_all_field_sets(cls) -> Dict[str, List[str]]:
        """Get all available field sets"""
        try:
            return {
                field_set.value: philosophy_field_registry.get_field_set(field_set)
                for field_set in PhilosophyFieldSet
            }
        except Exception as e:
            logger.error(f"Error getting all field sets: {e}")
            return {}

    # Properties for backward compatibility
    @property
    def ALL_FIELDS(self) -> Dict[str, PhilosophyExtractionField]:
        """All registered fields (backward compatibility)"""
        try:
            return philosophy_field_registry._fields
        except Exception as e:
            logger.error(f"Error getting all fields: {e}")
            return {}

    # Class method version
    @classmethod
    def get_all_fields(cls) -> Dict[str, PhilosophyExtractionField]:
        """Get all registered fields"""
        try:
            return philosophy_field_registry._fields
        except Exception as e:
            logger.error(f"Error getting all fields: {e}")
            return {}

    # Convenience methods
    @classmethod
    def get_basic_fields(cls) -> List[str]:
        """Get basic philosophy fields"""
        try:
            return philosophy_field_registry.get_field_set(PhilosophyFieldSet.BASIC)
        except Exception as e:
            logger.error(f"Error getting basic fields: {e}")
            return []

    @classmethod
    def get_comprehensive_fields(cls) -> List[str]:
        """Get comprehensive philosophy fields"""
        try:
            return philosophy_field_registry.get_field_set(
                PhilosophyFieldSet.COMPREHENSIVE
            )
        except Exception as e:
            logger.error(f"Error getting comprehensive fields: {e}")
            return []

    @classmethod
    def create_custom_field(
        cls, name: str, description: str, **kwargs
    ) -> Optional[PhilosophyExtractionField]:
        """Create and register a custom field"""
        try:
            field = create_philosophy_field(
                name=name, description=description, **kwargs
            )
            field_registry.register(field)
            return field
        except Exception as e:
            logger.error(f"Error creating custom field {name}: {e}")
            return None


# Maintain backward compatibility - create singleton instance
philosophy_field_library = PhilosophyFieldLibrary()

# Export commonly used items
__all__ = [
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
