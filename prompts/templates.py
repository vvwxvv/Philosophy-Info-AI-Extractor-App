"""Enhanced template management for philosophical extraction"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from extractors.fields import (
    PhilosophyExtractionField,
    PhilosophyFieldSet,
    philosophy_field_registry,
)
from extractors.types import (
    PhilosophySourceType,
    ExtractionDepth,
    PhilosophicalCategory,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhilosophyExtractionTemplate:
    """Template for philosophical content extraction"""

    name: str
    description: str
    source_type: PhilosophySourceType
    fields: List[str]  # List of field names
    categories: List[PhilosophicalCategory] = field(default_factory=list)
    extraction_depth: ExtractionDepth = ExtractionDepth.DETAILED
    language: str = "mixed"  # CN, EN, or mixed
    priority_level: int = 3  # 1-5 scale
    custom_fields: Dict[str, PhilosophyExtractionField] = field(default_factory=dict)
    post_processors: List[str] = field(default_factory=list)
    parent_template: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Extraction configuration
    include_examples: bool = True
    include_references: bool = True
    include_historical_context: bool = True
    include_criticisms: bool = True
    extract_implicit_content: bool = True

    def __post_init__(self):
        """Validate template after initialization"""
        self._validate_template()

    def _validate_template(self):
        """Validate template configuration"""
        if not self.name:
            raise ValueError("Template name cannot be empty")

        if not self.description:
            raise ValueError("Template description cannot be empty")

        if not self.fields:
            raise ValueError("Template must have at least one field")

        if not 1 <= self.priority_level <= 5:
            raise ValueError("Priority level must be between 1 and 5")

        if self.language not in ["CN", "EN", "mixed", "auto"]:
            raise ValueError("Language must be one of: CN, EN, mixed, auto")

        # Validate fields exist
        for field_name in self.fields:
            if (
                not philosophy_field_registry.get_field(field_name)
                and field_name not in self.custom_fields
            ):
                raise ValueError(f"Field '{field_name}' not found")

    def get_fields(self) -> Dict[str, PhilosophyExtractionField]:
        """Get all fields for this template including inherited fields"""
        result = {}

        # Get fields from parent template if exists
        if self.parent_template:
            parent = philosophy_template_library.get_template(self.parent_template)
            if parent:
                result.update(parent.get_fields())

        # Add template's own fields
        for field_name in self.fields:
            field = philosophy_field_registry.get_field(field_name)
            if field:
                result[field_name] = field

        # Add custom fields
        result.update(self.custom_fields)

        return result

    def get_required_fields(self) -> Dict[str, PhilosophyExtractionField]:
        """Get only required fields from this template"""
        return {
            name: field for name, field in self.get_fields().items() if field.required
        }

    def get_guidelines(self) -> List[str]:
        """Get extraction guidelines based on template configuration"""
        guidelines = [
            f"Extract {self.extraction_depth.description}",
            (
                f"Focus on {', '.join(cat.value for cat in self.categories)}"
                if self.categories
                else "Extract all philosophical content"
            ),
            f"Source type is {self.source_type.description}",
        ]

        if self.include_examples:
            guidelines.append("Include concrete examples where relevant")

        if self.include_references:
            guidelines.append("Extract all philosophical references and citations")

        if self.include_historical_context:
            guidelines.append("Provide historical context and period information")

        if self.include_criticisms:
            guidelines.append("Include criticisms and counter-arguments")

        if self.extract_implicit_content:
            guidelines.append(
                "Extract both explicit and implicit philosophical content"
            )

        return guidelines

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type.value,
            "fields": self.fields,
            "categories": [cat.value for cat in self.categories],
            "extraction_depth": self.extraction_depth.value,
            "language": self.language,
            "priority_level": self.priority_level,
            "custom_fields": {
                name: field.__dict__ for name, field in self.custom_fields.items()
            },
            "post_processors": self.post_processors,
            "parent_template": self.parent_template,
            "metadata": self.metadata,
            "configuration": {
                "include_examples": self.include_examples,
                "include_references": self.include_references,
                "include_historical_context": self.include_historical_context,
                "include_criticisms": self.include_criticisms,
                "extract_implicit_content": self.extract_implicit_content,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhilosophyExtractionTemplate":
        """Create template from dictionary"""
        custom_fields = {}
        if "custom_fields" in data:
            for name, field_data in data["custom_fields"].items():
                custom_fields[name] = PhilosophyExtractionField(**field_data)

        config = data.get("configuration", {})

        return cls(
            name=data["name"],
            description=data["description"],
            source_type=PhilosophySourceType(data["source_type"]),
            fields=data["fields"],
            categories=[
                PhilosophicalCategory(cat) for cat in data.get("categories", [])
            ],
            extraction_depth=ExtractionDepth(data.get("extraction_depth", "detailed")),
            language=data.get("language", "mixed"),
            priority_level=data.get("priority_level", 3),
            custom_fields=custom_fields,
            post_processors=data.get("post_processors", []),
            parent_template=data.get("parent_template"),
            metadata=data.get("metadata", {}),
            include_examples=config.get("include_examples", True),
            include_references=config.get("include_references", True),
            include_historical_context=config.get("include_historical_context", True),
            include_criticisms=config.get("include_criticisms", True),
            extract_implicit_content=config.get("extract_implicit_content", True),
        )


class PhilosophyTemplateLibrary:
    """Library of philosophical extraction templates"""

    def __init__(self):
        self._templates: Dict[str, PhilosophyExtractionTemplate] = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Initialize with default templates"""

        # Basic philosophy template
        self.register(
            PhilosophyExtractionTemplate(
                name="philosophy_basic",
                description="Basic philosophical content extraction",
                source_type=PhilosophySourceType.ESSAY,
                fields=philosophy_field_registry.get_field_set(
                    PhilosophyFieldSet.BASIC
                ),
                extraction_depth=ExtractionDepth.BASIC,
                priority_level=2,
                include_examples=False,
                include_criticisms=False,
            )
        )

        # Comprehensive philosophy template
        self.register(
            PhilosophyExtractionTemplate(
                name="comprehensive",
                description="Comprehensive philosophical analysis",
                source_type=PhilosophySourceType.TREATISE,
                fields=philosophy_field_registry.get_field_set(
                    PhilosophyFieldSet.COMPREHENSIVE
                ),
                extraction_depth=ExtractionDepth.EXPERT,
                priority_level=5,
                categories=[],  # All categories
                include_examples=True,
                include_references=True,
                include_historical_context=True,
                include_criticisms=True,
                extract_implicit_content=True,
            )
        )

        # Philosopher profile template
        self.register(
            PhilosophyExtractionTemplate(
                name="philosopher_profile",
                description="Extract philosopher biographical and intellectual information",
                source_type=PhilosophySourceType.ESSAY,
                fields=philosophy_field_registry.get_field_set(
                    PhilosophyFieldSet.PHILOSOPHER
                ),
                extraction_depth=ExtractionDepth.DETAILED,
                include_historical_context=True,
                include_references=True,
            )
        )

        # Philosophical argument template
        self.register(
            PhilosophyExtractionTemplate(
                name="philosophical_argument",
                description="Detailed argument structure analysis",
                source_type=PhilosophySourceType.ESSAY,
                fields=philosophy_field_registry.get_field_set(
                    PhilosophyFieldSet.ARGUMENT
                ),
                categories=[PhilosophicalCategory.LOGIC],
                extraction_depth=ExtractionDepth.DETAILED,
                priority_level=4,
            )
        )

        # Ethics-focused template
        self.register(
            PhilosophyExtractionTemplate(
                name="philosophy_ethical",
                description="Extract ethical and moral philosophy content",
                source_type=PhilosophySourceType.ESSAY,
                fields=[
                    "main_thesis",
                    "key_concepts",
                    "ethical_principles",
                    "key_arguments",
                    "criticisms",
                    "applications",
                ],
                categories=[PhilosophicalCategory.ETHICS],
                extraction_depth=ExtractionDepth.DETAILED,
            )
        )

        # Metaphysics template
        self.register(
            PhilosophyExtractionTemplate(
                name="philosophy_metaphysical",
                description="Extract metaphysical positions and arguments",
                source_type=PhilosophySourceType.TREATISE,
                fields=[
                    "main_thesis",
                    "key_concepts",
                    "metaphysical_positions",
                    "key_arguments",
                    "philosophical_tradition",
                    "criticisms",
                ],
                categories=[PhilosophicalCategory.METAPHYSICS],
                extraction_depth=ExtractionDepth.DETAILED,
            )
        )

        # Historical philosophy template
        self.register(
            PhilosophyExtractionTemplate(
                name="historical_philosophy",
                description="Extract philosophical content with historical focus",
                source_type=PhilosophySourceType.ESSAY,
                fields=[
                    "text_title",
                    "philosopher_name",
                    "historical_context",
                    "main_thesis",
                    "influences",
                    "influenced",
                    "legacy",
                ],
                extraction_depth=ExtractionDepth.DETAILED,
                include_historical_context=True,
                priority_level=4,
            )
        )

        # Philosophical concept template
        self.register(
            PhilosophyExtractionTemplate(
                name="philosophical_concept",
                description="Extract philosophical concept definitions and analysis",
                source_type=PhilosophySourceType.ESSAY,
                fields=[
                    "concept_name",
                    "definition",
                    "key_concepts",
                    "key_philosophers",
                    "related_concepts",
                    "examples",
                    "criticisms",
                    "applications",
                ],
                extraction_depth=ExtractionDepth.DETAILED,
                include_examples=True,
                include_references=True,
                priority_level=4,
            )
        )

    def register(self, template: PhilosophyExtractionTemplate):
        """Register a template"""
        self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[PhilosophyExtractionTemplate]:
        """Get template by name"""
        return self._templates.get(name)

    def get_all_templates(self) -> Dict[str, PhilosophyExtractionTemplate]:
        """Get all templates"""
        return self._templates.copy()

    def get_templates_by_category(
        self, category: PhilosophicalCategory
    ) -> List[PhilosophyExtractionTemplate]:
        """Get templates that focus on a specific category"""
        return [
            template
            for template in self._templates.values()
            if category in template.categories
            or not template.categories  # Empty categories means all
        ]

    def get_templates_by_source_type(
        self, source_type: PhilosophySourceType
    ) -> List[PhilosophyExtractionTemplate]:
        """Get templates for a specific source type"""
        return [
            template
            for template in self._templates.values()
            if template.source_type == source_type
        ]

    def create_custom_template(
        self, name: str, base_template: str, modifications: Dict[str, Any]
    ) -> PhilosophyExtractionTemplate:
        """Create a custom template based on an existing one"""
        base = self.get_template(base_template)
        if not base:
            raise ValueError(f"Base template '{base_template}' not found")

        # Convert to dict and apply modifications
        template_data = base.to_dict()
        template_data["name"] = name
        template_data["parent_template"] = base_template
        template_data.update(modifications)

        # Create new template
        custom_template = PhilosophyExtractionTemplate.from_dict(template_data)
        self.register(custom_template)

        return custom_template


# Global template library instance
philosophy_template_library = PhilosophyTemplateLibrary()
