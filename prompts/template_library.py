"""Enhanced template library with validation and composition"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field as dataclass_field
from enum import Enum


class PhilosophySourceType(str, Enum):
    """Source types for philosophical content"""

    ESSAY = "essay"
    DIALOGUE = "dialogue"
    TREATISE = "treatise"
    COMMENTARY = "commentary"
    LECTURE = "lecture"
    JOURNAL = "journal"
    BOOK = "book"
    FRAGMENT = "fragment"

    @property
    def description(self) -> str:
        """Get human-readable description"""
        descriptions = {
            self.ESSAY: "Philosophical essay or article",
            self.DIALOGUE: "Philosophical dialogue or conversation",
            self.TREATISE: "Philosophical treatise or systematic work",
            self.COMMENTARY: "Commentary on philosophical work",
            self.LECTURE: "Philosophical lecture or presentation",
            self.JOURNAL: "Academic journal article",
            self.BOOK: "Philosophical book or monograph",
            self.FRAGMENT: "Philosophical fragment or excerpt",
        }
        return descriptions.get(self, self.value)


@dataclass
class TemplateMetadata:
    """Metadata for extraction templates"""

    author: str = ""
    version: str = "1.0"
    created_date: str = ""
    last_modified: str = ""
    tags: List[str] = dataclass_field(default_factory=list)
    difficulty_level: int = 3  # 1-5 scale
    estimated_extraction_time: int = 60  # seconds


@dataclass
class ExtractionTemplate:
    """Enhanced template for philosophical content extraction"""

    name: str
    description: str
    source_type: PhilosophySourceType
    fields: List[str]
    required_fields: List[str] = dataclass_field(default_factory=list)
    optional_fields: List[str] = dataclass_field(default_factory=list)
    metadata: TemplateMetadata = dataclass_field(default_factory=TemplateMetadata)
    prerequisites: List[str] = dataclass_field(default_factory=list)
    output_examples: List[Dict[str, Any]] = dataclass_field(default_factory=list)

    def __post_init__(self):
        """Validate and process template"""
        self._validate()
        self._process_fields()

    def _validate(self):
        """Validate template configuration"""
        # Validate name
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"Invalid template name: {self.name}")

        # Validate fields
        if not self.fields:
            raise ValueError("Template must have at least one field")

        # Validate required fields are in fields list
        for req_field in self.required_fields:
            if req_field not in self.fields:
                raise ValueError(f"Required field '{req_field}' not in fields list")

        # Validate no duplicates
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("Duplicate fields detected")

    def _process_fields(self):
        """Process field lists"""
        # Auto-populate optional fields
        if not self.optional_fields:
            self.optional_fields = [
                f for f in self.fields if f not in self.required_fields
            ]

    @property
    def complexity_score(self) -> float:
        """Calculate template complexity score"""
        base_score = len(self.fields) * 0.1
        required_score = len(self.required_fields) * 0.2
        difficulty_score = self.metadata.difficulty_level * 0.15
        return min(base_score + required_score + difficulty_score, 1.0)

    def is_compatible_with(self, other: "ExtractionTemplate") -> bool:
        """Check if this template is compatible with another"""
        # Check source type compatibility
        compatible_types = {
            PhilosophySourceType.ESSAY: [
                PhilosophySourceType.ARTICLE,
                PhilosophySourceType.COMMENTARY,
            ],
            PhilosophySourceType.TREATISE: [PhilosophySourceType.BOOK],
            PhilosophySourceType.DIALOGUE: [PhilosophySourceType.CONVERSATION],
        }

        if self.source_type != other.source_type:
            other_compatible = compatible_types.get(self.source_type, [])
            if other.source_type not in other_compatible:
                return False

        # Check field overlap
        field_overlap = set(self.fields).intersection(set(other.fields))
        return len(field_overlap) >= min(3, len(self.fields) // 2)

    def merge_with(self, other: "ExtractionTemplate") -> "ExtractionTemplate":
        """Merge this template with another"""
        if not self.is_compatible_with(other):
            raise ValueError(
                f"Template '{self.name}' is not compatible with '{other.name}'"
            )

        # Merge fields
        merged_fields = list(set(self.fields + other.fields))
        merged_required = list(set(self.required_fields + other.required_fields))

        # Create merged template
        return ExtractionTemplate(
            name=f"{self.name}_merged_{other.name}",
            description=f"Merged: {self.description} & {other.description}",
            source_type=self.source_type,  # Keep original source type
            fields=merged_fields,
            required_fields=merged_required,
            metadata=TemplateMetadata(
                author="system",
                version="1.0",
                tags=list(set(self.metadata.tags + other.metadata.tags)),
                difficulty_level=max(
                    self.metadata.difficulty_level, other.metadata.difficulty_level
                ),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type.value,
            "fields": self.fields,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "metadata": {
                "author": self.metadata.author,
                "version": self.metadata.version,
                "tags": self.metadata.tags,
                "difficulty_level": self.metadata.difficulty_level,
                "complexity_score": self.complexity_score,
            },
        }


class TemplateBuilder:
    """Builder pattern for creating templates"""

    def __init__(self, name: str):
        self.name = name
        self.description = ""
        self.source_type = PhilosophySourceType.ESSAY
        self.fields = []
        self.required_fields = []
        self.metadata = TemplateMetadata()

    def with_description(self, description: str) -> "TemplateBuilder":
        """Set template description"""
        self.description = description
        return self

    def with_source_type(
        self, source_type: Union[str, PhilosophySourceType]
    ) -> "TemplateBuilder":
        """Set source type"""
        if isinstance(source_type, str):
            source_type = PhilosophySourceType(source_type)
        self.source_type = source_type
        return self

    def add_field(self, field_name: str, required: bool = False) -> "TemplateBuilder":
        """Add a field to the template"""
        if field_name not in self.fields:
            self.fields.append(field_name)
            if required:
                self.required_fields.append(field_name)
        return self

    def add_fields(
        self, field_names: List[str], required: bool = False
    ) -> "TemplateBuilder":
        """Add multiple fields"""
        for field_name in field_names:
            self.add_field(field_name, required)
        return self

    def with_metadata(self, **kwargs) -> "TemplateBuilder":
        """Set metadata properties"""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        return self

    def build(self) -> ExtractionTemplate:
        """Build the template"""
        return ExtractionTemplate(
            name=self.name,
            description=self.description,
            source_type=self.source_type,
            fields=self.fields,
            required_fields=self.required_fields,
            metadata=self.metadata,
        )


class PhilosophyTemplateLibrary:
    """Enhanced template library with builder pattern and validation"""

    # Template registry
    _templates: Dict[str, ExtractionTemplate] = {}
    _initialized = False

    @classmethod
    def _initialize_templates(cls):
        """Initialize default templates"""
        if cls._initialized:
            return

        # Basic template
        cls.register_template(
            TemplateBuilder("philosophy_basic")
            .with_description("Basic template for philosophical content extraction")
            .with_source_type(PhilosophySourceType.ESSAY)
            .add_fields(["text_title", "main_topic", "key_arguments"], required=True)
            .add_fields(["philosophical_tradition", "historical_context"])
            .with_metadata(difficulty_level=2, tags=["basic", "general"])
            .build()
        )

        # Dialogue template
        cls.register_template(
            TemplateBuilder("philosophy_dialogue")
            .with_description("Template for philosophical dialogues")
            .with_source_type(PhilosophySourceType.DIALOGUE)
            .add_fields(
                ["participants", "main_questions", "key_exchanges"], required=True
            )
            .add_fields(["dramatic_context", "philosophical_method"])
            .with_metadata(difficulty_level=3, tags=["dialogue", "socratic"])
            .build()
        )

        # Comprehensive template
        cls.register_template(
            TemplateBuilder("philosophy_comprehensive")
            .with_description("Comprehensive philosophical analysis template")
            .with_source_type(PhilosophySourceType.TREATISE)
            .add_fields(
                [
                    "text_title",
                    "author",
                    "main_thesis",
                    "key_arguments",
                    "philosophical_tradition",
                    "methodology",
                ],
                required=True,
            )
            .add_fields(
                [
                    "historical_context",
                    "influences",
                    "criticisms",
                    "contemporary_relevance",
                    "key_concepts",
                    "implications",
                ]
            )
            .with_metadata(
                difficulty_level=5, tags=["comprehensive", "advanced", "scholarly"]
            )
            .build()
        )

        # Concept analysis template
        cls.register_template(
            TemplateBuilder("philosophical_concept")
            .with_description("Template for analyzing philosophical concepts")
            .with_source_type(PhilosophySourceType.ESSAY)
            .add_fields(
                ["concept_name", "definition", "historical_development"], required=True
            )
            .add_fields(
                [
                    "etymology",
                    "related_concepts",
                    "distinctions",
                    "applications",
                    "criticisms",
                    "contemporary_usage",
                ]
            )
            .with_metadata(
                difficulty_level=4, tags=["concept", "analysis", "definition"]
            )
            .build()
        )

        cls._initialized = True

    @classmethod
    def register_template(cls, template: ExtractionTemplate) -> None:
        """Register a template in the library"""
        cls._templates[template.name] = template

    @classmethod
    def get_template(cls, template_name: str) -> Optional[ExtractionTemplate]:
        """Get a template by name"""
        cls._initialize_templates()
        return cls._templates.get(template_name)

    @classmethod
    def get_all_templates(cls) -> Dict[str, ExtractionTemplate]:
        """Get all available templates"""
        cls._initialize_templates()
        return cls._templates.copy()

    @classmethod
    def get_templates_by_source_type(
        cls, source_type: Union[str, PhilosophySourceType]
    ) -> Dict[str, ExtractionTemplate]:
        """Get templates for a specific source type"""
        cls._initialize_templates()
        if isinstance(source_type, str):
            source_type = PhilosophySourceType(source_type)

        return {
            name: template
            for name, template in cls._templates.items()
            if template.source_type == source_type
        }

    @classmethod
    def get_templates_by_difficulty(
        cls, max_difficulty: int
    ) -> Dict[str, ExtractionTemplate]:
        """Get templates up to a certain difficulty level"""
        cls._initialize_templates()
        return {
            name: template
            for name, template in cls._templates.items()
            if template.metadata.difficulty_level <= max_difficulty
        }

    @classmethod
    def search_templates(cls, query: str) -> Dict[str, ExtractionTemplate]:
        """Search templates by name, description, or tags"""
        cls._initialize_templates()
        query_lower = query.lower()
        results = {}

        for name, template in cls._templates.items():
            # Search in name
            if query_lower in name.lower():
                results[name] = template
                continue

            # Search in description
            if query_lower in template.description.lower():
                results[name] = template
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in template.metadata.tags):
                results[name] = template

        return results

    @classmethod
    def create_custom_template(
        cls,
        name: str,
        description: str,
        source_type: Union[str, PhilosophySourceType],
        fields: List[str],
        required_fields: Optional[List[str]] = None,
        **metadata_kwargs,
    ) -> ExtractionTemplate:
        """Create and register a custom template"""
        builder = (
            TemplateBuilder(name)
            .with_description(description)
            .with_source_type(source_type)
        )

        # Add fields
        for field in fields:
            is_required = required_fields and field in required_fields
            builder.add_field(field, required=is_required)

        # Add metadata
        if metadata_kwargs:
            builder.with_metadata(**metadata_kwargs)

        # Build and register
        template = builder.build()
        cls.register_template(template)

        return template
