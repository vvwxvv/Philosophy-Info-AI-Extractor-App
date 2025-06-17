"""
Philosophy Prompt Structure Module
Handles prompt building and structuring for philosophy extraction
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

# Import types
from extractors.types import (
    PhilosophySourceType,
    ExtractionDepth,
    TargetAudience,
    ExtractionMode,
    PhilosophicalCategory,
)

# Import only what we need from fields
from extractors.fields import philosophy_field_registry

# Import from templates
from prompts.templates import (
    PhilosophyExtractionTemplate,
    philosophy_template_library,
)

# Import from config and generator
from extractors.config import PhilosophyExtractorConfig
from extractors.generator import (
    PhilosophyExtractionPromptGenerator,
    AdvancedPhilosophyPromptBuilder,
)

logger = logging.getLogger(__name__)


class PhilosophyPromptBuilder:
    """Main prompt builder for philosophy extraction"""

    def __init__(self):
        self.prompt_generator = PhilosophyExtractionPromptGenerator()
        self.advanced_builder = AdvancedPhilosophyPromptBuilder()

    @staticmethod
    def build_prompt(config: PhilosophyExtractorConfig) -> str:
        """
        Build a philosophy extraction prompt from configuration

        Args:
            config: Extraction configuration

        Returns:
            Generated prompt string
        """
        generator = PhilosophyExtractionPromptGenerator()
        return generator.generate(config)

    def build_advanced_prompt(
        self, text: str, template_name: Optional[str] = None, **kwargs
    ) -> str:
        """
        Build an advanced prompt with custom parameters

        Args:
            text: Text to analyze
            template_name: Template to use
            **kwargs: Additional parameters

        Returns:
            Generated prompt string
        """
        return self.advanced_builder.build_prompt(
            text=text, template_name=template_name, **kwargs
        )

    def build_from_template(
        self,
        template: PhilosophyExtractionTemplate,
        text: str,
        language: str = "mixed",
        **kwargs,
    ) -> str:
        """
        Build prompt from a specific template

        Args:
            template: Template to use
            text: Text to analyze
            language: Target language
            **kwargs: Additional parameters

        Returns:
            Generated prompt string
        """
        config = PhilosophyExtractorConfig(
            template=template,
            source_type=template.source_type,
            language=language,
            **kwargs,
        )

        prompt = self.build_prompt(config)
        return f"{prompt}\n\n--- TEXT TO ANALYZE ---\n{text}"

    def build_validation_prompt(
        self, extracted_data: Dict[str, Any], original_text: str, template_name: str
    ) -> str:
        """Build a validation prompt"""
        return self.advanced_builder.build_validation_prompt(
            extracted_data=extracted_data,
            original_text=original_text,
            template_name=template_name,
        )

    def build_enhancement_prompt(
        self, partial_data: Dict[str, Any], missing_fields: List[str], text: str
    ) -> str:
        """Build an enhancement prompt"""
        return self.advanced_builder.build_enhancement_prompt(
            partial_data=partial_data, missing_fields=missing_fields, text=text
        )


def create_philosophy_extraction_prompt(
    text: str,
    source_type: Union[str, PhilosophySourceType] = PhilosophySourceType.ESSAY,
    extraction_mode: Union[str, ExtractionMode] = ExtractionMode.COMPREHENSIVE,
    depth_level: Union[str, ExtractionDepth] = ExtractionDepth.DETAILED,
    target_audience: Union[str, TargetAudience] = TargetAudience.ACADEMIC,
    categories: Optional[List[Union[str, PhilosophicalCategory]]] = None,
    template_name: Optional[str] = None,
    language: str = "mixed",
    **kwargs,
) -> str:
    """
    Create a philosophy extraction prompt with specified parameters

    Args:
        text: The philosophical text to analyze
        source_type: Type of philosophical source
        extraction_mode: Mode of extraction
        depth_level: Depth of analysis
        target_audience: Target audience for extraction
        categories: Philosophical categories to focus on
        template_name: Specific template to use
        language: Language for extraction
        **kwargs: Additional parameters

    Returns:
        Generated extraction prompt
    """
    # Convert string enums to proper types
    if isinstance(source_type, str):
        source_type = PhilosophySourceType(source_type)
    if isinstance(extraction_mode, str):
        extraction_mode = ExtractionMode(extraction_mode)
    if isinstance(depth_level, str):
        depth_level = ExtractionDepth(depth_level)
    if isinstance(target_audience, str):
        target_audience = TargetAudience(target_audience)

    # Convert category strings to enums
    if categories:
        category_enums = []
        for cat in categories:
            if isinstance(cat, str):
                try:
                    category_enums.append(PhilosophicalCategory(cat))
                except ValueError:
                    logger.warning(f"Invalid category: {cat}")
            else:
                category_enums.append(cat)
        categories = category_enums

    # Create configuration
    config = PhilosophyExtractorConfig(
        source_type=source_type,
        extraction_mode=extraction_mode,
        extraction_depth=depth_level,
        target_audience=target_audience,
        categories_focus=categories or [],
        language=language,
        **kwargs,
    )

    # If template specified, use it
    if template_name:
        template = philosophy_template_library.get_template(template_name)
        if template:
            config.template = template

    # Build prompt
    builder = PhilosophyPromptBuilder()
    prompt = builder.build_prompt(config)

    # Add the text to analyze
    return f"{prompt}\n\n--- TEXT TO ANALYZE ---\n{text}"


@dataclass
class PromptSection:
    """Represents a section of a structured prompt"""

    title: str
    content: str
    priority: int = 5
    required: bool = False

    def render(self) -> str:
        """Render the section as formatted text"""
        header = f"## {self.title}"
        if self.required:
            header += " [REQUIRED]"

        return f"{header}\n{self.content}\n"


class StructuredPromptBuilder:
    """Builder for creating structured prompts with sections"""

    def __init__(self):
        self.sections: List[PromptSection] = []
        self.metadata: Dict[str, Any] = {}

    def add_section(
        self, title: str, content: str, priority: int = 5, required: bool = False
    ) -> "StructuredPromptBuilder":
        """Add a section to the prompt"""
        self.sections.append(
            PromptSection(
                title=title, content=content, priority=priority, required=required
            )
        )
        return self

    def set_metadata(self, key: str, value: Any) -> "StructuredPromptBuilder":
        """Set metadata for the prompt"""
        self.metadata[key] = value
        return self

    def build(self) -> str:
        """Build the final prompt"""
        # Sort sections by priority
        sorted_sections = sorted(self.sections, key=lambda s: s.priority, reverse=True)

        # Build header
        header_lines = ["# PHILOSOPHY EXTRACTION PROMPT"]

        # Add metadata
        if self.metadata:
            header_lines.append("\n## METADATA")
            for key, value in self.metadata.items():
                header_lines.append(f"- {key}: {value}")

        # Add sections
        section_text = "\n".join(section.render() for section in sorted_sections)

        return "\n".join(header_lines) + "\n\n" + section_text


def build_structured_philosophy_prompt(
    config: PhilosophyExtractorConfig, text: str
) -> str:
    """
    Build a structured philosophy prompt from configuration

    Args:
        config: Extraction configuration
        text: Text to analyze

    Returns:
        Structured prompt string
    """
    builder = StructuredPromptBuilder()

    # Add metadata
    builder.set_metadata("Source Type", config.source_type.value)
    builder.set_metadata("Extraction Mode", config.extraction_mode.value)
    builder.set_metadata("Depth Level", config.extraction_depth.value)
    builder.set_metadata("Target Audience", config.target_audience.value)
    builder.set_metadata("Language", config.language)

    # Add instructions
    builder.add_section(
        "INSTRUCTIONS",
        "Extract structured philosophical information from the provided text "
        "according to the template and guidelines below.",
        priority=10,
        required=True,
    )

    # Add field descriptions
    if config.template:
        field_descriptions = []
        for field_name in config.template.fields:
            field = philosophy_field_registry.get_field(field_name)
            if field:
                desc = f"- **{field.name}**: {field.description}"
                if field.required:
                    desc += " [REQUIRED]"
                field_descriptions.append(desc)

        builder.add_section(
            "FIELD DESCRIPTIONS",
            "\n".join(field_descriptions),
            priority=8,
            required=True,
        )

    # Add extraction guidelines
    guidelines = config.get_extraction_guidelines()
    if guidelines:
        builder.add_section(
            "EXTRACTION GUIDELINES", "\n".join(f"- {g}" for g in guidelines), priority=7
        )

    # Add quality standards
    standards = config.get_quality_standards()
    if standards:
        builder.add_section(
            "QUALITY STANDARDS",
            "\n".join(f"- {k}: {v}" for k, v in standards.items()),
            priority=6,
        )

    # Add the text
    builder.add_section("TEXT TO ANALYZE", text, priority=1, required=True)

    return builder.build()


# Export main functions and classes
__all__ = [
    "PhilosophyPromptBuilder",
    "create_philosophy_extraction_prompt",
    "PromptSection",
    "StructuredPromptBuilder",
    "build_structured_philosophy_prompt",
]
