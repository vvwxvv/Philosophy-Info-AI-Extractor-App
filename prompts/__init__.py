"""
Prompts Package for Philosophy Extraction System

This package provides comprehensive prompt generation and template management
for philosophical text analysis and extraction.
"""

# Core prompt templates and factory
from .templates import (
    PhilosophyExtractionTemplate,
    PhilosophyTemplateLibrary,
    philosophy_template_library,
)

# Enhanced template library with validation
from .template_library import (
    TemplateMetadata,
    ExtractionTemplate,
    TemplateBuilder,
    PhilosophyTemplateLibrary as EnhancedTemplateLibrary,
    PhilosophySourceType,
)

# Advanced prompt system
from .prompts import (
    PromptTemplate,
    CategoryPromptTemplate,
    PhilosophyPromptFactory,
    PHILOSOPHY_INFO_EXTRACT_PROMPT,
    PHILOSOPHY_PROMPT_TEMPLATES,
)

# Prompt structure and builders
from .prompt_structure import (
    PhilosophyPromptBuilder,
    create_philosophy_extraction_prompt,
    PromptSection,
    StructuredPromptBuilder,
    build_structured_philosophy_prompt,
)

# Prompt elements and guidelines
from .prompt_elements import (
    # Core guidelines
    PHILOSOPHY_CORE_GUIDELINES,
    PHILOSOPHY_SYSTEM_ROLE,
    PHILOSOPHY_OUTPUT_FORMAT,
    PHILOSOPHY_FIELD_DESCRIPTIONS,
    PHILOSOPHY_TEMPLATES,
    # Category-specific guidelines
    ETHICAL_PHILOSOPHY_GUIDELINES,
    METAPHYSICAL_PHILOSOPHY_GUIDELINES,
    EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES,
    AESTHETIC_PHILOSOPHY_GUIDELINES,
    LANGUAGE_PHILOSOPHY_GUIDELINES,
    MIND_PHILOSOPHY_GUIDELINES,
    LOGIC_PHILOSOPHY_GUIDELINES,
    POLITICAL_PHILOSOPHY_GUIDELINES,
    # General extraction elements
    CORE_GUIDELINES,
    SYSTEM_ROLE,
    OUTPUT_FORMATS,
    INSTRUCTIONS,
)

# Utility function
from .print_prompt_section import print_prompt_section

# Version info
__version__ = "1.0.0"
__author__ = "Philosophy Extraction System"

# Main exports
__all__ = [
    # Template classes
    "PhilosophyExtractionTemplate",
    "PhilosophyTemplateLibrary",
    "TemplateMetadata",
    "ExtractionTemplate",
    "TemplateBuilder",
    "EnhancedTemplateLibrary",
    # Prompt classes
    "PromptTemplate",
    "CategoryPromptTemplate",
    "PhilosophyPromptFactory",
    # Builder classes
    "PhilosophyPromptBuilder",
    "PromptSection",
    "StructuredPromptBuilder",
    # Main functions
    "create_philosophy_extraction_prompt",
    "build_structured_philosophy_prompt",
    "print_prompt_section",
    # Template instances
    "philosophy_template_library",
    # Ready-to-use prompts
    "PHILOSOPHY_INFO_EXTRACT_PROMPT",
    "PHILOSOPHY_PROMPT_TEMPLATES",
    # Core elements
    "PHILOSOPHY_CORE_GUIDELINES",
    "PHILOSOPHY_SYSTEM_ROLE",
    "PHILOSOPHY_OUTPUT_FORMAT",
    "PHILOSOPHY_FIELD_DESCRIPTIONS",
    "PHILOSOPHY_TEMPLATES",
    # Category guidelines
    "ETHICAL_PHILOSOPHY_GUIDELINES",
    "METAPHYSICAL_PHILOSOPHY_GUIDELINES",
    "EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES",
    "AESTHETIC_PHILOSOPHY_GUIDELINES",
    "LANGUAGE_PHILOSOPHY_GUIDELINES",
    "MIND_PHILOSOPHY_GUIDELINES",
    "LOGIC_PHILOSOPHY_GUIDELINES",
    "POLITICAL_PHILOSOPHY_GUIDELINES",
    # General elements
    "CORE_GUIDELINES",
    "SYSTEM_ROLE",
    "OUTPUT_FORMATS",
    "INSTRUCTIONS",
    # Enums and types
    "PhilosophySourceType",
]


# Convenience functions for quick access
def get_template(name: str):
    """Get a template by name from the global library"""
    return philosophy_template_library.get_template(name)


def create_prompt(template_name: str, text: str, **kwargs):
    """Create a prompt using a template name"""
    return create_philosophy_extraction_prompt(
        text=text, template_name=template_name, **kwargs
    )


def list_templates():
    """List all available templates"""
    return list(philosophy_template_library.get_all_templates().keys())


def get_category_guidelines(category: str):
    """Get guidelines for a specific philosophical category"""
    guidelines_map = {
        "ethical": ETHICAL_PHILOSOPHY_GUIDELINES,
        "metaphysical": METAPHYSICAL_PHILOSOPHY_GUIDELINES,
        "epistemological": EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES,
        "aesthetic": AESTHETIC_PHILOSOPHY_GUIDELINES,
        "language": LANGUAGE_PHILOSOPHY_GUIDELINES,
        "mind": MIND_PHILOSOPHY_GUIDELINES,
        "logic": LOGIC_PHILOSOPHY_GUIDELINES,
        "political": POLITICAL_PHILOSOPHY_GUIDELINES,
    }
    return guidelines_map.get(category.lower(), [])


# Add convenience functions to __all__
__all__.extend(
    [
        "get_template",
        "create_prompt",
        "list_templates",
        "get_category_guidelines",
    ]
)

# Package-level documentation
__doc__ = """
Philosophy Prompts Package

This package provides a comprehensive system for generating prompts for philosophical
text analysis and extraction. It includes:

1. Template Management:
   - PhilosophyExtractionTemplate: Core template class
   - PhilosophyTemplateLibrary: Template registry and management
   - TemplateBuilder: Builder pattern for creating templates

2. Prompt Generation:
   - PhilosophyPromptBuilder: Main prompt builder
   - PromptTemplate: Advanced prompt templates with composition
   - CategoryPromptTemplate: Category-specific templates

3. Structured Prompts:
   - StructuredPromptBuilder: Build prompts with sections
   - PromptSection: Individual prompt sections

4. Guidelines and Elements:
   - Core philosophy guidelines for all categories
   - Category-specific guidelines (ethics, metaphysics, etc.)
   - Output formats and field descriptions

Usage Examples:

    # Create a basic prompt
    from prompts import create_philosophy_extraction_prompt
    prompt = create_philosophy_extraction_prompt(text, template_name="philosophy_basic")

    # Get a template
    from prompts import get_template
    template = get_template("comprehensive")

    # Use the prompt builder
    from prompts import PhilosophyPromptBuilder
    builder = PhilosophyPromptBuilder()
    prompt = builder.build_advanced_prompt(text, template_name="ethical")

    # Create custom templates
    from prompts import TemplateBuilder, PhilosophySourceType
    template = (TemplateBuilder("my_template")
                .with_description("Custom template")
                .with_source_type(PhilosophySourceType.ESSAY)
                .add_field("main_argument", required=True)
                .build())
"""
