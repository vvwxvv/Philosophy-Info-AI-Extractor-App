"""Enhanced prompt system with template composition and validation"""

from typing import Dict, List, Any, Optional, Set
import json
from prompts.prompt_elements import (
    PHILOSOPHY_FIELD_DESCRIPTIONS,
    PHILOSOPHY_OUTPUT_FORMAT,
    PHILOSOPHY_CORE_GUIDELINES,
    PHILOSOPHY_SYSTEM_ROLE,
)


class PromptTemplate:
    """Base class for prompt templates with validation"""

    def __init__(
        self,
        name: str,
        description: str,
        system_role: str,
        guidelines: List[str],
        output_format: Dict[str, Any],
        field_descriptions: str = "",
        required_sections: Set[str] = None,
    ):
        self.name = name
        self.description = description
        self.system_role = system_role
        self.guidelines = guidelines
        self.output_format = output_format
        self.field_descriptions = field_descriptions
        self.required_sections = required_sections or set()

        self._validate()

    def _validate(self):
        """Validate template structure"""
        if not self.name:
            raise ValueError("Template must have a name")
        if not self.system_role:
            raise ValueError("Template must have a system role")
        if not self.guidelines:
            raise ValueError("Template must have guidelines")

    def render(self, **kwargs) -> str:
        """Render the template with provided values"""
        # Build the complete prompt
        sections = []

        # System role
        sections.append(f"SYSTEM ROLE:\n{self.system_role}\n")

        # Guidelines
        sections.append("GUIDELINES:")
        for guideline in self.guidelines:
            sections.append(f"- {guideline}")
        sections.append("")

        # Field descriptions if provided
        if self.field_descriptions or kwargs.get("field_descriptions"):
            field_desc = kwargs.get("field_descriptions", self.field_descriptions)
            sections.append(f"FIELDS TO EXTRACT:\n{field_desc}\n")

        # Output format
        sections.append(f"OUTPUT FORMAT:\n{json.dumps(self.output_format, indent=2)}\n")

        # Additional sections from kwargs
        for key, value in kwargs.items():
            if key not in ["field_descriptions"] and value:
                section_name = key.replace("_", " ").upper()
                sections.append(f"{section_name}:\n{value}\n")

        return "\n".join(sections)

    def compose_with(self, other: "PromptTemplate") -> "PromptTemplate":
        """Compose this template with another"""
        return PromptTemplate(
            name=f"{self.name}+{other.name}",
            description=f"{self.description} combined with {other.description}",
            system_role=self.system_role,  # Keep original system role
            guidelines=list(
                set(self.guidelines + other.guidelines)
            ),  # Merge guidelines
            output_format={
                **self.output_format,
                **other.output_format,
            },  # Merge outputs
            field_descriptions=f"{self.field_descriptions}\n{other.field_descriptions}",
            required_sections=self.required_sections.union(other.required_sections),
        )


class CategoryPromptTemplate(PromptTemplate):
    """Template for specific philosophical categories"""

    def __init__(
        self,
        name: str,
        description: str,
        system_role: str,
        guidelines: List[str],
        output_format: Dict[str, Any],
        field_descriptions: str = "",
        required_sections: Set[str] = None,
        category: str = "",
        category_guidelines: List[str] = None,
    ):
        # Combine core guidelines with category-specific ones
        all_guidelines = PHILOSOPHY_CORE_GUIDELINES + (category_guidelines or [])

        # Call parent constructor
        super().__init__(
            name=name,
            description=description,
            system_role=system_role,
            guidelines=all_guidelines,
            output_format=output_format,
            field_descriptions=field_descriptions,
            required_sections=required_sections,
        )

        self.category = category
        self.category_guidelines = category_guidelines or []


# Enhanced template factory
class PhilosophyPromptFactory:
    """Factory for creating philosophical prompt templates"""

    # Template cache
    _templates: Dict[str, PromptTemplate] = {}

    @classmethod
    def create_general_template(cls) -> PromptTemplate:
        """Create general philosophy extraction template"""
        if "general" not in cls._templates:
            cls._templates["general"] = PromptTemplate(
                name="general_philosophy",
                description="General philosophical text analysis",
                system_role=PHILOSOPHY_SYSTEM_ROLE,
                guidelines=PHILOSOPHY_CORE_GUIDELINES,
                output_format=PHILOSOPHY_OUTPUT_FORMAT,
                field_descriptions=PHILOSOPHY_FIELD_DESCRIPTIONS.get("general", ""),
            )
        return cls._templates["general"]

    @classmethod
    def create_category_template(
        cls, category: str, guidelines: List[str]
    ) -> CategoryPromptTemplate:
        """Create category-specific template"""
        cache_key = f"category_{category}"
        if cache_key not in cls._templates:
            cls._templates[cache_key] = CategoryPromptTemplate(
                name=f"{category}_philosophy",
                description=f"Analysis for {category} philosophy",
                system_role=PHILOSOPHY_SYSTEM_ROLE,
                guidelines=PHILOSOPHY_CORE_GUIDELINES,
                output_format=PHILOSOPHY_OUTPUT_FORMAT,
                field_descriptions=PHILOSOPHY_FIELD_DESCRIPTIONS.get(category, ""),
                category=category,
                category_guidelines=guidelines,
            )
        return cls._templates[cache_key]

    @classmethod
    def create_composite_template(cls, categories: List[str]) -> PromptTemplate:
        """Create a composite template from multiple categories"""
        cache_key = f"composite_{'_'.join(sorted(categories))}"
        if cache_key not in cls._templates:
            # Start with general template
            composite = cls.create_general_template()

            # Compose with each category template
            for category in categories:
                if category in PHILOSOPHY_FIELD_DESCRIPTIONS:
                    cat_template = cls.create_category_template(category, [])
                    composite = composite.compose_with(cat_template)

            composite.name = f"composite_{'+'.join(categories)}"
            cls._templates[cache_key] = composite

        return cls._templates[cache_key]

    @classmethod
    def get_template(cls, name: str) -> Optional[PromptTemplate]:
        """Get a template by name"""
        return cls._templates.get(name)

    @classmethod
    def list_templates(cls) -> List[Dict[str, str]]:
        """List all available templates"""
        return [
            {
                "name": template.name,
                "description": template.description,
                "type": type(template).__name__,
            }
            for template in cls._templates.values()
        ]


# Create specific category templates
from prompts.prompt_elements import (
    ETHICAL_PHILOSOPHY_GUIDELINES,
    METAPHYSICAL_PHILOSOPHY_GUIDELINES,
    EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES,
    AESTHETIC_PHILOSOPHY_GUIDELINES,
    LANGUAGE_PHILOSOPHY_GUIDELINES,
    MIND_PHILOSOPHY_GUIDELINES,
    LOGIC_PHILOSOPHY_GUIDELINES,
    POLITICAL_PHILOSOPHY_GUIDELINES,
)

# Register category templates
PhilosophyPromptFactory.create_category_template(
    "ethical", ETHICAL_PHILOSOPHY_GUIDELINES
)
PhilosophyPromptFactory.create_category_template(
    "metaphysical", METAPHYSICAL_PHILOSOPHY_GUIDELINES
)
PhilosophyPromptFactory.create_category_template(
    "epistemological", EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES
)
PhilosophyPromptFactory.create_category_template(
    "aesthetic", AESTHETIC_PHILOSOPHY_GUIDELINES
)
PhilosophyPromptFactory.create_category_template(
    "language", LANGUAGE_PHILOSOPHY_GUIDELINES
)
PhilosophyPromptFactory.create_category_template("mind", MIND_PHILOSOPHY_GUIDELINES)
PhilosophyPromptFactory.create_category_template("logic", LOGIC_PHILOSOPHY_GUIDELINES)
PhilosophyPromptFactory.create_category_template(
    "political", POLITICAL_PHILOSOPHY_GUIDELINES
)


# Backward compatibility
PHILOSOPHY_INFO_EXTRACT_PROMPT = (
    PhilosophyPromptFactory.create_general_template().render()
)

# Dictionary of all prompt templates for backward compatibility
PHILOSOPHY_PROMPT_TEMPLATES = {
    "general": PHILOSOPHY_INFO_EXTRACT_PROMPT,
    "ethical": PhilosophyPromptFactory.create_category_template(
        "ethical", ETHICAL_PHILOSOPHY_GUIDELINES
    ).render(),
    "metaphysical": PhilosophyPromptFactory.create_category_template(
        "metaphysical", METAPHYSICAL_PHILOSOPHY_GUIDELINES
    ).render(),
    "epistemological": PhilosophyPromptFactory.create_category_template(
        "epistemological", EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES
    ).render(),
    "aesthetic": PhilosophyPromptFactory.create_category_template(
        "aesthetic", AESTHETIC_PHILOSOPHY_GUIDELINES
    ).render(),
    "language": PhilosophyPromptFactory.create_category_template(
        "language", LANGUAGE_PHILOSOPHY_GUIDELINES
    ).render(),
    "mind": PhilosophyPromptFactory.create_category_template(
        "mind", MIND_PHILOSOPHY_GUIDELINES
    ).render(),
    "logic": PhilosophyPromptFactory.create_category_template(
        "logic", LOGIC_PHILOSOPHY_GUIDELINES
    ).render(),
    "political": PhilosophyPromptFactory.create_category_template(
        "political", POLITICAL_PHILOSOPHY_GUIDELINES
    ).render(),
}
