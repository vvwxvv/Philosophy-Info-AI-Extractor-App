"""
Enhanced Philosophy Extraction Prompt Generator
Integrates with new philosophy modules for advanced prompt generation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re

# Import from enhanced modules
from extractors.config import PhilosophyExtractorConfig
from prompts.templates import (
    PhilosophyExtractionTemplate,
    philosophy_template_library,
)
from extractors.fields import (
    PhilosophyExtractionField,
    philosophy_field_registry,
)
from extractors.types import (
    ExtractionDepth,
    TargetAudience,
    ExtractionMode,
    PhilosophicalCategory,
)

logger = logging.getLogger(__name__)


class IPromptGenerator(ABC):
    """Interface for prompt generators"""

    @abstractmethod
    def generate(self, config: Any, context: Dict[str, Any] = None) -> str:
        """Generate extraction prompt"""
        pass


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    max_hints_per_field: int = 3
    include_examples: bool = True
    group_by_category: bool = True
    include_metadata: bool = True
    include_instructions: bool = True
    include_quality_standards: bool = True
    output_format: str = "json"
    language_style: str = "academic"  # academic, accessible, technical
    emphasis_markers: bool = True  # Use formatting for emphasis


class PhilosophyExtractionPromptGenerator(IPromptGenerator):
    """Advanced prompt generator for philosophical content extraction"""

    def __init__(
        self,
        custom_templates: Optional[Dict[str, str]] = None,
        prompt_version: str = "v3",
        config: Optional[PromptConfig] = None,
    ):
        """
        Initialize the prompt generator

        Args:
            custom_templates: Optional custom prompt templates
            prompt_version: Version of the prompt template to use
            config: Optional prompt configuration
        """
        self.custom_templates = custom_templates or {}
        self.prompt_version = prompt_version
        self.config = config or PromptConfig()
        self._prompt_cache = {}
        self._pattern_cache = {}

        logger.info(
            "Initialized PhilosophyExtractionPromptGenerator with version %s",
            prompt_version,
        )

    def generate(
        self,
        config: PhilosophyExtractorConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate extraction prompt based on configuration

        Args:
            config: Philosophy extraction configuration
            context: Optional context dictionary

        Returns:
            Generated prompt string
        """
        try:
            context = context or {}

            # Create cache key
            cache_key = self._create_cache_key(config)

            # Check cache first
            if cache_key in self._prompt_cache:
                logger.debug("Using cached prompt for key: %s", cache_key)
                return self._prompt_cache[cache_key]

            # Use the enhanced prompt builder
            # Create a temporary instance to build the prompt
            temp_builder = AdvancedPhilosophyPromptBuilder()
            prompt = temp_builder.build_prompt(
                text="",  # We'll add the text later
                template_name=(
                    config.template.name
                    if hasattr(config.template, "name")
                    else str(config.template)
                ),
                language=config.language,
                extraction_mode=config.extraction_mode.value,
                depth_level=config.extraction_depth.value,
                target_audience=config.target_audience.value,
                categories=(
                    [cat.value for cat in config.categories_focus]
                    if config.categories_focus
                    else None
                ),
            )

            # Add any custom enhancements
            if self.config.include_instructions:
                prompt = self._add_detailed_instructions(prompt, config)

            if self.config.include_quality_standards:
                prompt = self._add_quality_standards(prompt, config)

            # Cache the generated prompt
            self._prompt_cache[cache_key] = prompt

            logger.debug("Generated new prompt for configuration")
            return prompt

        except Exception as e:
            logger.error("Failed to generate prompt: %s", str(e), exc_info=True)
            raise ValueError(f"Prompt generation failed: {str(e)}")

    def generate_field_descriptions(
        self, fields: Union[Dict[str, PhilosophyExtractionField], List[str]]
    ) -> str:
        """
        Generate formatted field descriptions for prompt

        Args:
            fields: Dictionary of field names to field objects, or list of field names

        Returns:
            Formatted field descriptions string
        """
        try:
            # Convert list of field names to dictionary of field objects if needed
            if isinstance(fields, list):
                field_dict = {}
                for field_name in fields:
                    field = philosophy_field_registry.get_field(field_name)
                    if field:
                        field_dict[field_name] = field
                fields = field_dict

            if not fields:
                return ""

            if self.config.group_by_category:
                return self._generate_categorized_descriptions(fields)
            else:
                return self._generate_flat_descriptions(fields)

        except Exception as e:
            logger.error(
                "Failed to generate field descriptions: %s", str(e), exc_info=True
            )
            return ""

    def _generate_categorized_descriptions(
        self, fields: Dict[str, PhilosophyExtractionField]
    ) -> str:
        """Generate field descriptions grouped by category"""
        categorized = {}
        uncategorized = []

        # Group fields by category
        for field_name, field in fields.items():
            if field.category:
                if field.category not in categorized:
                    categorized[field.category] = []
                categorized[field.category].append((field_name, field))
            else:
                uncategorized.append((field_name, field))

        descriptions = []

        # Add categorized fields
        for category in PhilosophicalCategory:
            if category in categorized:
                descriptions.append(f"\n### {category.value.upper()} FIELDS")

                # Sort by priority
                cat_fields = sorted(categorized[category], key=lambda x: x[1].priority)

                for field_name, field in cat_fields:
                    descriptions.append(self._format_field_description(field))

        # Add uncategorized fields
        if uncategorized:
            descriptions.append("\n### GENERAL FIELDS")
            for field_name, field in sorted(uncategorized, key=lambda x: x[1].priority):
                descriptions.append(self._format_field_description(field))

        return "\n".join(descriptions)

    def _generate_flat_descriptions(
        self, fields: Dict[str, PhilosophyExtractionField]
    ) -> str:
        """Generate flat list of field descriptions"""
        # Sort fields by priority
        sorted_fields = sorted(fields.values(), key=lambda f: f.priority)

        descriptions = []
        for field in sorted_fields:
            descriptions.append(self._format_field_description(field))

        return "\n".join(descriptions)

    def _format_field_description(self, field: PhilosophyExtractionField) -> str:
        """Format a single field description"""
        parts = []

        # Field name and required status
        if field.required:
            if self.config.emphasis_markers:
                parts.append(f"- **{field.name}** [REQUIRED]:")
            else:
                parts.append(f"- {field.name} [REQUIRED]:")
        else:
            parts.append(f"- {field.name}:")

        # Description
        parts.append(f" {field.description}")

        # Add extraction hints
        if field.extraction_hints and self.config.max_hints_per_field > 0:
            hints = field.extraction_hints[: self.config.max_hints_per_field]
            hints_str = " | ".join(hints)
            parts.append(f"\n  Hints: {hints_str}")

        # Add examples if configured
        if self.config.include_examples and field.examples:
            # Get examples in the appropriate language
            lang_examples = field.examples.get("en", [])
            if lang_examples:
                examples_str = ", ".join(lang_examples[:3])
                parts.append(f"\n  Examples: {examples_str}")

        # Add keywords if present
        if field.contextual_keywords:
            keywords_str = ", ".join(field.contextual_keywords[:5])
            parts.append(f"\n  Keywords: {keywords_str}")

        return "".join(parts)

    def _create_cache_key(self, config: PhilosophyExtractorConfig) -> str:
        """Create a cache key for the configuration"""
        key_parts = [
            (
                config.template.name
                if isinstance(config.template, PhilosophyExtractionTemplate)
                else str(config.template)
            ),
            config.language,
            config.extraction_depth.value,
            config.extraction_mode.value,
            config.target_audience.value,
            ",".join(sorted([cat.value for cat in config.categories_focus])),
            str(config.confidence_threshold),
            str(self.prompt_version),
        ]
        return "_".join(key_parts)

    def _add_detailed_instructions(
        self, prompt: str, config: PhilosophyExtractorConfig
    ) -> str:
        """Add detailed extraction instructions"""
        instructions = [
            "\n\n## DETAILED EXTRACTION INSTRUCTIONS:",
            f"1. Read the entire text carefully, noting its {config.source_type.description}",
            "2. Identify the main philosophical thesis or argument first",
            "3. Extract concepts in order of philosophical significance",
            "4. Preserve the logical structure of arguments",
            "5. Note implicit assumptions and unstated premises",
            "6. Distinguish author's position from positions being discussed",
            "7. Maintain philosophical terminology precision",
        ]

        # Add mode-specific instructions
        mode_instructions = {
            ExtractionMode.CRITICAL: [
                "8. Evaluate argument validity and soundness",
                "9. Identify logical fallacies if present",
                "10. Assess the strength of evidence provided",
            ],
            ExtractionMode.HISTORICAL: [
                "8. Place ideas in historical philosophical context",
                "9. Identify influences and predecessors",
                "10. Note how ideas evolved over time",
            ],
            ExtractionMode.COMPARATIVE: [
                "8. Compare positions systematically",
                "9. Identify points of agreement and disagreement",
                "10. Synthesize comparative insights",
            ],
        }

        if config.extraction_mode in mode_instructions:
            instructions.extend(mode_instructions[config.extraction_mode])

        return prompt + "\n".join(instructions)

    def _add_quality_standards(
        self, prompt: str, config: PhilosophyExtractorConfig
    ) -> str:
        """Add quality standards section"""
        standards = [
            "\n\n## QUALITY STANDARDS:",
            f"- Confidence Threshold: {config.confidence_threshold}",
            f"- Extraction Depth: {config.extraction_depth.description}",
            f"- Target Audience: {config.target_audience.formality_level}",
            "- Accuracy: All extracted information must be traceable to the source",
            "- Completeness: Extract all relevant information for required fields",
            "- Consistency: Use consistent terminology throughout",
            "- Context: Provide sufficient context for understanding",
        ]

        if config.preserve_original_language:
            standards.append(
                "- Terminology: Preserve original philosophical terms with translations"
            )

        return prompt + "\n".join(standards)


class PhilosophyTemplateMatcher:
    """Intelligent template matching for philosophical texts"""

    def __init__(self):
        self.keyword_patterns = self._build_keyword_patterns()
        self.structure_patterns = self._build_structure_patterns()
        self._match_cache = {}

    def _build_keyword_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Build keyword patterns for each template"""
        patterns = {
            "philosophical_argument": {
                "strong_indicators": [
                    "therefore",
                    "thus",
                    "hence",
                    "it follows",
                    "consequently",
                ],
                "moderate_indicators": [
                    "because",
                    "since",
                    "given that",
                    "premise",
                    "conclusion",
                ],
                "weak_indicators": ["argue", "claim", "assert", "maintain", "contend"],
            },
            "philosopher_profile": {
                "strong_indicators": [
                    "born",
                    "died",
                    "life",
                    "biography",
                    "works include",
                ],
                "moderate_indicators": [
                    "influenced by",
                    "student of",
                    "taught at",
                    "wrote",
                ],
                "weak_indicators": ["philosopher", "thinker", "author", "scholar"],
            },
            "philosophical_concept": {
                "strong_indicators": ["concept of", "definition", "means", "refers to"],
                "moderate_indicators": [
                    "understanding",
                    "interpretation",
                    "analysis of",
                ],
                "weak_indicators": ["idea", "notion", "principle", "theory"],
            },
            "philosophy_ethical": {
                "strong_indicators": [
                    "moral",
                    "ethical",
                    "ought",
                    "duty",
                    "obligation",
                ],
                "moderate_indicators": ["right", "wrong", "good", "evil", "virtue"],
                "weak_indicators": ["should", "value", "principle", "justice"],
            },
            "philosophy_metaphysical": {
                "strong_indicators": [
                    "being",
                    "existence",
                    "reality",
                    "substance",
                    "essence",
                ],
                "moderate_indicators": [
                    "ontology",
                    "nature of",
                    "fundamental",
                    "ultimate",
                ],
                "weak_indicators": ["real", "actual", "possible", "necessary"],
            },
        }
        return patterns

    def _build_structure_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build structural patterns for template matching"""
        patterns = {
            "philosophical_argument": [
                re.compile(r"premise\s*\d+:", re.IGNORECASE),
                re.compile(r"(first|second|third)ly,", re.IGNORECASE),
                re.compile(r"in conclusion|to conclude", re.IGNORECASE),
            ],
            "philosopher_profile": [
                re.compile(r"\b\d{4}\s*-\s*\d{4}\b"),  # Birth-death dates
                re.compile(r"born\s+(in|on)\s+", re.IGNORECASE),
                re.compile(r"(early|later)\s+life", re.IGNORECASE),
            ],
            "philosophical_concept": [
                re.compile(r"is\s+defined\s+as", re.IGNORECASE),
                re.compile(r"by\s+\w+\s+(?:I|we)\s+mean", re.IGNORECASE),
                re.compile(
                    r"three\s+(?:key\s+)?(?:aspects|components|elements)", re.IGNORECASE
                ),
            ],
        }
        return patterns

    def match_template(
        self, text: str, hint: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Match text with the most appropriate template

        Args:
            text: Input text to match
            hint: Optional hint about text type

        Returns:
            Tuple of (template_name, confidence_score, match_details)
        """
        # Check cache
        cache_key = hash(text[:500])  # Use first 500 chars for cache key
        if cache_key in self._match_cache:
            return self._match_cache[cache_key]

        text_lower = text.lower()
        scores = {}
        details = {}

        # Score each template
        for template_name, keywords in self.keyword_patterns.items():
            score = 0
            matched_keywords = {"strong": [], "moderate": [], "weak": []}

            # Check keyword matches
            for level, words in keywords.items():
                weight = {
                    "strong_indicators": 3,
                    "moderate_indicators": 2,
                    "weak_indicators": 1,
                }[level]
                for word in words:
                    if word in text_lower:
                        score += weight
                        matched_keywords[level.split("_")[0]].append(word)

            # Check structural patterns
            structure_matches = 0
            if template_name in self.structure_patterns:
                for pattern in self.structure_patterns[template_name]:
                    if pattern.search(text):
                        score += 2
                        structure_matches += 1

            # Apply hint bonus
            if hint and hint.lower() in template_name.lower():
                score *= 1.5

            scores[template_name] = score
            details[template_name] = {
                "keyword_matches": matched_keywords,
                "structure_matches": structure_matches,
                "raw_score": score,
            }

        # Normalize scores
        max_score = max(scores.values()) if scores else 1

        # Handle case where all scores are zero
        if max_score == 0:
            # If no matches found, default to comprehensive template
            best_template = ("comprehensive", 1.0)
            details["comprehensive"] = {"fallback": True, "reason": "no_matches_found"}
        else:
            normalized_scores = {k: v / max_score for k, v in scores.items()}

            # Get best match
            best_template = max(normalized_scores.items(), key=lambda x: x[1])

            # Add comprehensive template if no strong match
            if best_template[1] < 0.3:
                best_template = ("comprehensive", 1.0)
                details["comprehensive"] = {
                    "fallback": True,
                    "reason": "low_confidence",
                }

        result = (best_template[0], best_template[1], details[best_template[0]])

        # Cache result
        self._match_cache[cache_key] = result

        logger.info(
            f"Matched template '{best_template[0]}' with confidence {best_template[1]:.2f}"
        )

        return result


class AdvancedPhilosophyPromptBuilder:
    """High-level interface for building philosophical extraction prompts"""

    def __init__(self):
        self.prompt_generator = PhilosophyExtractionPromptGenerator()
        self.template_matcher = PhilosophyTemplateMatcher()
        self._template_cache = {}

    def build_prompt(
        self,
        text: str,
        template_name: Optional[str] = None,
        language: str = "mixed",
        additional_fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        extraction_mode: Optional[str] = None,
        categories: Optional[List[str]] = None,
        depth_level: Optional[str] = None,
        target_audience: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Build extraction prompt for the given philosophical text
        """
        # Auto-detect template if not provided
        if not template_name:
            template_name, confidence, match_details = (
                self.template_matcher.match_template(
                    text, hint=kwargs.get("text_type_hint")
                )
            )
            logger.info(
                f"Auto-detected template: {template_name} (confidence: {confidence:.2f})"
            )
            logger.debug(f"Match details: {match_details}")
        else:
            confidence = 1.0

        # Get template
        template = philosophy_template_library.get_template(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found, using default")
            template = philosophy_template_library.get_template("philosophy_basic")

        # Filter valid config parameters
        valid_config_params = {
            "template": template,
            "source_type": template.source_type,
            "language": language,
            "additional_fields": additional_fields or [],
            "exclude_fields": exclude_fields or [],
            "extraction_mode": (
                ExtractionMode(extraction_mode)
                if extraction_mode
                else ExtractionMode.COMPREHENSIVE
            ),
            "categories_focus": [
                PhilosophicalCategory(cat) for cat in (categories or [])
            ],
            "extraction_depth": (
                ExtractionDepth(depth_level)
                if depth_level
                else template.extraction_depth
            ),
            "target_audience": (
                TargetAudience(target_audience)
                if target_audience
                else TargetAudience.ACADEMIC
            ),
        }

        # Add valid kwargs that match config fields
        valid_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(PhilosophyExtractorConfig, key):
                valid_kwargs[key] = value

        # Create configuration
        config = PhilosophyExtractorConfig(**valid_config_params, **valid_kwargs)

        # Build the prompt directly here (no recursion)
        # Use PhilosophyExtractionPromptGenerator for field descriptions only
        field_descriptions = self.prompt_generator.generate_field_descriptions(
            template.fields
        )
        prompt = f"""
# {template.name.upper()} PHILOSOPHY EXTRACTION PROMPT

## INSTRUCTIONS:
Extract structured philosophical information from the provided text according to the following template and guidelines.

## FIELD DESCRIPTIONS:
{field_descriptions}

## EXTRACTION DEPTH: {config.extraction_depth.value}
## TARGET AUDIENCE: {config.target_audience.value}
## EXTRACTION MODE: {config.extraction_mode.value}
"""
        # Add any custom enhancements (instructions, quality standards)
        if self.prompt_generator.config.include_instructions:
            prompt += self.prompt_generator._add_detailed_instructions("", config)
        if self.prompt_generator.config.include_quality_standards:
            prompt += self.prompt_generator._add_quality_standards("", config)

        # Add text to analyze
        final_prompt = f"{prompt}\n\n--- TEXT TO ANALYZE ---\n\n{text}"
        return final_prompt

    def build_validation_prompt(
        self, extracted_data: Dict[str, Any], original_text: str, template_name: str
    ) -> str:
        """Build prompt for validating extracted data"""
        validation_prompt = f"""
## VALIDATION TASK

Review the following philosophical extraction for accuracy and completeness.

### ORIGINAL TEXT (excerpt):
{original_text[:500]}...

### EXTRACTED DATA:
{extracted_data}

### VALIDATION CRITERIA:
1. Verify all required fields are properly extracted
2. Check accuracy of philosophical terminology
3. Confirm logical structure preservation
4. Validate interpretations against source text
5. Ensure completeness of extraction

Please identify any errors, omissions, or improvements needed.
"""
        return validation_prompt

    def build_enhancement_prompt(
        self, partial_data: Dict[str, Any], missing_fields: List[str], text: str
    ) -> str:
        """Build prompt for enhancing partial extraction"""
        field_descriptions = []
        for field_name in missing_fields:
            field = philosophy_field_registry.get_field(field_name)
            if field:
                field_descriptions.append(f"- {field_name}: {field.description}")

        enhancement_prompt = f"""
## ENHANCEMENT TASK

Complete the philosophical extraction by providing the missing fields.

### PARTIAL EXTRACTION:
{partial_data}

### MISSING FIELDS:
{chr(10).join(field_descriptions)}

### TEXT TO ANALYZE:
{text}

Please extract the missing information while maintaining consistency with the existing extraction.
"""
        return enhancement_prompt

    def list_available_templates(self) -> List[Dict[str, Any]]:
        """List all available templates with metadata"""
        templates = []
        for name, template in philosophy_template_library.get_all_templates().items():
            templates.append(
                {
                    "name": name,
                    "description": template.description,
                    "source_type": template.source_type.value,
                    "extraction_depth": template.extraction_depth.value,
                    "categories": [cat.value for cat in template.categories],
                    "field_count": len(template.fields),
                    "priority_level": template.priority_level,
                }
            )
        return sorted(templates, key=lambda x: x["priority_level"], reverse=True)

    def list_available_fields(
        self, category: Optional[str] = None, required_only: bool = False
    ) -> List[Dict[str, Any]]:
        """List available fields with full metadata"""
        fields = []

        if category:
            try:
                cat_enum = PhilosophicalCategory(category)
                field_dict = philosophy_field_registry.get_fields_by_category(cat_enum)
            except ValueError:
                logger.warning(f"Invalid category: {category}")
                field_dict = {}
        else:
            field_dict = philosophy_field_registry._fields

        for name, field in field_dict.items():
            if required_only and not field.required:
                continue

            fields.append(
                {
                    "name": name,
                    "description": field.description,
                    "category": field.category.value if field.category else "general",
                    "required": field.required,
                    "data_type": field.data_type.value,
                    "priority": field.priority,
                    "examples": field.examples,
                    "extraction_hints": field.extraction_hints,
                    "post_processors": field.post_processors,
                }
            )

        return sorted(fields, key=lambda x: x["priority"])

    def suggest_template(
        self,
        text: str,
        use_case: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Suggest the best template for given text and use case"""
        # Get template match
        template_name, confidence, details = self.template_matcher.match_template(text)

        # Get alternative suggestions
        alternatives = []

        # Suggest based on use case
        if use_case:
            use_case_templates = {
                "research": ["comprehensive", "philosophical_argument"],
                "education": ["philosophical_concept", "philosopher_profile"],
                "analysis": ["critical_analysis", "philosophical_argument"],
                "comparison": ["comparative_philosophy"],
                "history": ["historical_philosophy", "philosopher_profile"],
            }

            if use_case in use_case_templates:
                for alt_template in use_case_templates[use_case]:
                    if alt_template != template_name:
                        alternatives.append(alt_template)

        # Suggest based on categories
        if categories:
            for cat in categories:
                cat_templates = philosophy_template_library.get_templates_by_category(
                    PhilosophicalCategory(cat)
                )
                for template in cat_templates:
                    if (
                        template.name != template_name
                        and template.name not in alternatives
                    ):
                        alternatives.append(template.name)

        return {
            "recommended": template_name,
            "confidence": confidence,
            "match_details": details,
            "alternatives": alternatives[:3],  # Top 3 alternatives
            "reasoning": self._explain_recommendation(
                template_name, confidence, details
            ),
        }

    def _explain_recommendation(
        self, template_name: str, confidence: float, details: Dict[str, Any]
    ) -> str:
        """Explain why a template was recommended"""
        if confidence > 0.8:
            strength = "strongly matches"
        elif confidence > 0.5:
            strength = "matches"
        else:
            strength = "weakly matches"

        explanation = f"The text {strength} the '{template_name}' template"

        if "keyword_matches" in details:
            strong_matches = details["keyword_matches"].get("strong", [])
            if strong_matches:
                explanation += f" due to keywords like: {', '.join(strong_matches[:3])}"

        if details.get("fallback"):
            explanation = "No specific template strongly matched, using comprehensive template for thorough analysis"

        return explanation


# Create global instances for convenience
philosophy_prompt_generator = PhilosophyExtractionPromptGenerator()
philosophy_prompt_builder = AdvancedPhilosophyPromptBuilder()
