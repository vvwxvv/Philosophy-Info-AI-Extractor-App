from typing import List, Optional, Dict, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Import your existing prompt elements - add fallback if not available
try:
    from prompts.prompt_elements import *
    from prompts.prompt_structure import create_philosophy_extraction_prompt
except ImportError:
    logger.warning("Prompt elements not available, using fallback")
    # Fallback constants if prompt modules not available
    PHILOSOPHY_FIELD_DESCRIPTIONS = {}
    ETHICAL_PHILOSOPHY_GUIDELINES = []
    METAPHYSICAL_PHILOSOPHY_GUIDELINES = []
    EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES = []
    AESTHETIC_PHILOSOPHY_GUIDELINES = []
    LANGUAGE_PHILOSOPHY_GUIDELINES = []
    MIND_PHILOSOPHY_GUIDELINES = []
    LOGIC_PHILOSOPHY_GUIDELINES = []
    POLITICAL_PHILOSOPHY_GUIDELINES = []
    PHILOSOPHY_CORE_GUIDELINES = []
    PHILOSOPHY_SYSTEM_ROLE = "You are a philosophical analysis assistant."
    PHILOSOPHY_OUTPUT_FORMAT = {}

    def create_philosophy_extraction_prompt(**kwargs):
        return "Analyze the philosophical content of the text."


class PhilosophyCategory(Enum):
    """Enumeration of philosophy categories"""

    GENERAL = "general"
    ETHICAL = "ethical"
    METAPHYSICAL = "metaphysical"
    EPISTEMOLOGICAL = "epistemological"
    AESTHETIC = "aesthetic"
    LANGUAGE = "language"
    MIND = "mind"
    LOGIC = "logic"
    POLITICAL = "political"
    SCIENCE = "science"
    SOCIAL = "social"
    EXISTENTIAL = "existential"
    PHENOMENOLOGICAL = "phenomenological"


class ExtractionMode(Enum):
    """Different extraction modes for various use cases"""

    COMPREHENSIVE = "comprehensive"  # Extract all philosophical aspects
    FOCUSED = "focused"  # Extract specific category only
    EXPLORATORY = "exploratory"  # Discover philosophical themes
    COMPARATIVE = "comparative"  # Compare philosophical positions
    HISTORICAL = "historical"  # Extract with historical context
    APPLIED = "applied"  # Extract practical applications


@dataclass
class ExtractionContext:
    """Context information for extraction"""

    source_type: str = "text"
    language: str = "en"
    target_audience: str = "academic"
    depth_level: str = "detailed"  # basic, intermediate, detailed, expert
    extraction_mode: ExtractionMode = ExtractionMode.COMPREHENSIVE
    categories: List[PhilosophyCategory] = field(default_factory=list)
    custom_focus: Optional[str] = None
    historical_period: Optional[str] = None
    cultural_context: Optional[str] = None

    def __post_init__(self):
        """Validate and process context parameters"""
        # Ensure extraction_mode is enum
        if isinstance(self.extraction_mode, str):
            try:
                self.extraction_mode = ExtractionMode(self.extraction_mode)
            except ValueError:
                self.extraction_mode = ExtractionMode.COMPREHENSIVE

        # Ensure categories are enums
        if self.categories:
            converted_categories = []
            for cat in self.categories:
                if isinstance(cat, str):
                    try:
                        converted_categories.append(PhilosophyCategory(cat))
                    except ValueError:
                        continue
                else:
                    converted_categories.append(cat)
            self.categories = converted_categories


@dataclass
class PhilosophyPromptConfig:
    """Configuration for philosophy prompt generation"""

    category: PhilosophyCategory
    field_descriptions: Dict[str, str]
    guidelines: List[str]
    additional_instructions: Optional[str] = None
    examples: Optional[List[Dict[str, Any]]] = None
    validation_rules: Optional[List[str]] = None


class PhilosophyPromptBuilder:
    """Flexible builder for philosophy extraction prompts"""

    def __init__(self):
        self._category_configs = self._initialize_category_configs()
        self._mode_strategies = self._initialize_mode_strategies()

    def _initialize_category_configs(
        self,
    ) -> Dict[PhilosophyCategory, PhilosophyPromptConfig]:
        """Initialize configurations for each philosophy category"""
        return {
            PhilosophyCategory.ETHICAL: PhilosophyPromptConfig(
                category=PhilosophyCategory.ETHICAL,
                field_descriptions={
                    "ethical": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "ethical", "Ethical principles and moral reasoning"
                    )
                },
                guidelines=(
                    ETHICAL_PHILOSOPHY_GUIDELINES
                    if ETHICAL_PHILOSOPHY_GUIDELINES
                    else [
                        "Focus on moral principles and ethical frameworks",
                        "Identify normative and applied ethical considerations",
                        "Consider cultural ethical perspectives",
                    ]
                ),
                validation_rules=[
                    "Ensure moral frameworks are clearly identified",
                    "Include both normative and applied ethics when relevant",
                    "Consider cultural ethical perspectives",
                ],
            ),
            PhilosophyCategory.METAPHYSICAL: PhilosophyPromptConfig(
                category=PhilosophyCategory.METAPHYSICAL,
                field_descriptions={
                    "metaphysical": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "metaphysical",
                        "Claims about the nature of reality and existence",
                    )
                },
                guidelines=(
                    METAPHYSICAL_PHILOSOPHY_GUIDELINES
                    if METAPHYSICAL_PHILOSOPHY_GUIDELINES
                    else [
                        "Focus on claims about reality and existence",
                        "Identify ontological commitments",
                        "Analyze metaphysical assumptions",
                    ]
                ),
                validation_rules=[
                    "Distinguish between ontological and cosmological claims",
                    "Identify assumptions about reality's nature",
                ],
            ),
            PhilosophyCategory.EPISTEMOLOGICAL: PhilosophyPromptConfig(
                category=PhilosophyCategory.EPISTEMOLOGICAL,
                field_descriptions={
                    "epistemological": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "epistemological",
                        "Claims about knowledge, truth, and justification",
                    )
                },
                guidelines=(
                    EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES
                    if EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES
                    else [
                        "Focus on knowledge claims and justification",
                        "Identify epistemic frameworks",
                        "Analyze truth conditions",
                    ]
                ),
                validation_rules=[
                    "Clarify knowledge claims and justifications",
                    "Identify epistemic frameworks used",
                ],
            ),
            PhilosophyCategory.AESTHETIC: PhilosophyPromptConfig(
                category=PhilosophyCategory.AESTHETIC,
                field_descriptions={
                    "aesthetic": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "aesthetic", "Theories of beauty and aesthetic judgment"
                    )
                },
                guidelines=(
                    AESTHETIC_PHILOSOPHY_GUIDELINES
                    if AESTHETIC_PHILOSOPHY_GUIDELINES
                    else [
                        "Focus on aesthetic theories and judgments",
                        "Analyze concepts of beauty and art",
                        "Consider aesthetic experience",
                    ]
                ),
            ),
            PhilosophyCategory.LANGUAGE: PhilosophyPromptConfig(
                category=PhilosophyCategory.LANGUAGE,
                field_descriptions={
                    "language": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "language", "Philosophy of language and meaning"
                    )
                },
                guidelines=(
                    LANGUAGE_PHILOSOPHY_GUIDELINES
                    if LANGUAGE_PHILOSOPHY_GUIDELINES
                    else [
                        "Focus on language, meaning, and communication",
                        "Analyze semantic and pragmatic aspects",
                        "Consider linguistic philosophy",
                    ]
                ),
            ),
            PhilosophyCategory.MIND: PhilosophyPromptConfig(
                category=PhilosophyCategory.MIND,
                field_descriptions={
                    "mind": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "mind", "Philosophy of mind and consciousness"
                    )
                },
                guidelines=(
                    MIND_PHILOSOPHY_GUIDELINES
                    if MIND_PHILOSOPHY_GUIDELINES
                    else [
                        "Focus on consciousness and mental phenomena",
                        "Analyze mind-body problems",
                        "Consider cognitive philosophy",
                    ]
                ),
            ),
            PhilosophyCategory.LOGIC: PhilosophyPromptConfig(
                category=PhilosophyCategory.LOGIC,
                field_descriptions={
                    "logic": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "logic", "Logical reasoning and argument structure"
                    )
                },
                guidelines=(
                    LOGIC_PHILOSOPHY_GUIDELINES
                    if LOGIC_PHILOSOPHY_GUIDELINES
                    else [
                        "Focus on logical reasoning and argument structure",
                        "Analyze validity and soundness",
                        "Consider formal and informal logic",
                    ]
                ),
            ),
            PhilosophyCategory.POLITICAL: PhilosophyPromptConfig(
                category=PhilosophyCategory.POLITICAL,
                field_descriptions={
                    "political": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "political", "Political theory and social philosophy"
                    )
                },
                guidelines=(
                    POLITICAL_PHILOSOPHY_GUIDELINES
                    if POLITICAL_PHILOSOPHY_GUIDELINES
                    else [
                        "Focus on political theory and social philosophy",
                        "Analyze concepts of justice and power",
                        "Consider political institutions",
                    ]
                ),
            ),
            # Add general category
            PhilosophyCategory.GENERAL: PhilosophyPromptConfig(
                category=PhilosophyCategory.GENERAL,
                field_descriptions={
                    "general": PHILOSOPHY_FIELD_DESCRIPTIONS.get(
                        "general", "General philosophical analysis"
                    )
                },
                guidelines=(
                    PHILOSOPHY_CORE_GUIDELINES
                    if PHILOSOPHY_CORE_GUIDELINES
                    else [
                        "Provide comprehensive philosophical analysis",
                        "Identify key concepts and arguments",
                        "Consider multiple philosophical perspectives",
                    ]
                ),
            ),
        }

    def _initialize_mode_strategies(self) -> Dict[ExtractionMode, Callable]:
        """Initialize extraction strategies for different modes"""
        return {
            ExtractionMode.COMPREHENSIVE: self._build_comprehensive_prompt,
            ExtractionMode.FOCUSED: self._build_focused_prompt,
            ExtractionMode.EXPLORATORY: self._build_exploratory_prompt,
            ExtractionMode.COMPARATIVE: self._build_comparative_prompt,
            ExtractionMode.HISTORICAL: self._build_historical_prompt,
            ExtractionMode.APPLIED: self._build_applied_prompt,
        }

    def _merge_all_field_descriptions(self) -> str:
        """Merge all field descriptions for comprehensive extraction"""
        if not PHILOSOPHY_FIELD_DESCRIPTIONS:
            return "Analyze all philosophical aspects including concepts, arguments, and implications."

        descriptions = []
        for category, description in PHILOSOPHY_FIELD_DESCRIPTIONS.items():
            descriptions.append(f"{category.upper()}: {description}")
        return "\n\n".join(descriptions)

    def build_prompt(self, context: ExtractionContext) -> str:
        """Build a prompt based on the extraction context"""
        strategy = self._mode_strategies.get(
            context.extraction_mode, self._build_comprehensive_prompt
        )
        return strategy(context)

    def _build_comprehensive_prompt(self, context: ExtractionContext) -> str:
        """Build a comprehensive multi-aspect prompt"""
        # Merge guidelines from all relevant categories
        guidelines = (
            PHILOSOPHY_CORE_GUIDELINES.copy()
            if PHILOSOPHY_CORE_GUIDELINES
            else [
                "Provide comprehensive philosophical analysis",
                "Identify key concepts and arguments",
                "Consider multiple philosophical perspectives",
            ]
        )
        field_descriptions = self._merge_all_field_descriptions()

        if context.categories:
            # Use specified categories
            for category in context.categories:
                config = self._category_configs.get(category)
                if config:
                    guidelines.extend(config.guidelines)

        # Add context-specific guidelines
        contextual_guidelines = self._generate_contextual_guidelines(context)
        guidelines.extend(contextual_guidelines)

        return create_philosophy_extraction_prompt(
            field_descriptions=field_descriptions,
            source_type=context.source_type,
            language=context.language,
            custom_guidelines=guidelines,
            system_role=self._enhance_system_role(context),
            output_format="json",
            custom_output_format=PHILOSOPHY_OUTPUT_FORMAT,
        )

    def _build_focused_prompt(self, context: ExtractionContext) -> str:
        """Build a prompt focused on specific categories"""
        if not context.categories:
            raise ValueError("Focused mode requires at least one category")

        primary_category = context.categories[0]
        config = self._category_configs.get(primary_category)

        if not config:
            raise ValueError(f"Unknown category: {primary_category}")

        # Add focused extraction instructions
        focused_guidelines = config.guidelines.copy()
        focused_guidelines.extend(
            [
                f"Concentrate specifically on {primary_category.value} philosophical aspects",
                "Provide deep analysis within this domain",
                "Connect to other philosophical areas only when directly relevant",
            ]
        )

        if context.custom_focus:
            focused_guidelines.append(f"Special focus: {context.custom_focus}")

        return create_philosophy_extraction_prompt(
            field_descriptions=config.field_descriptions.get(
                primary_category.value, ""
            ),
            source_type=context.source_type,
            language=context.language,
            custom_guidelines=focused_guidelines,
            system_role=self._enhance_system_role(context),
            output_format="json",
            custom_output_format=PHILOSOPHY_OUTPUT_FORMAT,
        )

    def _build_exploratory_prompt(self, context: ExtractionContext) -> str:
        """Build a prompt for discovering philosophical themes"""
        exploratory_guidelines = (
            PHILOSOPHY_CORE_GUIDELINES.copy() if PHILOSOPHY_CORE_GUIDELINES else []
        )
        exploratory_guidelines.extend(
            [
                "Identify all philosophical themes present, even if subtle or implicit",
                "Look for emerging philosophical questions or problems",
                "Note connections between different philosophical domains",
                "Highlight novel or unconventional philosophical insights",
                "Consider interdisciplinary philosophical implications",
                "Map the philosophical landscape of the content",
            ]
        )

        return create_philosophy_extraction_prompt(
            field_descriptions=self._merge_all_field_descriptions(),
            source_type=context.source_type,
            language=context.language,
            custom_guidelines=exploratory_guidelines,
            system_role=self._enhance_system_role(context),
            output_format="json",
            custom_output_format=PHILOSOPHY_OUTPUT_FORMAT,
        )

    def _build_comparative_prompt(self, context: ExtractionContext) -> str:
        """Build a prompt for comparative philosophical analysis"""
        comparative_guidelines = (
            PHILOSOPHY_CORE_GUIDELINES.copy() if PHILOSOPHY_CORE_GUIDELINES else []
        )
        comparative_guidelines.extend(
            [
                "Identify different philosophical positions present in the content",
                "Compare and contrast philosophical approaches",
                "Note agreements and disagreements between viewpoints",
                "Analyze the strengths and weaknesses of each position",
                "Consider synthesis possibilities between different views",
                "Map philosophical debates and dialogues",
            ]
        )

        return create_philosophy_extraction_prompt(
            field_descriptions=self._merge_all_field_descriptions(),
            source_type=context.source_type,
            language=context.language,
            custom_guidelines=comparative_guidelines,
            system_role=self._enhance_system_role(context),
            output_format="json",
            custom_output_format=PHILOSOPHY_OUTPUT_FORMAT,
        )

    def _build_historical_prompt(self, context: ExtractionContext) -> str:
        """Build a prompt with historical philosophical context"""
        historical_guidelines = (
            PHILOSOPHY_CORE_GUIDELINES.copy() if PHILOSOPHY_CORE_GUIDELINES else []
        )
        historical_guidelines.extend(
            [
                "Situate philosophical ideas within their historical period",
                "Trace the genealogy of philosophical concepts",
                "Identify influences from earlier philosophical traditions",
                "Note how ideas have evolved or been transformed",
                "Consider the historical conditions that shaped the philosophy",
                "Connect to contemporary philosophical developments",
            ]
        )

        if context.historical_period:
            historical_guidelines.append(
                f"Focus on historical period: {context.historical_period}"
            )

        return create_philosophy_extraction_prompt(
            field_descriptions=self._merge_all_field_descriptions(),
            source_type=context.source_type,
            language=context.language,
            custom_guidelines=historical_guidelines,
            system_role=self._enhance_system_role(context),
            output_format="json",
            custom_output_format=PHILOSOPHY_OUTPUT_FORMAT,
        )

    def _build_applied_prompt(self, context: ExtractionContext) -> str:
        """Build a prompt for applied philosophical analysis"""
        applied_guidelines = (
            PHILOSOPHY_CORE_GUIDELINES.copy() if PHILOSOPHY_CORE_GUIDELINES else []
        )
        applied_guidelines.extend(
            [
                "Focus on practical applications of philosophical ideas",
                "Identify real-world implications and consequences",
                "Consider how philosophical concepts inform practice",
                "Analyze case studies and examples",
                "Bridge theory and application",
                "Evaluate practical effectiveness of philosophical approaches",
            ]
        )

        return create_philosophy_extraction_prompt(
            field_descriptions=self._merge_all_field_descriptions(),
            source_type=context.source_type,
            language=context.language,
            custom_guidelines=applied_guidelines,
            system_role=self._enhance_system_role(context),
            output_format="json",
            custom_output_format=PHILOSOPHY_OUTPUT_FORMAT,
        )

    def _generate_contextual_guidelines(self, context: ExtractionContext) -> List[str]:
        """Generate additional guidelines based on context"""
        guidelines = []

        # Depth level instructions
        depth_instructions = {
            "basic": "Provide clear, accessible explanations suitable for beginners",
            "intermediate": "Include moderate detail and some technical terminology",
            "detailed": "Provide comprehensive analysis with nuanced distinctions",
            "expert": "Include advanced theoretical frameworks and scholarly debates",
        }
        if context.depth_level in depth_instructions:
            guidelines.append(depth_instructions[context.depth_level])

        # Audience-specific instructions
        audience_instructions = {
            "academic": "Use scholarly conventions and cite relevant philosophical traditions",
            "general": "Explain concepts clearly without excessive jargon",
            "professional": "Focus on practical applications and decision-making frameworks",
            "educational": "Structure content for learning with clear examples",
        }
        if context.target_audience in audience_instructions:
            guidelines.append(audience_instructions[context.target_audience])

        # Cultural context
        if context.cultural_context:
            guidelines.append(
                f"Consider cultural perspective: {context.cultural_context}"
            )

        return guidelines

    def _enhance_system_role(self, context: ExtractionContext) -> str:
        """Enhance system role based on context"""
        base_role = (
            PHILOSOPHY_SYSTEM_ROLE
            if PHILOSOPHY_SYSTEM_ROLE
            else "You are a philosophical analysis assistant."
        )

        enhancements = []

        if context.extraction_mode == ExtractionMode.HISTORICAL:
            enhancements.append("with expertise in history of philosophy")
        elif context.extraction_mode == ExtractionMode.COMPARATIVE:
            enhancements.append("skilled in comparative philosophical analysis")
        elif context.extraction_mode == ExtractionMode.APPLIED:
            enhancements.append("focused on practical philosophical applications")

        if context.target_audience == "academic":
            enhancements.append("following scholarly standards")

        if enhancements:
            return f"{base_role} {', '.join(enhancements)}"

        return base_role


class AdvancedPhilosophyExtractor:
    """Advanced philosophy extractor with flexible extraction modes"""

    def __init__(self):
        self.prompt_builder = PhilosophyPromptBuilder()
        self._validators = self._initialize_validators()

    def _initialize_validators(self) -> Dict[str, Callable]:
        """Initialize output validators"""
        return {
            "structure": self._validate_structure,
            "content": self._validate_content,
            "completeness": self._validate_completeness,
            "consistency": self._validate_consistency,
        }

    def extract(
        self,
        text: str,
        context: Optional[ExtractionContext] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract philosophical content from text

        Args:
            text: Input text to analyze
            context: Extraction context (uses defaults if not provided)
            validate: Whether to validate the output

        Returns:
            Extracted philosophical content as a dictionary
        """
        if context is None:
            context = ExtractionContext()

        # Build appropriate prompt
        prompt_template = self.prompt_builder.build_prompt(context)

        # Format prompt with text
        formatted_prompt = f"{prompt_template}\n\nTEXT TO ANALYZE:\n{text}"

        # Here you would call your LLM with the prompt
        # For now, returning a structure example
        result = self._call_llm(formatted_prompt, text)

        if validate:
            self._validate_output(result, context)

        return result

    def extract_multiple_aspects(
        self,
        text: str,
        categories: List[PhilosophyCategory],
        merge_results: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract multiple philosophical aspects from text"""
        results = []

        for category in categories:
            context = ExtractionContext(
                categories=[category], extraction_mode=ExtractionMode.FOCUSED
            )
            result = self.extract(text, context)
            results.append({"category": category.value, "extraction": result})

        if merge_results:
            return self._merge_extractions(results)

        return results

    def discover_philosophy(
        self, text: str, depth_level: str = "detailed"
    ) -> Dict[str, Any]:
        """Discover philosophical themes in text"""
        context = ExtractionContext(
            extraction_mode=ExtractionMode.EXPLORATORY, depth_level=depth_level
        )
        return self.extract(text, context)

    def compare_philosophies(
        self,
        texts: List[str],
        focus_categories: Optional[List[PhilosophyCategory]] = None,
    ) -> Dict[str, Any]:
        """Compare philosophical positions across multiple texts"""
        # Combine texts with markers
        combined_text = "\n\n---TEXT BOUNDARY---\n\n".join(
            [f"[TEXT {i+1}]\n{text}" for i, text in enumerate(texts)]
        )

        context = ExtractionContext(
            extraction_mode=ExtractionMode.COMPARATIVE,
            categories=focus_categories or [],
        )

        return self.extract(combined_text, context)

    def _call_llm(self, prompt: str, text: str) -> Dict[str, Any]:
        """Call LLM with prompt and text - implement your LLM call here"""
        # This is where you'd integrate with your LLM
        # For now, returning a mock structure
        return {
            "philosophical_themes": [],
            "key_concepts": [],
            "arguments": [],
            "implications": [],
            "metadata": {},
        }

    def _validate_output(
        self, output: Dict[str, Any], context: ExtractionContext
    ) -> None:
        """Validate extracted output"""
        for validator_name, validator in self._validators.items():
            validator(output, context)

    def _validate_structure(
        self, output: Dict[str, Any], context: ExtractionContext
    ) -> None:
        """Validate output structure"""
        required_fields = ["philosophical_themes", "key_concepts", "arguments"]
        for field in required_fields:
            if field not in output:
                raise ValueError(f"Missing required field: {field}")

    def _validate_content(
        self, output: Dict[str, Any], context: ExtractionContext
    ) -> None:
        """Validate output content quality"""
        # Implement content validation logic
        pass

    def _validate_completeness(
        self, output: Dict[str, Any], context: ExtractionContext
    ) -> None:
        """Validate extraction completeness"""
        # Implement completeness validation logic
        pass

    def _validate_consistency(
        self, output: Dict[str, Any], context: ExtractionContext
    ) -> None:
        """Validate internal consistency"""
        # Implement consistency validation logic
        pass

    def _merge_extractions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple extraction results"""
        merged = {
            "philosophical_themes": [],
            "key_concepts": [],
            "arguments": [],
            "implications": [],
            "category_specific": {},
            "metadata": {
                "extraction_count": len(results),
                "categories_analyzed": [r["category"] for r in results],
            },
        }

        for result in results:
            category = result["category"]
            extraction = result["extraction"]

            # Merge common fields
            for field in [
                "philosophical_themes",
                "key_concepts",
                "arguments",
                "implications",
            ]:
                if field in extraction:
                    merged[field].extend(extraction[field])

            # Store category-specific data
            merged["category_specific"][category] = extraction

        # Remove duplicates from lists
        for field in [
            "philosophical_themes",
            "key_concepts",
            "arguments",
            "implications",
        ]:
            if field in merged and isinstance(merged[field], list):
                merged[field] = list(dict.fromkeys(merged[field]))

        return merged


# Convenience functions for common use cases
def extract_philosophy(
    text: str,
    mode: str = "comprehensive",
    categories: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for extracting philosophy from text

    Args:
        text: Input text
        mode: Extraction mode (comprehensive, focused, exploratory, etc.)
        categories: List of category names to focus on
        **kwargs: Additional context parameters

    Returns:
        Extracted philosophical content
    """
    extractor = AdvancedPhilosophyExtractor()

    # Convert string categories to enum
    category_enums = []
    if categories:
        for cat in categories:
            try:
                category_enums.append(PhilosophyCategory(cat))
            except ValueError:
                logger.warning(f"Invalid category: {cat}")

    context = ExtractionContext(
        extraction_mode=ExtractionMode(mode),
        categories=category_enums,
        **kwargs,
    )

    return extractor.extract(text, context)


# Example usage functions
def extract_ethical_philosophy(text: str, **kwargs) -> Dict[str, Any]:
    """Extract ethical philosophy from text"""
    return extract_philosophy(text, mode="focused", categories=["ethical"], **kwargs)


def discover_philosophical_themes(text: str, **kwargs) -> Dict[str, Any]:
    """Discover philosophical themes in text"""
    return extract_philosophy(text, mode="exploratory", **kwargs)


def compare_philosophical_texts(texts: List[str], **kwargs) -> Dict[str, Any]:
    """Compare philosophy across multiple texts"""
    extractor = AdvancedPhilosophyExtractor()
    return extractor.compare_philosophies(texts, **kwargs)
