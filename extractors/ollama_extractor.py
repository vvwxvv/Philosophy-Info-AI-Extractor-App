"""
Philosophy Extraction using Ollama API
Integrates philosophy prompts with Ollama for actual extraction
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from assets.ollama_api_client import OllamaAPIClient
from extractors.api import PhilosophyExtractorAPI
from extractors.types import (
    PhilosophyExtractionConfig,
    PhilosophyExtractionResult,
    ExtractionItem,
    ExtractionMetadata,
    PhilosophicalCategory,
    ExtractionDepth,
    TargetAudience,
    ExtractionMode,
)

from extractors.generator import PhilosophyExtractionPromptGenerator

logger = logging.getLogger(__name__)


class PhilosophyOllamaExtractor:
    """Extracts philosophical content using Ollama models"""

    def __init__(
        self,
        model_name: str = "deepseek-r1:7b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 300,
    ):
        """
        Initialize the Ollama-based philosophy extractor

        Args:
            model_name: Ollama model to use
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.ollama_client = OllamaAPIClient(base_url, timeout)
        self.prompt_generator = PhilosophyExtractionPromptGenerator()
        self.extractor_api = PhilosophyExtractorAPI()

        # Test connection
        if not self.ollama_client.test_connection():
            logger.warning("Cannot connect to Ollama API at %s", base_url)

    async def extract_philosophy(
        self, text: str, config: Optional[PhilosophyExtractionConfig] = None, **kwargs
    ) -> PhilosophyExtractionResult:
        """
        Extract philosophical content from text using Ollama

        Args:
            text: Text to analyze
            config: Extraction configuration
            **kwargs: Additional extraction parameters

        Returns:
            PhilosophyExtractionResult with extracted content
        """
        start_time = datetime.utcnow()

        # Create default config if not provided
        if config is None:
            config = self._create_default_config(**kwargs)

        # Generate the appropriate prompt
        prompt_data = self._generate_extraction_prompt(text, config, **kwargs)

        # Call Ollama API
        try:
            response = await self.ollama_client.call_api_with_json_response(
                model_name=self.model_name,
                prompt=prompt_data["user_prompt"],
                system_prompt=prompt_data["system_prompt"],
            )

            # Parse and structure the response
            extraction_result = self._parse_ollama_response(response, config)

            # Add metadata
            extraction_result.metadata.duration_seconds = (
                datetime.utcnow() - start_time
            ).total_seconds()
            extraction_result.metadata.parameters["model"] = self.model_name
            extraction_result.metadata.parameters["prompt_template"] = prompt_data.get(
                "template_name", "custom"
            )

            return extraction_result

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Return empty result on failure
            return self._create_empty_result(
                config=config,
                error_message=str(e),
                duration=(datetime.utcnow() - start_time).total_seconds(),
            )

    def _create_default_config(self, **kwargs) -> PhilosophyExtractionConfig:
        """Create default configuration from kwargs"""
        return PhilosophyExtractionConfig(
            source_type=kwargs.get("source_type", "essay"),
            language=kwargs.get("language", "EN"),
            extraction_depth=ExtractionDepth(kwargs.get("depth_level", "detailed")),
            target_audience=TargetAudience(kwargs.get("target_audience", "academic")),
            extraction_mode=ExtractionMode(
                kwargs.get("extraction_mode", "comprehensive")
            ),
            categories_focus=[
                PhilosophicalCategory(cat) for cat in kwargs.get("categories", [])
            ],
            include_examples=kwargs.get("include_examples", True),
            include_references=kwargs.get("include_references", True),
            include_historical_context=kwargs.get("include_historical_context", True),
            confidence_threshold=kwargs.get("confidence_threshold", 0.7),
        )

    def _generate_extraction_prompt(
        self, text: str, config: PhilosophyExtractionConfig, **kwargs
    ) -> Dict[str, str]:
        """Generate appropriate prompt based on configuration"""

        # Get template name based on extraction mode and categories
        template_name = kwargs.get("template_name")
        if not template_name:
            template_name = self._select_template(config)

        # Build the extraction request using the API
        extraction_result = self.extractor_api.extract(
            text=text,
            template_name=template_name,
            language=config.language,
            extraction_mode=config.extraction_mode.value,
            categories=[cat.value for cat in config.categories_focus],
            depth_level=config.extraction_depth.value,
            target_audience=config.target_audience.value,
            **kwargs,
        )

        # Extract the prompt from the result
        full_prompt = extraction_result.prompt

        # Split into system and user prompts
        parts = full_prompt.split("TEXT TO ANALYZE:", 1)
        if len(parts) == 2:
            system_prompt = parts[0].strip()
            user_prompt = f"Analyze the following philosophical text and extract the requested information in JSON format:\n\n{text}"
        else:
            system_prompt = "You are a philosophical text analysis system."
            user_prompt = full_prompt

        # Add specific extraction instructions
        user_prompt += "\n\nProvide your analysis as a valid JSON object with the following structure:"
        user_prompt += self._get_json_structure_prompt(config)

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "template_name": template_name,
        }

    def _select_template(self, config: PhilosophyExtractionConfig) -> str:
        """Select appropriate template based on configuration"""
        # Map extraction modes to templates
        mode_template_map = {
            ExtractionMode.COMPREHENSIVE: "comprehensive",
            ExtractionMode.FOCUSED: "philosophy_basic",
            ExtractionMode.CRITICAL: "philosophical_argument",
            ExtractionMode.THEMATIC: "philosophical_concept",
            ExtractionMode.COMPARATIVE: "philosophy_basic",
        }

        # Check if specific category template should be used
        if config.categories_focus:
            primary_category = config.categories_focus[0]
            category_template_map = {
                PhilosophicalCategory.ETHICS: "philosophy_ethical",
                PhilosophicalCategory.LOGIC: "philosophical_argument",
                PhilosophicalCategory.METAPHYSICS: "philosophy_treatise",
                PhilosophicalCategory.EPISTEMOLOGY: "philosophical_concept",
            }
            if primary_category in category_template_map:
                return category_template_map[primary_category]

        return mode_template_map.get(config.extraction_mode, "philosophy_basic")

    def _get_json_structure_prompt(self, config: PhilosophyExtractionConfig) -> str:
        """Get JSON structure prompt based on configuration"""
        structure = {
            "main_thesis": "The central philosophical claim or thesis",
            "key_arguments": [
                {
                    "premises": ["premise 1", "premise 2"],
                    "conclusion": "conclusion",
                    "type": "deductive/inductive/abductive",
                    "strength": "strong/moderate/weak",
                }
            ],
            "key_concepts": [
                {
                    "term": "concept name",
                    "definition": "concept definition",
                    "context": "usage context",
                }
            ],
            "philosophers_mentioned": ["philosopher names"],
            "philosophical_tradition": "identified philosophical school or tradition",
        }

        # Add category-specific fields
        if PhilosophicalCategory.ETHICS in config.categories_focus:
            structure["ethical_principles"] = ["identified ethical principles"]
            structure["moral_arguments"] = ["moral claims and justifications"]

        if PhilosophicalCategory.EPISTEMOLOGY in config.categories_focus:
            structure["epistemological_claims"] = ["knowledge-related claims"]
            structure["justification_methods"] = ["methods of justification discussed"]

        if PhilosophicalCategory.METAPHYSICS in config.categories_focus:
            structure["metaphysical_positions"] = ["claims about reality/existence"]
            structure["ontological_commitments"] = ["what the text commits to existing"]

        if PhilosophicalCategory.LOGIC in config.categories_focus:
            structure["logical_structures"] = ["identified logical forms"]
            structure["validity_assessment"] = "assessment of argument validity"

        # Add optional fields based on config
        if config.include_historical_context:
            structure["historical_context"] = "relevant historical background"

        if config.include_influences:
            structure["influences"] = ["philosophical influences identified"]

        if config.include_criticisms:
            structure["criticisms"] = ["criticisms or counterarguments presented"]

        return f"\n```json\n{json.dumps(structure, indent=2)}\n```"

    def _parse_ollama_response(
        self, response: Dict[str, Any], config: PhilosophyExtractionConfig
    ) -> PhilosophyExtractionResult:
        """Parse Ollama response into PhilosophyExtractionResult"""

        # Check for errors
        if "error" in response:
            logger.error(f"Ollama returned error: {response}")
            return self._create_empty_result(
                config, error_message=response.get("message", "Unknown error")
            )

        # Create result object
        result = PhilosophyExtractionResult(
            metadata=ExtractionMetadata(
                extraction_id=f"ollama_phil_{datetime.utcnow().timestamp()}",
                model_version=self.model_name,
                parameters=config.to_dict(),
            )
        )

        # Extract main thesis
        result.main_thesis = response.get("main_thesis")

        # Extract arguments
        if "key_arguments" in response:
            result.arguments = self._parse_arguments(response["key_arguments"])

        # Extract concepts
        if "key_concepts" in response:
            result.key_concepts = self._parse_concepts(response["key_concepts"])

        # Extract philosophers
        if "philosophers_mentioned" in response:
            result.philosophers = [
                ExtractionItem(value=phil, confidence=0.9)
                for phil in response["philosophers_mentioned"]
            ]

        # Extract tradition
        result.philosophical_tradition = response.get("philosophical_tradition")

        # Extract category-specific content
        if "ethical_principles" in response:
            result.ethical_principles = [
                ExtractionItem(value=principle, confidence=0.8)
                for principle in response["ethical_principles"]
            ]

        if "epistemological_claims" in response:
            result.epistemological_claims = [
                ExtractionItem(value=claim, confidence=0.8)
                for claim in response["epistemological_claims"]
            ]

        if "metaphysical_positions" in response:
            result.metaphysical_positions = [
                ExtractionItem(value=position, confidence=0.8)
                for position in response["metaphysical_positions"]
            ]

        if "logical_structures" in response:
            result.logical_structures = [
                ExtractionItem(value=structure, confidence=0.85)
                for structure in response["logical_structures"]
            ]

        # Extract context
        if "historical_context" in response:
            result.historical_context = [
                ExtractionItem(value=response["historical_context"], confidence=0.75)
            ]

        if "influences" in response:
            result.influences = [
                ExtractionItem(value=influence, confidence=0.8)
                for influence in response["influences"]
            ]

        if "criticisms" in response:
            result.criticisms = [
                ExtractionItem(value=criticism, confidence=0.8)
                for criticism in response["criticisms"]
            ]

        # Update statistics
        result.metadata.statistics = result.get_statistics()

        return result

    def _parse_arguments(self, arguments_data: List[Dict]) -> List[ExtractionItem]:
        """Parse arguments from response"""
        arguments = []

        for arg in arguments_data:
            if isinstance(arg, dict):
                argument_value = {
                    "premises": arg.get("premises", []),
                    "conclusion": arg.get("conclusion", ""),
                    "type": arg.get("type", "unknown"),
                    "strength": arg.get("strength", "unknown"),
                }
                arguments.append(
                    ExtractionItem(
                        value=argument_value,
                        confidence=0.85,
                        metadata={"logical_structure": arg.get("logical_structure")},
                    )
                )
            elif isinstance(arg, str):
                # Simple string argument
                arguments.append(
                    ExtractionItem(value={"description": arg}, confidence=0.7)
                )

        return arguments

    def _parse_concepts(self, concepts_data: List[Any]) -> List[ExtractionItem]:
        """Parse concepts from response"""
        concepts = []

        for concept in concepts_data:
            if isinstance(concept, dict):
                concepts.append(
                    ExtractionItem(
                        value=concept.get("term", ""),
                        confidence=0.9,
                        context=concept.get("definition", ""),
                        metadata={"usage_context": concept.get("context")},
                    )
                )
            elif isinstance(concept, str):
                concepts.append(ExtractionItem(value=concept, confidence=0.7))

        return concepts

    def _create_empty_result(
        self,
        config: PhilosophyExtractionConfig,
        error_message: str = "",
        duration: float = 0.0,
    ) -> PhilosophyExtractionResult:
        """Create an empty result for error cases"""
        return PhilosophyExtractionResult(
            metadata=ExtractionMetadata(
                extraction_id=f"error_{datetime.utcnow().timestamp()}",
                duration_seconds=duration,
                parameters=config.to_dict(),
                statistics={"error": error_message},
            )
        )

    async def batch_extract(
        self,
        texts: List[str],
        config: Optional[PhilosophyExtractionConfig] = None,
        **kwargs,
    ) -> List[PhilosophyExtractionResult]:
        """Extract philosophy from multiple texts"""
        tasks = [self.extract_philosophy(text, config, **kwargs) for text in texts]
        return await asyncio.gather(*tasks)

    async def extract_with_categories(
        self, text: str, categories: List[str], **kwargs
    ) -> Dict[str, PhilosophyExtractionResult]:
        """Extract text analyzing specific philosophical categories"""
        results = {}

        for category in categories:
            config = self._create_default_config(**kwargs)
            config.categories_focus = [PhilosophicalCategory(category)]
            config.extraction_mode = ExtractionMode.FOCUSED

            result = await self.extract_philosophy(text, config, **kwargs)
            results[category] = result

        return results


# Convenience functions
async def extract_philosophical_text(
    text: str,
    model_name: str = "deepseek-r1:7b",
    depth: str = "detailed",
    categories: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Simple function to extract philosophical content

    Args:
        text: Text to analyze
        model_name: Ollama model to use
        depth: Extraction depth (basic/intermediate/detailed/expert)
        categories: Optional philosophical categories to focus on
        **kwargs: Additional parameters

    Returns:
        Dictionary with extracted philosophical content
    """
    extractor = PhilosophyOllamaExtractor(model_name=model_name)

    result = await extractor.extract_philosophy(
        text=text, depth_level=depth, categories=categories or [], **kwargs
    )

    return result.to_dict()


# Usage examples
async def example_usage():
    """Example usage of the philosophy extractor with Ollama"""

    # Initialize extractor
    extractor = PhilosophyOllamaExtractor(model_name="deepseek-r1:7b")

    # Example 1: Basic extraction
    text1 = """
    The concept of free will has been central to philosophical discourse. 
    If determinism is true and every event is caused by prior events, 
    then human actions are simply part of a causal chain. However, we 
    experience ourselves as making genuine choices. This apparent 
    contradiction has led philosophers to various positions: hard 
    determinism denies free will, libertarianism affirms it, while 
    compatibilism attempts to reconcile both views.
    """

    print("Example 1: Basic philosophical extraction")
    result1 = await extractor.extract_philosophy(
        text=text1, extraction_mode="comprehensive", depth_level="detailed"
    )
    print(json.dumps(result1.to_dict(), indent=2))

    # Example 2: Category-focused extraction
    text2 = """
    Kant's categorical imperative states that one should act only 
    according to maxims that could become universal laws. This principle 
    provides a test for moral actions: if everyone acting on your maxim 
    would lead to contradiction or an undesirable world, the action is 
    morally wrong. For instance, lying fails this test because universal 
    lying would undermine the very possibility of communication.
    """

    print("\nExample 2: Ethics-focused extraction")
    config2 = PhilosophyExtractionConfig(
        source_type="essay",
        extraction_depth=ExtractionDepth.DETAILED,
        categories_focus=[PhilosophicalCategory.ETHICS],
        extraction_mode=ExtractionMode.FOCUSED,
    )

    result2 = await extractor.extract_philosophy(text=text2, config=config2)
    print(json.dumps(result2.to_dict(), indent=2))

    # Example 3: Multi-category analysis
    text3 = """
    Descartes' cogito ergo sum establishes the thinking self as the 
    foundation of knowledge. This move simultaneously makes a metaphysical 
    claim about the nature of the self as a thinking substance, and an 
    epistemological claim about the certainty of self-awareness. The 
    argument's logical structure - I think, therefore I am - has been 
    both celebrated as undeniable and criticized as circular.
    """

    print("\nExample 3: Multi-category extraction")
    results3 = await extractor.extract_with_categories(
        text=text3,
        categories=["epistemology", "metaphysics", "logic"],
        depth_level="expert",
    )

    for category, result in results3.items():
        print(f"\n{category.upper()} Analysis:")
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_usage())
