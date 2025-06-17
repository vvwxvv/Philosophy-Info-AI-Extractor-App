"""
Enhanced Philosophy Extractor API with comprehensive integration
"""

from typing import Dict, List, Any, Optional, Union, Protocol
from dataclasses import dataclass, field
import logging
import asyncio
import json
import re
from datetime import datetime
import hashlib

# Core imports
from extractors.types import (
    PhilosophySourceType,
    PhilosophyExtractionConfig,
    ExtractionDepth,
    TargetAudience,
    ExtractionMode as TypesExtractionMode,
    PhilosophicalCategory,
    PhilosophyExtractionResult,
    ValidationResult,
    ExtractionItem,
    ExtractionMetadata,
)

# Enhanced modules
from extractors.fields import (
    PhilosophyExtractionField,
    PhilosophyFieldSet,
    philosophy_field_registry,
)
from prompts.templates import (
    PhilosophyExtractionTemplate,
    philosophy_template_library,
)
from extractors.generator import (
    AdvancedPhilosophyPromptBuilder,
    PhilosophyTemplateMatcher,
)
from extractors.system import (
    PhilosophySystemIntegration,
    validate_field_compatibility,
    validate_extraction_data,
    analyze_field_coverage,
    create_extraction_result,
)

# Knowledge base integration
try:
    from knowledges import (
        enhance_extraction_with_knowledge,
        extract_entities_from_text,
        get_concept_definition,
        validate_knowledge_base,
    )

    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Knowledge base not available")

# Advanced extractor
try:
    from extractors.advanced_extractor import (
        AdvancedPhilosophyExtractor,
        ExtractionContext,
        ExtractionMode,
        PhilosophyCategory,
    )

    ADVANCED_EXTRACTOR_AVAILABLE = True
except ImportError:
    AdvancedPhilosophyExtractor = None
    ExtractionContext = None
    ExtractionMode = TypesExtractionMode
    PhilosophyCategory = PhilosophicalCategory
    ADVANCED_EXTRACTOR_AVAILABLE = False

# Ollama integration
try:
    from extractors.ollama_extractor import PhilosophyOllamaExtractor

    OLLAMA_AVAILABLE = True
except ImportError:
    PhilosophyOllamaExtractor = None
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Use AdvancedPhilosophyPromptBuilder as PhilosophyPromptBuilder
PhilosophyPromptBuilder = AdvancedPhilosophyPromptBuilder


@dataclass
class ExtractionRequest:
    """Enhanced request object for extraction operations"""

    text: str
    template_name: Optional[str] = None
    language: str = "mixed"
    additional_fields: List[str] = field(default_factory=list)
    exclude_fields: List[str] = field(default_factory=list)
    extraction_mode: Union[str, TypesExtractionMode] = TypesExtractionMode.COMPREHENSIVE
    categories: List[Union[str, PhilosophicalCategory]] = field(default_factory=list)
    depth_level: str = "detailed"
    target_audience: str = "academic"
    custom_focus: Optional[str] = None
    historical_period: Optional[str] = None
    cultural_context: Optional[str] = None

    # Enhanced parameters
    confidence_threshold: float = 0.7
    max_extraction_time: int = 300
    enable_knowledge_enhancement: bool = True
    enable_validation: bool = True
    auto_template_selection: bool = False
    batch_mode: bool = False
    preserve_context: bool = True

    def __post_init__(self):
        """Convert string enums to proper enum types and validate"""
        if isinstance(self.extraction_mode, str):
            try:
                self.extraction_mode = TypesExtractionMode(self.extraction_mode)
            except ValueError:
                logger.warning(f"Invalid extraction mode: {self.extraction_mode}")
                self.extraction_mode = TypesExtractionMode.COMPREHENSIVE

        # Convert string categories to enum
        converted_categories = []
        for cat in self.categories:
            if isinstance(cat, str):
                try:
                    converted_categories.append(PhilosophicalCategory(cat))
                except ValueError:
                    logger.warning(f"Invalid category: {cat}")
                    continue
            else:
                converted_categories.append(cat)
        self.categories = converted_categories

        # Validate parameters
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        if self.max_extraction_time < 10:
            raise ValueError("Max extraction time must be at least 10 seconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return {
            "text_length": len(self.text),
            "template_name": self.template_name,
            "language": self.language,
            "extraction_mode": self.extraction_mode.value,
            "categories": [cat.value for cat in self.categories],
            "depth_level": self.depth_level,
            "target_audience": self.target_audience,
            "confidence_threshold": self.confidence_threshold,
            "enable_knowledge_enhancement": self.enable_knowledge_enhancement,
            "enable_validation": self.enable_validation,
        }

    def get_cache_key(self) -> str:
        """Generate cache key for request"""
        key_data = {
            "text_hash": hashlib.md5(self.text.encode()).hexdigest()[:16],
            "template": self.template_name,
            "mode": self.extraction_mode.value,
            "categories": sorted([cat.value for cat in self.categories]),
            "depth": self.depth_level,
            "language": self.language,
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


@dataclass
class ExtractionResult:
    """Enhanced structured result of an extraction operation"""

    prompt: str
    extraction_request: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Enhanced result data
    validation_result: Optional[ValidationResult] = None
    knowledge_enhanced: bool = False
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: Optional[float] = None
    template_used: Optional[str] = None
    strategy_used: Optional[str] = None
    field_coverage: Dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_prompt: bool = False) -> Dict[str, Any]:
        """Convert result to dictionary with options"""
        result = {
            "success": self.success,
            "extracted_data": self.extracted_data,
            "metadata": self.metadata or {},
            "knowledge_enhanced": self.knowledge_enhanced,
            "confidence_scores": self.confidence_scores,
            "processing_time": self.processing_time,
            "template_used": self.template_used,
            "strategy_used": self.strategy_used,
            "field_coverage": self.field_coverage,
        }

        if include_prompt:
            result["prompt"] = self.prompt

        if self.error:
            result["error"] = self.error

        if self.validation_result:
            result["validation"] = self.validation_result.to_dict()

        return result

    def get_high_confidence_data(self, threshold: float = 0.8) -> Dict[str, Any]:
        """Get only high confidence extracted data"""
        if not self.extracted_data:
            return {}

        filtered_data = {}
        for field_name, value in self.extracted_data.items():
            field_confidence = self.confidence_scores.get(field_name, 1.0)
            if field_confidence >= threshold:
                filtered_data[field_name] = value

        return filtered_data

    def to_philosophy_extraction_result(self) -> PhilosophyExtractionResult:
        """Convert to PhilosophyExtractionResult format"""
        # Create metadata
        metadata = ExtractionMetadata(
            extraction_id=f"api_{datetime.utcnow().timestamp()}",
            duration_seconds=self.processing_time or 0.0,
            parameters=self.extraction_request,
            statistics=self.confidence_scores,
        )

        # Create PhilosophyExtractionResult
        result = PhilosophyExtractionResult(metadata=metadata)

        if self.extracted_data:
            # Map extracted data to result fields
            def create_extraction_items(values, confidence=0.9):
                if not isinstance(values, list):
                    values = [values] if values else []

                items = []
                for value in values:
                    if isinstance(value, dict):
                        item = ExtractionItem(
                            value=value.get("value", value),
                            confidence=value.get("confidence", confidence),
                            context=value.get("context"),
                            metadata=value.get("metadata", {}),
                        )
                    else:
                        item = ExtractionItem(value=value, confidence=confidence)
                    items.append(item)
                return items

            # Map common fields
            if "key_concepts" in self.extracted_data:
                result.key_concepts = create_extraction_items(
                    self.extracted_data["key_concepts"],
                    self.confidence_scores.get("key_concepts", 0.9),
                )

            if "arguments" in self.extracted_data:
                result.arguments = create_extraction_items(
                    self.extracted_data["arguments"],
                    self.confidence_scores.get("arguments", 0.9),
                )

            if "philosophers" in self.extracted_data:
                result.philosophers = create_extraction_items(
                    self.extracted_data["philosophers"],
                    self.confidence_scores.get("philosophers", 0.9),
                )

            # Set analysis fields
            result.main_thesis = self.extracted_data.get("main_thesis")
            result.philosophical_tradition = self.extracted_data.get(
                "philosophical_tradition"
            )
            result.argument_type = self.extracted_data.get("argument_type")
            result.methodology = self.extracted_data.get("methodology")

        return result


class ExtractionStrategy(Protocol):
    """Protocol for extraction strategies"""

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Execute extraction based on request"""
        ...


class EnhancedLegacyExtractionStrategy:
    """Enhanced legacy extraction strategy with better integration"""

    def __init__(self):
        self.prompt_builder = AdvancedPhilosophyPromptBuilder()
        self.template_matcher = PhilosophyTemplateMatcher()
        self._cache = {}

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Execute enhanced legacy extraction"""
        start_time = datetime.utcnow()

        logger.debug(
            "Starting enhanced legacy extraction with template: %s",
            request.template_name,
        )

        try:
            # Auto-select template if requested
            if request.auto_template_selection and not request.template_name:
                template_name, confidence, details = (
                    self.template_matcher.match_template(request.text)
                )
                request.template_name = template_name
                logger.info(
                    f"Auto-selected template: {template_name} (confidence: {confidence:.2f})"
                )

            # Get template from library
            template = self._get_template(request.template_name)

            # Validate field compatibility using PhilosophyFieldSet
            field_set_name = self._get_field_set_for_template(template)
            is_compatible, validation = validate_field_compatibility(
                field_set=field_set_name,
                source_type=template.source_type,
                extraction_depth=ExtractionDepth(request.depth_level),
            )

            if not is_compatible:
                logger.warning(f"Field compatibility issues: {validation.warnings}")

            # Create enhanced configuration
            config = self._create_config(request, template)

            # Generate enhanced prompt
            prompt = self._generate_enhanced_prompt(request, config, template)

            # Get expected fields using PhilosophyFieldSet
            expected_fields = self._get_expected_fields(template, request)

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            result = ExtractionResult(
                prompt=prompt,
                extraction_request=request.to_dict(),
                success=True,
                template_used=request.template_name,
                strategy_used="enhanced_legacy",
                processing_time=processing_time,
                metadata={
                    "template_info": (
                        template.to_dict() if hasattr(template, "to_dict") else {}
                    ),
                    "expected_fields": expected_fields,
                    "field_compatibility": validation.to_dict(),
                    "auto_selected": request.auto_template_selection,
                    "field_set_used": field_set_name,
                },
            )

            return result

        except Exception as e:
            logger.error(f"Enhanced legacy extraction failed: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ExtractionResult(
                prompt="",
                extraction_request=request.to_dict(),
                success=False,
                error=str(e),
                strategy_used="enhanced_legacy",
                processing_time=processing_time,
            )

    def _get_template(
        self, template_name: Optional[str]
    ) -> PhilosophyExtractionTemplate:
        """Get template with fallback"""
        template = philosophy_template_library.get_template(template_name)
        if not template:
            template_name = template_name or "philosophy_basic"
            template = philosophy_template_library.get_template(template_name)
            if not template:
                raise ValueError(f"Template '{template_name}' not found")
        return template

    def _get_field_set_for_template(
        self, template: PhilosophyExtractionTemplate
    ) -> str:
        """Get appropriate field set for template"""
        # Map template names to field sets
        template_to_fieldset = {
            "philosophy_basic": PhilosophyFieldSet.BASIC.value,
            "comprehensive": PhilosophyFieldSet.COMPREHENSIVE.value,
            "philosopher_profile": PhilosophyFieldSet.PHILOSOPHER.value,
            "philosophical_argument": PhilosophyFieldSet.ARGUMENT.value,
            "philosophical_concept": PhilosophyFieldSet.CONCEPT.value,
            "philosophy_ethical": PhilosophyFieldSet.BASIC.value,  # Use basic for category-specific
            "historical_philosophy": PhilosophyFieldSet.HISTORICAL.value,
            "comparative_philosophy": PhilosophyFieldSet.COMPARATIVE.value,
            "critical_analysis": PhilosophyFieldSet.CRITICAL.value,
        }

        template_name = template.name if hasattr(template, "name") else str(template)
        return template_to_fieldset.get(template_name, PhilosophyFieldSet.BASIC.value)

    def _get_expected_fields(
        self, template: PhilosophyExtractionTemplate, request: ExtractionRequest
    ) -> List[str]:
        """Get expected fields for template"""
        # Get field set
        field_set_name = self._get_field_set_for_template(template)
        try:
            field_set = PhilosophyFieldSet(field_set_name)
            expected_fields = philosophy_field_registry.get_field_set(field_set)
        except ValueError:
            # Fallback to template fields if field set not found
            expected_fields = template.fields if hasattr(template, "fields") else []

        # Add additional fields from request
        if request.additional_fields:
            expected_fields.extend(request.additional_fields)

        # Remove excluded fields
        if request.exclude_fields:
            expected_fields = [
                f for f in expected_fields if f not in request.exclude_fields
            ]

        return expected_fields

    def _create_config(
        self, request: ExtractionRequest, template: PhilosophyExtractionTemplate
    ) -> PhilosophyExtractionConfig:
        """Create enhanced configuration"""
        return PhilosophyExtractionConfig(
            template=template,
            source_type=template.source_type,
            language=request.language,
            additional_fields=request.additional_fields,
            exclude_fields=request.exclude_fields,
            extraction_depth=ExtractionDepth(request.depth_level),
            target_audience=TargetAudience(request.target_audience),
            extraction_mode=request.extraction_mode,
            categories_focus=request.categories,
            confidence_threshold=request.confidence_threshold,
            max_extraction_time=request.max_extraction_time,
            enable_validation=request.enable_validation,
            custom_guidelines=[request.custom_focus] if request.custom_focus else [],
        )

    def _generate_enhanced_prompt(
        self,
        request: ExtractionRequest,
        config: PhilosophyExtractionConfig,
        template: PhilosophyExtractionTemplate,
    ) -> str:
        """Generate enhanced prompt with all context"""
        prompt = self.prompt_builder.build_prompt(
            text="",  # We'll add the text later
            template_name=template.name if hasattr(template, "name") else str(template),
            language=config.language,
            extraction_mode=config.extraction_mode.value,
            depth_level=config.extraction_depth.value,
            target_audience=config.target_audience.value,
            categories=(
                [cat.value for cat in config.categories_focus]
                if config.categories_focus
                else None
            ),
            historical_period=request.historical_period,
            cultural_context=request.cultural_context,
        )

        # Add confidence threshold instruction
        confidence_instruction = (
            f"\n\nCONFIDENCE THRESHOLD: {request.confidence_threshold}\n"
        )
        confidence_instruction += (
            "Include confidence scores (0.0-1.0) for each extracted field.\n"
        )

        # Add knowledge base instruction if available
        if KNOWLEDGE_BASE_AVAILABLE and request.enable_knowledge_enhancement:
            kb_instruction = "\nKNOWLEDGE BASE: Use philosophical knowledge base for entity recognition and concept validation.\n"
            prompt += kb_instruction

        full_prompt = (
            f"{prompt}{confidence_instruction}\n\nText to analyze:\n{request.text}"
        )
        return full_prompt


# Create alias for LegacyExtractionStrategy
LegacyExtractionStrategy = EnhancedLegacyExtractionStrategy


class AdvancedExtractionStrategy:
    """Enhanced advanced extraction strategy"""

    def __init__(self, advanced_extractor):
        self.advanced_extractor = advanced_extractor
        self._performance_cache = {}

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Execute enhanced advanced extraction"""
        start_time = datetime.utcnow()

        try:
            if not ADVANCED_EXTRACTOR_AVAILABLE:
                return self._fallback_extraction(request, start_time)

            # Build enhanced extraction context
            context = self._create_extraction_context(request)

            # Execute extraction with context
            raw_result = self.advanced_extractor.extract(request.text, context)

            # Convert to PhilosophyExtractionResult if needed
            if not isinstance(raw_result, PhilosophyExtractionResult):
                # Create PhilosophyExtractionResult using system integration
                philosophy_result = create_extraction_result(
                    extracted_data=raw_result,
                    config=self._create_config_for_result(request),
                    extraction_time=(datetime.utcnow() - start_time).total_seconds(),
                )
            else:
                philosophy_result = raw_result

            # Enhance with knowledge base if available
            if KNOWLEDGE_BASE_AVAILABLE and request.enable_knowledge_enhancement:
                enhanced_data = enhance_extraction_with_knowledge(
                    philosophy_result.to_dict(), request.text, request.language
                )
                raw_result = enhanced_data

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Create enhanced result
            result = ExtractionResult(
                prompt=(
                    self.advanced_extractor.prompt_builder.build_prompt(context)
                    if context
                    else ""
                ),
                extraction_request=request.to_dict(),
                success=True,
                extracted_data=raw_result,
                knowledge_enhanced=KNOWLEDGE_BASE_AVAILABLE
                and request.enable_knowledge_enhancement,
                strategy_used="advanced",
                processing_time=processing_time,
                metadata={
                    "context": context.__dict__ if context else {},
                    "advanced_features_used": True,
                    "philosophy_result_metadata": (
                        philosophy_result.metadata.to_dict()
                        if hasattr(philosophy_result, "metadata")
                        else {}
                    ),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Advanced extraction failed: {e}")
            return self._fallback_extraction(request, start_time, error=str(e))

    def _create_extraction_context(self, request: ExtractionRequest) -> Optional[Any]:
        """Create extraction context if available"""
        if not ExtractionContext:
            return None

        return ExtractionContext(
            source_type="philosophical_text",
            language=request.language,
            target_audience=request.target_audience,
            depth_level=request.depth_level,
            extraction_mode=ExtractionMode(request.extraction_mode.value),
            categories=[PhilosophyCategory(cat.value) for cat in request.categories],
            custom_focus=request.custom_focus,
            historical_period=request.historical_period,
            cultural_context=request.cultural_context,
        )

    def _create_config_for_result(
        self, request: ExtractionRequest
    ) -> PhilosophyExtractionConfig:
        """Create config for result creation"""
        return PhilosophyExtractionConfig(
            source_type=PhilosophySourceType.ESSAY,
            language=request.language,
            extraction_depth=ExtractionDepth(request.depth_level),
            target_audience=TargetAudience(request.target_audience),
            extraction_mode=request.extraction_mode,
            categories_focus=request.categories,
        )

    def _fallback_extraction(
        self,
        request: ExtractionRequest,
        start_time: datetime,
        error: Optional[str] = None,
    ) -> ExtractionResult:
        """Fallback extraction when advanced extractor not available"""
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Use legacy strategy as fallback
        legacy_strategy = LegacyExtractionStrategy()
        result = legacy_strategy.extract(request)
        result.strategy_used = "advanced_fallback"
        result.metadata = result.metadata or {}
        result.metadata["fallback_reason"] = error or "Advanced extractor not available"

        return result


class OllamaExtractionStrategy:
    """Enhanced Ollama extraction strategy with better error handling"""

    def __init__(self, model_name: str = "deepseek-r1:7b"):
        self.model_name = model_name
        self.extractor = None
        self._connection_tested = False
        self._connection_status = False

        # Try to import and initialize Ollama extractor
        if OLLAMA_AVAILABLE:
            try:
                self.extractor = PhilosophyOllamaExtractor(model_name=model_name)
                self._test_connection()
            except Exception as e:
                logger.warning(f"Ollama extractor initialization failed: {e}")
        else:
            logger.warning("Ollama extractor not available")

    def _test_connection(self):
        """Test Ollama connection"""
        if self.extractor and not self._connection_tested:
            self._connection_status = self.extractor.ollama_client.test_connection()
            self._connection_tested = True
            if not self._connection_status:
                logger.warning("Ollama connection test failed")

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Execute enhanced Ollama extraction"""
        start_time = datetime.utcnow()

        if not self.extractor or not self._connection_status:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ExtractionResult(
                prompt="",
                extraction_request=request.to_dict(),
                success=False,
                error="Ollama extractor not available or connection failed",
                strategy_used="ollama",
                processing_time=processing_time,
            )

        try:
            # Create enhanced config from request
            config = self._create_ollama_config(request)

            # Run async extraction in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                extraction_result = loop.run_until_complete(
                    self.extractor.extract_philosophy(
                        text=request.text,
                        config=config,
                        template_name=request.template_name,
                        custom_focus=request.custom_focus,
                    )
                )
            finally:
                loop.close()

            # Convert PhilosophyExtractionResult to dict
            if isinstance(extraction_result, PhilosophyExtractionResult):
                extracted_data = extraction_result.to_dict()
                confidence_scores = extraction_result.get_statistics()
            else:
                extracted_data = extraction_result
                confidence_scores = {}

            # Enhance with knowledge base if available
            if KNOWLEDGE_BASE_AVAILABLE and request.enable_knowledge_enhancement:
                extracted_data = enhance_extraction_with_knowledge(
                    extracted_data, request.text, request.language
                )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Create enhanced result
            result = ExtractionResult(
                prompt="Generated by Ollama extraction pipeline",
                extraction_request=request.to_dict(),
                extracted_data=extracted_data,
                success=True,
                knowledge_enhanced=KNOWLEDGE_BASE_AVAILABLE
                and request.enable_knowledge_enhancement,
                strategy_used="ollama",
                processing_time=processing_time,
                template_used=request.template_name,
                confidence_scores=confidence_scores,
                metadata={
                    "ollama_model": self.model_name,
                    "extraction_config": (
                        config.to_dict() if hasattr(config, "to_dict") else {}
                    ),
                },
            )

            return result

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Ollama extraction failed: {e}")
            return ExtractionResult(
                prompt="",
                extraction_request=request.to_dict(),
                success=False,
                error=str(e),
                strategy_used="ollama",
                processing_time=processing_time,
            )

    def _create_ollama_config(
        self, request: ExtractionRequest
    ) -> PhilosophyExtractionConfig:
        """Create Ollama-specific configuration"""
        return PhilosophyExtractionConfig(
            source_type=PhilosophySourceType.from_text_length(
                len(request.text.split())
            ),
            language=request.language,
            extraction_depth=ExtractionDepth(request.depth_level),
            target_audience=TargetAudience(request.target_audience),
            extraction_mode=request.extraction_mode,
            categories_focus=request.categories,
            confidence_threshold=request.confidence_threshold,
            max_extraction_time=request.max_extraction_time,
            enable_validation=request.enable_validation,
            custom_guidelines=[request.custom_focus] if request.custom_focus else [],
        )


class PhilosophyExtractorAPI:
    """Enhanced main API for philosophical information extraction"""

    def __init__(
        self,
        default_language: str = "mixed",
        default_template: str = "philosophy_basic",
        use_advanced_extractor: bool = True,
        use_ollama: bool = False,
        ollama_model: str = "deepseek-r1:7b",
        enable_caching: bool = True,
        cache_size: int = 128,
    ):
        """
        Initialize the enhanced philosophy extractor API
        """
        self.default_language = default_language
        self.default_template = default_template
        self.use_advanced_extractor = use_advanced_extractor
        self.use_ollama = use_ollama
        self.enable_caching = enable_caching

        # Initialize caching
        if enable_caching:
            self._cache = {}
            self._cache_size = cache_size

        # Initialize strategies
        self.legacy_strategy = LegacyExtractionStrategy()

        # Initialize advanced extractor if enabled and available
        if self.use_advanced_extractor and ADVANCED_EXTRACTOR_AVAILABLE:
            try:
                self.advanced_extractor = AdvancedPhilosophyExtractor()
                self.advanced_strategy = AdvancedExtractionStrategy(
                    self.advanced_extractor
                )
            except Exception as e:
                logger.warning(f"Advanced extractor initialization failed: {e}")
                self.advanced_extractor = None
                self.advanced_strategy = None
        else:
            self.advanced_extractor = None
            self.advanced_strategy = None

        # Initialize Ollama strategy if enabled
        if use_ollama and OLLAMA_AVAILABLE:
            self.ollama_strategy = OllamaExtractionStrategy(ollama_model)
        else:
            self.ollama_strategy = None

        # Initialize utility components
        self.template_matcher = PhilosophyTemplateMatcher()
        self.system_integration = PhilosophySystemIntegration()

        logger.info(
            "Enhanced PhilosophyExtractorAPI initialized - Advanced: %s, Ollama: %s, KB: %s",
            ADVANCED_EXTRACTOR_AVAILABLE,
            OLLAMA_AVAILABLE and use_ollama,
            KNOWLEDGE_BASE_AVAILABLE,
        )

    def extract(
        self,
        text: str,
        template_name: Optional[str] = None,
        language: Optional[str] = None,
        additional_fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        # Advanced extraction parameters
        extraction_mode: Optional[str] = None,
        categories: Optional[List[str]] = None,
        depth_level: Optional[str] = None,
        target_audience: Optional[str] = None,
        custom_focus: Optional[str] = None,
        historical_period: Optional[str] = None,
        cultural_context: Optional[str] = None,
        # Enhanced parameters
        confidence_threshold: Optional[float] = None,
        auto_template_selection: bool = False,
        enable_knowledge_enhancement: Optional[bool] = None,
        enable_validation: Optional[bool] = None,
        max_extraction_time: Optional[int] = None,
    ) -> ExtractionResult:
        """
        Enhanced extract method with comprehensive parameter support
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")

            # Create enhanced extraction request
            request = ExtractionRequest(
                text=text,
                template_name=template_name or self.default_template,
                language=language or self.default_language,
                additional_fields=additional_fields or [],
                exclude_fields=exclude_fields or [],
                extraction_mode=extraction_mode
                or TypesExtractionMode.COMPREHENSIVE.value,
                categories=categories or [],
                depth_level=depth_level or "detailed",
                target_audience=target_audience or "academic",
                custom_focus=custom_focus,
                historical_period=historical_period,
                cultural_context=cultural_context,
                confidence_threshold=confidence_threshold or 0.7,
                auto_template_selection=auto_template_selection,
                enable_knowledge_enhancement=(
                    enable_knowledge_enhancement
                    if enable_knowledge_enhancement is not None
                    else KNOWLEDGE_BASE_AVAILABLE
                ),
                enable_validation=(
                    enable_validation if enable_validation is not None else True
                ),
                max_extraction_time=max_extraction_time or 300,
            )

            # Check cache first
            if self.enable_caching:
                cache_key = request.get_cache_key()
                if cache_key in self._cache:
                    logger.debug("Returning cached result")
                    cached_result = self._cache[cache_key]
                    cached_result.metadata = cached_result.metadata or {}
                    cached_result.metadata["from_cache"] = True
                    return cached_result

            # Select and execute strategy
            strategy = self._select_strategy(request)
            result = strategy.extract(request)

            # Post-process result
            result = self._post_process_result(result, request)

            # Cache result if enabled
            if self.enable_caching and result.success:
                self._cache_result(cache_key, result)

            return result

        except Exception as e:
            logger.error("Extraction failed: %s", str(e), exc_info=True)
            return ExtractionResult(
                prompt="",
                extraction_request={},
                success=False,
                error=str(e),
                strategy_used="error",
            )

    def _select_strategy(self, request: ExtractionRequest) -> ExtractionStrategy:
        """Enhanced strategy selection with better logic"""
        # Prefer Ollama if available and enabled
        if self.use_ollama and self.ollama_strategy:
            return self.ollama_strategy

        # Use advanced strategy if enabled and request has advanced parameters
        if self.advanced_strategy and self._requires_advanced_extraction(request):
            return self.advanced_strategy

        # Default to enhanced legacy strategy
        return self.legacy_strategy

    def _requires_advanced_extraction(self, request: ExtractionRequest) -> bool:
        """Enhanced check for advanced extraction requirements"""
        return any(
            [
                request.extraction_mode != TypesExtractionMode.COMPREHENSIVE,
                request.categories,
                request.depth_level not in ["detailed", "basic"],
                request.target_audience != "academic",
                request.custom_focus,
                request.historical_period,
                request.cultural_context,
                request.confidence_threshold != 0.7,
            ]
        )

    def _post_process_result(
        self, result: ExtractionResult, request: ExtractionRequest
    ) -> ExtractionResult:
        """Post-process extraction result with validation and enhancement"""
        if not result.success:
            return result

        try:
            # Validate extraction if enabled using proper field set
            if request.enable_validation and result.extracted_data:
                field_set_name = self._determine_field_set(request.template_name)
                validation = validate_extraction_data(
                    result.extracted_data, field_set_name
                )
                result.validation_result = validation

            # Analyze field coverage
            if result.extracted_data and request.template_name:
                expected_fields = self._get_expected_fields_for_template(
                    request.template_name, request
                )
                coverage = analyze_field_coverage(
                    result.extracted_data, expected_fields
                )
                result.field_coverage = coverage

            # Calculate confidence scores if not present
            if not result.confidence_scores and result.extracted_data:
                result.confidence_scores = self._calculate_confidence_scores(
                    result.extracted_data
                )

        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")

        return result

    def _determine_field_set(self, template_name: Optional[str]) -> str:
        """Determine appropriate field set for validation"""
        if not template_name:
            return PhilosophyFieldSet.BASIC.value

        # Map template names to field sets
        template_to_fieldset = {
            "philosophy_basic": PhilosophyFieldSet.BASIC.value,
            "comprehensive": PhilosophyFieldSet.COMPREHENSIVE.value,
            "philosopher_profile": PhilosophyFieldSet.PHILOSOPHER.value,
            "philosophical_argument": PhilosophyFieldSet.ARGUMENT.value,
            "philosophical_concept": PhilosophyFieldSet.CONCEPT.value,
            "historical_philosophy": PhilosophyFieldSet.HISTORICAL.value,
            "comparative_philosophy": PhilosophyFieldSet.COMPARATIVE.value,
            "critical_analysis": PhilosophyFieldSet.CRITICAL.value,
        }

        return template_to_fieldset.get(template_name, PhilosophyFieldSet.BASIC.value)

    def _get_expected_fields_for_template(
        self, template_name: Optional[str], request: ExtractionRequest
    ) -> List[str]:
        """Get expected fields for a template"""
        if not template_name:
            return []

        # Get field set
        field_set_name = self._determine_field_set(template_name)
        try:
            field_set = PhilosophyFieldSet(field_set_name)
            expected_fields = philosophy_field_registry.get_field_set(field_set)
        except ValueError:
            # Fallback to basic fields
            expected_fields = philosophy_field_registry.get_field_set(
                PhilosophyFieldSet.BASIC
            )

        # Add additional fields and remove excluded fields
        if request.additional_fields:
            expected_fields.extend(request.additional_fields)
        if request.exclude_fields:
            expected_fields = [
                f for f in expected_fields if f not in request.exclude_fields
            ]

        return expected_fields

    def _calculate_confidence_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate basic confidence scores for extracted data"""
        scores = {}
        for field_name, value in data.items():
            if value is None or value == "":
                scores[field_name] = 0.0
            elif isinstance(value, list):
                scores[field_name] = min(1.0, len(value) * 0.2)
            elif isinstance(value, str):
                scores[field_name] = min(1.0, len(value.split()) * 0.1)
            else:
                scores[field_name] = 0.8
        return scores

    def _cache_result(self, cache_key: str, result: ExtractionResult):
        """Cache extraction result"""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = result

    def get_fields(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all available fields using PhilosophyExtractionField"""
        try:
            fields = []

            if category:
                # Get fields by category
                try:
                    cat_enum = PhilosophicalCategory(category)
                    field_dict = philosophy_field_registry.get_fields_by_category(
                        cat_enum
                    )
                    for name, field in field_dict.items():
                        fields.append(self._format_field_info(field))
                except ValueError:
                    logger.warning(f"Invalid category: {category}")
            else:
                # Get all fields
                for name, field in philosophy_field_registry._fields.items():
                    fields.append(self._format_field_info(field))

            return fields
        except Exception as e:
            logger.error("Failed to get fields: %s", str(e), exc_info=True)
            return []

    def _format_field_info(self, field: PhilosophyExtractionField) -> Dict[str, Any]:
        """Helper method to format field information consistently"""
        return {
            "name": field.name,
            "description": field.description,
            "category": field.category.value if field.category else "general",
            "required": field.required,
            "data_type": field.data_type.value,
            "priority": field.priority,
            "extraction_hints": field.extraction_hints,
            "post_processors": field.post_processors,
            "examples": field.examples,
            "validation_rules": (
                [rule.name for rule in field.validation_rules]
                if hasattr(field, "validation_rules")
                else []
            ),
        }

    def get_field_sets(self) -> List[Dict[str, Any]]:
        """Get available field sets using PhilosophyFieldSet"""
        return [
            {
                "name": field_set.value,
                "description": field_set.description,
                "required_fields": field_set.required_fields,
                "field_count": len(philosophy_field_registry.get_field_set(field_set)),
                "complexity": self._calculate_field_set_complexity(field_set),
            }
            for field_set in PhilosophyFieldSet
        ]

    def _calculate_field_set_complexity(self, field_set: PhilosophyFieldSet) -> str:
        """Calculate complexity level of a field set"""
        field_count = len(philosophy_field_registry.get_field_set(field_set))
        if field_count <= 5:
            return "basic"
        elif field_count <= 10:
            return "intermediate"
        elif field_count <= 15:
            return "detailed"
        else:
            return "expert"

    def get_templates(self) -> List[Dict[str, str]]:
        """Get all available templates with enhanced information"""
        try:
            templates = []
            for (
                name,
                template,
            ) in philosophy_template_library.get_all_templates().items():
                template_info = {
                    "name": name,
                    "description": template.description,
                    "source_type": template.source_type.value,
                    "extraction_depth": template.extraction_depth.value,
                    "categories": [cat.value for cat in template.categories],
                    "field_count": len(template.fields),
                    "priority_level": template.priority_level,
                    "language": template.language,
                    "field_set": self._determine_field_set(name),
                }

                # Add usage recommendations
                template_info["recommended_for"] = self._get_template_recommendations(
                    template
                )
                templates.append(template_info)

            return sorted(templates, key=lambda x: x["priority_level"], reverse=True)
        except Exception as e:
            logger.error("Failed to get templates: %s", str(e), exc_info=True)
            return []

    def _get_template_recommendations(self, template) -> List[str]:
        """Get recommendations for when to use a template"""
        recommendations = []

        if hasattr(template, "categories") and template.categories:
            cat_names = [cat.value for cat in template.categories]
            if "ethics" in cat_names:
                recommendations.append("moral philosophy analysis")
            if "logic" in cat_names:
                recommendations.append("argument analysis")
            if "epistemology" in cat_names:
                recommendations.append("knowledge claims analysis")

        if hasattr(template, "extraction_depth"):
            if template.extraction_depth.value == "basic":
                recommendations.append("quick overviews")
            elif template.extraction_depth.value == "expert":
                recommendations.append("detailed academic research")

        return recommendations

    def get_extraction_modes(self) -> List[Dict[str, str]]:
        """Get available extraction modes"""
        return [
            {
                "name": mode.value,
                "description": mode.description,
            }
            for mode in TypesExtractionMode
        ]

    def get_philosophy_categories(self) -> List[Dict[str, str]]:
        """Get available philosophy categories"""
        return [
            {
                "name": cat.value,
                "description": cat.name.lower().replace("_", " ").title(),
                "subcategories": cat.subcategories,
            }
            for cat in PhilosophicalCategory
        ]

    def discover_philosophy(
        self, text: str, depth_level: str = "detailed"
    ) -> Dict[str, Any]:
        """Enhanced philosophical theme discovery"""
        result = self.extract(
            text=text,
            extraction_mode="exploratory",
            depth_level=depth_level,
            template_name="comprehensive",
            enable_knowledge_enhancement=True,
            auto_template_selection=True,
        )

        discovery_result = {
            "themes_discovered": result.extracted_data if result.success else [],
            "extraction_result": result,
            "confidence_analysis": self._analyze_discovery_confidence(result),
            "knowledge_entities": {},
            "recommendations": {},
        }

        # Add knowledge base entities if available
        if KNOWLEDGE_BASE_AVAILABLE and result.success:
            try:
                entities = extract_entities_from_text(text, confidence_threshold=0.6)
                discovery_result["knowledge_entities"] = entities

                # Generate recommendations based on discovered themes
                discovery_result["recommendations"] = (
                    self._generate_discovery_recommendations(
                        result.extracted_data, entities
                    )
                )
            except Exception as e:
                logger.warning(f"Knowledge base entity extraction failed: {e}")

        return discovery_result

    def _analyze_discovery_confidence(self, result: ExtractionResult) -> Dict[str, Any]:
        """Analyze confidence of discovered themes"""
        if not result.success or not result.confidence_scores:
            return {"overall": 0.0, "distribution": {}}

        scores = list(result.confidence_scores.values())
        return {
            "overall": sum(scores) / len(scores) if scores else 0.0,
            "distribution": {
                "high_confidence": len([s for s in scores if s >= 0.8]),
                "medium_confidence": len([s for s in scores if 0.5 <= s < 0.8]),
                "low_confidence": len([s for s in scores if s < 0.5]),
            },
            "field_confidence": result.confidence_scores,
        }

    def _generate_discovery_recommendations(
        self, extracted_data: Dict[str, Any], entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendations based on discovered themes"""
        recommendations = {
            "further_analysis": [],
            "related_concepts": [],
            "suggested_categories": [],
            "recommended_templates": [],
        }

        try:
            # Analyze extracted concepts for further exploration
            if "key_concepts" in extracted_data:
                concepts = extracted_data["key_concepts"]
                for concept in concepts[:5]:  # Top 5 concepts
                    if KNOWLEDGE_BASE_AVAILABLE:
                        related = get_concept_definition(str(concept))
                        if related:
                            recommendations["related_concepts"].append(
                                {
                                    "concept": concept,
                                    "definition": related,
                                    "explore_further": True,
                                }
                            )

            # Suggest categories based on entities
            if entities:
                if entities.get("theories"):
                    recommendations["suggested_categories"].extend(
                        ["metaphysics", "epistemology"]
                    )
                if entities.get("schools"):
                    recommendations["suggested_categories"].append("historical")
                if "ethical" in str(entities).lower():
                    recommendations["suggested_categories"].append("ethics")

            # Remove duplicates
            recommendations["suggested_categories"] = list(
                set(recommendations["suggested_categories"])
            )

            # Recommend templates based on discovered content
            if "arguments" in extracted_data:
                recommendations["recommended_templates"].append(
                    "philosophical_argument"
                )
            if "philosophers" in extracted_data:
                recommendations["recommended_templates"].append("philosopher_profile")

        except Exception as e:
            logger.warning(f"Failed to generate discovery recommendations: {e}")

        return recommendations

    def compare_philosophies(
        self, texts: List[str], focus_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enhanced philosophical comparison across multiple texts"""
        comparisons = {
            "texts": [],
            "similarities": [],
            "differences": [],
            "synthesis": "",
            "comparative_analysis": {},
            "confidence_matrix": {},
            "category_analysis": {},
            "philosophical_mapping": {},
        }

        try:
            # Extract from each text with enhanced parameters
            for i, text in enumerate(texts):
                result = self.extract(
                    text=text,
                    categories=focus_categories or [],
                    extraction_mode="comparative",
                    template_name="comparative_philosophy",
                    enable_knowledge_enhancement=True,
                    confidence_threshold=0.7,
                )

                comparisons["texts"].append(
                    {
                        "index": i,
                        "result": result,
                        "confidence": result.confidence_scores,
                        "processing_time": result.processing_time,
                        "knowledge_enhanced": result.knowledge_enhanced,
                    }
                )

            # Enhanced comparative analysis
            if len(comparisons["texts"]) >= 2:
                comparisons["comparative_analysis"] = (
                    self._enhanced_comparative_analysis(
                        comparisons["texts"], focus_categories
                    )
                )

                # Generate confidence matrix
                comparisons["confidence_matrix"] = self._generate_confidence_matrix(
                    comparisons["texts"]
                )

                # Category-specific analysis
                if focus_categories:
                    comparisons["category_analysis"] = self._analyze_by_categories(
                        comparisons["texts"], focus_categories
                    )

            # Generate synthesis
            comparisons["synthesis"] = self._generate_philosophical_synthesis(
                comparisons["texts"]
            )

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            comparisons["error"] = str(e)

        return comparisons

    def _enhanced_comparative_analysis(
        self, text_results: List[Dict], focus_categories: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Enhanced analysis of similarities and differences"""
        analysis = {
            "common_concepts": [],
            "unique_concepts": {},
            "argument_patterns": [],
            "philosophical_positions": {},
            "conceptual_overlap": {},
            "thematic_connections": [],
            "divergence_points": [],
        }

        try:
            # Extract and analyze concepts
            all_concepts = {}
            all_arguments = {}

            for i, text_data in enumerate(text_results):
                result = text_data["result"]
                if result.success and result.extracted_data:
                    # Collect concepts
                    concepts = result.extracted_data.get("key_concepts", [])
                    all_concepts[i] = set(str(c) for c in concepts)

                    # Collect arguments
                    arguments = result.extracted_data.get("arguments", [])
                    all_arguments[i] = arguments

            # Find conceptual overlaps and divergences
            if len(all_concepts) >= 2:
                # Common concepts across all texts
                concept_sets = list(all_concepts.values())
                if concept_sets:
                    common = set.intersection(*concept_sets)
                    analysis["common_concepts"] = list(common)

                    # Unique concepts per text
                    for i, concepts in all_concepts.items():
                        others = set()
                        for j, other_concepts in all_concepts.items():
                            if i != j:
                                others.update(other_concepts)
                        unique = concepts - others
                        if unique:
                            analysis["unique_concepts"][f"text_{i}"] = list(unique)

                    # Calculate conceptual overlap percentages
                    analysis["conceptual_overlap"] = self._calculate_conceptual_overlap(
                        all_concepts
                    )

            # Analyze argument patterns
            analysis["argument_patterns"] = self._analyze_argument_patterns(
                all_arguments
            )

            # Find thematic connections
            analysis["thematic_connections"] = self._find_thematic_connections(
                text_results
            )

        except Exception as e:
            logger.warning(f"Enhanced comparative analysis failed: {e}")

        return analysis

    def _calculate_conceptual_overlap(
        self, all_concepts: Dict[int, set]
    ) -> Dict[str, float]:
        """Calculate percentage overlap between concept sets"""
        overlap_matrix = {}

        for i, concepts_i in all_concepts.items():
            for j, concepts_j in all_concepts.items():
                if i != j:
                    key = f"text_{i}_vs_text_{j}"
                    intersection = len(concepts_i.intersection(concepts_j))
                    union = len(concepts_i.union(concepts_j))
                    overlap_percentage = (
                        (intersection / union * 100) if union > 0 else 0
                    )
                    overlap_matrix[key] = round(overlap_percentage, 2)

        return overlap_matrix

    def _analyze_argument_patterns(
        self, all_arguments: Dict[int, List]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in philosophical arguments"""
        patterns = []

        try:
            # Look for common argument structures
            argument_types = {}
            for text_idx, arguments in all_arguments.items():
                for arg in arguments:
                    if isinstance(arg, dict):
                        arg_type = arg.get("type", "unknown")
                        if arg_type not in argument_types:
                            argument_types[arg_type] = []
                        argument_types[arg_type].append(
                            {"text_index": text_idx, "argument": arg}
                        )

            # Identify patterns
            for arg_type, instances in argument_types.items():
                if len(instances) > 1:
                    patterns.append(
                        {
                            "type": arg_type,
                            "frequency": len(instances),
                            "texts_involved": list(
                                set(inst["text_index"] for inst in instances)
                            ),
                            "pattern_strength": (
                                "strong" if len(instances) >= 3 else "moderate"
                            ),
                        }
                    )

        except Exception as e:
            logger.warning(f"Argument pattern analysis failed: {e}")

        return patterns

    def _find_thematic_connections(
        self, text_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Find thematic connections between texts"""
        connections = []

        try:
            # Extract main themes from each text
            themes_by_text = {}
            for i, text_data in enumerate(text_results):
                result = text_data["result"]
                if result.success and result.extracted_data:
                    main_thesis = result.extracted_data.get("main_thesis", "")
                    key_concepts = result.extracted_data.get("key_concepts", [])
                    themes_by_text[i] = {
                        "thesis": main_thesis,
                        "concepts": key_concepts,
                    }

            # Find connections between themes
            for i, themes_i in themes_by_text.items():
                for j, themes_j in themes_by_text.items():
                    if i < j:  # Avoid duplicate comparisons
                        connection = self._analyze_thematic_connection(
                            themes_i, themes_j
                        )
                        if (
                            connection["strength"] > 0.3
                        ):  # Threshold for meaningful connection
                            connections.append(
                                {
                                    "texts": [i, j],
                                    "connection_type": connection["type"],
                                    "strength": connection["strength"],
                                    "shared_elements": connection["shared_elements"],
                                }
                            )

        except Exception as e:
            logger.warning(f"Thematic connection analysis failed: {e}")

        return connections

    def _analyze_thematic_connection(
        self, themes_i: Dict, themes_j: Dict
    ) -> Dict[str, Any]:
        """Analyze connection between two sets of themes"""
        connection = {"type": "conceptual", "strength": 0.0, "shared_elements": []}

        try:
            # Compare concepts
            concepts_i = set(str(c) for c in themes_i.get("concepts", []))
            concepts_j = set(str(c) for c in themes_j.get("concepts", []))

            shared_concepts = concepts_i.intersection(concepts_j)
            if shared_concepts:
                connection["shared_elements"].extend(list(shared_concepts))

                # Calculate strength based on overlap
                total_concepts = len(concepts_i.union(concepts_j))
                connection["strength"] = (
                    len(shared_concepts) / total_concepts if total_concepts > 0 else 0
                )

            # Compare thesis statements (simple keyword matching)
            thesis_i = themes_i.get("thesis", "").lower()
            thesis_j = themes_j.get("thesis", "").lower()

            if thesis_i and thesis_j:
                # Simple keyword matching for thesis connection
                words_i = set(thesis_i.split())
                words_j = set(thesis_j.split())
                shared_words = words_i.intersection(words_j)

                # Filter out common words
                common_words = {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                }
                meaningful_shared = shared_words - common_words

                if len(meaningful_shared) >= 2:
                    connection["type"] = "thesis_related"
                    connection["strength"] = max(connection["strength"], 0.4)

        except Exception as e:
            logger.warning(f"Thematic connection analysis failed: {e}")

        return connection

    def _generate_confidence_matrix(self, text_results: List[Dict]) -> Dict[str, Any]:
        """Generate confidence matrix for comparison results"""
        matrix = {
            "individual_confidence": {},
            "comparative_reliability": 0.0,
            "confidence_variance": 0.0,
        }

        try:
            confidences = []
            for i, text_data in enumerate(text_results):
                result = text_data["result"]
                if result.confidence_scores:
                    avg_confidence = sum(result.confidence_scores.values()) / len(
                        result.confidence_scores
                    )
                    matrix["individual_confidence"][f"text_{i}"] = {
                        "average": avg_confidence,
                        "field_count": len(result.confidence_scores),
                        "high_confidence_fields": len(
                            [s for s in result.confidence_scores.values() if s >= 0.8]
                        ),
                    }
                    confidences.append(avg_confidence)

            if confidences:
                matrix["comparative_reliability"] = sum(confidences) / len(confidences)
                # Calculate variance
                mean = matrix["comparative_reliability"]
                variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
                matrix["confidence_variance"] = variance

        except Exception as e:
            logger.warning(f"Confidence matrix generation failed: {e}")

        return matrix

    def _analyze_by_categories(
        self, text_results: List[Dict], focus_categories: List[str]
    ) -> Dict[str, Any]:
        """Analyze comparison results by philosophical categories"""
        category_analysis = {}

        for category in focus_categories:
            category_analysis[category] = {
                "texts_with_content": [],
                "common_themes": [],
                "divergent_approaches": [],
                "category_confidence": 0.0,
            }

            try:
                category_data = []
                for i, text_data in enumerate(text_results):
                    result = text_data["result"]
                    if result.success and result.extracted_data:
                        # Extract category-specific content
                        category_content = self._extract_category_content(
                            result.extracted_data, category
                        )
                        if category_content:
                            category_analysis[category]["texts_with_content"].append(i)
                            category_data.append(category_content)

                # Analyze common themes within category
                if len(category_data) >= 2:
                    common_themes = self._find_common_category_themes(category_data)
                    category_analysis[category]["common_themes"] = common_themes

                    # Calculate category confidence
                    confidences = [
                        text_data["result"].confidence_scores.get(
                            f"{category}_content", 0.0
                        )
                        for text_data in text_results
                        if text_data["result"].confidence_scores
                    ]
                    if confidences:
                        category_analysis[category]["category_confidence"] = sum(
                            confidences
                        ) / len(confidences)

            except Exception as e:
                logger.warning(f"Category analysis failed for {category}: {e}")

        return category_analysis

    def _extract_category_content(
        self, extracted_data: Dict[str, Any], category: str
    ) -> Optional[Dict[str, Any]]:
        """Extract content specific to a philosophical category"""
        category_fields = {
            "ethics": ["ethical_principles", "moral_arguments", "ethical_frameworks"],
            "epistemology": [
                "epistemological_claims",
                "knowledge_claims",
                "justification_methods",
            ],
            "metaphysics": [
                "metaphysical_positions",
                "ontological_commitments",
                "reality_claims",
            ],
            "logic": ["logical_structures", "argument_forms", "validity_assessment"],
            "aesthetics": [
                "aesthetic_theories",
                "beauty_concepts",
                "artistic_judgments",
            ],
            "political": ["political_theories", "justice_concepts", "power_analysis"],
        }

        content = {}
        category_specific_fields = category_fields.get(category, [])

        for field in category_specific_fields:
            if field in extracted_data:
                content[field] = extracted_data[field]

        # Also look for general content that might be category-relevant
        general_content = extracted_data.get("key_concepts", [])
        if general_content:
            relevant_concepts = [
                concept
                for concept in general_content
                if category.lower() in str(concept).lower()
            ]
            if relevant_concepts:
                content["relevant_concepts"] = relevant_concepts

        return content if content else None

    def _find_common_category_themes(
        self, category_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Find common themes within a specific category"""
        common_themes = []

        try:
            # Collect all themes from all texts in this category
            all_themes = []
            for data in category_data:
                for field, values in data.items():
                    if isinstance(values, list):
                        all_themes.extend([str(v) for v in values])
                    else:
                        all_themes.append(str(values))

            # Find themes that appear in multiple texts
            theme_counts = {}
            for theme in all_themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

            # Themes appearing in at least 2 texts are considered common
            common_themes = [
                theme for theme, count in theme_counts.items() if count >= 2
            ]

        except Exception as e:
            logger.warning(f"Common theme finding failed: {e}")

        return common_themes

    def _generate_philosophical_synthesis(self, text_results: List[Dict]) -> str:
        """Generate a philosophical synthesis of the compared texts"""
        try:
            synthesis_elements = []

            # Collect main theses
            theses = []
            for text_data in text_results:
                result = text_data["result"]
                if result.success and result.extracted_data:
                    thesis = result.extracted_data.get("main_thesis")
                    if thesis:
                        theses.append(thesis)

            if theses:
                synthesis_elements.append(
                    f"The texts present {len(theses)} distinct philosophical positions."
                )

                # Simple synthesis based on common themes
                all_words = []
                for thesis in theses:
                    all_words.extend(thesis.lower().split())

                # Find frequently mentioned concepts
                word_counts = {}
                for word in all_words:
                    if len(word) > 4:  # Filter short words
                        word_counts[word] = word_counts.get(word, 0) + 1

                common_concepts = [
                    word for word, count in word_counts.items() if count >= 2
                ][
                    :5
                ]  # Top 5 common concepts

                if common_concepts:
                    synthesis_elements.append(
                        f"Common philosophical themes include: {', '.join(common_concepts)}."
                    )

            return (
                " ".join(synthesis_elements)
                if synthesis_elements
                else "No clear synthesis could be generated."
            )

        except Exception as e:
            logger.warning(f"Synthesis generation failed: {e}")
            return "Synthesis generation failed due to processing error."

    def analyze_text_for_recommendations(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis for template and parameter recommendations"""
        recommendations = {
            "templates": [],
            "categories": [],
            "depth_level": "detailed",
            "extraction_mode": "comprehensive",
            "estimated_processing_time": 60,
            "confidence": 0.0,
            "text_analysis": {},
            "field_recommendations": [],
            "optimization_suggestions": [],
        }

        try:
            # Enhanced text analysis
            word_count = len(text.split())
            sentence_count = len([s for s in text.split(".") if s.strip()])

            recommendations["text_analysis"] = {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "estimated_complexity": self._estimate_text_complexity(text),
                "detected_language": self._detect_primary_language(text),
                "philosophical_indicators": self._detect_philosophical_indicators(text),
            }

            # Template matching with enhanced analysis
            template_name, confidence, details = self.template_matcher.match_template(
                text
            )
            recommendations["templates"] = [template_name]
            recommendations["confidence"] = confidence

            # Add alternative templates
            if confidence < 0.8:
                alternatives = self._get_alternative_templates(text, word_count)
                recommendations["templates"].extend(alternatives)

            # Enhanced category analysis
            if KNOWLEDGE_BASE_AVAILABLE:
                try:
                    entities = extract_entities_from_text(
                        text, confidence_threshold=0.5
                    )
                    category_suggestions = self._analyze_entities_for_categories(
                        entities
                    )
                    recommendations["categories"].extend(category_suggestions)
                except Exception as e:
                    logger.warning(f"Knowledge base category analysis failed: {e}")

            # Text complexity-based recommendations
            complexity = recommendations["text_analysis"]["estimated_complexity"]
            if complexity == "high":
                recommendations["depth_level"] = "expert"
                recommendations["estimated_processing_time"] = 180
                recommendations["optimization_suggestions"].append(
                    "Consider breaking into smaller sections"
                )
            elif complexity == "low":
                recommendations["depth_level"] = "basic"
                recommendations["estimated_processing_time"] = 30

            # Philosophical indicator-based recommendations
            indicators = recommendations["text_analysis"]["philosophical_indicators"]
            if indicators["argument_density"] > 0.3:
                recommendations["extraction_mode"] = "focused"
                recommendations["categories"].append("logic")

            if indicators["concept_density"] > 0.2:
                if "philosophical_concept" not in recommendations["templates"]:
                    recommendations["templates"].append("philosophical_concept")

            # Field recommendations based on content analysis
            recommendations["field_recommendations"] = self._recommend_fields_for_text(
                text, indicators
            )

            # Remove duplicates and validate
            recommendations["categories"] = list(set(recommendations["categories"]))
            recommendations["templates"] = list(set(recommendations["templates"]))

        except Exception as e:
            logger.warning(f"Text analysis failed: {e}")
            recommendations["error"] = str(e)

        return recommendations

    def _estimate_text_complexity(self, text: str) -> str:
        """Estimate the complexity of philosophical text"""
        # Simple heuristics for complexity estimation
        word_count = len(text.split())
        avg_word_length = (
            sum(len(word) for word in text.split()) / word_count
            if word_count > 0
            else 0
        )

        # Count philosophical terms (simple approach)
        philosophical_terms = [
            "epistemology",
            "metaphysics",
            "ontology",
            "phenomenology",
            "hermeneutics",
            "dialectical",
            "categorical",
            "transcendental",
            "existential",
            "nihilism",
            "pragmatism",
            "empiricism",
            "rationalism",
            "materialism",
            "idealism",
        ]

        term_count = sum(
            1 for term in philosophical_terms if term.lower() in text.lower()
        )
        term_density = term_count / word_count if word_count > 0 else 0

        # Complex sentences (simple heuristic)
        complex_sentence_indicators = [
            ";",
            "however",
            "nevertheless",
            "furthermore",
            "moreover",
        ]
        complex_indicators = sum(
            text.lower().count(indicator) for indicator in complex_sentence_indicators
        )

        if avg_word_length > 6 and term_density > 0.02 and complex_indicators > 3:
            return "high"
        elif avg_word_length > 5 or term_density > 0.01 or complex_indicators > 1:
            return "medium"
        else:
            return "low"

    def _detect_primary_language(self, text: str) -> str:
        """Detect the primary language of the text"""
        # Simple language detection
        chinese_chars = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > 0.3:
            return "CN"
        elif chinese_ratio > 0.1:
            return "mixed"
        else:
            return "EN"

    def _detect_philosophical_indicators(self, text: str) -> Dict[str, float]:
        """Detect various philosophical indicators in text"""
        indicators = {
            "argument_density": 0.0,
            "concept_density": 0.0,
            "reference_density": 0.0,
            "question_density": 0.0,
        }

        try:
            word_count = len(text.split())
            if word_count == 0:
                return indicators

            # Argument indicators
            argument_words = [
                "therefore",
                "thus",
                "hence",
                "because",
                "since",
                "if",
                "then",
                "follows",
            ]
            argument_count = sum(text.lower().count(word) for word in argument_words)
            indicators["argument_density"] = argument_count / word_count

            # Concept indicators (capitalized terms, quotes)
            concept_patterns = [
                r'"[^"]+"',  # Quoted terms
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Capitalized terms
            ]
            concept_count = sum(
                len(re.findall(pattern, text)) for pattern in concept_patterns
            )
            indicators["concept_density"] = concept_count / word_count

            # Reference indicators
            reference_patterns = [
                r"\b\d{4}\b",  # Years
                r"\([^)]*\d{4}[^)]*\)",  # Citations with years
                r"\b[A-Z][a-z]+\s+argues?\b",  # "Author argues"
            ]
            reference_count = sum(
                len(re.findall(pattern, text)) for pattern in reference_patterns
            )
            indicators["reference_density"] = reference_count / word_count

            # Question density
            question_count = text.count("?")
            indicators["question_density"] = question_count / word_count

        except Exception as e:
            logger.warning(f"Philosophical indicator detection failed: {e}")

        return indicators

    def _get_alternative_templates(self, text: str, word_count: int) -> List[str]:
        """Get alternative template suggestions"""
        alternatives = []

        # Based on text length
        if word_count < 500:
            alternatives.extend(["philosophy_basic", "philosophical_concept"])
        elif word_count > 2000:
            alternatives.extend(["comprehensive", "historical_philosophy"])

        # Based on content patterns
        if "argument" in text.lower() or "premise" in text.lower():
            alternatives.append("philosophical_argument")

        if any(
            name in text.lower()
            for name in ["plato", "aristotle", "kant", "hegel", "descartes"]
        ):
            alternatives.append("philosopher_profile")

        return alternatives[:3]  # Limit to top 3 alternatives

    def _analyze_entities_for_categories(self, entities: Dict[str, Any]) -> List[str]:
        """Analyze knowledge base entities to suggest categories"""
        categories = []

        try:
            if entities.get("theories"):
                # Map theory types to categories
                theory_category_map = {
                    "ethical": "ethics",
                    "epistemological": "epistemology",
                    "metaphysical": "metaphysics",
                    "political": "political",
                    "aesthetic": "aesthetics",
                }

                for theory in entities["theories"][:5]:  # Check first 5 theories
                    theory_name = str(theory).lower()
                    for theory_type, category in theory_category_map.items():
                        if theory_type in theory_name:
                            categories.append(category)

            if entities.get("schools"):
                # Historical analysis for schools
                categories.append("historical")

            if entities.get("concepts"):
                concepts = entities["concepts"]
                concept_text = " ".join(str(c) for c in concepts[:10]).lower()

                # Analyze concept content
                if any(
                    word in concept_text
                    for word in ["moral", "ethical", "virtue", "duty"]
                ):
                    categories.append("ethics")
                if any(
                    word in concept_text for word in ["knowledge", "truth", "belief"]
                ):
                    categories.append("epistemology")
                if any(
                    word in concept_text for word in ["being", "existence", "reality"]
                ):
                    categories.append("metaphysics")

        except Exception as e:
            logger.warning(f"Entity category analysis failed: {e}")

        return list(set(categories))

    def _recommend_fields_for_text(
        self, text: str, indicators: Dict[str, float]
    ) -> List[str]:
        """Recommend specific fields based on text analysis"""
        field_recommendations = []

        try:
            # Always recommend core fields
            field_recommendations.extend(["main_thesis", "key_concepts"])

            # Based on argument density
            if indicators.get("argument_density", 0) > 0.2:
                field_recommendations.extend(
                    ["arguments", "premises", "conclusion", "logical_structure"]
                )

            # Based on reference density
            if indicators.get("reference_density", 0) > 0.1:
                field_recommendations.extend(
                    ["philosophers", "influences", "historical_context"]
                )

            # Based on concept density
            if indicators.get("concept_density", 0) > 0.15:
                field_recommendations.extend(
                    ["key_concepts", "related_concepts", "definitions"]
                )

            # Based on question density (exploratory nature)
            if indicators.get("question_density", 0) > 0.05:
                field_recommendations.extend(
                    ["key_questions", "philosophical_problems"]
                )

            # Content-based recommendations
            text_lower = text.lower()
            if any(
                word in text_lower for word in ["moral", "ethical", "ought", "should"]
            ):
                field_recommendations.extend(["ethical_principles", "moral_arguments"])

            if any(
                word in text_lower
                for word in ["knowledge", "truth", "belief", "justified"]
            ):
                field_recommendations.extend(
                    ["epistemological_claims", "justification_methods"]
                )

            if any(
                word in text_lower
                for word in ["being", "existence", "reality", "substance"]
            ):
                field_recommendations.extend(
                    ["metaphysical_positions", "ontological_commitments"]
                )

        except Exception as e:
            logger.warning(f"Field recommendation failed: {e}")

        return list(set(field_recommendations))

    def batch_extract(
        self,
        texts: List[str],
        batch_size: int = 5,
        max_concurrent: int = 3,
        **extraction_params,
    ) -> List[ExtractionResult]:
        """Enhanced batch extraction with better concurrency control and progress tracking"""

        async def extract_batch_async():
            semaphore = asyncio.Semaphore(max_concurrent)
            results = []

            async def extract_single(text: str, index: int) -> ExtractionResult:
                async with semaphore:
                    try:
                        # Add progress tracking
                        logger.info(f"Processing text {index + 1}/{len(texts)}")

                        # Run extraction in thread to avoid blocking
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, lambda: self.extract(text, **extraction_params)
                        )

                        # Add batch metadata
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata.update(
                            {
                                "batch_index": index,
                                "batch_size": len(texts),
                                "batch_processing": True,
                            }
                        )

                        return result

                    except Exception as e:
                        logger.error(f"Error processing text {index}: {e}")
                        return ExtractionResult(
                            prompt="",
                            extraction_request={},
                            success=False,
                            error=str(e),
                            metadata={"batch_index": index, "batch_error": True},
                        )

            # Process in batches with progress tracking
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_start_idx = i

                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                )

                batch_tasks = [
                    extract_single(text, batch_start_idx + j)
                    for j, text in enumerate(batch)
                ]

                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                for result in batch_results:
                    if isinstance(result, Exception):
                        error_result = ExtractionResult(
                            prompt="",
                            extraction_request={},
                            success=False,
                            error=str(result),
                            metadata={"batch_error": True},
                        )
                        results.append(error_result)
                    else:
                        results.append(result)

            return results

        # Run batch extraction
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            start_time = datetime.utcnow()
            results = loop.run_until_complete(extract_batch_async())
            total_time = (datetime.utcnow() - start_time).total_seconds()

            # Add batch summary
            successful = sum(1 for r in results if r.success)
            logger.info(
                f"Batch extraction completed: {successful}/{len(results)} successful in {total_time:.2f}s"
            )

            return results
        finally:
            loop.close()

    def extract_multiple_aspects(
        self, text: str, categories: List[str], merge_results: bool = True
    ) -> Dict[str, Any]:
        """Enhanced multi-aspect extraction with better integration"""
        results = {}

        for category in categories:
            result = self.extract(
                text=text,
                categories=[category],
                extraction_mode="focused",
                enable_knowledge_enhancement=True,
                confidence_threshold=0.6,  # Lower threshold for focused extraction
                auto_template_selection=True,
            )
            results[category] = result

        if merge_results:
            merged = {
                "text": text,
                "aspects": results,
                "summary": self._enhanced_summarize_aspects(results),
                "overall_confidence": self._calculate_overall_confidence(results),
                "processing_time": sum(
                    r.processing_time or 0 for r in results.values()
                ),
                "cross_aspect_analysis": self._analyze_cross_aspects(results),
                "knowledge_enhancement_status": all(
                    r.knowledge_enhanced for r in results.values() if r.success
                ),
            }
            return merged

        return results

    def _enhanced_summarize_aspects(
        self, results: Dict[str, ExtractionResult]
    ) -> Dict[str, Any]:
        """Enhanced aspect summarization with cross-analysis"""
        summary = {
            "total_aspects": len(results),
            "successful_extractions": sum(1 for r in results.values() if r.success),
            "main_themes": [],
            "key_concepts": [],
            "philosophers_mentioned": [],
            "confidence_distribution": {},
            "processing_stats": {},
            "aspect_correlations": {},
            "unified_themes": [],
        }

        try:
            all_concepts = []
            all_philosophers = []

            # Enhanced aggregation with cross-referencing
            for aspect, result in results.items():
                if result.success and result.extracted_data:
                    data = result.extracted_data

                    # Collect themes with enhanced context
                    if "main_thesis" in data:
                        theme_entry = {
                            "aspect": aspect,
                            "thesis": data["main_thesis"],
                            "confidence": result.confidence_scores.get(
                                "main_thesis", 0.0
                            ),
                            "supporting_concepts": data.get("key_concepts", [])[
                                :3
                            ],  # Top 3 supporting concepts
                        }
                        summary["main_themes"].append(theme_entry)

                    # Enhanced concept collection with deduplication
                    if "key_concepts" in data:
                        concepts_with_context = []
                        for concept in data["key_concepts"]:
                            concept_entry = {
                                "concept": concept,
                                "aspect": aspect,
                                "confidence": result.confidence_scores.get(
                                    "key_concepts", 0.0
                                ),
                                "cross_referenced": False,  # Will be updated in cross-analysis
                            }
                            concepts_with_context.append(concept_entry)
                            all_concepts.append(str(concept).lower())
                        summary["key_concepts"].extend(concepts_with_context)

                    # Philosopher collection
                    if "philosophers" in data:
                        for philosopher in data["philosophers"]:
                            all_philosophers.append(str(philosopher).lower())

                    # Enhanced confidence distribution
                    if result.confidence_scores:
                        summary["confidence_distribution"][aspect] = {
                            "average": sum(result.confidence_scores.values())
                            / len(result.confidence_scores),
                            "min": min(result.confidence_scores.values()),
                            "max": max(result.confidence_scores.values()),
                            "field_count": len(result.confidence_scores),
                        }

                    # Processing statistics
                    summary["processing_stats"][aspect] = {
                        "processing_time": result.processing_time,
                        "strategy_used": result.strategy_used,
                        "knowledge_enhanced": result.knowledge_enhanced,
                        "field_coverage": (
                            len(result.field_coverage) if result.field_coverage else 0
                        ),
                    }

            # Cross-aspect analysis
            summary["aspect_correlations"] = self._calculate_aspect_correlations(
                results
            )

            # Find unified themes across aspects
            summary["unified_themes"] = self._find_unified_themes(
                summary["main_themes"]
            )

            # Update cross-referenced status for concepts
            concept_counts = {}
            for concept in all_concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

            for concept_entry in summary["key_concepts"]:
                concept_key = str(concept_entry["concept"]).lower()
                if concept_counts.get(concept_key, 0) > 1:
                    concept_entry["cross_referenced"] = True

            # Deduplicate philosophers
            summary["philosophers_mentioned"] = list(set(all_philosophers))

        except Exception as e:
            logger.warning(f"Enhanced aspect summarization failed: {e}")

        return summary

    def _calculate_aspect_correlations(
        self, results: Dict[str, ExtractionResult]
    ) -> Dict[str, Any]:
        """Calculate correlations between different aspects"""
        correlations = {
            "concept_overlap": {},
            "theme_similarity": {},
            "confidence_correlation": 0.0,
        }

        try:
            aspects = list(results.keys())

            # Calculate concept overlap between aspects
            for i, aspect1 in enumerate(aspects):
                for aspect2 in aspects[i + 1 :]:
                    result1 = results[aspect1]
                    result2 = results[aspect2]

                    if (
                        result1.success
                        and result2.success
                        and result1.extracted_data
                        and result2.extracted_data
                    ):
                        concepts1 = set(
                            str(c).lower()
                            for c in result1.extracted_data.get("key_concepts", [])
                        )
                        concepts2 = set(
                            str(c).lower()
                            for c in result2.extracted_data.get("key_concepts", [])
                        )

                        if concepts1 and concepts2:
                            overlap = len(concepts1.intersection(concepts2))
                            total = len(concepts1.union(concepts2))
                            overlap_ratio = overlap / total if total > 0 else 0

                            correlations["concept_overlap"][
                                f"{aspect1}_vs_{aspect2}"
                            ] = {
                                "overlap_ratio": overlap_ratio,
                                "shared_concepts": list(
                                    concepts1.intersection(concepts2)
                                ),
                                "total_concepts": total,
                            }

            # Calculate confidence correlation
            confidences = []
            for result in results.values():
                if result.success and result.confidence_scores:
                    avg_confidence = sum(result.confidence_scores.values()) / len(
                        result.confidence_scores
                    )
                    confidences.append(avg_confidence)

            if len(confidences) > 1:
                mean_confidence = sum(confidences) / len(confidences)
                variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(
                    confidences
                )
                correlations["confidence_correlation"] = (
                    1 - variance
                )  # Higher correlation = lower variance

        except Exception as e:
            logger.warning(f"Aspect correlation calculation failed: {e}")

        return correlations

    def _find_unified_themes(
        self, main_themes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find themes that appear across multiple aspects"""
        unified_themes = []

        try:
            if len(main_themes) < 2:
                return unified_themes

            # Simple keyword-based unification
            theme_keywords = {}
            for theme in main_themes:
                thesis = theme.get("thesis", "").lower()
                words = [
                    word for word in thesis.split() if len(word) > 4
                ]  # Filter short words

                for word in words:
                    if word not in theme_keywords:
                        theme_keywords[word] = []
                    theme_keywords[word].append(theme)

            # Find keywords that appear in multiple themes
            for keyword, associated_themes in theme_keywords.items():
                if len(associated_themes) >= 2:  # Appears in at least 2 themes
                    unified_theme = {
                        "keyword": keyword,
                        "aspects_involved": [
                            theme["aspect"] for theme in associated_themes
                        ],
                        "unified_confidence": sum(
                            theme["confidence"] for theme in associated_themes
                        )
                        / len(associated_themes),
                        "theme_count": len(associated_themes),
                    }
                    unified_themes.append(unified_theme)

            # Sort by theme count and confidence
            unified_themes.sort(
                key=lambda x: (x["theme_count"], x["unified_confidence"]), reverse=True
            )

        except Exception as e:
            logger.warning(f"Unified theme finding failed: {e}")

        return unified_themes[:5]  # Return top 5 unified themes

    def _analyze_cross_aspects(
        self, results: Dict[str, ExtractionResult]
    ) -> Dict[str, Any]:
        """Analyze relationships and connections across different aspects"""
        cross_analysis = {
            "interconnections": [],
            "contradictions": [],
            "complementary_insights": [],
            "coverage_gaps": [],
        }

        try:
            successful_results = {
                k: v for k, v in results.items() if v.success and v.extracted_data
            }

            if len(successful_results) < 2:
                return cross_analysis

            aspects = list(successful_results.keys())

            # Find interconnections
            for i, aspect1 in enumerate(aspects):
                for aspect2 in aspects[i + 1 :]:
                    interconnection = self._find_aspect_interconnection(
                        successful_results[aspect1],
                        successful_results[aspect2],
                        aspect1,
                        aspect2,
                    )
                    if interconnection:
                        cross_analysis["interconnections"].append(interconnection)

            # Identify coverage gaps
            all_expected_fields = set()
            covered_fields = set()

            for result in successful_results.values():
                if result.field_coverage:
                    for field, coverage in result.field_coverage.items():
                        all_expected_fields.add(field)
                        if coverage > 0.5:  # Consider field covered if > 50% coverage
                            covered_fields.add(field)

            uncovered_fields = all_expected_fields - covered_fields
            if uncovered_fields:
                cross_analysis["coverage_gaps"] = list(uncovered_fields)

        except Exception as e:
            logger.warning(f"Cross-aspect analysis failed: {e}")

        return cross_analysis

    def _find_aspect_interconnection(
        self,
        result1: ExtractionResult,
        result2: ExtractionResult,
        aspect1: str,
        aspect2: str,
    ) -> Optional[Dict[str, Any]]:
        """Find interconnections between two aspect extraction results"""
        try:
            data1 = result1.extracted_data
            data2 = result2.extracted_data

            interconnection = {
                "aspects": [aspect1, aspect2],
                "connection_type": "conceptual",
                "strength": 0.0,
                "shared_elements": [],
                "relationship": "neutral",
            }

            # Find shared concepts
            concepts1 = set(str(c).lower() for c in data1.get("key_concepts", []))
            concepts2 = set(str(c).lower() for c in data2.get("key_concepts", []))
            shared_concepts = concepts1.intersection(concepts2)

            if shared_concepts:
                interconnection["shared_elements"] = list(shared_concepts)
                interconnection["strength"] = len(shared_concepts) / len(
                    concepts1.union(concepts2)
                )
                interconnection["connection_type"] = "conceptual_overlap"

            # Check for complementary insights
            thesis1 = data1.get("main_thesis", "").lower()
            thesis2 = data2.get("main_thesis", "").lower()

            if thesis1 and thesis2:
                # Simple complementarity check
                if "supports" in thesis1 and aspect2 in thesis1:
                    interconnection["relationship"] = "supportive"
                    interconnection["strength"] = max(interconnection["strength"], 0.7)
                elif "contradicts" in thesis1 and aspect2 in thesis1:
                    interconnection["relationship"] = "contradictory"
                    interconnection["strength"] = max(interconnection["strength"], 0.6)

            # Return interconnection if strength is meaningful
            return interconnection if interconnection["strength"] > 0.2 else None

        except Exception as e:
            logger.warning(f"Interconnection analysis failed: {e}")
            return None

    def _calculate_overall_confidence(
        self, results: Dict[str, ExtractionResult]
    ) -> float:
        """Calculate overall confidence across all aspect results"""
        total_confidence = 0.0
        count = 0

        for result in results.values():
            if result.success and result.confidence_scores:
                avg_confidence = sum(result.confidence_scores.values()) / len(
                    result.confidence_scores
                )
                total_confidence += avg_confidence
                count += 1

        return total_confidence / count if count > 0 else 0.0

    def clear_cache(self):
        """Clear the extraction cache"""
        if self.enable_caching:
            self._cache.clear()
            logger.info("Extraction cache cleared")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            "cache_hit_rate": 0.0,
            "average_processing_time": 0.0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_extractions": 0,
            "strategy_usage": {
                "ollama": 0,
                "advanced": 0,
                "enhanced_legacy": 0,
                "error": 0,
            },
            "knowledge_enhancement_rate": 0.0,
            "average_confidence": 0.0,
        }

        # This would be implemented with proper metrics collection
        # For now, return basic structure with cache information
        if self.enable_caching:
            metrics["cache_size"] = len(self._cache)
            metrics["cache_max_size"] = self._cache_size

        return metrics

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "version": "2.0.0",
            "components": {
                "advanced_extractor": ADVANCED_EXTRACTOR_AVAILABLE,
                "ollama_integration": OLLAMA_AVAILABLE and self.use_ollama,
                "knowledge_base": KNOWLEDGE_BASE_AVAILABLE,
                "caching": self.enable_caching,
            },
            "features": {
                "auto_template_selection": True,
                "batch_processing": True,
                "knowledge_enhancement": KNOWLEDGE_BASE_AVAILABLE,
                "validation": True,
                "confidence_scoring": True,
                "field_coverage_analysis": True,
                "cross_aspect_analysis": True,
                "template_recommendation": True,
            },
            "statistics": {
                "templates_available": len(
                    philosophy_template_library.get_all_templates()
                ),
                "fields_available": len(philosophy_field_registry._fields),
                "field_sets_available": len(list(PhilosophyFieldSet)),
                "categories_available": len(list(PhilosophicalCategory)),
                "cache_size": len(self._cache) if self.enable_caching else 0,
            },
            "configuration": {
                "default_language": self.default_language,
                "default_template": self.default_template,
                "use_advanced_extractor": self.use_advanced_extractor,
                "use_ollama": self.use_ollama,
                "cache_enabled": self.enable_caching,
                "cache_size": self._cache_size if self.enable_caching else 0,
            },
        }

        # Add knowledge base statistics if available
        if KNOWLEDGE_BASE_AVAILABLE:
            try:
                kb_stats = validate_knowledge_base()
                info["knowledge_base"] = kb_stats
            except Exception as e:
                info["knowledge_base"] = {"error": str(e)}

        # Add Ollama connection status if enabled
        if self.use_ollama and self.ollama_strategy:
            info["ollama_status"] = {
                "connected": self.ollama_strategy._connection_status,
                "model": self.ollama_strategy.model_name,
                "tested": self.ollama_strategy._connection_tested,
            }

        return info


# Export main classes and functions
__all__ = [
    "PhilosophyExtractorAPI",
    "ExtractionRequest",
    "ExtractionResult",
    "ExtractionStrategy",
]
