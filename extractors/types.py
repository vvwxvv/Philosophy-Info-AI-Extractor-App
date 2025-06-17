from enum import Enum
from typing import List, Optional, Dict, Any, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum


class DataType(Enum):
    """Data types for field values"""

    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    YEAR = "year"
    DIMENSION = "dimension"
    CURRENCY = "currency"
    URL = "url"
    ARRAY = "array"
    BOOLEAN = "boolean"  # 添加 BOOLEAN 类型
    OBJECT = "object"  # Add OBJECT type

    @property
    def pattern(self) -> str:
        """Get regex pattern for validation"""
        patterns = {
            self.STRING: r".*",  # 匹配任意字符串
            self.NUMBER: r"^-?\d+(\.\d+)?$",  # 匹配整数或浮点数
            self.DATE: r"^\d{4}-\d{2}-\d{2}$",  # 匹配日期格式 YYYY-MM-DD
            self.YEAR: r"^\d{4}$",  # 匹配四位年份
            self.DIMENSION: r"^\d+(\.\d+)?\s*[x×]\s*\d+(\.\d+)?\s*(cm|m|mm|inch|英尺)?$",  # 匹配尺寸格式
            self.CURRENCY: r"^[A-Z]{3}$",  # 匹配三位货币代码
            self.URL: r"^https?://.*",  # 匹配 URL
            self.ARRAY: r".*",  # 匹配任意数组
            self.BOOLEAN: r"^(True|False)$",  # 匹配布尔值 True 或 False
            self.OBJECT: r".*",  # 匹配任意对象
        }
        return patterns[self]


class PhilosophySourceType(str, Enum):
    """Types of philosophical content that can be analyzed"""

    ESSAY = "essay"
    DIALOGUE = "dialogue"
    TREATISE = "treatise"
    LECTURE = "lecture"
    COMMENTARY = "commentary"
    CRITIQUE = "critique"
    MANIFESTO = "manifesto"
    APHORISM = "aphorism"
    JOURNAL_ARTICLE = "journal_article"
    BOOK_CHAPTER = "book_chapter"
    FRAGMENT = "fragment"
    LETTER = "letter"
    NOTEBOOK = "notebook"

    @property
    def description(self) -> str:
        """Get a human-readable description of the source type"""
        descriptions = {
            "essay": "A philosophical essay or article exploring specific ideas",
            "dialogue": "A philosophical dialogue or conversation between interlocutors",
            "treatise": "A formal, systematic philosophical treatise or book",
            "lecture": "A philosophical lecture or academic presentation",
            "commentary": "A commentary on existing philosophical works",
            "critique": "A critical analysis of philosophical ideas or systems",
            "manifesto": "A philosophical manifesto or declaration of principles",
            "aphorism": "A collection of philosophical aphorisms or maxims",
            "journal_article": "An academic journal article on philosophy",
            "book_chapter": "A chapter from a philosophical book",
            "fragment": "A philosophical fragment or incomplete text",
            "letter": "A philosophical letter or correspondence",
            "notebook": "Philosophical notebooks or personal reflections",
        }
        return descriptions.get(self.value, f"A {self.value.replace('_', ' ')}")

    @property
    def typical_length(self) -> str:
        """Get typical length for this source type"""
        lengths = {
            "essay": "5-30 pages",
            "dialogue": "10-50 pages",
            "treatise": "100-500 pages",
            "lecture": "10-30 pages",
            "commentary": "20-100 pages",
            "critique": "10-40 pages",
            "manifesto": "5-20 pages",
            "aphorism": "1-10 pages",
            "journal_article": "15-40 pages",
            "book_chapter": "20-50 pages",
            "fragment": "1-5 pages",
            "letter": "2-10 pages",
            "notebook": "varies",
        }
        return lengths.get(self.value, "varies")

    @classmethod
    def from_text_length(cls, word_count: int) -> "PhilosophySourceType":
        """Suggest source type based on text length"""
        if word_count < 500:
            return cls.FRAGMENT
        elif word_count < 2000:
            return cls.APHORISM
        elif word_count < 10000:
            return cls.ESSAY
        elif word_count < 50000:
            return cls.JOURNAL_ARTICLE
        else:
            return cls.TREATISE


class ExtractionDepth(str, Enum):
    """Depth levels for extraction"""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    DETAILED = "detailed"
    EXPERT = "expert"

    @property
    def complexity_score(self) -> int:
        """Get complexity score (1-5)"""
        scores = {"basic": 1, "intermediate": 2, "detailed": 3, "expert": 5}
        return scores[self.value]

    @property
    def description(self) -> str:
        """Get description of depth level"""
        descriptions = {
            "basic": "Essential concepts and main arguments only",
            "intermediate": "Core ideas with some context and analysis",
            "detailed": "Comprehensive analysis with full context",
            "expert": "Deep scholarly analysis with all nuances",
        }
        return descriptions[self.value]


class TargetAudience(str, Enum):
    """Target audiences for extraction output"""

    ACADEMIC = "academic"
    GENERAL = "general"
    PROFESSIONAL = "professional"
    EDUCATIONAL = "educational"
    RESEARCH = "research"

    @property
    def formality_level(self) -> str:
        """Get expected formality level"""
        levels = {
            "academic": "highly formal",
            "general": "accessible",
            "professional": "formal",
            "educational": "clear and structured",
            "research": "technical",
        }
        return levels[self.value]


class ExtractionMode(str, Enum):
    """Modes of philosophical extraction"""

    COMPREHENSIVE = "comprehensive"
    FOCUSED = "focused"
    EXPLORATORY = "exploratory"
    COMPARATIVE = "comparative"
    THEMATIC = "thematic"
    HISTORICAL = "historical"
    CRITICAL = "critical"

    @property
    def description(self) -> str:
        """Get mode description"""
        descriptions = {
            "comprehensive": "Extract all philosophical content systematically",
            "focused": "Focus on specific philosophical aspects",
            "exploratory": "Explore philosophical themes without preconceptions",
            "comparative": "Compare philosophical positions and arguments",
            "thematic": "Extract content organized by themes",
            "historical": "Focus on historical development and context",
            "critical": "Critical analysis of arguments and positions",
        }
        return descriptions[self.value]


class PhilosophicalCategory(str, Enum):
    """Categories of philosophical content"""

    METAPHYSICS = "metaphysics"
    EPISTEMOLOGY = "epistemology"
    ETHICS = "ethics"
    AESTHETICS = "aesthetics"
    LOGIC = "logic"
    POLITICAL = "political"
    PHILOSOPHY_OF_MIND = "philosophy_of_mind"
    PHILOSOPHY_OF_LANGUAGE = "philosophy_of_language"
    PHILOSOPHY_OF_SCIENCE = "philosophy_of_science"
    PHILOSOPHY_OF_RELIGION = "philosophy_of_religion"
    CONTINENTAL = "continental"
    ANALYTIC = "analytic"
    EASTERN = "eastern"

    @property
    def subcategories(self) -> List[str]:
        """Get subcategories"""
        subcats = {
            "metaphysics": ["ontology", "cosmology", "philosophy_of_time", "modality"],
            "epistemology": [
                "skepticism",
                "rationalism",
                "empiricism",
                "phenomenology",
            ],
            "ethics": [
                "normative_ethics",
                "metaethics",
                "applied_ethics",
                "virtue_ethics",
            ],
            "logic": [
                "formal_logic",
                "informal_logic",
                "modal_logic",
                "philosophical_logic",
            ],
            "political": [
                "political_theory",
                "social_philosophy",
                "philosophy_of_law",
                "justice",
            ],
        }
        return subcats.get(self.value, [])


class OutputFormat(str, Enum):
    """Available output formats for extraction results"""

    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    MARKDOWN = "markdown"
    TABLE = "table"
    CUSTOM = "custom"

    @property
    def description(self) -> str:
        """Get format description"""
        descriptions = {
            "json": "Structured JSON format with nested objects and arrays",
            "csv": "Comma-separated values format for tabular data",
            "xml": "Extensible Markup Language format with tags",
            "yaml": "YAML format with human-readable structure",
            "markdown": "Markdown format for documentation",
            "table": "Simple table format with headers and rows",
            "custom": "Custom format defined by user",
        }
        return descriptions.get(self.value, f"{self.value} format")

    @property
    def mime_type(self) -> str:
        """Get MIME type for the format"""
        mime_types = {
            "json": "application/json",
            "csv": "text/csv",
            "xml": "application/xml",
            "yaml": "application/x-yaml",
            "markdown": "text/markdown",
            "table": "text/plain",
            "custom": "text/plain",
        }
        return mime_types.get(self.value, "text/plain")

    @property
    def file_extension(self) -> str:
        """Get file extension for the format"""
        extensions = {
            "json": ".json",
            "csv": ".csv",
            "xml": ".xml",
            "yaml": ".yaml",
            "markdown": ".md",
            "table": ".txt",
            "custom": ".txt",
        }
        return extensions.get(self.value, ".txt")


@dataclass
class ValidationResult:
    """Result of validation operations"""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message"""
        self.warnings.append(warning)

    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggestion"""
        self.suggestions.append(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


@dataclass
class ExtractionMetadata:
    """Metadata for extraction operations"""

    extraction_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: Optional[float] = None
    model_version: str = "2.0"
    extractor_version: str = "1.0"
    source_hash: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "extraction_id": self.extraction_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "model_version": self.model_version,
            "extractor_version": self.extractor_version,
            "source_hash": self.source_hash,
            "parameters": self.parameters,
            "statistics": self.statistics,
        }


@dataclass
class PhilosophyExtractionConfig:
    """Enhanced configuration for philosophical text extraction"""

    source_type: PhilosophySourceType
    language: str = "mixed"
    extraction_depth: ExtractionDepth = ExtractionDepth.DETAILED
    target_audience: TargetAudience = TargetAudience.ACADEMIC
    extraction_mode: ExtractionMode = ExtractionMode.COMPREHENSIVE
    categories_focus: List[PhilosophicalCategory] = field(default_factory=list)

    # Feature flags
    include_examples: bool = True
    include_references: bool = True
    include_historical_context: bool = True
    include_influences: bool = True
    include_criticisms: bool = True
    include_applications: bool = True
    include_cross_references: bool = True

    # Advanced options
    confidence_threshold: float = 0.7
    max_extraction_time: int = 300  # seconds
    preserve_original_language: bool = True
    extract_implicit_content: bool = True

    # Output format configuration
    output_format: OutputFormat = OutputFormat.JSON
    custom_format_template: Optional[str] = None
    include_metadata_in_output: bool = True
    pretty_print: bool = True

    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration"""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters"""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        if self.max_extraction_time < 10:
            raise ValueError("Maximum extraction time must be at least 10 seconds")

        if self.language not in ["CN", "EN", "mixed", "auto"]:
            raise ValueError("Language must be CN, EN, mixed, or auto")

    def validate(self) -> ValidationResult:
        """Comprehensive validation with result object"""
        result = ValidationResult(is_valid=True)

        # Validate confidence threshold
        if self.confidence_threshold < 0.5:
            result.add_warning(
                "Low confidence threshold may include unreliable extractions"
            )

        # Validate extraction time
        if self.max_extraction_time > 600:
            result.add_warning("Very long extraction time may cause timeouts")

        # Validate category focus
        if len(self.categories_focus) > 5:
            result.add_suggestion(
                "Consider focusing on fewer categories for better results"
            )

        # Check feature flag combinations
        if (
            self.extraction_depth == ExtractionDepth.BASIC
            and self.include_cross_references
        ):
            result.add_warning("Cross-references may not be available at basic depth")

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "source_type": self.source_type.value,
            "language": self.language,
            "extraction_depth": self.extraction_depth.value,
            "target_audience": self.target_audience.value,
            "extraction_mode": self.extraction_mode.value,
            "categories_focus": [cat.value for cat in self.categories_focus],
            "features": {
                "include_examples": self.include_examples,
                "include_references": self.include_references,
                "include_historical_context": self.include_historical_context,
                "include_influences": self.include_influences,
                "include_criticisms": self.include_criticisms,
                "include_applications": self.include_applications,
                "include_cross_references": self.include_cross_references,
            },
            "advanced": {
                "confidence_threshold": self.confidence_threshold,
                "max_extraction_time": self.max_extraction_time,
                "preserve_original_language": self.preserve_original_language,
                "extract_implicit_content": self.extract_implicit_content,
            },
            "output_format": self.output_format.value,
            "custom_format_template": self.custom_format_template,
            "include_metadata_in_output": self.include_metadata_in_output,
            "pretty_print": self.pretty_print,
            "custom_parameters": self.custom_parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhilosophyExtractionConfig":
        """Create config from dictionary"""
        # Extract nested structures
        features = data.get("features", {})
        advanced = data.get("advanced", {})
        output_format = data.get("output_format", OutputFormat.JSON)

        return cls(
            source_type=PhilosophySourceType(data["source_type"]),
            language=data.get("language", "mixed"),
            extraction_depth=ExtractionDepth(data.get("extraction_depth", "detailed")),
            target_audience=TargetAudience(data.get("target_audience", "academic")),
            extraction_mode=ExtractionMode(
                data.get("extraction_mode", "comprehensive")
            ),
            categories_focus=[
                PhilosophicalCategory(cat) for cat in data.get("categories_focus", [])
            ],
            include_examples=features.get("include_examples", True),
            include_references=features.get("include_references", True),
            include_historical_context=features.get("include_historical_context", True),
            include_influences=features.get("include_influences", True),
            include_criticisms=features.get("include_criticisms", True),
            include_applications=features.get("include_applications", True),
            include_cross_references=features.get("include_cross_references", True),
            confidence_threshold=advanced.get("confidence_threshold", 0.7),
            max_extraction_time=advanced.get("max_extraction_time", 300),
            preserve_original_language=advanced.get("preserve_original_language", True),
            extract_implicit_content=advanced.get("extract_implicit_content", True),
            output_format=output_format,
            custom_format_template=data.get("custom_format_template"),
            include_metadata_in_output=data.get("include_metadata_in_output", True),
            pretty_print=data.get("pretty_print", True),
            custom_parameters=data.get("custom_parameters", {}),
        )


T = TypeVar("T")


@dataclass
class ExtractionItem(Generic[T]):
    """Generic extraction item with metadata"""

    value: T
    confidence: float = 1.0
    source_location: Optional[Dict[str, int]] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if item meets confidence threshold"""
        return self.confidence >= threshold


@dataclass
class PhilosophyExtractionResult:
    """Enhanced result of a philosophical text extraction"""

    # Core philosophical content
    key_concepts: List[ExtractionItem[str]] = field(default_factory=list)
    arguments: List[ExtractionItem[Dict[str, Any]]] = field(default_factory=list)
    philosophers: List[ExtractionItem[str]] = field(default_factory=list)

    # Philosophical categories
    ethical_principles: List[ExtractionItem[str]] = field(default_factory=list)
    epistemological_claims: List[ExtractionItem[str]] = field(default_factory=list)
    metaphysical_positions: List[ExtractionItem[str]] = field(default_factory=list)
    logical_structures: List[ExtractionItem[str]] = field(default_factory=list)
    aesthetic_theories: List[ExtractionItem[str]] = field(default_factory=list)
    political_theories: List[ExtractionItem[str]] = field(default_factory=list)

    # Contextual information
    historical_context: List[ExtractionItem[str]] = field(default_factory=list)
    influences: List[ExtractionItem[str]] = field(default_factory=list)
    criticisms: List[ExtractionItem[str]] = field(default_factory=list)
    applications: List[ExtractionItem[str]] = field(default_factory=list)

    # Analysis results
    philosophical_tradition: Optional[str] = None
    argument_type: Optional[str] = None
    methodology: Optional[str] = None
    main_thesis: Optional[str] = None

    # Metadata
    metadata: ExtractionMetadata = field(
        default_factory=lambda: ExtractionMetadata(
            extraction_id=f"phil_{datetime.utcnow().timestamp()}"
        )
    )
    validation_result: Optional[ValidationResult] = None

    def filter_by_confidence(
        self, threshold: float = 0.8
    ) -> "PhilosophyExtractionResult":
        """Return a new result with only high-confidence items"""
        filtered = PhilosophyExtractionResult(
            philosophical_tradition=self.philosophical_tradition,
            argument_type=self.argument_type,
            methodology=self.methodology,
            main_thesis=self.main_thesis,
            metadata=self.metadata,
        )

        # Filter all list fields
        for field_name in [
            "key_concepts",
            "arguments",
            "philosophers",
            "ethical_principles",
            "epistemological_claims",
            "metaphysical_positions",
            "logical_structures",
            "aesthetic_theories",
            "political_theories",
            "historical_context",
            "influences",
            "criticisms",
            "applications",
        ]:
            items = getattr(self, field_name)
            filtered_items = [
                item for item in items if item.is_high_confidence(threshold)
            ]
            setattr(filtered, field_name, filtered_items)

        return filtered

    def get_statistics(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return {
            "total_concepts": len(self.key_concepts),
            "total_arguments": len(self.arguments),
            "total_philosophers": len(self.philosophers),
            "total_ethical_principles": len(self.ethical_principles),
            "total_epistemological_claims": len(self.epistemological_claims),
            "total_metaphysical_positions": len(self.metaphysical_positions),
            "high_confidence_items": sum(
                len(
                    [
                        item
                        for item in getattr(self, field_name)
                        if hasattr(item, "is_high_confidence")
                        and item.is_high_confidence()
                    ]
                )
                for field_name in dir(self)
                if isinstance(getattr(self, field_name), list)
            ),
        }

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Convert result to dictionary"""

        def convert_items(items: List[ExtractionItem]) -> List[Any]:
            """Convert extraction items to dict format"""
            return [
                (
                    {
                        "value": item.value,
                        "confidence": item.confidence,
                        "context": item.context,
                        "metadata": item.metadata,
                    }
                    if include_metadata
                    else item.value
                )
                for item in items
            ]

        result = {
            "key_concepts": convert_items(self.key_concepts),
            "arguments": convert_items(self.arguments),
            "philosophers": convert_items(self.philosophers),
            "philosophical_analysis": {
                "tradition": self.philosophical_tradition,
                "argument_type": self.argument_type,
                "methodology": self.methodology,
                "main_thesis": self.main_thesis,
            },
        }

        # Add category-specific content
        categories = {
            "ethics": {"principles": convert_items(self.ethical_principles)},
            "epistemology": {"claims": convert_items(self.epistemological_claims)},
            "metaphysics": {"positions": convert_items(self.metaphysical_positions)},
            "logic": {"structures": convert_items(self.logical_structures)},
            "aesthetics": {"theories": convert_items(self.aesthetic_theories)},
            "political": {"theories": convert_items(self.political_theories)},
        }

        # Only include non-empty categories
        result["categories"] = {k: v for k, v in categories.items() if any(v.values())}

        # Add context
        result["context"] = {
            "historical": convert_items(self.historical_context),
            "influences": convert_items(self.influences),
            "criticisms": convert_items(self.criticisms),
            "applications": convert_items(self.applications),
        }

        # Add metadata if requested
        if include_metadata:
            result["metadata"] = self.metadata.to_dict()
            result["statistics"] = self.get_statistics()
            if self.validation_result:
                result["validation"] = self.validation_result.to_dict()

        return result

    def to_json(self, include_metadata: bool = True, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(include_metadata), indent=indent, default=str)


# Type aliases for convenience
ConfigDict = Dict[str, Any]
ResultDict = Dict[str, Any]
FieldDict = Dict[str, List[str]]
