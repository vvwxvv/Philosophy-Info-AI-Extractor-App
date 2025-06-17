"""Enhanced field definitions for philosophical content extraction"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .types import DataType, PhilosophicalCategory, ValidationResult


# ===== VALIDATION RULES (Moved before usage) =====


@dataclass
class ValidationRule(ABC):
    """Base validation rule"""

    name: str

    @abstractmethod
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate value and return (is_valid, error_message)"""
        pass


@dataclass
class MinMaxRule(ValidationRule):
    """Validate numeric value is within range"""

    min_value: float
    max_value: float
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"min_max_{self.min_value}_{self.max_value}"

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        try:
            num_value = float(value)
            if self.min_value <= num_value <= self.max_value:
                return True, None
            return False, f"Value must be between {self.min_value} and {self.max_value}"
        except (TypeError, ValueError):
            return False, "Value must be numeric"


@dataclass
class PatternRule(ValidationRule):
    """Validate value matches regex pattern"""

    pattern: str
    name: str = field(init=False)
    compiled_pattern: re.Pattern = field(init=False)

    def __post_init__(self):
        self.name = f"pattern_rule_{hash(self.pattern) % 1000}"
        self.compiled_pattern = re.compile(self.pattern)

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if self.compiled_pattern.match(str(value)):
            return True, None
        return False, f"Value does not match required pattern: {self.pattern}"


@dataclass
class LengthRule(ValidationRule):
    """Validate string length"""

    min_length: int = 0
    max_length: int = 10000
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"length_{self.min_length}_{self.max_length}"

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if not isinstance(value, str):
            return False, "Value must be a string"

        length = len(value)
        if self.min_length <= length <= self.max_length:
            return True, None
        return False, f"Length must be between {self.min_length} and {self.max_length}"


@dataclass
class OptionsRule(ValidationRule):
    """Validate value is in allowed options"""

    allowed_options: List[str]
    case_sensitive: bool = False
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"options_rule_{len(self.allowed_options)}"
        if not self.case_sensitive:
            self.allowed_options = [opt.lower() for opt in self.allowed_options]

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        check_value = str(value)
        if not self.case_sensitive:
            check_value = check_value.lower()

        if check_value in self.allowed_options:
            return True, None
        return False, f"Value must be one of: {', '.join(self.allowed_options)}"


@dataclass
class RequiredFieldsRule(ValidationRule):
    """Validate that an object has required fields"""

    required_fields: List[str]
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"required_fields_{len(self.required_fields)}"

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if not isinstance(value, dict):
            return False, "Value must be a dictionary"

        missing_fields = [
            f for f in self.required_fields if f not in value or not value[f]
        ]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        return True, None


# ===== FIELD DEFINITION =====


@dataclass
class PhilosophyExtractionField:
    """Enhanced field definition for philosophical content"""

    name: str
    description: str
    data_type: DataType = DataType.STRING
    required: bool = False
    default_value: Any = None
    validation_rules: List[ValidationRule] = field(default_factory=list)
    extraction_hints: List[str] = field(default_factory=list)
    multilingual: bool = True
    language_detection: bool = True
    priority: int = 1  # 1 = highest priority
    category: Optional[PhilosophicalCategory] = None
    aliases: List[str] = field(default_factory=list)
    post_processors: List[str] = field(default_factory=list)
    extraction_patterns: List[str] = field(default_factory=list)
    contextual_keywords: List[str] = field(default_factory=list)
    examples: Dict[str, List[str]] = field(default_factory=dict)
    related_fields: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set up additional field properties"""
        if not self.category and self.name:
            # Auto-detect category from field name
            category_prefixes = {
                "ethics_": PhilosophicalCategory.ETHICS,
                "metaphysics_": PhilosophicalCategory.METAPHYSICS,
                "epistemology_": PhilosophicalCategory.EPISTEMOLOGY,
                "logic_": PhilosophicalCategory.LOGIC,
                "aesthetics_": PhilosophicalCategory.AESTHETICS,
                "political_": PhilosophicalCategory.POLITICAL,
                "mind_": PhilosophicalCategory.PHILOSOPHY_OF_MIND,
                "language_": PhilosophicalCategory.PHILOSOPHY_OF_LANGUAGE,
                "science_": PhilosophicalCategory.PHILOSOPHY_OF_SCIENCE,
                "religion_": PhilosophicalCategory.PHILOSOPHY_OF_RELIGION,
            }
            for prefix, cat in category_prefixes.items():
                if self.name.startswith(prefix):
                    self.category = cat
                    break

    def validate_value(self, value: Any) -> ValidationResult:
        """Validate a value against all field rules"""
        result = ValidationResult(is_valid=True)

        # Check required field
        if self.required and (
            value is None
            or value == ""
            or (isinstance(value, list) and len(value) == 0)
        ):
            result.add_error(f"Field '{self.name}' is required")
            return result

        # Skip validation for null values on optional fields
        if value is None and not self.required:
            return result

        # Check data type pattern
        if value is not None and self.data_type != DataType.ARRAY:
            pattern_match = re.match(self.data_type.pattern, str(value))
            if not pattern_match:
                result.add_error(
                    f"Field '{self.name}' does not match expected format for type {self.data_type.value}"
                )

        # Check validation rules
        for rule in self.validation_rules:
            is_valid, error_msg = rule.validate(value)
            if not is_valid:
                result.add_error(
                    error_msg
                    or f"Field '{self.name}' failed validation rule '{rule.name}'"
                )

        return result

    def apply_post_processors(self, value: Any) -> Any:
        """Apply post-processors to the value"""
        for processor_name in self.post_processors:
            processor = PhilosophyFieldProcessor.get_processor(processor_name)
            if processor:
                value = processor(value)
        return value


# ===== FIELD PROCESSORS =====


class PhilosophyFieldProcessor:
    """Post-processing functions for philosophical fields"""

    @staticmethod
    def normalize_philosopher_name(value: str) -> str:
        """Normalize philosopher names"""
        if not isinstance(value, str):
            return value

        # Remove titles
        value = re.sub(
            r"\b(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.|Sir|Lord|Saint|St\.)\s*", "", value
        )

        # Standardize spacing
        value = " ".join(value.split())

        # Handle "of Location" patterns
        value = re.sub(r"\s+of\s+", " of ", value)

        # Capitalize properly
        parts = value.split()
        normalized_parts = []
        for part in parts:
            if part.lower() in ["of", "the", "von", "de", "van", "der"]:
                normalized_parts.append(part.lower())
            else:
                normalized_parts.append(part.capitalize())

        return " ".join(normalized_parts).strip()

    @staticmethod
    def extract_philosophical_tradition(value: str) -> str:
        """Extract philosophical tradition from text"""
        if not isinstance(value, str):
            return value

        traditions = {
            "existentialism": [
                "existential",
                "existence",
                "authenticity",
                "absurd",
                "angst",
            ],
            "pragmatism": [
                "pragmatic",
                "practical",
                "usefulness",
                "instrumentalism",
                "james",
                "dewey",
            ],
            "stoicism": [
                "stoic",
                "virtue",
                "indifferent",
                "apatheia",
                "marcus aurelius",
                "epictetus",
            ],
            "platonism": ["forms", "ideal", "platonic", "plato", "academy", "idea"],
            "aristotelianism": [
                "aristotelian",
                "virtue",
                "golden mean",
                "aristotle",
                "peripatetic",
            ],
            "kantian": [
                "categorical imperative",
                "synthetic a priori",
                "kant",
                "transcendental",
                "noumenal",
            ],
            "utilitarianism": [
                "utility",
                "greatest happiness",
                "consequentialism",
                "mill",
                "bentham",
            ],
            "phenomenology": [
                "phenomenological",
                "intentionality",
                "lived experience",
                "husserl",
                "heidegger",
            ],
            "empiricism": [
                "empirical",
                "experience",
                "hume",
                "locke",
                "berkeley",
                "sensation",
            ],
            "rationalism": [
                "rational",
                "reason",
                "descartes",
                "spinoza",
                "leibniz",
                "innate ideas",
            ],
            "idealism": [
                "ideal",
                "mind",
                "spirit",
                "hegel",
                "berkeley",
                "consciousness",
            ],
            "materialism": [
                "material",
                "matter",
                "physical",
                "marx",
                "epicurus",
                "democritus",
            ],
        }

        value_lower = value.lower()
        matches = []

        for tradition, keywords in traditions.items():
            score = sum(1 for keyword in keywords if keyword in value_lower)
            if score > 0:
                matches.append((tradition, score))

        if matches:
            # Return the tradition with highest score
            best_match = max(matches, key=lambda x: x[1])
            return best_match[0].title()

        return value

    @staticmethod
    def parse_argument_structure(value: str) -> Dict[str, Any]:
        """Parse argument structure from text"""
        if not isinstance(value, str):
            return {"raw": value}

        # Initialize structure
        structure = {
            "premises": [],
            "conclusion": None,
            "type": "unknown",
            "indicators": [],
            "raw": value,
        }

        # Extract premises
        premise_patterns = [
            (
                r"premise[s]?\s*\d*\s*[:：]\s*(.+?)(?=premise|conclusion|therefore|$)",
                "explicit",
            ),
            (
                r"(?:since|because|given that|for|as)\s+(.+?)(?=therefore|thus|hence|consequently|so|$)",
                "causal",
            ),
            (
                r"(?:first|second|third|furthermore|moreover)\s*,?\s*(.+?)(?=[.;]|$)",
                "enumerated",
            ),
        ]

        for pattern, indicator_type in premise_patterns:
            matches = re.finditer(pattern, value, re.IGNORECASE | re.DOTALL)
            for match in matches:
                premise_text = match.group(1).strip()
                if premise_text:
                    structure["premises"].append(premise_text)
                    if indicator_type not in structure["indicators"]:
                        structure["indicators"].append(indicator_type)

        # Extract conclusion
        conclusion_patterns = [
            (r"conclusion\s*[:：]\s*(.+?)(?:[.!?]|$)", "explicit"),
            (
                r"(?:therefore|thus|hence|consequently|so)\s+(.+?)(?:[.!?]|$)",
                "inferential",
            ),
            (r"(?:it follows that|we can conclude that)\s+(.+?)(?:[.!?]|$)", "formal"),
        ]

        for pattern, indicator_type in conclusion_patterns:
            match = re.search(pattern, value, re.IGNORECASE | re.DOTALL)
            if match:
                structure["conclusion"] = match.group(1).strip()
                if indicator_type not in structure["indicators"]:
                    structure["indicators"].append(indicator_type)
                break

        # Determine argument type based on indicators and content
        if "causal" in structure["indicators"]:
            structure["type"] = "causal"
        elif (
            "inferential" in structure["indicators"]
            or "formal" in structure["indicators"]
        ):
            structure["type"] = "deductive"
        elif len(structure["premises"]) > 2:
            structure["type"] = "inductive"

        return structure

    @staticmethod
    def detect_language(value: str) -> str:
        """Detect if text is primarily Chinese, English, or other"""
        if not isinstance(value, str) or not value.strip():
            return "unknown"

        # Count character types
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", value))
        latin_chars = len(re.findall(r"[a-zA-Z]", value))
        greek_chars = len(re.findall(r"[\u0370-\u03ff]", value))
        total_chars = len(re.sub(r"\s+", "", value))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars
        latin_ratio = latin_chars / total_chars
        greek_ratio = greek_chars / total_chars

        if chinese_ratio > 0.3:
            return "zh"
        elif greek_ratio > 0.1:
            return "el"  # Greek
        elif latin_ratio > 0.5:
            return "en"
        else:
            return "mixed"

    @staticmethod
    def extract_historical_period(value: str) -> str:
        """Extract historical period from text"""
        if not isinstance(value, str):
            return value

        # Look for century patterns
        century_patterns = [
            (r"(\d{1,2})(?:st|nd|rd|th)\s+century\s*(?:BCE?|CE|AD)?", "century"),
            (r"(\d{3,4})\s*(?:BCE?|CE|AD)", "year"),
            (r"(?:circa|c\.)\s*(\d{3,4})", "circa"),
        ]

        for pattern, period_type in century_patterns:
            match = re.search(pattern, value, re.IGNORECASE)
            if match:
                if period_type == "century":
                    return f"{match.group(1)}th century"
                else:
                    return match.group(0)

        # Look for named periods
        periods = {
            "ancient": [
                "ancient",
                "classical",
                "antiquity",
                "pre-socratic",
                "hellenistic",
            ],
            "medieval": [
                "medieval",
                "middle ages",
                "scholastic",
                "dark ages",
                "byzantine",
            ],
            "renaissance": ["renaissance", "humanism", "reformation", "early modern"],
            "enlightenment": [
                "enlightenment",
                "age of reason",
                "18th century",
                "lumières",
            ],
            "modern": ["modern", "19th century", "industrial", "romantic"],
            "contemporary": [
                "contemporary",
                "20th century",
                "21st century",
                "postmodern",
                "current",
            ],
        }

        value_lower = value.lower()
        for period, keywords in periods.items():
            if any(keyword in value_lower for keyword in keywords):
                return period.title()

        return value

    @staticmethod
    def extract_key_terms(value: str) -> List[str]:
        """Extract key philosophical terms from text"""
        if not isinstance(value, str):
            return []

        # Common philosophical terms to look for
        term_patterns = [
            r"\b[A-Z][a-z]+(?:ism|ology|istic|ian)\b",  # -ism, -ology words
            r'"([^"]+)"',  # Quoted terms
            r"'([^']+)'",  # Single quoted terms
            r"\b(?:concept|theory|principle|doctrine)\s+of\s+(\w+)",  # "concept of X"
        ]

        terms = set()
        for pattern in term_patterns:
            matches = re.finditer(pattern, value)
            for match in matches:
                term = match.group(1) if match.lastindex else match.group(0)
                terms.add(term.strip())

        return list(terms)

    @staticmethod
    def standardize_philosophical_terms(value: str) -> str:
        """Standardize common philosophical terms"""
        if not isinstance(value, str):
            return value

        # Common variations to standardize
        standardizations = {
            r"\b(?:platonic|platonist)\b": "Platonic",
            r"\b(?:aristotelian|aristotelean)\b": "Aristotelian",
            r"\b(?:kantian|kant\'s)\b": "Kantian",
            r"\b(?:hegelian|hegel\'s)\b": "Hegelian",
            r"\b(?:cartesian|descartes\')\b": "Cartesian",
            r"\b(?:humean|hume\'s)\b": "Humean",
            r"\b(?:socratic|socrates\')\b": "Socratic",
        }

        result = value
        for pattern, replacement in standardizations.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    @staticmethod
    def get_processor(name: str):
        """Get processor function by name"""
        processors = {
            "normalize_philosopher": PhilosophyFieldProcessor.normalize_philosopher_name,
            "extract_tradition": PhilosophyFieldProcessor.extract_philosophical_tradition,
            "parse_argument": PhilosophyFieldProcessor.parse_argument_structure,
            "detect_language": PhilosophyFieldProcessor.detect_language,
            "extract_period": PhilosophyFieldProcessor.extract_historical_period,
            "extract_key_terms": PhilosophyFieldProcessor.extract_key_terms,
            "standardize_terms": PhilosophyFieldProcessor.standardize_philosophical_terms,
        }
        return processors.get(name)


# ===== FIELD SETS =====


class PhilosophyFieldSet(str, Enum):
    """Predefined field sets for philosophical extraction"""

    BASIC = "philosophy_basic"
    PHILOSOPHER = "philosopher_profile"
    CONCEPT = "philosophical_concept"
    ARGUMENT = "philosophical_argument"
    THEORY = "philosophical_theory"
    TEXT_ANALYSIS = "text_analysis"
    COMPARATIVE = "comparative_philosophy"
    HISTORICAL = "historical_philosophy"
    CRITICAL = "critical_analysis"
    COMPREHENSIVE = "comprehensive_philosophy"

    @property
    def description(self) -> str:
        """Get field set description"""
        descriptions = {
            "philosophy_basic": "Essential philosophical information",
            "philosopher_profile": "Detailed philosopher biographical and intellectual profile",
            "philosophical_concept": "In-depth philosophical concept analysis",
            "philosophical_argument": "Detailed argument structure and logic analysis",
            "philosophical_theory": "Comprehensive theory examination",
            "text_analysis": "Philosophical text interpretation and analysis",
            "comparative_philosophy": "Comparative analysis of philosophical positions",
            "historical_philosophy": "Historical development and context analysis",
            "critical_analysis": "Critical examination of philosophical claims",
            "comprehensive_philosophy": "Exhaustive philosophical content extraction",
        }
        return descriptions.get(self.value, self.value)

    @property
    def required_fields(self) -> List[str]:
        """Get required fields for this set"""
        required_map = {
            "philosophy_basic": [
                "main_topic",
                "key_concepts",
                "philosophical_tradition",
            ],
            "philosopher_profile": [
                "philosopher_name",
                "main_ideas",
                "philosophical_school",
            ],
            "philosophical_concept": ["concept_name", "definition", "key_philosophers"],
            "philosophical_argument": ["argument_summary", "premises", "conclusion"],
            "philosophical_theory": [
                "theory_name",
                "main_principles",
                "key_proponents",
            ],
            "text_analysis": ["text_title", "main_thesis", "key_arguments"],
            "comparative_philosophy": [
                "positions_compared",
                "similarities",
                "differences",
            ],
            "historical_philosophy": [
                "historical_period",
                "key_developments",
                "influences",
            ],
            "critical_analysis": ["thesis_evaluated", "strengths", "weaknesses"],
            "comprehensive_philosophy": [
                "main_thesis",
                "key_concepts",
                "arguments",
                "tradition",
            ],
        }
        return required_map.get(self.value, [])


# ===== FIELD DEFINITIONS (Now with MinMaxRule defined) =====

# Define actual philosophical fields
PHILOSOPHY_FIELDS = {
    # Basic identification
    "text_title": PhilosophyExtractionField(
        name="text_title",
        description="Title or name of the philosophical text",
        required=True,
        extraction_patterns=[r"(?:title|work)[:：]\s*(.+)", r"《(.+?)》", r'"(.+?)"'],
        examples={
            "en": ["Being and Time", "Meditations", "Critique of Pure Reason"],
            "zh": ["存在与时间", "沉思录", "纯粹理性批判"],
        },
        validation_rules=[LengthRule(1, 500)],
    ),
    "philosopher_name": PhilosophyExtractionField(
        name="philosopher_name",
        description="Name of the philosopher",
        required=True,
        post_processors=["normalize_philosopher"],
        extraction_patterns=[
            r"(?:by|author)[:：]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+(?:work|philosophy|theory)",
        ],
        examples={
            "en": ["Immanuel Kant", "Aristotle", "Simone de Beauvoir"],
            "zh": ["康德", "亚里士多德", "西蒙娜·德·波伏娃"],
        },
        validation_rules=[PatternRule(r"^[A-Za-z\s\-\.]+$")],
    ),
    # Core philosophical content
    "main_thesis": PhilosophyExtractionField(
        name="main_thesis",
        description="Central thesis or main claim of the text",
        required=True,
        priority=1,
        extraction_hints=[
            "Look for the primary argument or position",
            "Identify the central claim",
        ],
        contextual_keywords=[
            "argues",
            "claims",
            "thesis",
            "position",
            "maintains",
            "contends",
            "holds",
        ],
        validation_rules=[LengthRule(10, 1000)],
    ),
    "main_topic": PhilosophyExtractionField(
        name="main_topic",
        description="Main topic or subject of philosophical discussion",
        required=True,
        priority=1,
        extraction_hints=["Identify the central topic", "Find the main subject matter"],
        validation_rules=[LengthRule(3, 200)],
    ),
    "key_concepts": PhilosophyExtractionField(
        name="key_concepts",
        description="Key philosophical concepts discussed",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=[
            "Identify technical terms and central ideas",
            "Look for defined concepts",
        ],
        examples={
            "en": ["Being", "Dasein", "categorical imperative", "eudaimonia"],
            "zh": ["存在", "此在", "绝对命令", "幸福"],
        },
        post_processors=["extract_key_terms", "standardize_terms"],
    ),
    "key_arguments": PhilosophyExtractionField(
        name="key_arguments",
        description="Main arguments presented in the text",
        data_type=DataType.ARRAY,
        required=True,
        post_processors=["parse_argument"],
        extraction_hints=[
            "Extract premise-conclusion structures",
            "Identify logical reasoning",
        ],
        validation_rules=[RequiredFieldsRule(["premises", "conclusion"])],
    ),
    "philosophical_tradition": PhilosophyExtractionField(
        name="philosophical_tradition",
        description="Philosophical school or tradition",
        post_processors=["extract_tradition"],
        extraction_patterns=[r"(?:tradition|school)[:：]\s*(\w+)"],
        examples={
            "en": ["Existentialism", "Pragmatism", "Stoicism", "Empiricism"],
            "zh": ["存在主义", "实用主义", "斯多葛主义", "经验主义"],
        },
        validation_rules=[
            OptionsRule(
                [
                    "existentialism",
                    "pragmatism",
                    "stoicism",
                    "platonism",
                    "aristotelianism",
                    "kantian",
                    "utilitarianism",
                    "phenomenology",
                    "empiricism",
                    "rationalism",
                    "idealism",
                    "materialism",
                    "other",
                ]
            )
        ],
    ),
    # Argument analysis fields
    "argument_summary": PhilosophyExtractionField(
        name="argument_summary",
        description="Summary of the philosophical argument",
        required=True,
        priority=1,
        extraction_hints=[
            "Summarize the main argumentative thrust",
            "Capture the essential reasoning",
        ],
        validation_rules=[LengthRule(20, 500)],
    ),
    "premises": PhilosophyExtractionField(
        name="premises",
        description="Premises of philosophical arguments",
        data_type=DataType.ARRAY,
        category=PhilosophicalCategory.LOGIC,
        extraction_hints=[
            "Identify supporting statements and assumptions",
            "Find the basis of arguments",
        ],
        related_fields=["conclusion", "argument_type"],
    ),
    "conclusion": PhilosophyExtractionField(
        name="conclusion",
        description="Conclusion of the argument",
        category=PhilosophicalCategory.LOGIC,
        extraction_hints=[
            "Find what follows from the premises",
            "Identify the final claim",
        ],
        related_fields=["premises", "argument_type"],
    ),
    "argument_type": PhilosophyExtractionField(
        name="argument_type",
        description="Type of philosophical argument",
        category=PhilosophicalCategory.LOGIC,
        examples={
            "en": [
                "deductive",
                "inductive",
                "abductive",
                "transcendental",
                "dialectical",
            ]
        },
        validation_rules=[
            OptionsRule(
                [
                    "deductive",
                    "inductive",
                    "abductive",
                    "transcendental",
                    "dialectical",
                    "analogical",
                ]
            )
        ],
    ),
    "logical_form": PhilosophyExtractionField(
        name="logical_form",
        description="Formal logical structure of the argument",
        category=PhilosophicalCategory.LOGIC,
        examples={
            "en": [
                "modus ponens",
                "modus tollens",
                "disjunctive syllogism",
                "hypothetical syllogism",
            ],
            "zh": ["肯定前件式", "否定后件式", "选言三段论", "假言三段论"],
        },
        extraction_hints=["Identify the formal structure", "Look for logical patterns"],
        contextual_keywords=[
            "if-then",
            "either-or",
            "all-some",
            "necessary-sufficient",
        ],
    ),
    "validity_assessment": PhilosophyExtractionField(
        name="validity_assessment",
        description="Assessment of logical validity and soundness",
        category=PhilosophicalCategory.LOGIC,
        extraction_hints=[
            "Evaluate if conclusion follows from premises",
            "Check for logical fallacies",
        ],
        contextual_keywords=["valid", "invalid", "sound", "unsound", "fallacy"],
    ),
    # Concept analysis fields
    "concept_name": PhilosophyExtractionField(
        name="concept_name",
        description="Name of the philosophical concept being analyzed",
        required=True,
        extraction_patterns=[
            r"concept\s+of\s+['\"]?([^'\"]+)['\"]?",
            r"([A-Za-z]+)\s+is\s+defined\s+as",
        ],
        validation_rules=[LengthRule(1, 100)],
    ),
    "definition": PhilosophyExtractionField(
        name="definition",
        description="Definition or explanation of the concept",
        required=True,
        extraction_patterns=[
            r"defined\s+as\s+(.+)",
            r"means\s+(.+)",
            r"refers\s+to\s+(.+)",
        ],
        validation_rules=[LengthRule(10, 500)],
    ),
    "related_concepts": PhilosophyExtractionField(
        name="related_concepts",
        description="Concepts related to the main concept",
        data_type=DataType.ARRAY,
        extraction_hints=[
            "Find conceptually related terms",
            "Identify connected ideas",
        ],
        post_processors=["extract_key_terms"],
    ),
    "examples": PhilosophyExtractionField(
        name="examples",
        description="Examples illustrating the concept or argument",
        data_type=DataType.ARRAY,
        extraction_hints=["Look for concrete instances", "Find illustrative cases"],
        contextual_keywords=["for example", "for instance", "such as", "like", "e.g."],
    ),
    # Profile and biographical fields
    "main_ideas": PhilosophyExtractionField(
        name="main_ideas",
        description="Main philosophical ideas or contributions",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=[
            "Identify key philosophical contributions",
            "Find central ideas",
        ],
    ),
    "philosophical_school": PhilosophyExtractionField(
        name="philosophical_school",
        description="Philosophical school or movement associated with",
        extraction_patterns=[r"(?:school|movement|tradition)\s+of\s+(\w+)"],
        post_processors=["extract_tradition"],
    ),
    # Theory fields
    "theory_name": PhilosophyExtractionField(
        name="theory_name",
        description="Name of the philosophical theory",
        required=True,
        extraction_patterns=[r"theory\s+of\s+(.+)", r"(.+)\s+theory"],
        validation_rules=[LengthRule(3, 100)],
    ),
    "main_principles": PhilosophyExtractionField(
        name="main_principles",
        description="Main principles or tenets of the theory",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=["Identify core principles", "Find fundamental tenets"],
    ),
    "key_proponents": PhilosophyExtractionField(
        name="key_proponents",
        description="Key proponents or defenders of the position",
        data_type=DataType.ARRAY,
        required=True,
        post_processors=["normalize_philosopher"],
    ),
    # Comparative fields
    "positions_compared": PhilosophyExtractionField(
        name="positions_compared",
        description="Philosophical positions being compared",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=[
            "Identify the positions under comparison",
            "List what's being contrasted",
        ],
    ),
    "similarities": PhilosophyExtractionField(
        name="similarities",
        description="Similarities between the compared positions",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=["Find points of agreement", "Identify common ground"],
    ),
    "differences": PhilosophyExtractionField(
        name="differences",
        description="Differences between the compared positions",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=["Find points of disagreement", "Identify contrasts"],
    ),
    # Historical analysis fields
    "historical_context": PhilosophyExtractionField(
        name="historical_context",
        description="Historical background and period",
        post_processors=["extract_period"],
        extraction_hints=[
            "Identify time period and historical circumstances",
            "Note cultural context",
        ],
        category=PhilosophicalCategory.POLITICAL,
    ),
    "historical_period": PhilosophyExtractionField(
        name="historical_period",
        description="Historical period of the philosophy",
        required=True,
        post_processors=["extract_period"],
        extraction_hints=["Identify time period", "Find historical context"],
    ),
    "key_developments": PhilosophyExtractionField(
        name="key_developments",
        description="Key philosophical developments in the period",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=[
            "Identify major philosophical advances",
            "Find important changes",
        ],
    ),
    "influences": PhilosophyExtractionField(
        name="influences",
        description="Philosophical influences on the work",
        data_type=DataType.ARRAY,
        extraction_hints=[
            "Look for references to other philosophers or schools",
            "Identify intellectual heritage",
        ],
        post_processors=["normalize_philosopher"],
    ),
    "influenced": PhilosophyExtractionField(
        name="influenced",
        description="Philosophers or movements influenced by this",
        data_type=DataType.ARRAY,
        extraction_hints=[
            "Find who was influenced",
            "Identify intellectual descendants",
        ],
        post_processors=["normalize_philosopher"],
    ),
    # Critical analysis fields
    "thesis_evaluated": PhilosophyExtractionField(
        name="thesis_evaluated",
        description="The thesis or position being critically evaluated",
        required=True,
        extraction_hints=[
            "Identify what's being critiqued",
            "Find the target of analysis",
        ],
    ),
    "strengths": PhilosophyExtractionField(
        name="strengths",
        description="Strengths of the philosophical position",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=["Identify positive aspects", "Find what works well"],
    ),
    "weaknesses": PhilosophyExtractionField(
        name="weaknesses",
        description="Weaknesses or problems with the position",
        data_type=DataType.ARRAY,
        required=True,
        extraction_hints=["Identify problems", "Find vulnerabilities"],
    ),
    "criticisms": PhilosophyExtractionField(
        name="criticisms",
        description="Criticisms or counter-arguments presented",
        data_type=DataType.ARRAY,
        extraction_hints=[
            "Identify objections and opposing views",
            "Find counter-arguments",
        ],
        priority=2,
    ),
    # Additional comprehensive fields
    "arguments": PhilosophyExtractionField(
        name="arguments",
        description="Philosophical arguments presented",
        data_type=DataType.ARRAY,
        extraction_hints=[
            "Extract argumentative structures",
            "Identify reasoning patterns",
        ],
        post_processors=["parse_argument"],
    ),
    "tradition": PhilosophyExtractionField(
        name="tradition",
        description="Philosophical tradition or school",
        extraction_hints=["Identify philosophical tradition", "Find school of thought"],
        post_processors=["extract_tradition"],
    ),
    "key_philosophers": PhilosophyExtractionField(
        name="key_philosophers",
        description="Key philosophers associated with this concept or tradition",
        data_type=DataType.ARRAY,
        extraction_hints=["Identify important philosophers", "Find major contributors"],
        post_processors=["normalize_philosopher"],
    ),
    "author_position": PhilosophyExtractionField(
        name="author_position",
        description="The author's philosophical position or stance",
        extraction_hints=[
            "Identify what the author argues for",
            "Find the author's view",
        ],
        priority=2,
    ),
    "legacy": PhilosophyExtractionField(
        name="legacy",
        description="Philosophical legacy and influence",
        extraction_hints=["Identify lasting impact", "Find continuing influence"],
    ),
    # Category-specific fields
    "ethical_principles": PhilosophyExtractionField(
        name="ethical_principles",
        description="Ethical principles or moral claims",
        data_type=DataType.ARRAY,
        category=PhilosophicalCategory.ETHICS,
        contextual_keywords=[
            "ought",
            "should",
            "duty",
            "moral",
            "ethical",
            "right",
            "wrong",
            "virtue",
            "vice",
            "obligation",
            "responsibility",
            "justice",
        ],
        examples={
            "en": ["categorical imperative", "utilitarianism", "virtue ethics"],
            "zh": ["绝对命令", "功利主义", "德性伦理学"],
        },
    ),
    "epistemological_claims": PhilosophyExtractionField(
        name="epistemological_claims",
        description="Claims about knowledge and justification",
        data_type=DataType.ARRAY,
        category=PhilosophicalCategory.EPISTEMOLOGY,
        contextual_keywords=[
            "knowledge",
            "truth",
            "belief",
            "justification",
            "evidence",
            "certainty",
            "doubt",
            "skepticism",
            "empirical",
            "a priori",
        ],
    ),
    "metaphysical_positions": PhilosophyExtractionField(
        name="metaphysical_positions",
        description="Positions on the nature of reality",
        data_type=DataType.ARRAY,
        category=PhilosophicalCategory.METAPHYSICS,
        contextual_keywords=[
            "being",
            "existence",
            "reality",
            "substance",
            "essence",
            "ontology",
            "mind",
            "matter",
            "dualism",
            "monism",
        ],
    ),
    # Metadata fields
    "language_detected": PhilosophyExtractionField(
        name="language_detected",
        description="Primary language of the text",
        post_processors=["detect_language"],
        required=False,
        validation_rules=[OptionsRule(["en", "zh", "el", "mixed", "unknown"])],
    ),
    "extraction_confidence": PhilosophyExtractionField(
        name="extraction_confidence",
        description="Confidence level of extraction",
        data_type=DataType.NUMBER,
        default_value=0.0,
        validation_rules=[MinMaxRule(0.0, 1.0)],
        metadata={
            "unit": "ratio",
            "description": "0.0 = no confidence, 1.0 = full confidence",
        },
    ),
    # Additional fields
    "philosophical_method": PhilosophyExtractionField(
        name="philosophical_method",
        description="Philosophical method or approach used",
        examples={
            "en": ["phenomenological", "analytical", "dialectical", "hermeneutical"],
            "zh": ["现象学的", "分析的", "辩证的", "诠释学的"],
        },
        category=PhilosophicalCategory.LOGIC,
    ),
    "key_questions": PhilosophyExtractionField(
        name="key_questions",
        description="Central philosophical questions addressed",
        data_type=DataType.ARRAY,
        extraction_hints=[
            "Look for interrogative forms",
            "Identify problem formulations",
        ],
        priority=2,
    ),
    "applications": PhilosophyExtractionField(
        name="applications",
        description="Practical applications or implications",
        data_type=DataType.ARRAY,
        extraction_hints=[
            "Find real-world applications",
            "Identify practical consequences",
        ],
        contextual_keywords=["applied", "practical", "implications", "consequences"],
    ),
}


# ===== FIELD REGISTRY =====


class PhilosophyFieldRegistry:
    """Registry for philosophical extraction fields"""

    def __init__(self):
        self._fields = PHILOSOPHY_FIELDS.copy()
        self._field_sets = {}
        self._category_index = {}
        self._initialize_field_sets()
        self._build_indexes()

    def _initialize_field_sets(self):
        """Initialize predefined field sets"""
        self._field_sets = {
            PhilosophyFieldSet.BASIC: [
                "text_title",
                "main_thesis",
                "key_concepts",
                "philosophical_tradition",
                "key_arguments",
                "main_topic",
            ],
            PhilosophyFieldSet.PHILOSOPHER: [
                "philosopher_name",
                "main_thesis",
                "main_ideas",
                "key_concepts",
                "philosophical_tradition",
                "philosophical_school",
                "influences",
                "historical_context",
            ],
            PhilosophyFieldSet.CONCEPT: [
                "concept_name",
                "definition",
                "key_concepts",
                "key_philosophers",
                "related_concepts",
                "examples",
                "criticisms",
            ],
            PhilosophyFieldSet.ARGUMENT: [
                "argument_summary",
                "premises",
                "conclusion",
                "argument_type",
                "logical_form",
                "validity_assessment",
            ],
            PhilosophyFieldSet.THEORY: [
                "theory_name",
                "main_principles",
                "key_proponents",
                "key_concepts",
                "philosophical_tradition",
                "criticisms",
                "applications",
            ],
            PhilosophyFieldSet.TEXT_ANALYSIS: [
                "text_title",
                "main_thesis",
                "key_arguments",
                "key_concepts",
                "philosophical_tradition",
                "author_position",
            ],
            PhilosophyFieldSet.COMPARATIVE: [
                "positions_compared",
                "similarities",
                "differences",
                "key_concepts",
                "philosophical_tradition",
            ],
            PhilosophyFieldSet.HISTORICAL: [
                "historical_period",
                "key_developments",
                "influences",
                "key_philosophers",
                "philosophical_tradition",
            ],
            PhilosophyFieldSet.CRITICAL: [
                "thesis_evaluated",
                "strengths",
                "weaknesses",
                "arguments",
                "conclusion",
            ],
            PhilosophyFieldSet.COMPREHENSIVE: list(PHILOSOPHY_FIELDS.keys()),
        }

    def _build_indexes(self):
        """Build indexes for efficient lookups"""
        # Build category index
        for field_name, field in self._fields.items():
            if field.category:
                if field.category not in self._category_index:
                    self._category_index[field.category] = []
                self._category_index[field.category].append(field_name)

    def get_field(self, name: str) -> Optional[PhilosophyExtractionField]:
        """Get field by name"""
        return self._fields.get(name)

    def get_field_set(self, set_name: PhilosophyFieldSet) -> List[str]:
        """Get list of field names for a field set"""
        return self._field_sets.get(set_name, [])

    def register_field(self, field: PhilosophyExtractionField):
        """Register a new field"""
        self._fields[field.name] = field
        # Update indexes
        if field.category and field.category not in self._category_index:
            self._category_index[field.category] = []
        if field.category:
            self._category_index[field.category].append(field.name)

    def get_fields_by_category(
        self, category: PhilosophicalCategory
    ) -> Dict[str, PhilosophyExtractionField]:
        """Get all fields for a specific philosophical category"""
        field_names = self._category_index.get(category, [])
        return {
            name: self._fields[name] for name in field_names if name in self._fields
        }

    def get_required_fields(self) -> Dict[str, PhilosophyExtractionField]:
        """Get all required fields"""
        return {name: field for name, field in self._fields.items() if field.required}

    def search_fields(self, query: str) -> List[PhilosophyExtractionField]:
        """Search fields by name, description, or keywords"""
        query_lower = query.lower()
        results = []

        for field in self._fields.values():
            if (
                query_lower in field.name.lower()
                or query_lower in field.description.lower()
                or any(
                    query_lower in keyword.lower()
                    for keyword in field.contextual_keywords
                )
            ):
                results.append(field)

        return results


# Global registry instance
philosophy_field_registry = PhilosophyFieldRegistry()
