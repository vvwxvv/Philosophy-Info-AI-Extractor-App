from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json


class ArgumentType(str, Enum):
    """Types of philosophical arguments"""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    TRANSCENDENTAL = "transcendental"


class ArgumentStrength(str, Enum):
    """Strength levels for arguments"""

    CONCLUSIVE = "conclusive"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    FALLACIOUS = "fallacious"


class ValidityStatus(str, Enum):
    """Logical validity status"""

    VALID = "valid"
    INVALID = "invalid"
    SOUND = "sound"
    UNSOUND = "unsound"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class BaseElement:
    """Base class for all philosophical elements"""

    id: Optional[str] = None
    confidence: float = 1.0
    source_location: Optional[Dict[str, int]] = None
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        """Generate ID if not provided"""
        if self.id is None:
            import uuid

            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def validate(self) -> List[str]:
        """Validate element and return list of errors"""
        errors = []
        if self.confidence < 0 or self.confidence > 1:
            errors.append(f"Confidence must be between 0 and 1, got {self.confidence}")
        return errors


@dataclass
class ArgumentElement(BaseElement):
    """Enhanced structure for philosophical arguments"""

    premises: List[str] = field(default_factory=list)
    conclusion: str = ""
    type: ArgumentType = ArgumentType.DEDUCTIVE
    strength: ArgumentStrength = ArgumentStrength.MODERATE
    logical_structure: str = ""
    validity: ValidityStatus = ValidityStatus.NOT_APPLICABLE
    counterarguments: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate argument structure"""
        errors = super().validate()
        if not self.premises:
            errors.append("Argument must have at least one premise")
        if not self.conclusion:
            errors.append("Argument must have a conclusion")
        if (
            self.type == ArgumentType.DEDUCTIVE
            and self.validity == ValidityStatus.NOT_APPLICABLE
        ):
            errors.append("Deductive arguments must have a validity status")
        return errors

    @property
    def is_complete(self) -> bool:
        """Check if argument has all required components"""
        return bool(self.premises and self.conclusion and self.logical_structure)


@dataclass
class ConceptElement(BaseElement):
    """Enhanced structure for philosophical concepts"""

    term: str = ""
    definition: str = ""
    context: str = ""
    semantic_analysis: str = ""
    etymology: Optional[str] = None
    related_terms: List[str] = field(default_factory=list)
    distinctions: Dict[str, str] = field(default_factory=dict)
    historical_evolution: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate concept structure"""
        errors = super().validate()
        if not self.term:
            errors.append("Concept must have a term")
        if not self.definition:
            errors.append("Concept must have a definition")
        return errors

    def add_distinction(self, other_term: str, distinction: str):
        """Add a distinction from another term"""
        self.distinctions[other_term] = distinction


@dataclass
class TraditionElement(BaseElement):
    """Enhanced structure for philosophical traditions"""

    school: str = ""
    period: str = ""
    influences: List[str] = field(default_factory=list)
    methodology: str = ""
    key_figures: List[str] = field(default_factory=list)
    core_texts: List[str] = field(default_factory=list)
    central_claims: List[str] = field(default_factory=list)
    historical_context: str = ""

    def validate(self) -> List[str]:
        """Validate tradition structure"""
        errors = super().validate()
        if not self.school:
            errors.append("Tradition must have a school name")
        if not self.period:
            errors.append("Tradition must have a historical period")
        return errors


@dataclass
class EthicalElement(BaseElement):
    """Enhanced structure for ethical considerations"""

    principle: str = ""
    application: str = ""
    implications: List[str] = field(default_factory=list)
    framework: str = ""
    moral_status: Optional[str] = None
    justification: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate ethical element"""
        errors = super().validate()
        if not self.principle:
            errors.append("Ethical element must have a principle")
        if not self.framework:
            errors.append("Ethical element must specify a framework")
        return errors


@dataclass
class PhilosophicalAnalysis:
    """Complete philosophical analysis containing all elements"""

    document_id: str
    title: str
    analyzed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Core elements
    main_arguments: List[ArgumentElement] = field(default_factory=list)
    key_concepts: List[ConceptElement] = field(default_factory=list)
    philosophical_tradition: Optional[TraditionElement] = None

    # Additional elements
    ethical_considerations: List[EthicalElement] = field(default_factory=list)
    epistemological_assumptions: List[Dict[str, Any]] = field(default_factory=list)
    metaphysical_claims: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Dict[str, List[str]]:
        """Validate entire analysis"""
        validation_errors = {}

        # Validate arguments
        for i, arg in enumerate(self.main_arguments):
            errors = arg.validate()
            if errors:
                validation_errors[f"argument_{i}"] = errors

        # Validate concepts
        for i, concept in enumerate(self.key_concepts):
            errors = concept.validate()
            if errors:
                validation_errors[f"concept_{i}"] = errors

        # Validate tradition
        if self.philosophical_tradition:
            errors = self.philosophical_tradition.validate()
            if errors:
                validation_errors["tradition"] = errors

        # Validate ethical elements
        for i, ethical in enumerate(self.ethical_considerations):
            errors = ethical.validate()
            if errors:
                validation_errors[f"ethical_{i}"] = errors

        return validation_errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire analysis to dictionary"""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "analyzed_at": self.analyzed_at,
            "main_arguments": [arg.to_dict() for arg in self.main_arguments],
            "key_concepts": [concept.to_dict() for concept in self.key_concepts],
            "philosophical_tradition": (
                self.philosophical_tradition.to_dict()
                if self.philosophical_tradition
                else None
            ),
            "ethical_considerations": [
                ethical.to_dict() for ethical in self.ethical_considerations
            ],
            "epistemological_assumptions": self.epistemological_assumptions,
            "metaphysical_claims": self.metaphysical_claims,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhilosophicalAnalysis":
        """Create analysis from dictionary"""
        analysis = cls(
            document_id=data["document_id"],
            title=data["title"],
            analyzed_at=data.get("analyzed_at", datetime.utcnow().isoformat()),
        )

        # Reconstruct elements
        for arg_data in data.get("main_arguments", []):
            arg = ArgumentElement(**arg_data)
            analysis.main_arguments.append(arg)

        for concept_data in data.get("key_concepts", []):
            concept = ConceptElement(**concept_data)
            analysis.key_concepts.append(concept)

        if data.get("philosophical_tradition"):
            analysis.philosophical_tradition = TraditionElement(
                **data["philosophical_tradition"]
            )

        for ethical_data in data.get("ethical_considerations", []):
            ethical = EthicalElement(**ethical_data)
            analysis.ethical_considerations.append(ethical)

        analysis.epistemological_assumptions = data.get(
            "epistemological_assumptions", []
        )
        analysis.metaphysical_claims = data.get("metaphysical_claims", [])
        analysis.metadata = data.get("metadata", {})

        return analysis


# Element factory for creating elements
class ElementFactory:
    """Factory for creating philosophical elements"""

    @staticmethod
    def create_argument(
        premises: List[str],
        conclusion: str,
        type: Union[str, ArgumentType] = ArgumentType.DEDUCTIVE,
        **kwargs,
    ) -> ArgumentElement:
        """Create an argument element"""
        if isinstance(type, str):
            type = ArgumentType(type)
        return ArgumentElement(
            premises=premises, conclusion=conclusion, type=type, **kwargs
        )

    @staticmethod
    def create_concept(term: str, definition: str, **kwargs) -> ConceptElement:
        """Create a concept element"""
        return ConceptElement(term=term, definition=definition, **kwargs)

    @staticmethod
    def create_tradition(school: str, period: str, **kwargs) -> TraditionElement:
        """Create a tradition element"""
        return TraditionElement(school=school, period=period, **kwargs)

    @staticmethod
    def create_ethical(principle: str, framework: str, **kwargs) -> EthicalElement:
        """Create an ethical element"""
        return EthicalElement(principle=principle, framework=framework, **kwargs)
