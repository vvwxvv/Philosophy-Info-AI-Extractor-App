"""
Knowledges Package for Philosophy Extraction System

This package provides comprehensive philosophical knowledge base components
including schools, theories, periods, concepts, and extraction patterns.
"""

# Core knowledge base infrastructure
from .knowledge_base import (
    PhilosophyKnowledgeBase,
    PhilosophicalEntity,
    PhilosophicalTheory,
    PhilosophicalSchool,
    PhilosophicalPeriod,
    PhilosophicalMethod,
    PhilosophicalConcept,
    PhilosophicalBranch,
    PhilosophicalArgument,
    Philosopher,
    philosophy_kb,
    search_philosophy,
    get_philosopher,
    get_concept,
    get_theory,
)

# Knowledge integration and extraction
from .knowledge_integration import (
    ExtractionContext,
    PhilosophyKnowledgeExtractor,
    enhance_extraction_with_knowledge,
)

# Pattern recognition for extraction
from .patterns import (
    PhilosophicalPatterns,
    initialize_patterns,
)

# Specific knowledge domains
from .schools import (
    PhilosophicalSchool as SchoolEntity,
    initialize_schools,
)

from .theories import (
    PhilosophicalSchool as TheoryEntity,  # Note: theories.py appears to contain schools
    initialize_schools as initialize_theories,  # Mapping for consistency
)

# Knowledge definitions and mappings
from .periods import PHILOSOPHICAL_PERIODS
from .methods import PHILOSOPHICAL_METHODS
from .concepts import PHILOSOPHICAL_CONCEPTS
from .categories import (
    PHILOSOPHY_CATEGORIES,
    PHILOSOPHY_EVENT_TYPES,
    PHILOSOPHY_FIELDS_TYPE,
)
from .branches import PHILOSOPHICAL_BRANCHES
from .arguments import PHILOSOPHICAL_ARGUMENTS
from .type import (
    PHILOSOPHY_BRANCHES as BRANCH_TYPES,
    PHILOSOPHY_PERIODS as PERIOD_TYPES,
    PHILOSOPHY_MOVEMENTS,
    PHILOSOPHY_CONCEPTS as CONCEPT_TYPES,
)
from .country_name import COUNTRY_NAME_EN_CN

# Version info
__version__ = "1.0.0"
__author__ = "Philosophy Knowledge Base System"

# Main exports
__all__ = [
    # Core classes and infrastructure
    "PhilosophyKnowledgeBase",
    "PhilosophicalEntity",
    "PhilosophicalTheory",
    "PhilosophicalSchool",
    "PhilosophicalPeriod",
    "PhilosophicalMethod",
    "PhilosophicalConcept",
    "PhilosophicalBranch",
    "PhilosophicalArgument",
    "Philosopher",
    # Knowledge integration
    "ExtractionContext",
    "PhilosophyKnowledgeExtractor",
    "enhance_extraction_with_knowledge",
    # Pattern extraction
    "PhilosophicalPatterns",
    "initialize_patterns",
    # Domain-specific entities
    "SchoolEntity",
    "TheoryEntity",
    "initialize_schools",
    "initialize_theories",
    # Global knowledge base instance
    "philosophy_kb",
    # Convenience functions
    "search_philosophy",
    "get_philosopher",
    "get_concept",
    "get_theory",
    # Knowledge definitions
    "PHILOSOPHICAL_PERIODS",
    "PHILOSOPHICAL_METHODS",
    "PHILOSOPHICAL_CONCEPTS",
    "PHILOSOPHY_CATEGORIES",
    "PHILOSOPHY_EVENT_TYPES",
    "PHILOSOPHY_FIELDS_TYPE",
    "PHILOSOPHICAL_BRANCHES",
    "PHILOSOPHICAL_ARGUMENTS",
    # Type definitions
    "BRANCH_TYPES",
    "PERIOD_TYPES",
    "PHILOSOPHY_MOVEMENTS",
    "CONCEPT_TYPES",
    # Utility mappings
    "COUNTRY_NAME_EN_CN",
]


# Convenience functions for quick access
def get_knowledge_stats():
    """Get statistics about the knowledge base"""
    return philosophy_kb.get_statistics()


def search_by_category(category: str, query: str = ""):
    """Search within a specific philosophical category"""
    if category in PHILOSOPHY_CATEGORIES:
        concepts = PHILOSOPHY_CATEGORIES[category]
        if query:
            return [concept for concept in concepts if query.lower() in concept.lower()]
        return concepts
    return []


def get_philosophers_by_period(period: str):
    """Get philosophers from a specific period"""
    if period in PHILOSOPHICAL_PERIODS:
        period_data = PHILOSOPHICAL_PERIODS[period]
        return period_data.get("key_philosophers", [])
    return []


def get_school_info(school_name: str):
    """Get information about a philosophical school"""
    school = philosophy_kb.schools.get(school_name.lower())
    if school:
        return school.to_dict()
    return None


def get_concept_definition(concept_name: str):
    """Get definition of a philosophical concept"""
    if concept_name in PHILOSOPHICAL_CONCEPTS:
        return PHILOSOPHICAL_CONCEPTS[concept_name].get("definition", "")

    concept = philosophy_kb.concepts.get(concept_name.lower())
    if concept:
        return concept.definition
    return None


def get_argument_structure(argument_name: str):
    """Get the logical structure of a philosophical argument"""
    if argument_name in PHILOSOPHICAL_ARGUMENTS:
        arg = PHILOSOPHICAL_ARGUMENTS[argument_name]
        return {
            "premises": arg.get("premises", []),
            "conclusion": arg.get("conclusion", {}),
            "logical_structure": arg.get("logical_structure", {}),
            "counter_arguments": arg.get("counter_arguments", []),
        }
    return None


def extract_entities_from_text(
    text: str, language: str = "en", confidence_threshold: float = 0.7
):
    """Extract philosophical entities from text using the knowledge base"""
    extractor = PhilosophyKnowledgeExtractor()
    context = ExtractionContext(
        text=text,
        language=language,
        confidence_threshold=confidence_threshold,
        include_related=True,
    )
    return extractor.extract_entities(context)


def get_related_concepts(concept_name: str):
    """Get concepts related to a given concept"""
    if concept_name in PHILOSOPHICAL_CONCEPTS:
        return PHILOSOPHICAL_CONCEPTS[concept_name].get("related_elements", [])

    concept = philosophy_kb.concepts.get(concept_name.lower())
    if concept:
        return concept.related_concepts
    return []


def validate_knowledge_base():
    """Validate the integrity of the knowledge base"""
    stats = get_knowledge_stats()
    issues = []

    # Check if knowledge base is properly loaded
    if stats["total_entities"] == 0:
        issues.append("Knowledge base appears to be empty")

    # Check for missing required data
    if stats["philosophers"] == 0:
        issues.append("No philosophers loaded")

    if stats["concepts"] == 0:
        issues.append("No concepts loaded")

    return {"valid": len(issues) == 0, "issues": issues, "statistics": stats}


# Add convenience functions to exports
__all__.extend(
    [
        "get_knowledge_stats",
        "search_by_category",
        "get_philosophers_by_period",
        "get_school_info",
        "get_concept_definition",
        "get_argument_structure",
        "extract_entities_from_text",
        "get_related_concepts",
        "validate_knowledge_base",
    ]
)

# Package-level documentation
__doc__ = """
Philosophy Knowledges Package

This package provides a comprehensive knowledge base system for philosophical
content extraction and analysis. It includes:

1. Knowledge Base Infrastructure:
   - PhilosophyKnowledgeBase: Central knowledge repository
   - PhilosophicalEntity: Base class for all philosophical entities
   - Specialized entity classes for different philosophical domains

2. Domain-Specific Knowledge:
   - Schools: Philosophical schools and traditions
   - Theories: Philosophical theories and systems
   - Periods: Historical periods in philosophy
   - Concepts: Key philosophical concepts and definitions
   - Arguments: Classical philosophical arguments
   - Methods: Philosophical methodologies and approaches

3. Extraction and Integration:
   - PhilosophyKnowledgeExtractor: Extract entities from text
   - Pattern recognition for identifying philosophical content
   - Integration with other extraction systems

4. Utility Functions:
   - Search and retrieval functions
   - Category-based organization
   - Relationship mapping between entities

Usage Examples:

    # Search the knowledge base
    from knowledges import search_philosophy, philosophy_kb
    results = search_philosophy("stoicism")

    # Extract entities from text
    from knowledges import extract_entities_from_text
    entities = extract_entities_from_text("Socrates developed the Socratic method...")

    # Get specific information
    from knowledges import get_concept_definition, get_school_info
    definition = get_concept_definition("free_will")
    school_data = get_school_info("stoicism")

    # Enhance extraction results
    from knowledges import enhance_extraction_with_knowledge
    enhanced = enhance_extraction_with_knowledge(ollama_results, text)

    # Validate knowledge base
    from knowledges import validate_knowledge_base
    validation = validate_knowledge_base()
"""

# Initialize knowledge base on import
try:
    # Trigger knowledge base initialization
    _ = philosophy_kb.get_statistics()
    print(f"Philosophy knowledge base loaded successfully")
except Exception as e:
    print(f"Warning: Failed to initialize knowledge base: {e}")
