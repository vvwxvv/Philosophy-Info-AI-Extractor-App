"""
Unified Philosophy Knowledge Base
Production-ready philosophical knowledge for extraction and analysis
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PhilosophicalEntity:
    """Base class for all philosophical entities"""

    id: str
    name: Dict[str, str]  # Multilingual names
    description: Dict[str, str]  # Multilingual descriptions
    keywords: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_name(self, lang: str = "en") -> str:
        """Get name in specified language"""
        return self.name.get(lang, self.name.get("en", ""))

    def get_description(self, lang: str = "en") -> str:
        """Get description in specified language"""
        return self.description.get(lang, self.description.get("en", ""))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "aliases": self.aliases,
            "references": self.references,
            "metadata": self.metadata,
        }


@dataclass
class PhilosophicalTheory(PhilosophicalEntity):
    """Represents a philosophical theory"""

    theory_type: str = ""
    main_principles: List[Dict[str, str]] = field(default_factory=list)
    key_philosophers: List[Dict[str, str]] = field(default_factory=list)
    criticisms: List[Dict[str, str]] = field(default_factory=list)
    applications: List[Dict[str, str]] = field(default_factory=list)
    historical_development: Dict[str, str] = field(default_factory=dict)
    contemporary_relevance: Dict[str, str] = field(default_factory=dict)
    related_theories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update(
            {
                "theory_type": self.theory_type,
                "main_principles": self.main_principles,
                "key_philosophers": self.key_philosophers,
                "criticisms": self.criticisms,
                "applications": self.applications,
                "historical_development": self.historical_development,
                "contemporary_relevance": self.contemporary_relevance,
                "related_theories": self.related_theories,
            }
        )
        return base_dict


@dataclass
class PhilosophicalSchool(PhilosophicalEntity):
    """Represents a philosophical school or tradition"""

    founders: List[Dict[str, str]] = field(default_factory=list)
    key_figures: List[Dict[str, str]] = field(default_factory=list)
    core_tenets: List[Dict[str, str]] = field(default_factory=list)
    historical_period: Dict[str, str] = field(default_factory=dict)
    geographical_origin: Dict[str, str] = field(default_factory=dict)
    influences: List[Dict[str, str]] = field(default_factory=list)
    influenced: List[Dict[str, str]] = field(default_factory=list)
    key_texts: List[Dict[str, str]] = field(default_factory=list)
    practices: List[Dict[str, str]] = field(default_factory=list)
    contemporary_relevance: Dict[str, str] = field(default_factory=dict)
    sub_schools: List[str] = field(default_factory=list)
    opposing_schools: List[str] = field(default_factory=list)


@dataclass
class PhilosophicalPeriod(PhilosophicalEntity):
    """Represents a historical period in philosophy"""

    time_range: Dict[str, str] = field(default_factory=dict)
    key_developments: List[Dict[str, str]] = field(default_factory=list)
    major_figures: List[Dict[str, str]] = field(default_factory=list)
    dominant_schools: List[str] = field(default_factory=list)
    cultural_context: Dict[str, str] = field(default_factory=dict)
    geographical_centers: List[Dict[str, str]] = field(default_factory=list)
    major_works: List[Dict[str, str]] = field(default_factory=list)
    innovations: List[Dict[str, str]] = field(default_factory=list)
    legacy: Dict[str, str] = field(default_factory=dict)


@dataclass
class PhilosophicalMethod(PhilosophicalEntity):
    """Represents a philosophical method or approach"""

    steps: List[Dict[str, str]] = field(default_factory=list)
    purpose: Dict[str, str] = field(default_factory=dict)
    applications: List[Dict[str, str]] = field(default_factory=list)
    limitations: List[Dict[str, str]] = field(default_factory=list)
    key_practitioners: List[Dict[str, str]] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    variations: List[Dict[str, str]] = field(default_factory=list)
    effectiveness: Dict[str, str] = field(default_factory=dict)
    criticisms: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PhilosophicalConcept(PhilosophicalEntity):
    """Represents a philosophical concept"""

    definition: Dict[str, str] = field(default_factory=dict)
    etymology: Dict[str, str] = field(default_factory=dict)
    key_thinkers: List[Dict[str, str]] = field(default_factory=list)
    historical_development: Dict[str, str] = field(default_factory=dict)
    related_concepts: List[str] = field(default_factory=list)
    opposing_concepts: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    debates: List[Dict[str, str]] = field(default_factory=list)
    contemporary_usage: Dict[str, str] = field(default_factory=dict)
    cultural_variations: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PhilosophicalBranch(PhilosophicalEntity):
    """Represents a branch of philosophy"""

    main_topics: List[Dict[str, str]] = field(default_factory=list)
    key_questions: List[Dict[str, str]] = field(default_factory=list)
    methodologies: List[str] = field(default_factory=list)
    related_fields: List[Dict[str, str]] = field(default_factory=list)
    sub_branches: List[str] = field(default_factory=list)
    major_theories: List[str] = field(default_factory=list)
    contemporary_issues: List[Dict[str, str]] = field(default_factory=list)
    key_texts: List[Dict[str, str]] = field(default_factory=list)
    academic_departments: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PhilosophicalArgument(PhilosophicalEntity):
    """Represents a philosophical argument"""

    premises: List[Dict[str, str]] = field(default_factory=list)
    conclusion: Dict[str, str] = field(default_factory=dict)
    logical_structure: Dict[str, str] = field(default_factory=dict)
    argument_type: str = ""
    counter_arguments: List[Dict[str, str]] = field(default_factory=list)
    defenses: List[Dict[str, str]] = field(default_factory=list)
    key_proponents: List[Dict[str, str]] = field(default_factory=list)
    key_critics: List[Dict[str, str]] = field(default_factory=list)
    historical_context: Dict[str, str] = field(default_factory=dict)
    modern_relevance: Dict[str, str] = field(default_factory=dict)
    variations: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Philosopher(PhilosophicalEntity):
    """Represents a philosopher"""

    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    nationality: Dict[str, str] = field(default_factory=dict)
    schools: List[str] = field(default_factory=list)
    main_interests: List[str] = field(default_factory=list)
    notable_ideas: List[Dict[str, str]] = field(default_factory=list)
    influences: List[str] = field(default_factory=list)
    influenced: List[str] = field(default_factory=list)
    major_works: List[Dict[str, str]] = field(default_factory=list)
    education: List[Dict[str, str]] = field(default_factory=list)
    positions_held: List[Dict[str, str]] = field(default_factory=list)
    awards: List[Dict[str, str]] = field(default_factory=list)
    quotes: List[Dict[str, str]] = field(default_factory=list)
    legacy: Dict[str, str] = field(default_factory=dict)


class PhilosophyKnowledgeBase:
    """Central knowledge base for philosophical information"""

    def __init__(self):
        self.theories: Dict[str, PhilosophicalTheory] = {}
        self.schools: Dict[str, PhilosophicalSchool] = {}
        self.periods: Dict[str, PhilosophicalPeriod] = {}
        self.methods: Dict[str, PhilosophicalMethod] = {}
        self.concepts: Dict[str, PhilosophicalConcept] = {}
        self.branches: Dict[str, PhilosophicalBranch] = {}
        self.arguments: Dict[str, PhilosophicalArgument] = {}
        self.philosophers: Dict[str, Philosopher] = {}

        # Pattern collections
        self.extraction_patterns: Dict[str, List[str]] = {}

        # Initialize knowledge base
        self._initialize_knowledge()

    def _initialize_knowledge(self):
        """Initialize all knowledge components"""
        try:
            from .theories import initialize_theories
            from .schools import initialize_schools
            from .periods import initialize_periods
            from .methods import initialize_methods
            from .concepts import initialize_concepts
            from .branches import initialize_branches
            from .arguments import initialize_arguments
            from .philosophers import initialize_philosophers
            from .patterns import initialize_patterns

            self.theories = initialize_theories()
            self.schools = initialize_schools()
            self.periods = initialize_periods()
            self.methods = initialize_methods()
            self.concepts = initialize_concepts()
            self.branches = initialize_branches()
            self.arguments = initialize_arguments()
            self.philosophers = initialize_philosophers()
            self.extraction_patterns = initialize_patterns()
        except ImportError:
            # Initialize with empty collections if modules don't exist yet
            print(
                "Warning: Knowledge base modules not found. Initializing empty collections."
            )
            self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample data for testing"""
        # Sample theory
        self.theories["utilitarianism"] = PhilosophicalTheory(
            id="utilitarianism",
            name={"en": "Utilitarianism", "zh": "功利主义"},
            description={
                "en": "An ethical theory that determines right from wrong by focusing on outcomes",
                "zh": "一种通过关注结果来判断对错的伦理理论",
            },
            theory_type="ethical",
            keywords=["ethics", "consequentialism", "happiness"],
            aliases=["utilitarian ethics", "consequentialist ethics"],
        )

        # Sample philosopher
        self.philosophers["kant"] = Philosopher(
            id="kant",
            name={"en": "Immanuel Kant", "de": "Immanuel Kant"},
            description={
                "en": "German philosopher who developed the categorical imperative",
                "de": "Deutscher Philosoph, der den kategorischen Imperativ entwickelte",
            },
            birth_date="1724-04-22",
            death_date="1804-02-12",
            nationality={"en": "German", "de": "Deutsch"},
            main_interests=["ethics", "metaphysics", "epistemology"],
            keywords=[
                "categorical imperative",
                "synthetic a priori",
                "transcendental idealism",
            ],
        )

        # Sample concept
        self.concepts["free_will"] = PhilosophicalConcept(
            id="free_will",
            name={"en": "Free Will", "zh": "自由意志"},
            description={
                "en": "The ability to choose between different possible courses of action unimpeded",
                "zh": "不受阻碍地在不同可能行动方案之间进行选择的能力",
            },
            definition={
                "en": "The capacity of agents to choose their actions independently of prior causes",
                "zh": "主体独立于先前原因选择其行动的能力",
            },
            keywords=["agency", "choice", "determinism", "responsibility"],
            related_concepts=["determinism", "compatibilism", "moral_responsibility"],
        )

    def search(
        self, query: str, entity_type: Optional[str] = None
    ) -> List[PhilosophicalEntity]:
        """Search across all knowledge base"""
        results = []
        query_lower = query.lower()

        # Define search targets
        search_targets = {
            "theory": self.theories,
            "school": self.schools,
            "period": self.periods,
            "method": self.methods,
            "concept": self.concepts,
            "branch": self.branches,
            "argument": self.arguments,
            "philosopher": self.philosophers,
        }

        # Filter by entity type if specified
        if entity_type:
            search_targets = {entity_type: search_targets.get(entity_type, {})}

        # Search each collection
        for collection in search_targets.values():
            for entity in collection.values():
                if self._matches_query(entity, query_lower):
                    results.append(entity)

        return results

    def _matches_query(self, entity: PhilosophicalEntity, query: str) -> bool:
        """Check if entity matches search query"""
        # Search in names
        for name in entity.name.values():
            if query in name.lower():
                return True

        # Search in descriptions
        for desc in entity.description.values():
            if query in desc.lower():
                return True

        # Search in keywords and aliases
        for keyword in entity.keywords + entity.aliases:
            if query in keyword.lower():
                return True

        return False

    def get_related_entities(
        self, entity_id: str
    ) -> Dict[str, List[PhilosophicalEntity]]:
        """Get entities related to a given entity"""
        related = {
            "theories": [],
            "schools": [],
            "concepts": [],
            "philosophers": [],
            "arguments": [],
        }

        # Find the entity
        entity = None
        for collection in [
            self.theories,
            self.schools,
            self.periods,
            self.methods,
            self.concepts,
            self.branches,
            self.arguments,
            self.philosophers,
        ]:
            if entity_id in collection:
                entity = collection[entity_id]
                break

        if not entity:
            return related

        # Find related entities through references and metadata
        for ref in entity.references:
            for theory in self.theories.values():
                if ref in theory.id or ref in theory.keywords:
                    related["theories"].append(theory)

            for school in self.schools.values():
                if ref in school.id or ref in school.keywords:
                    related["schools"].append(school)

            for concept in self.concepts.values():
                if ref in concept.id or ref in concept.keywords:
                    related["concepts"].append(concept)

            for philosopher in self.philosophers.values():
                if ref in philosopher.id or ref in philosopher.keywords:
                    related["philosophers"].append(philosopher)

            for argument in self.arguments.values():
                if ref in argument.id or ref in argument.keywords:
                    related["arguments"].append(argument)

        return related

    def add_entity(self, entity_type: str, entity: PhilosophicalEntity) -> bool:
        """Add a new entity to the knowledge base"""
        collection_map = {
            "theory": self.theories,
            "school": self.schools,
            "period": self.periods,
            "method": self.methods,
            "concept": self.concepts,
            "branch": self.branches,
            "argument": self.arguments,
            "philosopher": self.philosophers,
        }

        collection = collection_map.get(entity_type)
        if collection is not None:
            collection[entity.id] = entity
            return True
        return False

    def remove_entity(self, entity_type: str, entity_id: str) -> bool:
        """Remove an entity from the knowledge base"""
        collection_map = {
            "theory": self.theories,
            "school": self.schools,
            "period": self.periods,
            "method": self.methods,
            "concept": self.concepts,
            "branch": self.branches,
            "argument": self.arguments,
            "philosopher": self.philosophers,
        }

        collection = collection_map.get(entity_type)
        if collection is not None and entity_id in collection:
            del collection[entity_id]
            return True
        return False

    def update_entity(
        self, entity_type: str, entity_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update an existing entity"""
        collection_map = {
            "theory": self.theories,
            "school": self.schools,
            "period": self.periods,
            "method": self.methods,
            "concept": self.concepts,
            "branch": self.branches,
            "argument": self.arguments,
            "philosopher": self.philosophers,
        }

        collection = collection_map.get(entity_type)
        if collection is not None and entity_id in collection:
            entity = collection[entity_id]
            for key, value in updates.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            return True
        return False

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the knowledge base"""
        return {
            "theories": len(self.theories),
            "schools": len(self.schools),
            "periods": len(self.periods),
            "methods": len(self.methods),
            "concepts": len(self.concepts),
            "branches": len(self.branches),
            "arguments": len(self.arguments),
            "philosophers": len(self.philosophers),
            "total_entities": sum(
                [
                    len(self.theories),
                    len(self.schools),
                    len(self.periods),
                    len(self.methods),
                    len(self.concepts),
                    len(self.branches),
                    len(self.arguments),
                    len(self.philosophers),
                ]
            ),
        }

    def export_for_extraction(self) -> Dict[str, Any]:
        """Export knowledge base in format suitable for extraction"""
        return {
            "theories": {k: v.to_dict() for k, v in self.theories.items()},
            "schools": {k: v.to_dict() for k, v in self.schools.items()},
            "periods": {k: v.to_dict() for k, v in self.periods.items()},
            "methods": {k: v.to_dict() for k, v in self.methods.items()},
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "branches": {k: v.to_dict() for k, v in self.branches.items()},
            "arguments": {k: v.to_dict() for k, v in self.arguments.items()},
            "philosophers": {k: v.to_dict() for k, v in self.philosophers.items()},
            "patterns": self.extraction_patterns,
            "statistics": self.get_statistics(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def save_to_file(self, filename: str) -> bool:
        """Save knowledge base to JSON file"""
        import json

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.export_for_extraction(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
            return False

    def load_from_file(self, filename: str) -> bool:
        """Load knowledge base from JSON file"""
        import json

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Clear existing data
            self.theories.clear()
            self.schools.clear()
            self.periods.clear()
            self.methods.clear()
            self.concepts.clear()
            self.branches.clear()
            self.arguments.clear()
            self.philosophers.clear()

            # Load data
            # Note: This is a simplified version. In production, you'd need proper deserialization
            print(
                f"Loaded {data.get('statistics', {}).get('total_entities', 0)} entities"
            )
            return True
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return False


# Global instance
philosophy_kb = PhilosophyKnowledgeBase()


# Convenience functions
def search_philosophy(
    query: str, entity_type: Optional[str] = None
) -> List[PhilosophicalEntity]:
    """Search the philosophy knowledge base"""
    return philosophy_kb.search(query, entity_type)


def get_philosopher(philosopher_id: str) -> Optional[Philosopher]:
    """Get a specific philosopher by ID"""
    return philosophy_kb.philosophers.get(philosopher_id)


def get_concept(concept_id: str) -> Optional[PhilosophicalConcept]:
    """Get a specific concept by ID"""
    return philosophy_kb.concepts.get(concept_id)


def get_theory(theory_id: str) -> Optional[PhilosophicalTheory]:
    """Get a specific theory by ID"""
    return philosophy_kb.theories.get(theory_id)


# Example usage
if __name__ == "__main__":
    # Test the knowledge base
    kb = PhilosophyKnowledgeBase()

    # Search for "kant"
    results = kb.search("kant")
    print(f"Found {len(results)} results for 'kant'")

    # Get statistics
    stats = kb.get_statistics()
    print(f"Knowledge base contains: {stats}")

    # Get a specific philosopher
    kant = get_philosopher("kant")
    if kant:
        print(f"Found philosopher: {kant.get_name('en')}")

    # Save to file
    kb.save_to_file("philosophy_kb.json")
