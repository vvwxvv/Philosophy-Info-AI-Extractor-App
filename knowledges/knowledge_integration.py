"""Integration module for philosophical knowledge extraction"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from knowledge_base import philosophy_kb, PhilosophicalEntity
from knowledges.patterns import PhilosophicalPatterns


@dataclass
class ExtractionContext:
    """Context for knowledge-based extraction"""

    text: str
    language: str = "en"
    confidence_threshold: float = 0.7
    max_results: int = 10
    include_related: bool = True


class PhilosophyKnowledgeExtractor:
    """Extracts philosophical entities using the knowledge base"""

    def __init__(self):
        self.kb = philosophy_kb
        self.patterns = PhilosophicalPatterns.compile_patterns()

    def extract_entities(
        self, context: ExtractionContext
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all philosophical entities from text"""
        results = {
            "philosophers": self._extract_philosophers(context),
            "theories": self._extract_theories(context),
            "schools": self._extract_schools(context),
            "concepts": self._extract_concepts(context),
            "arguments": self._extract_arguments(context),
            "periods": self._extract_periods(context),
            "methods": self._extract_methods(context),
        }

        # Add relationships if requested
        if context.include_related:
            results["relationships"] = self._extract_relationships(results)

        return results

    def _extract_philosophers(self, context: ExtractionContext) -> List[Dict[str, Any]]:
        """Extract philosopher mentions"""
        philosophers = []
        seen = set()

        # Use patterns to find philosopher names
        for pattern_list in self.patterns["philosopher"].values():
            for pattern in pattern_list:
                for match in pattern.finditer(context.text):
                    name = match.group(1).strip()
                    if name.lower() not in seen:
                        seen.add(name.lower())

                        # Check if philosopher exists in KB
                        philosopher_data = self._find_philosopher(name)

                        philosophers.append(
                            {
                                "name": name,
                                "confidence": (
                                    philosopher_data["confidence"]
                                    if philosopher_data
                                    else 0.6
                                ),
                                "context": context.text[
                                    max(0, match.start() - 50) : match.end() + 50
                                ],
                                "kb_data": (
                                    philosopher_data["data"]
                                    if philosopher_data
                                    else None
                                ),
                                "position": {
                                    "start": match.start(),
                                    "end": match.end(),
                                },
                            }
                        )

        return sorted(philosophers, key=lambda x: x["confidence"], reverse=True)[
            : context.max_results
        ]

    def _extract_theories(self, context: ExtractionContext) -> List[Dict[str, Any]]:
        """Extract theory mentions"""
        theories = []

        # Check for theory names and keywords
        for theory_id, theory in self.kb.theories.items():
            score = self._calculate_mention_score(theory, context.text)

            if score > context.confidence_threshold:
                # Find specific mentions
                mentions = self._find_entity_mentions(theory, context.text)

                for mention in mentions:
                    theories.append(
                        {
                            "theory_id": theory_id,
                            "name": theory.get_name(context.language),
                            "type": theory.theory_type,
                            "confidence": score,
                            "context": mention["context"],
                            "position": mention["position"],
                            "kb_data": theory.to_dict(),
                        }
                    )

        return sorted(theories, key=lambda x: x["confidence"], reverse=True)[
            : context.max_results
        ]

    def _extract_schools(self, context: ExtractionContext) -> List[Dict[str, Any]]:
        """Extract philosophical school mentions"""
        schools = []

        # Pattern-based extraction
        for pattern_list in self.patterns["school"]["movement"]:
            for match in pattern_list.finditer(context.text):
                school_name = match.group(1).strip()

                # Find in knowledge base
                school_data = self._find_school(school_name)

                if school_data:
                    schools.append(
                        {
                            "school_id": school_data["id"],
                            "name": school_data["name"],
                            "confidence": school_data["confidence"],
                            "context": context.text[
                                max(0, match.start() - 50) : match.end() + 50
                            ],
                            "position": {"start": match.start(), "end": match.end()},
                            "kb_data": school_data["data"],
                        }
                    )

        return sorted(schools, key=lambda x: x["confidence"], reverse=True)[
            : context.max_results
        ]

    def _extract_concepts(self, context: ExtractionContext) -> List[Dict[str, Any]]:
        """Extract philosophical concepts"""
        concepts = []

        # Use both patterns and KB matching
        for pattern_type, pattern_list in self.patterns["concept"].items():
            for pattern in pattern_list:
                for match in pattern.finditer(context.text):
                    concept_term = match.group(1).strip()

                    # Check KB for concept
                    concept_data = self._find_concept(concept_term)

                    concepts.append(
                        {
                            "term": concept_term,
                            "type": pattern_type,
                            "confidence": (
                                concept_data["confidence"] if concept_data else 0.7
                            ),
                            "definition": (
                                match.group(2) if match.lastindex > 1 else None
                            ),
                            "context": context.text[
                                max(0, match.start() - 50) : match.end() + 50
                            ],
                            "position": {"start": match.start(), "end": match.end()},
                            "kb_data": concept_data["data"] if concept_data else None,
                        }
                    )

        return self._deduplicate_concepts(concepts)[: context.max_results]

    def _extract_arguments(self, context: ExtractionContext) -> List[Dict[str, Any]]:
        """Extract philosophical arguments"""
        arguments = []

        # Extract premises and conclusions
        premises = []
        conclusions = []

        for pattern in self.patterns["argument"]["premise_indicators"]:
            for match in pattern.finditer(context.text):
                premises.append(
                    {
                        "text": match.group(1).strip(),
                        "position": {"start": match.start(), "end": match.end()},
                    }
                )

        for pattern in self.patterns["argument"]["conclusion_indicators"]:
            for match in pattern.finditer(context.text):
                conclusions.append(
                    {
                        "text": match.group(1).strip(),
                        "position": {"start": match.start(), "end": match.end()},
                    }
                )

        # Match premises with conclusions
        for conclusion in conclusions:
            related_premises = self._find_related_premises(conclusion, premises)

            # Check if this matches a known argument in KB
            argument_data = self._find_argument_pattern(related_premises, conclusion)

            arguments.append(
                {
                    "premises": [p["text"] for p in related_premises],
                    "conclusion": conclusion["text"],
                    "type": self._classify_argument_type(related_premises, conclusion),
                    "confidence": 0.8,
                    "kb_match": argument_data,
                    "position": conclusion["position"],
                }
            )

        return arguments[: context.max_results]

    def _extract_periods(self, context: ExtractionContext) -> List[Dict[str, Any]]:
        """Extract historical period references"""
        periods = []

        for pattern_type, pattern_list in self.patterns["period"].items():
            for pattern in pattern_list:
                for match in pattern.finditer(context.text):
                    period_text = match.group(0)

                    # Match with KB periods
                    period_data = self._find_period(period_text)

                    periods.append(
                        {
                            "period": period_text,
                            "type": pattern_type,
                            "confidence": (
                                period_data["confidence"] if period_data else 0.7
                            ),
                            "context": context.text[
                                max(0, match.start() - 50) : match.end() + 50
                            ],
                            "position": {"start": match.start(), "end": match.end()},
                            "kb_data": period_data["data"] if period_data else None,
                        }
                    )

        return periods[: context.max_results]

    def _extract_methods(self, context: ExtractionContext) -> List[Dict[str, Any]]:
        """Extract philosophical methods"""
        methods = []

        # Check for method keywords and descriptions
        for method_id, method in self.kb.methods.items():
            score = self._calculate_mention_score(method, context.text)

            if score > context.confidence_threshold:
                mentions = self._find_entity_mentions(method, context.text)

                for mention in mentions:
                    methods.append(
                        {
                            "method_id": method_id,
                            "name": method.get_name(context.language),
                            "confidence": score,
                            "context": mention["context"],
                            "position": mention["position"],
                            "kb_data": method.to_dict(),
                        }
                    )

        return methods[: context.max_results]

    def _extract_relationships(
        self, entities: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []

        # Find philosopher-theory relationships
        for philosopher in entities.get("philosophers", []):
            for theory in entities.get("theories", []):
                if self._are_related(philosopher, theory):
                    relationships.append(
                        {
                            "type": "philosopher_theory",
                            "subject": philosopher["name"],
                            "object": theory["name"],
                            "relation": "developed",
                            "confidence": 0.8,
                        }
                    )

        # Find school-philosopher relationships
        for school in entities.get("schools", []):
            for philosopher in entities.get("philosophers", []):
                if self._are_related(school, philosopher):
                    relationships.append(
                        {
                            "type": "school_philosopher",
                            "subject": school["name"],
                            "object": philosopher["name"],
                            "relation": "includes",
                            "confidence": 0.8,
                        }
                    )

        return relationships

    # Helper methods
    def _calculate_mention_score(self, entity: PhilosophicalEntity, text: str) -> float:
        """Calculate how likely an entity is mentioned in text"""
        score = 0.0
        text_lower = text.lower()

        # Check names
        for name in entity.name.values():
            if name.lower() in text_lower:
                score += 0.5

        # Check keywords
        for keyword in entity.keywords:
            if keyword.lower() in text_lower:
                score += 0.2

        # Check aliases
        for alias in entity.aliases:
            if alias.lower() in text_lower:
                score += 0.3

        return min(score, 1.0)

    def _find_entity_mentions(
        self, entity: PhilosophicalEntity, text: str
    ) -> List[Dict[str, Any]]:
        """Find specific mentions of an entity in text"""
        mentions = []
        text_lower = text.lower()

        # Search for each name variant
        for name in entity.name.values():
            start = 0
            while True:
                pos = text_lower.find(name.lower(), start)
                if pos == -1:
                    break

                mentions.append(
                    {
                        "text": name,
                        "position": {"start": pos, "end": pos + len(name)},
                        "context": text[max(0, pos - 50) : pos + len(name) + 50],
                    }
                )
                start = pos + 1

        return mentions

    def _find_philosopher(self, name: str) -> Optional[Dict[str, Any]]:
        """Find philosopher in knowledge base"""
        name_lower = name.lower()

        for phil_id, philosopher in self.kb.philosophers.items():
            # Check all name variants
            for phil_name in philosopher.name.values():
                if name_lower in phil_name.lower() or phil_name.lower() in name_lower:
                    return {
                        "confidence": 0.9,
                        "data": philosopher.to_dict(),
                        "id": phil_id,
                    }

            # Check aliases
            for alias in philosopher.aliases:
                if name_lower in alias.lower() or alias.lower() in name_lower:
                    return {
                        "confidence": 0.8,
                        "data": philosopher.to_dict(),
                        "id": phil_id,
                    }

        return None

    def _find_school(self, name: str) -> Optional[Dict[str, Any]]:
        """Find philosophical school in knowledge base"""
        name_lower = name.lower()

        for school_id, school in self.kb.schools.items():
            for school_name in school.name.values():
                if (
                    name_lower in school_name.lower()
                    or school_name.lower() in name_lower
                ):
                    return {
                        "id": school_id,
                        "name": school_name,
                        "confidence": 0.9,
                        "data": school.to_dict(),
                    }

        return None

    def _find_concept(self, term: str) -> Optional[Dict[str, Any]]:
        """Find concept in knowledge base"""
        term_lower = term.lower()

        for concept_id, concept in self.kb.concepts.items():
            for concept_name in concept.name.values():
                if (
                    term_lower in concept_name.lower()
                    or concept_name.lower() in term_lower
                ):
                    return {
                        "confidence": 0.9,
                        "data": concept.to_dict(),
                        "id": concept_id,
                    }

        return None

    def _deduplicate_concepts(
        self, concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate concept extractions"""
        seen = set()
        unique = []

        for concept in sorted(concepts, key=lambda x: x["confidence"], reverse=True):
            key = concept["term"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(concept)

        return unique

    def _find_related_premises(
        self, conclusion: Dict[str, Any], premises: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find premises related to a conclusion"""
        related = []

        # Simple proximity-based matching
        for premise in premises:
            if premise["position"]["end"] < conclusion["position"]["start"]:
                # Premise comes before conclusion
                distance = conclusion["position"]["start"] - premise["position"]["end"]
                if distance < 500:  # Within ~500 characters
                    related.append(premise)

        return related

    def _classify_argument_type(
        self, premises: List[Dict[str, Any]], conclusion: Dict[str, Any]
    ) -> str:
        """Classify the type of argument"""
        # Simple heuristic classification
        conclusion_text = conclusion["text"].lower()

        if any(
            word in conclusion_text for word in ["must", "necessarily", "certainly"]
        ):
            return "deductive"
        elif any(
            word in conclusion_text for word in ["probably", "likely", "suggests"]
        ):
            return "inductive"
        elif any(word in conclusion_text for word in ["best explanation", "explains"]):
            return "abductive"
        else:
            return "unknown"

    def _find_argument_pattern(
        self, premises: List[Dict[str, Any]], conclusion: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Match argument pattern with known arguments in KB"""
        # This would match extracted arguments with known argument forms
        # For now, return None
        return None

    def _find_period(self, period_text: str) -> Optional[Dict[str, Any]]:
        """Find historical period in knowledge base"""
        period_lower = period_text.lower()

        for period_id, period in self.kb.periods.items():
            for period_name in period.name.values():
                if (
                    period_lower in period_name.lower()
                    or period_name.lower() in period_lower
                ):
                    return {
                        "confidence": 0.9,
                        "data": period.to_dict(),
                        "id": period_id,
                    }

        return None

    def _are_related(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if two entities are related"""
        # Simple proximity check
        if abs(entity1["position"]["start"] - entity2["position"]["start"]) < 200:
            return True

        # Check KB relationships
        if entity1.get("kb_data") and entity2.get("kb_data"):
            # Check if one references the other
            entity1_refs = entity1["kb_data"].get("references", [])
            entity2_refs = entity2["kb_data"].get("references", [])

            if (
                entity2.get("name") in entity1_refs
                or entity1.get("name") in entity2_refs
            ):
                return True

        return False


# Integration with Ollama extractor
def enhance_extraction_with_knowledge(
    extraction_result: Dict[str, Any], text: str, language: str = "en"
) -> Dict[str, Any]:
    """Enhance Ollama extraction results with knowledge base data"""

    # Initialize knowledge extractor
    kb_extractor = PhilosophyKnowledgeExtractor()

    # Create extraction context
    context = ExtractionContext(
        text=text, language=language, confidence_threshold=0.6, include_related=True
    )

    # Extract entities using knowledge base
    kb_entities = kb_extractor.extract_entities(context)

    # Merge with Ollama results
    enhanced_result = extraction_result.copy()

    # Enhance philosophers
    if "philosophers" in enhanced_result:
        for i, philosopher in enumerate(enhanced_result["philosophers"]):
            # Find matching KB data
            for kb_phil in kb_entities["philosophers"]:
                if philosopher.lower() in kb_phil["name"].lower():
                    enhanced_result["philosophers"][i] = {
                        "name": philosopher,
                        "kb_data": kb_phil["kb_data"],
                        "confidence": kb_phil["confidence"],
                    }
                    break

    # Enhance concepts
    if "key_concepts" in enhanced_result:
        enhanced_concepts = []
        for concept in enhanced_result["key_concepts"]:
            # Find matching KB data
            kb_match = None
            for kb_concept in kb_entities["concepts"]:
                if concept.get("term", "").lower() in kb_concept["term"].lower():
                    kb_match = kb_concept
                    break

            if kb_match:
                enhanced_concepts.append(
                    {
                        **concept,
                        "kb_data": kb_match["kb_data"],
                        "confidence": kb_match["confidence"],
                    }
                )
            else:
                enhanced_concepts.append(concept)

        enhanced_result["key_concepts"] = enhanced_concepts

    # Add KB-only entities that weren't in Ollama results
    enhanced_result["kb_extracted"] = {
        "theories": kb_entities["theories"],
        "schools": kb_entities["schools"],
        "periods": kb_entities["periods"],
        "methods": kb_entities["methods"],
        "relationships": kb_entities["relationships"],
    }

    # Add extraction metadata
    enhanced_result["enhancement_metadata"] = {
        "kb_version": "1.0",
        "language": language,
        "entities_found": sum(len(v) for v in kb_entities.values()),
    }

    return enhanced_result
