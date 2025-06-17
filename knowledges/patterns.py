"""Enhanced extraction patterns for philosophical content"""

from typing import Dict, List, Pattern
import re


class PhilosophicalPatterns:
    """Comprehensive patterns for philosophical text extraction"""

    # Enhanced philosopher name patterns
    PHILOSOPHER_PATTERNS = {
        "direct_mention": [
            r"\b(?:philosopher|thinker|theorist|author)\s*[:：]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"\b(?:哲学家|思想家|理论家|作者)\s*[:：]\s*([\u4e00-\u9fa5]+)",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:argues?|claims?|states?|believes?|wrote|proposed)\b",
            r"\b(?:according to|per|following)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+(?:theory|philosophy|view|argument|concept)",
            r"\b([A-Z][a-z]+)\s+\((?:\d{4}[-–]\d{4}|\d{3,4}\s*BCE?[-–]\d{3,4}\s*CE?)\)",
        ],
        "with_dates": [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((?:born\s+)?(\d{4}|\d{3,4}\s*BCE?)(?:\s*[-–]\s*(\d{4}|\d{3,4}\s*CE?))?\)",
            r"([\u4e00-\u9fa5]+)\s*[\(（](?:生于\s*)?(\d{4}|\d{3,4}\s*BCE?)(?:\s*[-–]\s*(\d{4}|\d{3,4}\s*CE?))?[\)）]",
        ],
    }

    # Argument structure patterns
    ARGUMENT_PATTERNS = {
        "premise_indicators": [
            r"(?:since|because|given that|assuming that|for|as)\s+(.+?)(?=[.;,]|\s+therefore|\s+thus)",
            r"(?:因为|由于|鉴于|假设|既然)\s*(.+?)(?=[。；，]|\s+所以|\s+因此)",
            r"(?:first|second|third|furthermore|moreover|additionally),?\s*(.+?)(?=[.;,])",
            r"(?:premise|assumption|given):\s*(.+?)(?=[.;,\n])",
            r"(?:前提|假设|给定)[:：]\s*(.+?)(?=[。；，\n])",
        ],
        "conclusion_indicators": [
            r"(?:therefore|thus|hence|consequently|so|it follows that|we can conclude)\s+(.+?)(?=[.!?]|$)",
            r"(?:所以|因此|故|从而|由此可见|我们可以得出)\s*(.+?)(?=[。！？]|$)",
            r"(?:in conclusion|this shows that|this proves that|which means)\s+(.+?)(?=[.!?]|$)",
            r"(?:conclusion|result):\s*(.+?)(?=[.!?\n]|$)",
        ],
        "conditional": [
            r"if\s+(.+?)\s*,?\s*then\s+(.+?)(?=[.;,])",
            r"(?:如果|若)\s*(.+?)\s*[，,]?\s*(?:那么|则)\s*(.+?)(?=[。；，])",
            r"(?:whenever|when)\s+(.+?)\s*,?\s*(.+?)(?=[.;,])",
            r"(?:provided that|given that)\s+(.+?)\s*,?\s*(.+?)(?=[.;,])",
        ],
    }

    # Concept extraction patterns
    CONCEPT_PATTERNS = {
        "definition": [
            r"(?:the\s+)?(?:concept|notion|idea|principle)\s+of\s+([a-z]+(?:\s+[a-z]+)*)",
            r"([a-z]+(?:\s+[a-z]+)*)\s+(?:is defined as|means|refers to|denotes)\s+(.+?)(?=[.;,])",
            r"(?:by|what I mean by)\s+([a-z]+(?:\s+[a-z]+)*)\s+(?:is|I mean)\s+(.+?)(?=[.;,])",
            r'"([^"]+)"\s+(?:is|means|refers to)\s+(.+?)(?=[.;,])',
            r"(?:概念|观念|思想|原则)\s*[:：]\s*([\u4e00-\u9fa5]+)",
            r"([\u4e00-\u9fa5]+)\s*(?:被定义为|意味着|指的是|表示)\s*(.+?)(?=[。；，])",
        ],
        "technical_terms": [
            r"\b([a-z]+(?:-[a-z]+)*)\s+\([A-Z][a-z]+(?:\s+[a-z]+)*\)",  # term (Translation)
            r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b(?=\s+(?:is|are|was|were)\s+)",  # CamelCase terms
            r"(?:technical term|terminus technicus):\s*([a-z]+(?:\s+[a-z]+)*)",
            r"(?:术语|专门用语)[:：]\s*([\u4e00-\u9fa5]+)",
        ],
    }

    # School/tradition patterns
    SCHOOL_PATTERNS = {
        "movement": [
            r"\b([A-Z][a-z]+ism)\b",  # Matches philosophical -isms
            r"\b([A-Z][a-z]+(?:ean|ian|ist|ite))\s+(?:school|tradition|philosophy)",
            r"(?:school|tradition|movement)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:学派|传统|运动)\s*[:：]\s*([\u4e00-\u9fa5]+)",
            r"([\u4e00-\u9fa5]+(?:主义|学派|传统))",
        ],
        "adherents": [
            r"(?:followers?|adherents?|proponents?)\s+of\s+([A-Z][a-z]+(?:ism)?)",
            r"([A-Z][a-z]+(?:ean|ian|ist))\s+(?:philosopher|thinker)",
            r"(?:追随者|信徒|支持者)\s*[:：]\s*([\u4e00-\u9fa5]+)",
        ],
    }

    # Time period patterns
    PERIOD_PATTERNS = {
        "century": [
            r"(\d{1,2}(?:st|nd|rd|th))\s+century(?:\s+(BCE?|CE?))?",
            r"(\d{1,2})\s*世纪",
            r"(?:early|mid|late)\s+(\d{1,2}(?:st|nd|rd|th))\s+century",
        ],
        "era": [
            r"(?:Ancient|Medieval|Renaissance|Modern|Contemporary)\s+(?:period|era|philosophy)",
            r"(?:古代|中世纪|文艺复兴|现代|当代)\s*(?:时期|时代|哲学)",
            r"(?:Pre-Socratic|Hellenistic|Scholastic|Enlightenment)\s+(?:period|era)",
            r"(?:前苏格拉底|希腊化|经院|启蒙)\s*(?:时期|时代)",
        ],
        "specific_dates": [
            r"(?:from|between)\s+(\d{3,4}\s*BCE?)\s+(?:to|and)\s+(\d{3,4}\s*CE?)",
            r"(?:从|自)\s*(?:公元前)?\s*(\d{3,4})\s*(?:年)?\s*(?:到|至)\s*(?:公元)?\s*(\d{3,4})\s*(?:年)?",
        ],
    }

    @classmethod
    def compile_patterns(cls) -> Dict[str, Dict[str, List[Pattern]]]:
        """Compile all patterns for efficient matching"""
        compiled = {}

        for category, pattern_dict in [
            ("philosopher", cls.PHILOSOPHER_PATTERNS),
            ("argument", cls.ARGUMENT_PATTERNS),
            ("concept", cls.CONCEPT_PATTERNS),
            ("school", cls.SCHOOL_PATTERNS),
            ("period", cls.PERIOD_PATTERNS),
        ]:
            compiled[category] = {}
            for subcategory, patterns in pattern_dict.items():
                compiled[category][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    for pattern in patterns
                ]

        return compiled


def initialize_patterns() -> Dict[str, List[str]]:
    """Initialize pattern dictionary for knowledge base"""
    patterns = PhilosophicalPatterns()

    # Flatten patterns for export
    flattened = {}
    for category, pattern_dict in [
        ("philosopher", patterns.PHILOSOPHER_PATTERNS),
        ("argument", patterns.ARGUMENT_PATTERNS),
        ("concept", patterns.CONCEPT_PATTERNS),
        ("school", patterns.SCHOOL_PATTERNS),
        ("period", patterns.PERIOD_PATTERNS),
    ]:
        for subcategory, pattern_list in pattern_dict.items():
            key = f"{category}_{subcategory}"
            flattened[key] = pattern_list

    return flattened
