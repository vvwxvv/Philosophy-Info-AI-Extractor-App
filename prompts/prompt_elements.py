# Core guidelines that apply to all extractions
CORE_GUIDELINES = [
    "Extract ONLY what is explicitly stated in the text",
    "Return `null` for missing information",
    "Maintain original language of the text",
    "List multiple possible values as an array",
    "Preserve exact quotes when available",
    "Include confidence scores for uncertain extractions",
    "Maintain hierarchical relationships in the data",
    "Handle both explicit and implicit information appropriately",
]

# Default system role for extraction tasks
SYSTEM_ROLE = """You are an expert information extraction system. Your task is to extract structured information from text while maintaining accuracy, consistency, and completeness. Follow these guidelines precisely:"""

# Output format definitions
OUTPUT_FORMATS = {
    "csv": {
        "name": "CSV",
        "description": "For spreadsheet applications",
        "example": "1,2,3",
    },
    "json": {
        "name": "JSON",
        "description": "For data exchange and APIs",
        "example": '{"key": "value"}',
    },
    "txt": {
        "name": "Plain Text",
        "description": "For plain text output",
        "example": "This is a plain text example",
    },
    "markdown": {
        "name": "Markdown",
        "description": "For documentation and web content",
        "example": "**Bold text** and *italic text*",
    },
    "html": {
        "name": "HTML",
        "description": "For web display",
        "example": "<p>This is an HTML example</p>",
    },
}

INSTRUCTIONS = [
    # General Extraction Rules
    "1. Information Extraction Rules:",
    "   - Extract only factual, verifiable information",
    "   - Maintain source attribution when available",
    "   - Preserve temporal and causal relationships",
    "   - Handle ambiguous cases with appropriate confidence scores",
    "   - Include metadata about extraction context",
    # Data Quality Standards
    "2. Data Quality Requirements:",
    "   - Ensure extracted data is complete and accurate",
    "   - Validate against known constraints and rules",
    "   - Cross-reference related information when available",
    "   - Flag potential inconsistencies or contradictions",
    "   - Maintain data lineage and provenance",
    # Formatting and Structure
    "3. Output Formatting:",
    "   - Use consistent structure for all extractions",
    "   - Include confidence scores (0.0-1.0) for each field",
    "   - Provide extraction timestamps",
    "   - Include field-level metadata",
    "   - Maintain hierarchical relationships in nested structures",
    # Error Handling
    "4. Error and Edge Cases:",
    "   - Handle missing or incomplete information gracefully",
    "   - Flag uncertain or ambiguous extractions",
    "   - Provide alternative interpretations when relevant",
    "   - Include error codes and descriptions",
    "   - Document assumptions made during extraction",
    # Validation and Verification
    "5. Validation Requirements:",
    "   - Verify extracted information against source text",
    "   - Cross-check related fields for consistency",
    "   - Validate against domain-specific rules",
    "   - Ensure temporal and logical consistency",
    "   - Verify hierarchical relationships",
    # Performance Considerations
    "6. Performance Guidelines:",
    "   - Optimize for processing speed",
    "   - Handle large text inputs efficiently",
    "   - Maintain memory efficiency",
    "   - Scale appropriately with input size",
    "   - Provide progress indicators for long extractions",
    # Security and Privacy
    "7. Security Requirements:",
    "   - Handle sensitive information appropriately",
    "   - Follow data privacy guidelines",
    "   - Implement access controls",
    "   - Log security-relevant events",
    "   - Sanitize output for sensitive data",
    # Integration Guidelines
    "8. Integration Requirements:",
    "   - Provide clear API documentation",
    "   - Include version information",
    "   - Support standard data formats",
    "   - Enable easy integration with other systems",
    "   - Provide clear error messages and codes",
    # Monitoring and Logging
    "9. Monitoring Requirements:",
    "   - Log all extraction attempts",
    "   - Track success and failure rates",
    "   - Monitor performance metrics",
    "   - Record confidence scores",
    "   - Track data quality metrics",
    # Maintenance and Updates
    "10. Maintenance Guidelines:",
    "    - Support version control",
    "    - Enable easy updates and modifications",
    "    - Maintain backward compatibility",
    "    - Document all changes",
    "    - Support configuration management",
]

# Get CSV format description
csv_desc = OUTPUT_FORMATS["csv"]["description"]

# Get JSON example
json_example = OUTPUT_FORMATS["json"]["example"]
# Core philosophy guidelines
PHILOSOPHY_CORE_GUIDELINES = [
    "Extract philosophical arguments with clear premise-conclusion structure",
    "Identify and categorize key philosophical concepts with precise definitions",
    "Map philosophical traditions and schools of thought systematically",
    "Extract logical relationships and argumentative structures",
    "Identify epistemological assumptions and knowledge claims",
    "Extract metaphysical positions and ontological commitments",
    "Map ethical frameworks and moral principles",
    "Identify methodological approaches and philosophical tools",
    "Extract historical context and intellectual influences",
    "Categorize philosophical positions and their implications",
    "Identify conceptual relationships and philosophical hierarchies",
    "Extract argumentative patterns and logical forms",
    "Map philosophical terminology and technical vocabulary",
    "Identify philosophical problems and their proposed solutions",
]

# Category-specific guidelines
ETHICAL_PHILOSOPHY_GUIDELINES = [
    "Identify the ethical framework or theory being used",
    "Extract moral principles and values",
    "Note any ethical dilemmas or conflicts",
    "Identify the criteria for moral judgment",
    "Extract consequences and their moral significance",
    "Note any deontological considerations",
    "Identify virtue-based elements",
    "Extract utilitarian calculations if present",
    "Note any rights-based arguments",
    "Identify the scope of moral consideration",
]

METAPHYSICAL_PHILOSOPHY_GUIDELINES = [
    "Identify the fundamental nature of reality",
    "Extract claims about existence and being",
    "Note any ontological commitments",
    "Identify the nature of time and space",
    "Extract claims about causality",
    "Note any mind-body relationship claims",
    "Identify claims about free will",
    "Extract claims about universals and particulars",
    "Note any claims about substance",
    "Identify claims about possibility and necessity",
]

EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES = [
    "Identify the theory of knowledge being used",
    "Extract claims about justification",
    "Note any skepticism or doubt",
    "Identify the sources of knowledge",
    "Extract claims about truth",
    "Note any claims about perception",
    "Identify claims about reason and rationality",
    "Extract claims about a priori knowledge",
    "Note any claims about testimony",
    "Identify claims about certainty",
]

AESTHETIC_PHILOSOPHY_GUIDELINES = [
    "Identify the aesthetic theory or approach",
    "Extract claims about beauty",
    "Note any claims about taste",
    "Identify claims about art",
    "Extract claims about aesthetic experience",
    "Note any claims about aesthetic judgment",
    "Identify claims about aesthetic value",
    "Extract claims about aesthetic properties",
    "Note any claims about aesthetic pleasure",
    "Identify claims about aesthetic meaning",
]

LANGUAGE_PHILOSOPHY_GUIDELINES = [
    "Analyze the meaning and reference of key terms",
    "Identify semantic theories or approaches",
    "Extract claims about language use",
    "Note any claims about meaning and truth",
    "Identify claims about sense and reference",
    "Extract claims about linguistic meaning",
    "Note any claims about language games",
    "Identify claims about speech acts",
    "Extract claims about interpretation",
    "Note any claims about translation",
]

MIND_PHILOSOPHY_GUIDELINES = [
    "Identify theories of consciousness",
    "Extract claims about mental states",
    "Note any claims about qualia",
    "Identify claims about intentionality",
    "Extract claims about the mind-body problem",
    "Note any claims about mental causation",
    "Identify claims about personal identity",
    "Extract claims about mental representation",
    "Note any claims about consciousness",
    "Identify claims about mental content",
]

LOGIC_PHILOSOPHY_GUIDELINES = [
    "Identify the logical structure of arguments",
    "Extract claims about reasoning",
    "Note any logical fallacies",
    "Identify claims about validity",
    "Extract claims about soundness",
    "Note any claims about inference",
    "Identify claims about deduction",
    "Extract claims about induction",
    "Note any claims about abduction",
    "Identify claims about logical form",
]

POLITICAL_PHILOSOPHY_GUIDELINES = [
    "Identify political theories or approaches",
    "Extract claims about justice",
    "Note any claims about rights",
    "Identify claims about liberty",
    "Extract claims about equality",
    "Note any claims about power",
    "Identify claims about authority",
    "Extract claims about democracy",
    "Note any claims about citizenship",
    "Identify claims about social contract",
]

# Combine all guidelines
PHILOSOPHY_GUIDELINES = (
    PHILOSOPHY_CORE_GUIDELINES
    + ETHICAL_PHILOSOPHY_GUIDELINES
    + METAPHYSICAL_PHILOSOPHY_GUIDELINES
    + EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES
    + AESTHETIC_PHILOSOPHY_GUIDELINES
    + LANGUAGE_PHILOSOPHY_GUIDELINES
    + MIND_PHILOSOPHY_GUIDELINES
    + LOGIC_PHILOSOPHY_GUIDELINES
    + POLITICAL_PHILOSOPHY_GUIDELINES
)

# System role for philosophy extraction
PHILOSOPHY_SYSTEM_ROLE = """You are a philosophical knowledge extraction system designed to analyze texts and extract structured philosophical information. Your task is to identify and categorize philosophical elements including arguments, concepts, traditions, and methodologies. Focus on extracting precise, structured knowledge that can be used for further analysis and processing."""

# Output format for philosophy extraction
PHILOSOPHY_OUTPUT_FORMAT = {
    "name": "Philosophy JSON",
    "description": "Structured JSON format for philosophical text analysis",
    "example": """{
    "main_arguments": [
        {
            "premise": "string",
            "conclusion": "string",
            "type": "string",
            "strength": "string",
            "logical_structure": "string"
        }
    ],
    "key_concepts": [
        {
            "term": "string",
            "definition": "string",
            "context": "string",
            "semantic_analysis": "string"
        }
    ],
    "philosophical_tradition": {
        "school": "string",
        "period": "string",
        "influences": ["string"],
        "methodology": "string"
    },
    "ethical_considerations": [
        {
            "principle": "string",
            "application": "string",
            "implications": ["string"],
            "framework": "string"
        }
    ],
    "epistemological_assumptions": [
        {
            "claim": "string",
            "justification": "string",
            "type": "string",
            "evidence": "string"
        }
    ],
    "metaphysical_claims": [
        {
            "claim": "string",
            "scope": "string",
            "implications": ["string"],
            "ontological_status": "string"
        }
    ],
    "aesthetic_judgments": [
        {
            "judgment": "string",
            "criteria": "string",
            "context": "string",
            "theory": "string"
        }
    ],
    "language_analysis": [
        {
            "term": "string",
            "meaning": "string",
            "reference": "string",
            "usage": "string"
        }
    ],
    "mental_states": [
        {
            "state": "string",
            "description": "string",
            "consciousness": "string",
            "qualia": "string"
        }
    ],
    "logical_analysis": [
        {
            "structure": "string",
            "validity": "string",
            "fallacies": ["string"],
            "inference_type": "string"
        }
    ],
    "political_theory": [
        {
            "theory": "string",
            "principles": ["string"],
            "implications": ["string"],
            "critique": "string"
        }
    ],
    "methodology": {
        "approach": "string",
        "techniques": ["string"],
        "limitations": ["string"],
        "justification": "string"
    },
    "context": {
        "historical": "string",
        "cultural": "string",
        "intellectual": "string",
        "influences": ["string"]
    }
}""",
}

# Field descriptions for different categories
PHILOSOPHY_FIELD_DESCRIPTIONS = {
    "general": """Extract structured philosophical knowledge including: main arguments (premises, conclusions, logical structure), key concepts (definitions, relationships, hierarchies), philosophical traditions (schools, periods, influences), epistemological assumptions (knowledge claims, justification methods), metaphysical positions (ontological commitments, fundamental claims), ethical frameworks (principles, values, moral theories), and methodological approaches (analysis techniques, philosophical tools).""",
    "ethical": """Extract structured ethical knowledge including: ethical frameworks (deontological, consequentialist, virtue-based), moral principles (duties, rights, values), ethical dilemmas (conflicts, trade-offs), moral judgments (evaluations, criteria), ethical implications (consequences, applications), and moral reasoning (justification, argumentation).""",
    "metaphysical": """Extract structured metaphysical knowledge including: ontological commitments (what exists, fundamental entities), claims about reality (nature of being, substance), temporal and spatial claims (time, space, causality), mind-body relationships (consciousness, physicalism), claims about universals and particulars (abstraction, instantiation), and modal claims (necessity, possibility, contingency).""",
    "epistemological": """Extract structured epistemological knowledge including: theories of knowledge (justification, sources, limits), claims about truth (correspondence, coherence, pragmatic), skepticism and doubt (challenges, responses), perception and experience (sensory knowledge, phenomenology), reason and rationality (logical reasoning, a priori knowledge), and certainty and probability (degrees of belief, confidence).""",
    "aesthetic": """Extract structured aesthetic knowledge including: aesthetic theories (beauty, art, taste), aesthetic judgments (criteria, evaluations), aesthetic experience (perception, emotion), aesthetic properties (form, content, expression), aesthetic value (intrinsic, instrumental), and aesthetic meaning (interpretation, significance).""",
    "language": """Extract structured linguistic knowledge including: semantic theories (meaning, reference, truth), language use (communication, speech acts), linguistic meaning (sense, reference, context), language games (rules, practices), interpretation (understanding, translation), and linguistic analysis (structure, function).""",
    "mind": """Extract structured mental knowledge including: theories of consciousness (phenomenal, access, self), mental states (beliefs, desires, emotions), qualia (subjective experience, phenomenal properties), intentionality (aboutness, mental content), mind-body problem (dualism, physicalism, functionalism), and mental causation (causal relationships, mental processes).""",
    "logic": """Extract structured logical knowledge including: argument structure (premises, conclusions, validity), reasoning patterns (deduction, induction, abduction), logical fallacies (errors, weaknesses), inference types (formal, informal, material), logical form (structure, validity, soundness), and logical analysis (evaluation, assessment).""",
    "political": """Extract structured political knowledge including: political theories (justice, rights, liberty), principles of governance (democracy, authority, power), social organization (equality, citizenship, community), political institutions (state, law, rights), social contract (agreement, obligation, consent), and political implications (applications, consequences).""",
}

# Templates for different categories
PHILOSOPHY_TEMPLATES = {
    "general": {
        "name": "General Philosophy Analysis",
        "description": "Comprehensive analysis of philosophical texts",
        "guidelines": PHILOSOPHY_GUIDELINES,
    },
    "ethical": {
        "name": "Ethical Philosophy Analysis",
        "description": "Analysis of ethical arguments and moral philosophy",
        "guidelines": ETHICAL_PHILOSOPHY_GUIDELINES,
    },
    "metaphysical": {
        "name": "Metaphysical Philosophy Analysis",
        "description": "Analysis of metaphysical claims and ontology",
        "guidelines": METAPHYSICAL_PHILOSOPHY_GUIDELINES,
    },
    "epistemological": {
        "name": "Epistemological Philosophy Analysis",
        "description": "Analysis of knowledge claims and epistemology",
        "guidelines": EPISTEMOLOGICAL_PHILOSOPHY_GUIDELINES,
    },
    "aesthetic": {
        "name": "Aesthetic Philosophy Analysis",
        "description": "Analysis of aesthetic judgments and art philosophy",
        "guidelines": AESTHETIC_PHILOSOPHY_GUIDELINES,
    },
    "language": {
        "name": "Philosophy of Language Analysis",
        "description": "Analysis of meaning, reference, and language use",
        "guidelines": LANGUAGE_PHILOSOPHY_GUIDELINES,
    },
    "mind": {
        "name": "Philosophy of Mind Analysis",
        "description": "Analysis of consciousness and mental states",
        "guidelines": MIND_PHILOSOPHY_GUIDELINES,
    },
    "logic": {
        "name": "Logical Analysis",
        "description": "Analysis of argument structure and logical form",
        "guidelines": LOGIC_PHILOSOPHY_GUIDELINES,
    },
    "political": {
        "name": "Political Philosophy Analysis",
        "description": "Analysis of political theory and social philosophy",
        "guidelines": POLITICAL_PHILOSOPHY_GUIDELINES,
    },
}

# Categories for filtering guidelines
PHILOSOPHY_CATEGORIES = {
    "ethical": ["ethics", "moral", "virtue", "duty", "rights"],
    "metaphysical": ["metaphysics", "ontology", "being", "reality", "existence"],
    "epistemological": [
        "epistemology",
        "knowledge",
        "truth",
        "justification",
        "belief",
    ],
    "aesthetic": ["aesthetics", "beauty", "art", "taste", "experience"],
    "language": ["language", "meaning", "reference", "semantics", "sign"],
    "mind": ["mind", "consciousness", "mental", "qualia", "intentionality"],
    "logic": ["logic", "reasoning", "argument", "validity", "inference"],
    "political": ["politics", "justice", "rights", "power", "authority"],
}
