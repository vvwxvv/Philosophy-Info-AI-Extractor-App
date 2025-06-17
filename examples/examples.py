# Import the enhanced modules
from extractors.generator import AdvancedPhilosophyPromptBuilder
from extractors.config import PhilosophyExtractorConfig
from extractors.types import (
    PhilosophySourceType,
    ExtractionDepth,
    TargetAudience,
    ExtractionMode,
    PhilosophicalCategory,
)

from prompts.print_prompt_section import print_prompt_section


class PhilosophyPromptExamples:
    """Examples of prompt generation for different philosophical scenarios"""

    def __init__(self):
        self.prompt_builder = AdvancedPhilosophyPromptBuilder()

    def generate_basic_philosophy_prompt(self):
        """Generate a basic philosophy extraction prompt"""

        # Sample text snippet (just for context, not full analysis)
        sample_text = (
            "The categorical imperative is Kant's central philosophical concept..."
        )

        # Generate prompt for basic extraction
        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            template_name="philosophy_basic",
            language="EN",
            depth_level="basic",
            target_audience="general",
        )

        print_prompt_section("Basic Philosophy", prompt)
        return prompt

    def generate_academic_research_prompt(self):
        """Generate prompt for academic philosophical research"""

        sample_text = "In Being and Time, Heidegger explores Dasein's existence..."

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            template_name="comprehensive",
            language="mixed",  # Preserve original terms
            extraction_mode="comprehensive",
            depth_level="expert",
            target_audience="academic",
            categories=["metaphysics", "continental"],
            include_references=True,
            include_historical_context=True,
            preserve_original_language=True,
        )

        print_prompt_section("Academic Research", prompt)
        return prompt

    def generate_ethical_analysis_prompt(self):
        """Generate prompt for ethical philosophy analysis"""

        sample_text = "The trolley problem presents a moral dilemma..."

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            template_name="philosophy_ethical",
            extraction_mode="focused",
            categories=["ethics"],
            depth_level="detailed",
            target_audience="professional",
            custom_focus="moral dilemmas and ethical decision-making",
            include_applications=True,
        )

        print_prompt_section("Ethical Analysis", prompt)
        return prompt

    def generate_argument_analysis_prompt(self):
        """Generate prompt for philosophical argument analysis"""

        sample_text = (
            "Premise 1: All humans are mortal. Premise 2: Socrates is human..."
        )

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            template_name="philosophical_argument",
            extraction_mode="critical",
            categories=["logic"],
            depth_level="detailed",
            include_criticisms=True,
            custom_focus="logical validity and soundness",
        )

        print_prompt_section("Argument Analysis", prompt)
        return prompt

    def generate_comparative_philosophy_prompt(self):
        """Generate prompt for comparing philosophical positions"""

        sample_text = "While Plato argues for eternal Forms, Aristotle proposes..."

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            extraction_mode="comparative",
            categories=["metaphysics"],
            depth_level="detailed",
            target_audience="educational",
            custom_focus="comparing Platonic and Aristotelian metaphysics",
        )

        print_prompt_section("Comparative Philosophy", prompt)
        return prompt

    def generate_historical_philosophy_prompt(self):
        """Generate prompt for historical philosophical analysis"""

        sample_text = (
            "The Enlightenment period saw a shift in philosophical thinking..."
        )

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            template_name="historical_philosophy",
            extraction_mode="historical",
            depth_level="detailed",
            historical_period="Enlightenment",
            cultural_context="18th century Europe",
            include_influences=True,
            include_historical_context=True,
        )

        print_prompt_section("Historical Philosophy", prompt)
        return prompt

    def generate_concept_analysis_prompt(self):
        """Generate prompt for philosophical concept analysis"""

        sample_text = "The concept of 'authenticity' in existentialism refers to..."

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            template_name="philosophical_concept",
            extraction_mode="focused",
            categories=["continental"],
            depth_level="expert",
            custom_focus="existentialist conception of authenticity",
        )

        print_prompt_section("Concept Analysis", prompt)
        return prompt

    def generate_philosopher_profile_prompt(self):
        """Generate prompt for philosopher profile extraction"""

        sample_text = "Jean-Paul Sartre (1905-1980) was a French existentialist..."

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            template_name="philosopher_profile",
            extraction_mode="comprehensive",
            depth_level="detailed",
            include_influences=True,
            include_historical_context=True,
        )

        print_prompt_section("Philosopher Profile", prompt)
        return prompt

    def generate_exploratory_prompt(self):
        """Generate prompt for exploratory philosophical analysis"""

        sample_text = "What does it mean to live a good life? Different cultures..."

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            extraction_mode="exploratory",
            depth_level="intermediate",
            target_audience="general",
            extract_implicit_content=True,
        )

        print_prompt_section("Exploratory Philosophy", prompt)
        return prompt

    def generate_critical_analysis_prompt(self):
        """Generate prompt for critical philosophical analysis"""

        sample_text = (
            "The hard problem of consciousness challenges physicalist theories..."
        )

        prompt = self.prompt_builder.build_prompt(
            text=sample_text,
            extraction_mode="critical",
            categories=["philosophy_of_mind"],
            depth_level="expert",
            target_audience="research",
            include_criticisms=True,
            custom_focus="evaluating arguments about consciousness",
        )

        print_prompt_section("Critical Analysis", prompt)
        return prompt


def demonstrate_prompt_customization():
    """Demonstrate how to customize prompts for specific needs"""

    builder = AdvancedPhilosophyPromptBuilder()

    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CUSTOMIZATION EXAMPLES                        │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    )

    # Example 1: Minimal prompt for quick extraction
    print("\n▸ 1. MINIMAL PROMPT (Quick extraction)")
    print("  " + "─" * 40)

    config = PhilosophyExtractorConfig.create_for_use_case(
        "quick_analysis", source_type=PhilosophySourceType.ESSAY, language="EN"
    )

    prompt = builder.prompt_generator.generate(config)
    print("  " + prompt[:500].replace("\n", "\n  ") + "...")

    # Example 2: Detailed prompt with specific focus
    print("\n\n▸ 2. DETAILED PROMPT (Specific philosophical school)")
    print("  " + "─" * 40)

    config = PhilosophyExtractorConfig(
        template="comprehensive",
        source_type=PhilosophySourceType.TREATISE,
        language="mixed",
        extraction_mode=ExtractionMode.FOCUSED,
        categories_focus=[PhilosophicalCategory.CONTINENTAL],
        custom_guidelines=[
            "Focus on phenomenological aspects",
            "Preserve German philosophical terms",
            "Identify Heideggerian influences",
        ],
    )

    prompt = builder.prompt_generator.generate(config)
    print("  " + prompt[:500].replace("\n", "\n  ") + "...")

    # Example 3: Educational prompt
    print("\n\n▸ 3. EDUCATIONAL PROMPT (For teaching)")
    print("  " + "─" * 40)

    config = PhilosophyExtractorConfig.create_for_use_case(
        "academic_research",
        target_audience=TargetAudience.EDUCATIONAL,
        extraction_depth=ExtractionDepth.INTERMEDIATE,
        include_examples=True,
        custom_guidelines=[
            "Explain concepts clearly for students",
            "Provide concrete examples",
            "Include learning objectives",
        ],
    )

    prompt = builder.prompt_generator.generate(config)
    print("  " + prompt[:500].replace("\n", "\n  ") + "...")


def demonstrate_template_suggestions():
    """Demonstrate template suggestion system"""

    builder = AdvancedPhilosophyPromptBuilder()

    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TEMPLATE SUGGESTION EXAMPLES                        │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    )

    # Test texts for different scenarios
    test_scenarios = [
        {
            "text": "Therefore, we must conclude that free will is incompatible with determinism.",
            "use_case": "analysis",
        },
        {
            "text": "Aristotle was born in 384 BCE in Stagira. He studied under Plato...",
            "use_case": "education",
        },
        {
            "text": "The concept of 'Being-in-the-world' (In-der-Welt-sein) represents...",
            "use_case": "research",
        },
        {
            "text": "Is it morally permissible to lie to save a life? Kant would argue...",
            "use_case": None,
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n▸ {i}. Text: '{scenario['text'][:50]}...'")
        print(f"     Use case: {scenario['use_case'] or 'Not specified'}")

        suggestion = builder.suggest_template(
            text=scenario["text"], use_case=scenario["use_case"]
        )

        print(f"     • Recommended: {suggestion['recommended']}")
        print(f"     • Confidence: {suggestion['confidence']:.2f}")
        print(f"     • Reasoning: {suggestion['reasoning']}")

        if suggestion["alternatives"]:
            print(f"     • Alternatives: {', '.join(suggestion['alternatives'])}")


def generate_batch_prompts():
    """Generate prompts for batch processing different text types"""

    builder = AdvancedPhilosophyPromptBuilder()

    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BATCH PROMPT GENERATION                            │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    )

    # Different text types to process
    text_types = [
        {
            "name": "Ancient Philosophy",
            "sample": "Socrates claims that the unexamined life is not worth living...",
            "config": {
                "historical_period": "Ancient Greece",
                "categories": ["ethics", "epistemology"],
                "extraction_mode": "historical",
            },
        },
        {
            "name": "Modern Ethics",
            "sample": "The utilitarian calculus requires us to maximize happiness...",
            "config": {
                "categories": ["ethics"],
                "extraction_mode": "focused",
                "custom_focus": "consequentialist ethics",
            },
        },
        {
            "name": "Contemporary Metaphysics",
            "sample": "The mind-body problem in the context of modern neuroscience...",
            "config": {
                "categories": ["philosophy_of_mind", "metaphysics"],
                "extraction_mode": "comprehensive",
                "target_audience": "research",
            },
        },
        {
            "name": "Eastern Philosophy",
            "sample": "The Dao that can be spoken is not the true Dao...",
            "config": {
                "cultural_context": "Chinese philosophy",
                "extraction_mode": "exploratory",
                "preserve_original_language": True,
            },
        },
    ]

    prompts = []

    for text_type in text_types:
        print(f"\n▸ Generating prompt for: {text_type['name']}")
        print("  " + "─" * 40)

        prompt = builder.build_prompt(text=text_type["sample"], **text_type["config"])

        prompts.append({"type": text_type["name"], "prompt": prompt})

        # Show first 300 characters of prompt
        print("  " + prompt[:300].replace("\n", "\n  ") + "...")

    return prompts


def generate_validation_and_enhancement_prompts():
    """Generate prompts for validation and enhancement tasks"""

    builder = AdvancedPhilosophyPromptBuilder()

    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│                  ✓ VALIDATION AND ENHANCEMENT PROMPTS                       │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    )

    # Sample extracted data for validation
    sample_extraction = {
        "main_thesis": "Knowledge is justified true belief",
        "key_concepts": ["knowledge", "justification", "belief"],
        "philosophical_tradition": "Epistemology",
        "missing_fields": ["counterarguments", "historical_context"],
    }

    original_text = "The traditional analysis of knowledge as justified true belief..."

    # Generate validation prompt
    print("\n▸ 1. VALIDATION PROMPT")
    print("  " + "─" * 40)
    validation_prompt = builder.build_validation_prompt(
        extracted_data=sample_extraction,
        original_text=original_text,
        template_name="philosophy_basic",
    )
    print("  " + validation_prompt.replace("\n", "\n  "))

    # Generate enhancement prompt
    print("\n▸ 2. ENHANCEMENT PROMPT")
    print("  " + "─" * 40)
    enhancement_prompt = builder.build_enhancement_prompt(
        partial_data=sample_extraction,
        missing_fields=sample_extraction["missing_fields"],
        text=original_text,
    )
    print("  " + enhancement_prompt.replace("\n", "\n  "))


def main():
    """Run all prompt generation examples"""

    print(
        """
╭─────────────────────────────────────────────────────────────────────────────╮
│                   PHILOSOPHY PROMPT GENERATION EXAMPLES                  │
│                                                                             │
│  Demonstrating prompt generation for various philosophical scenarios        │
│  No actual extraction performed - only showing generated prompts            │
╰─────────────────────────────────────────────────────────────────────────────╯
"""
    )

    # Create example generator
    examples = PhilosophyPromptExamples()

    # Generate prompts for different scenarios
    scenarios = [
        ("Basic Philosophy", examples.generate_basic_philosophy_prompt),
        ("Academic Research", examples.generate_academic_research_prompt),
        ("Ethical Analysis", examples.generate_ethical_analysis_prompt),
        ("Argument Analysis", examples.generate_argument_analysis_prompt),
        ("Comparative Philosophy", examples.generate_comparative_philosophy_prompt),
        ("Historical Philosophy", examples.generate_historical_philosophy_prompt),
        ("Concept Analysis", examples.generate_concept_analysis_prompt),
        ("Philosopher Profile", examples.generate_philosopher_profile_prompt),
        ("Exploratory Analysis", examples.generate_exploratory_prompt),
        ("Critical Analysis", examples.generate_critical_analysis_prompt),
    ]

    # Generate each type of prompt
    generated_prompts = {}
    for name, generator_func in scenarios:
        print(f"\n{'─' * 80}")
        print(f"Generating: {name}")
        print("─" * 80)
        try:
            prompt = generator_func()
            generated_prompts[name] = prompt
        except Exception as e:
            print(f"❌ Error generating {name} prompt: {e}")

    # Additional demonstrations
    demonstrate_prompt_customization()
    demonstrate_template_suggestions()
    batch_prompts = generate_batch_prompts()
    generate_validation_and_enhancement_prompts()

    # Summary
    print(
        """
╔═════════════════════════════════════════════════════════════════════════════╗
║                               SUMMARY                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
    )
    print(f"  ✓ Total prompts generated: {len(generated_prompts)}")
    print(f"  ✓ Batch prompts generated: {len(batch_prompts)}")
    print("\n  Available templates:")
    for template in examples.prompt_builder.list_available_templates():
        print(f"    • {template['name']}: {template['description']}")

    return generated_prompts


if __name__ == "__main__":
    # Simple usage - generate a single prompt
    builder = AdvancedPhilosophyPromptBuilder()

    # Quick prompt generation
    quick_prompt = builder.build_prompt(
        text="What is the meaning of existence?",
        extraction_mode="exploratory",
        target_audience="general",
    )

    print(
        """
╭─────────────────────────────────────────────────────────────────────────────╮
│                         QUICK PROMPT EXAMPLE                                │
╰─────────────────────────────────────────────────────────────────────────────╯
"""
    )
    print(quick_prompt)

    # Run all examples
    print("\n\n🚀 RUNNING ALL EXAMPLES...")
    all_prompts = main()
