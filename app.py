
"""
Philosophy Information AI Extractor - Simple Demo Application

This application demonstrates the philosophy prompt generation capabilities
by importing and running examples from the examples module.
"""

import sys
import os

# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from examples.examples import (
    PhilosophyPromptExamples,
    demonstrate_prompt_customization,
    demonstrate_template_suggestions,
    generate_batch_prompts,
    generate_validation_and_enhancement_prompts,
    main as run_all_examples,
)


def show_menu():
    """Display the main menu options."""
    print(
        """
================================================================================
                    PHILOSOPHY PROMPT GENERATOR DEMO
================================================================================

Available demonstrations:

1.  Quick Single Prompt Demo
2.  Basic Philosophy Examples
3.  Academic Research Examples  
4.  Ethical Analysis Examples
5.  Argument Analysis Examples
6.  Comparative Philosophy Examples
7.  Historical Philosophy Examples
8.  Concept Analysis Examples
9.  Philosopher Profile Examples
10. Exploratory Analysis Examples
11. Critical Analysis Examples
12. Prompt Customization Demo
13. Template Suggestions Demo
14. Batch Prompt Generation
15. Validation & Enhancement Demo
16. Output Format Demonstrations
17. Custom Format Demo
18. Run All Examples
19. Exit

================================================================================
"""
    )


def quick_demo():
    """Run a quick single prompt demonstration."""
    print(
        """
================================================================================
                           QUICK PROMPT DEMO
================================================================================
"""
    )

    from extractors.generator import AdvancedPhilosophyPromptBuilder

    builder = AdvancedPhilosophyPromptBuilder()

    # Sample philosophical text
    sample_text = "What is the meaning of existence? This question has puzzled philosophers for millennia..."

    print(f"Sample text: {sample_text}")
    print("\nGenerating prompt...")

    prompt = builder.build_prompt(
        text=sample_text,
        extraction_mode="exploratory",
        target_audience="general",
        depth_level="intermediate",
    )

    print("\nGenerated Prompt:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)


def run_specific_example(example_name, example_func):
    """Run a specific example function."""
    print(f"\n{'=' * 80}")
    print(f"Running: {example_name}")
    print("=" * 80)

    try:
        result = example_func()
        print(f"\n{example_name} completed successfully!")
        return result
    except Exception as e:
        print(f"Error running {example_name}: {e}")
        return None


def main():
    """Main demo application loop."""
    examples = PhilosophyPromptExamples()

    while True:
        show_menu()

        try:
            choice = input("Enter your choice (1-19): ").strip()

            if choice == "1":
                quick_demo()

            elif choice == "2":
                run_specific_example(
                    "Basic Philosophy", examples.generate_basic_philosophy_prompt
                )

            elif choice == "3":
                run_specific_example(
                    "Academic Research", examples.generate_academic_research_prompt
                )

            elif choice == "4":
                run_specific_example(
                    "Ethical Analysis", examples.generate_ethical_analysis_prompt
                )

            elif choice == "5":
                run_specific_example(
                    "Argument Analysis", examples.generate_argument_analysis_prompt
                )

            elif choice == "6":
                run_specific_example(
                    "Comparative Philosophy",
                    examples.generate_comparative_philosophy_prompt,
                )

            elif choice == "7":
                run_specific_example(
                    "Historical Philosophy",
                    examples.generate_historical_philosophy_prompt,
                )

            elif choice == "8":
                run_specific_example(
                    "Concept Analysis", examples.generate_concept_analysis_prompt
                )

            elif choice == "9":
                run_specific_example(
                    "Philosopher Profile", examples.generate_philosopher_profile_prompt
                )

            elif choice == "10":
                run_specific_example(
                    "Exploratory Analysis", examples.generate_exploratory_prompt
                )

            elif choice == "11":
                run_specific_example(
                    "Critical Analysis", examples.generate_critical_analysis_prompt
                )

            elif choice == "12":
                run_specific_example(
                    "Prompt Customization", demonstrate_prompt_customization
                )

            elif choice == "13":
                run_specific_example(
                    "Template Suggestions", demonstrate_template_suggestions
                )

            elif choice == "14":
                run_specific_example("Batch Prompt Generation", generate_batch_prompts)

            elif choice == "15":
                run_specific_example(
                    "Validation & Enhancement",
                    generate_validation_and_enhancement_prompts,
                )

            elif choice == "16":
                run_specific_example(
                    "Output Format Demonstrations", examples.demonstrate_output_formats
                )

            elif choice == "17":
                run_specific_example(
                    "Custom Format Demo", examples.demonstrate_custom_format
                )

            elif choice == "18":
                print("\nRunning all examples...")
                run_specific_example("All Examples", run_all_examples)

            elif choice == "19":
                print("\nThank you for using the Philosophy Prompt Generator Demo!")
                print("Goodbye!")
                break

            else:
                print("\nInvalid choice. Please enter a number between 1 and 19.")

        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

        # Pause before showing menu again
        if choice != "19":
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    print(
        """
================================================================================
                    PHILOSOPHY INFORMATION AI EXTRACTOR
                              Simple Demo Application
================================================================================

Welcome to the Philosophy Prompt Generator Demo!

This application demonstrates various capabilities of the philosophy prompt
generation system, including different types of philosophical analysis,
customization options, and batch processing features.

================================================================================
"""
    )

    main()
