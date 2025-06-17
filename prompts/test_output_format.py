#!/usr/bin/env python3
"""
Test script for output format functionality
"""

from extractors.generator import AdvancedPhilosophyPromptBuilder


def test_output_formats():
    """Test different output formats"""
    builder = AdvancedPhilosophyPromptBuilder()

    sample_text = "Kant's categorical imperative states that one should act only according to maxims that could be universalized."

    formats = ["json", "csv", "xml", "yaml", "markdown", "table"]

    for output_format in formats:
        print(f"\n{'='*60}")
        print(f"TESTING OUTPUT FORMAT: {output_format.upper()}")
        print(f"{'='*60}")

        try:
            prompt = builder.build_prompt(text=sample_text, output_format=output_format)

            # Check if output format instructions are included
            has_output_instructions = "OUTPUT FORMAT INSTRUCTIONS:" in prompt
            print(f"Output format instructions included: {has_output_instructions}")

            if has_output_instructions:
                # Find the output format section
                start_idx = prompt.find("## OUTPUT FORMAT INSTRUCTIONS:")
                if start_idx != -1:
                    end_idx = prompt.find("--- TEXT TO ANALYZE ---", start_idx)
                    if end_idx != -1:
                        output_section = prompt[start_idx:end_idx].strip()
                        print(f"Output format section found:")
                        print(
                            output_section[:200] + "..."
                            if len(output_section) > 200
                            else output_section
                        )
                    else:
                        print("Could not find end of output format section")
                else:
                    print("Could not find output format section")
            else:
                print("No output format instructions found in prompt")

        except Exception as e:
            print(f"Error testing {output_format}: {e}")


if __name__ == "__main__":
    test_output_formats()
