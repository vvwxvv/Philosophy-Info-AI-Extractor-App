def print_prompt_section(title: str, prompt: str):
    """Pretty print a prompt section with ASCII art"""
    width = 80

    # ASCII art headers for different sections
    headers = {
        "Basic Philosophy": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          BASIC PHILOSOPHY PROMPT                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Academic Research": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ACADEMIC RESEARCH PROMPT                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Ethical Analysis": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ETHICAL ANALYSIS PROMPT                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Argument Analysis": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        ARGUMENT ANALYSIS PROMPT                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Comparative Philosophy": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      COMPARATIVE PHILOSOPHY PROMPT                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Historical Philosophy": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      HISTORICAL PHILOSOPHY PROMPT                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Concept Analysis": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       CONCEPT ANALYSIS PROMPT                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Philosopher Profile": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       PHILOSOPHER PROFILE PROMPT                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Exploratory Philosophy": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      PHILOSOPHY PROMPT                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
        "Critical Analysis": """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       CRITICAL ANALYSIS PROMPT                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""",
    }

    # Get header or create default
    header = headers.get(
        title.replace(" Extraction Prompt", ""),
        f"\n{'─' * width}\n▶ {title}\n{'─' * width}\n",
    )

    print(header)
    print(prompt)
    print(f"\n{'─' * width}\n")
