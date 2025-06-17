# Philosophy-Info-AI-Extractor-App

# Philosophy Info AI Extractor App

A comprehensive AI-powered tool for extracting, analyzing, and processing philosophical information from various sources. This application provides advanced prompt generation, data extraction, and CSV processing capabilities for philosophical research and analysis.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Features](#features)
- [API Integration](#api-integration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- API keys for AI services (OpenAI, Ollama, etc.)

### Installation

1. **Clone or download the project**

   ```bash
   cd PhilosophyInfoAIExtractorApp
   ```

2. **Set up virtual environment**

   ```bash
   # Create virtual environment
   python -m venv appenv
   
   # Activate virtual environment
   # Windows:
   appenv\Scripts\activate
   # macOS/Linux:
   source appenv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   # Or install manually:
   pip install aiohttp requests pandas numpy
   ```

4. **Configure API keys**

   ```bash
   # Set environment variables or create a .env file
   export OPENAI_API_KEY="your_openai_api_key"
   export OLLAMA_BASE_URL="http://localhost:11434"  # if using Ollama locally
   ```

## Usage

### Basic Usage

The app provides multiple ways to extract philosophical information:

#### Running the Main Application

```bash
python app.py
```

This will run all example scenarios and demonstrate:

- Basic philosophy prompt generation
- Academic research prompts
- Ethical analysis prompts
- Argument analysis prompts
- Comparative philosophy prompts
- Historical philosophy prompts
- Concept analysis prompts
- Philosopher profile prompts
- Exploratory analysis prompts
- Critical analysis prompts

#### Quick Prompt Generation

```python
from extractors.generator import AdvancedPhilosophyPromptBuilder

# Create a prompt builder
builder = AdvancedPhilosophyPromptBuilder()

# Generate a simple prompt
prompt = builder.build_prompt(
    text="What is the meaning of existence?",
    extraction_mode="exploratory",
    target_audience="general"
)

print(prompt)
```

### Advanced Usage

#### Custom Prompt Generation

```python
from extractors.generator import AdvancedPhilosophyPromptBuilder
from extractors.types import ExtractionMode, TargetAudience, ExtractionDepth

builder = AdvancedPhilosophyPromptBuilder()

# Generate academic research prompt
academic_prompt = builder.build_prompt(
    text="In Being and Time, Heidegger explores Dasein's existence...",
    template_name="comprehensive",
    language="mixed",
    extraction_mode="comprehensive",
    depth_level="expert",
    target_audience="academic",
    categories=["metaphysics", "continental"],
    include_references=True,
    include_historical_context=True,
    preserve_original_language=True
)
```

#### Ethical Analysis

```python
ethical_prompt = builder.build_prompt(
    text="The trolley problem presents a moral dilemma...",
    template_name="philosophy_ethical",
    extraction_mode="focused",
    categories=["ethics"],
    depth_level="detailed",
    target_audience="professional",
    custom_focus="moral dilemmas and ethical decision-making",
    include_applications=True
)
```

#### Historical Philosophy Analysis

```python
historical_prompt = builder.build_prompt(
    text="The Enlightenment period saw a shift in philosophical thinking...",
    template_name="historical_philosophy",
    extraction_mode="historical",
    depth_level="detailed",
    historical_period="Enlightenment",
    cultural_context="18th century Europe",
    include_influences=True,
    include_historical_context=True
)
```

### CSV Processing

The app includes powerful CSV processing capabilities:

#### Basic CSV Operations

```python
from assets.csv_handler import CSVHandler
from assets.csv_processor import CSVProcessor

# Initialize handlers
csv_handler = CSVHandler()
csv_processor = CSVProcessor()

# Read CSV file
data = csv_handler.read_csv("philosophy_data.csv")

# Process data
processed_data = csv_processor.process_philosophy_data(data)

# Save processed data
csv_handler.save_csv(processed_data, "processed_philosophy_data.csv")
```

#### CSV Batch Processing

```python
from assets.csv_convenience_functions import batch_process_philosophy_files

# Process multiple CSV files
results = batch_process_philosophy_files(
    input_directory="raw_data/",
    output_directory="processed_data/",
    extraction_mode="comprehensive"
)
```

#### CSV Processing Issues

## Configuration

### Available Templates

The app provides several pre-built templates:

| Template | Description |
|----------|-------------|
| `philosophy_basic` | Basic philosophical analysis |
| `comprehensive` | Comprehensive academic analysis |
| `philosophy_ethical` | Ethical philosophy focus |
| `philosophical_argument` | Argument analysis |
| `historical_philosophy` | Historical context analysis |
| `philosophical_concept` | Concept analysis |
| `philosopher_profile` | Philosopher biography extraction |

### Extraction Modes

| Mode | Description |
|------|-------------|
| `basic` | Simple extraction |
| `comprehensive` | Full analysis |
| `focused` | Targeted extraction |
| `critical` | Critical analysis |
| `comparative` | Comparative analysis |
| `historical` | Historical context |
| `exploratory` | Exploratory analysis |

### Target Audiences

| Audience | Description |
|----------|-------------|
| `general` | General audience |
| `educational` | Educational content |
| `academic` | Academic research |
| `professional` | Professional analysis |
| `research` | Research purposes |

### Depth Levels

| Level | Description |
|-------|-------------|
| `basic` | Basic information |
| `intermediate` | Moderate detail |
| `detailed` | Comprehensive detail |
| `expert` | Expert-level analysis |

## Project Structure

```text
PhilosophyInfoAIExtractorApp/
├── app.py                          # Main application file
├── extractors/                     # Core extraction modules
│   ├── generator.py               # Prompt generation
│   ├── advanced_extractor.py      # Advanced extraction logic
│   ├── api.py                     # API integration
│   ├── config.py                  # Configuration management
│   ├── types.py                   # Type definitions
│   └── ...
├── assets/                        # Utility modules
│   ├── csv_handler.py            # CSV file operations
│   ├── csv_processor.py          # CSV data processing
│   ├── openai_chat_client.py     # OpenAI integration
│   └── ollama_api_client.py      # Ollama integration
├── knowledges/                    # Knowledge base modules
├── prompts/                       # Prompt templates
└── appenv/                        # Virtual environment
```

## Features

### Core Capabilities

- **Advanced Prompt Generation**: Create sophisticated prompts for philosophical analysis
- **Multi-AI Integration**: Support for OpenAI, Ollama, and other AI services
- **CSV Processing**: Comprehensive CSV handling and data processing
- **Batch Operations**: Process multiple files simultaneously
- **Template System**: Pre-built templates for common philosophical analyses
- **Validation System**: Built-in validation and enhancement capabilities

### User Interface

#### Command Line Interface

The app provides a rich command-line interface with:

- Color-coded output for different sections
- Progress indicators for batch operations
- Formatted output with proper structure
- Clear error handling and suggestions

#### Example Output Format

```text
╭─────────────────────────────────────────────────────────────────────────────╮
│                   PHILOSOPHY PROMPT GENERATION EXAMPLES                  │
│                                                                             │
│  Demonstrating prompt generation for various philosophical scenarios        │
│  No actual extraction performed - only showing generated prompts            │
╰─────────────────────────────────────────────────────────────────────────────╯

▸ Generating: Basic Philosophy
───────────────────────────────────────────────────────────────────────────────
[Generated prompt content...]

▸ Generating: Academic Research
───────────────────────────────────────────────────────────────────────────────
[Generated prompt content...]
```

### Interactive Features

- Template suggestions and recommendations
- Parameter validation and error checking
- Batch processing with real-time progress tracking
- Comprehensive logging and debugging

## API Integration

### OpenAI Integration

```python
from assets.openai_chat_client import OpenAIChatClient

client = OpenAIChatClient(api_key="your_api_key")

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a philosophy expert."},
        {"role": "user", "content": "Explain Kant's categorical imperative."}
    ]
)
```

### Ollama Integration

```python
from assets.ollama_api_client import OllamaAPIClient

client = OllamaAPIClient(base_url="http://localhost:11434")

response = client.chat_completion(
    model="llama2",
    messages=[
        {"role": "system", "content": "You are a philosophy expert."},
        {"role": "user", "content": "Explain Kant's categorical imperative."}
    ]
)
```

## Use Cases

### Academic Research

- Extract philosophical concepts from academic papers
- Generate comprehensive analysis prompts
- Process historical philosophical texts
- Compare philosophical positions

### Educational Content

- Create educational materials from philosophical texts
- Generate study guides and summaries
- Extract key concepts for curriculum development

### Content Analysis

- Analyze philosophical arguments
- Extract ethical frameworks
- Process philosophical debates
- Generate critical analysis prompts

### Data Processing

- Convert philosophical texts to structured data
- Process large datasets of philosophical content
- Generate CSV reports for analysis

## Advanced Features

### Validation and Enhancement

```python
# Validate extracted data
validation_prompt = builder.build_validation_prompt(
    extracted_data=sample_extraction,
    original_text=original_text,
    template_name="philosophy_basic"
)

# Enhance partial data
enhancement_prompt = builder.build_enhancement_prompt(
    partial_data=sample_extraction,
    missing_fields=["counterarguments", "historical_context"],
    text=original_text
)
```

### Custom Field Libraries

```python
from extractors.enhanced_field_library import EnhancedFieldLibrary

field_lib = EnhancedFieldLibrary()

# Add custom fields
field_lib.add_custom_field(
    name="philosophical_tradition",
    description="The philosophical tradition or school of thought",
    extraction_rules=["identify_tradition", "categorize_school"]
)
```

### Advanced Batch Processing

```python
# Process multiple texts with different configurations
text_types = [
    {
        "name": "Ancient Philosophy",
        "sample": "Socrates claims that the unexamined life...",
        "config": {
            "historical_period": "Ancient Greece",
            "categories": ["ethics", "epistemology"],
            "extraction_mode": "historical"
        }
    },
    # Additional configurations...
]

for text_type in text_types:
    prompt = builder.build_prompt(
        text=text_type["sample"], 
        **text_type["config"]
    )
```

## Examples

### Example 1: Basic Philosophy Analysis

```python
from extractors.generator import AdvancedPhilosophyPromptBuilder

builder = AdvancedPhilosophyPromptBuilder()

prompt = builder.build_prompt(
    text="The categorical imperative is Kant's central philosophical concept...",
    template_name="philosophy_basic",
    language="EN",
    depth_level="basic",
    target_audience="general"
)

print(prompt)
```

### Example 2: Academic Research

```python
prompt = builder.build_prompt(
    text="In Being and Time, Heidegger explores Dasein's existence...",
    template_name="comprehensive",
    language="mixed",
    extraction_mode="comprehensive",
    depth_level="expert",
    target_audience="academic",
    categories=["metaphysics", "continental"],
    include_references=True,
    include_historical_context=True,
    preserve_original_language=True
)
```

### Example 3: CSV Processing

```python
from assets.csv_handler import CSVHandler
from assets.csv_processor import CSVProcessor

# Read and process CSV
handler = CSVHandler()
processor = CSVProcessor()

data = handler.read_csv("philosophy_texts.csv")
processed = processor.process_philosophy_data(data)
handler.save_csv(processed, "processed_data.csv")
```

## Troubleshooting

### Common Issues

#### API Key  

- Ensure your API keys are properly set in environment variables
- Check API key permissions and quotas
- Verify network connectivity to API endpoints

#### Import  

- Confirm you're in the correct virtual environment
- Verify all dependencies are installed: `pip list`
- Check Python version compatibility

#### CSV  

- Verify file permissions and access rights
- Ensure CSV files are properly formatted
- Check file paths are correct and accessible
- Validate CSV structure and encoding

#### Memory Issues

- For large files, consider processing in smaller chunks
- Use batch processing for multiple files
- Monitor system resources during processing

### Getting Help

1. **Check the logs**: Review detailed error messages in console output
2. **Verify configuration**: Ensure all settings and paths are correct
3. **Test with sample data**: Start with smaller datasets for troubleshooting
4. **Check dependencies**: Confirm all required packages are properly installed
5. **Review API quotas**: Ensure API limits haven't been exceeded

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

We welcome contributions to improve this project. Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes with appropriate tests
5. Ensure code follows project style guidelines
6. Submit a pull request with a clear description

### Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for all functions and classes
- Add unit tests for new functionality
- Update documentation as needed

### Reporting Issues

When reporting issues, please include:

- Python version and operating system
- Full error traceback
- Steps to reproduce the issue
- Expected vs actual behavior

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with modern Python libraries and best practices
- Integrates with popular AI services (OpenAI, Ollama)
- Designed for extensibility and customization
- Focused on philosophical content analysis and extraction

---
