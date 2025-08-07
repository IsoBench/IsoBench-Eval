# IsoBench Evaluation Framework

A comprehensive evaluation framework for the IsoBench dataset supporting multiple foundation models (OpenAI GPT, Google Gemini, Anthropic Claude) across all tasks and modalities (image and text representations).

## Overview

IsoBench is a benchmark dataset designed to evaluate multimodal reasoning capabilities of foundation models. This framework provides:

- **Modular Design**: Separate components for each model and task type
- **Multi-Modal Support**: Both image and text modality evaluation
- **Comprehensive Tasks**: Mathematics, science, algorithms, and games
- **Detailed Reporting**: Per-task and aggregate performance metrics
- **Easy Configuration**: Command-line interface with sensible defaults

## Features

- ✅ **Multiple Foundation Models**: OpenAI GPT (including GPT-5 default), Google Gemini, Anthropic Claude
- ✅ **Complete Task Coverage**: All IsoBench tasks across 4 domains
- ✅ **Dual Modality**: Text and image representation evaluation
- ✅ **Flexible Configuration**: Command-line arguments for customization
- ✅ **Results Export**: JSON and CSV output formats
- ✅ **Table 1 Reproduction**: Generate reports similar to the original paper
- ✅ **Resume Functionality**: Skip completed evaluations with intelligent caching
- ✅ **Comprehensive Logging**: Detailed JSON logs with full evaluation traces
- ✅ **Multi-Model Aggregation**: Compare multiple models with dedicated aggregation script
- ✅ **Long Prompt Support**: Use detailed prompts from paper appendix for better results

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages (install via pip):

```bash
pip install openai google-generativeai anthropic datasets pandas numpy pillow
```

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd IsoBench-Eval
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # We'll create this
```

3. Set up API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"  # or GOOGLE_API_KEY
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Usage

### Basic Usage

Run evaluation with GPT-5 (default model):
```bash
python eval.py
```

### Command Line Options

```bash
python eval.py [options]

Options:
  --model MODEL         Model to evaluate (default: gpt-5)
                       Options: gpt-5, gpt-4, gemini-2.0-flash-exp, gemini-1.5-pro, claude-3-opus
  --tasks TASKS         Specific tasks to evaluate (default: all tasks)
  --modalities {text,image}  Modalities to evaluate (default: text image)
  --max-samples N       Maximum samples per task (default: all samples)
  --output-dir DIR      Output directory for results (default: isobench_results)
  --api-key KEY         API key for the model (can also use env vars)
  --verbose             Enable verbose logging
  --help               Show help message
```

### Example Commands

1. **Full evaluation with GPT-5** (default):
```bash
python eval.py
```

2. **Evaluate specific tasks with GPT-4**:
```bash
python eval.py --model gpt-4 --tasks math_parity math_convexity chemistry
```

3. **Quick test with limited samples**:
```bash
python eval.py --model gemini-2.0-flash-exp --max-samples 50
```

4. **Text modality only**:
```bash
python eval.py --modalities text --output-dir text_only_results
```

5. **Use long prompts (paper appendix style)**:
```bash
python eval.py --long-prompts
```

6. **Resume previous evaluation**:
```bash
python eval.py --model gpt-4 --resume
```

7. **Fresh start (clear cache)**:
```bash
python eval.py --model gpt-4 --fresh-start
```

8. **Combine multiple options**:
```bash
python eval.py --model claude-3-opus-20240229 --tasks math_parity graph_connectivity --long-prompts --max-samples 100 --verbose
```

## Available Tasks

### Mathematics Tasks
- `math_parity`: Function parity classification (even/odd/neither)
- `math_convexity`: Function convexity analysis
- `math_breakpoint`: Breakpoint counting in piecewise functions

### Science Tasks  
- `chemistry`: Chemical reaction and molecular analysis
- `physics`: Physics problem solving

### Algorithm Tasks
- `graph_connectivity`: Graph connectivity analysis
- `graph_maxflow`: Maximum flow computation
- `graph_isomorphism`: Graph isomorphism detection

### Game Tasks
- `winner_id`: Game winner prediction
- `puzzle`: Puzzle solving

## Features

### Long Prompts Support
The framework supports both short and long prompts:

- **Short prompts** (default): Concise task descriptions for efficient evaluation
- **Long prompts** (`--long-prompts`): Detailed prompts from the paper appendix that include:
  - Comprehensive task definitions and examples
  - Step-by-step reasoning instructions
  - Mathematical definitions and concepts
  - Visual analysis guidelines for image tasks

Long prompts are particularly useful for:
- More detailed model reasoning
- Better performance on complex mathematical tasks
- Reproducing paper results that used detailed instructions

Example long prompt for math parity:
```
You are given a mathematical function f(x) = x^2 + 3x.

Your task is to determine whether this function has even symmetry, odd symmetry, or neither.

Recall the definitions:
- A function f(x) is EVEN if f(-x) = f(x) for all x in the domain...
- A function f(x) is ODD if f(-x) = -f(x) for all x in the domain...
...
```

## Project Structure

```
IsoBench-Eval/
├── eval.py                    # Main evaluation script and CLI
├── aggregate_results.py       # Multi-model results aggregation  
├── src/                       # Core evaluation package
│   ├── __init__.py           # Package exports and initialization
│   ├── models.py             # Model implementations (OpenAI, Gemini, Claude)
│   ├── evaluator.py          # Main evaluator and result aggregation  
│   ├── task_evaluators.py    # Task-specific evaluation logic with caching
│   └── data_structures.py    # Data classes for structured results
├── isobench_results/          # Default output directory
│   └── model_name/           # Per-model results and logs
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
└── LICENSE                  # License information
```

### Module Descriptions

- **`eval.py`**: Main entry point with comprehensive CLI and evaluation orchestration
- **`aggregate_results.py`**: Aggregates individual model results into comparative reports
- **`src/models.py`**: Abstract base class and model implementations with intelligent response parsing
- **`src/evaluator.py`**: Core evaluation logic, result aggregation, and report generation with resume support
- **`src/task_evaluators.py`**: Specialized evaluators for different task categories with caching and detailed logging
- **`src/data_structures.py`**: Data classes for structured result storage and type safety

## Output Structure

The framework generates a structured output directory with comprehensive logging and reporting:

```
isobench_results/
├── model_name/                    # e.g., gpt-5, gpt-4, gemini-1.5-pro
│   ├── math_parity.json          # Detailed task logs with predictions
│   ├── math_convexity.json       # Full evaluation data per task
│   ├── chemistry.json            
│   ├── ...                       # One JSON file per evaluated task
│   ├── evaluation_summary.json   # Aggregated statistics and accuracies
│   └── individual_report.csv     # Table 1 format for this model only
├── table1_report.csv             # Combined report (multi-model only)
├── table1_comprehensive_report.csv # Detailed aggregation (via aggregate script)
├── task_breakdown_report.csv     # Task-by-task analysis (via aggregate script)  
└── isobench_evaluation.log       # Execution log
```

### Files Generated

1. **Task-level JSON logs** (`{task_name}.json`): Complete evaluation results with:
   - Dataset samples and ground truth
   - Model inputs and outputs  
   - Parsing results and correctness
   - Timestamps and metadata

2. **Evaluation summary** (`evaluation_summary.json`): Statistical summary with:
   - Overall and per-task accuracies
   - Text vs image modality breakdown
   - Sample counts and performance metrics

3. **Individual model report** (`individual_report.csv`): Table 1 format for single model
4. **Combined reports**: Multi-model Table 1 comparison (when applicable)
5. **Execution log** (`isobench_evaluation.log`): Detailed run information

### Result Format

**Summary Report Example:**
```
Model          Text Accuracy  Image Accuracy  Gap (Text - Image)  Gap (Points)
GPT-5          85.2%          78.9%           6.3%                6.3
GPT-4          82.1%          75.4%           6.7%                6.7
Gemini-Pro     79.8%          73.2%           6.6%                6.6
```

**Task-level Results Example:**
```
=== Task-level Results for GPT-5 ===

math_parity:
  Text: 0.892
  Image: 0.834

math_convexity:
  Text: 0.876
  Image: 0.812
  
chemistry:
  Text: 0.823
  Image: 0.756
```

## Supported Models

### OpenAI GPT Models
- **GPT-5** (default): Latest GPT model (currently maps to GPT-4o)
- **GPT-4**: GPT-4 Turbo with vision capabilities
- **GPT-3.5**: GPT-3.5 Turbo

### Google Gemini Models  
- **Gemini 2.0 Flash Experimental**: Latest experimental model with fast response times
- **Gemini 1.5 Pro**: Advanced multimodal capabilities and long context
- **Gemini Ultra**: Most capable model (when available)

### Anthropic Claude Models
- **Claude-3 Opus**: Most capable Claude model
- **Claude-3 Sonnet**: Balanced performance/cost
- **Claude-3 Haiku**: Fast and efficient

## Configuration

### Environment Variables

Set API keys using environment variables:
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Google Gemini  
export GEMINI_API_KEY="AI..."
# Alternative: GOOGLE_API_KEY also supported for backward compatibility

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Rate Limiting

The framework includes built-in rate limiting (1 second delay between API calls) to respect API limits. Modify `rate_limit_delay` in model classes if needed.

## Multi-Model Analysis

### Aggregating Results from Multiple Models

When you've evaluated multiple models separately, use the aggregation script to combine results:

```bash
# Aggregate all models in the output directory
python aggregate_results.py --output-dir isobench_results

# Aggregate specific models only  
python aggregate_results.py --models gpt-5 gpt-4 gemini-1.5-pro

# Include detailed task-by-task breakdown
python aggregate_results.py --include-task-breakdown --verbose
```

This generates:
- **`table1_comprehensive_report.csv`**: Full comparison with sample counts
- **`table1_report.csv`**: Clean summary (Model, Text Accuracy, Image Accuracy, Gap)  
- **`task_breakdown_report.csv`**: Per-task performance analysis (optional)

### Resume Functionality

The framework supports resuming interrupted evaluations:

```bash
# Resume from where you left off (default behavior)
python eval.py --model gpt-4 --resume

# Start completely fresh (clear all cache)  
python eval.py --model gpt-4 --fresh-start

# Disable resume but keep existing cache
python eval.py --model gpt-4 --no-resume
```

The system automatically detects completed task-modality combinations and skips them unless specified otherwise.

## Performance Tips

1. **Start Small**: Use `--max-samples` for initial testing
2. **Single Modality**: Use `--modalities text` for faster evaluation
3. **Specific Tasks**: Use `--tasks` to focus on particular areas
4. **Verbose Mode**: Use `--verbose` for debugging

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure environment variables are set correctly
   - Check API key validity and permissions

2. **Dataset Loading Issues**:
   - Verify internet connection
   - Check if datasets library is installed: `pip install datasets`

3. **Memory Issues**:
   - Use `--max-samples` to limit evaluation size
   - Process tasks individually with `--tasks`

4. **Rate Limiting**:
   - Framework includes automatic rate limiting
   - Increase delay in model classes if needed

### Debug Mode

Run with verbose logging for detailed information:
```bash
python eval.py --verbose
```

Check the log file `isobench_evaluation.log` for complete execution details.

### Detailed Evaluation Logs

Each task generates comprehensive JSON logs containing:

- **Dataset samples**: Original problem data with LaTeX, code, images
- **Model inputs**: Complete prompts sent to the model  
- **Model outputs**: Raw responses before parsing
- **Evaluation details**: Parsed predictions, ground truth, correctness
- **Metadata**: Timestamps, task names, modalities, prompt types

Example log entry structure:
```json
{
  "sample_index": 0,
  "task_name": "math_parity", 
  "modality": "text",
  "timestamp": "2025-08-07T13:40:52.805620",
  "dataset_sample": {
    "label": "odd",
    "latex": "$$f(x) = -\\frac{2x^5}{...}$$",
    "code": "f(x) = -2*x**5/(...)",
    "image_available": true
  },
  "evaluation": {
    "input_prompt": "You are given a mathematical function...",
    "model_response": "Answer: odd\n\nReasoning: ...",
    "parsed_prediction": "odd", 
    "ground_truth": "odd",
    "is_correct": true,
    "prompt_type": "long"
  }
}
```

This detailed logging enables:
- **Debugging model errors** by examining exact inputs/outputs
- **Analyzing prompt effectiveness** across different formulations  
- **Understanding failure modes** through response patterns
- **Reproducing specific results** with complete evaluation traces

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Submit a pull request

### Adding New Models

To add support for new models:

1. Create a new model class in `models.py` inheriting from `BaseModel`
2. Implement `predict_text` and `predict_image_text` methods
3. Add model creation logic in `eval.py`
4. Update documentation

### Adding New Tasks

To add support for new tasks:

1. Create a new task evaluator in `task_evaluators.py` 
2. Add task name to appropriate category in `evaluator.py`
3. Implement task-specific prompt generation
4. Test with existing models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this evaluation framework, please cite:

```bibtex
@inproceedings{fu2024isobench,
      title={{I}so{B}ench: Benchmarking Multimodal Foundation Models on Isomorphic Representations}, 
      author={Deqing Fu and Ruohao Guo and Ghazal Khalighinejad and Ollie Liu and Bhuwan Dhingra and Dani Yogatama and Robin Jia and Willie Neiswanger},
      booktitle={First Conference on Language Modeling (COLM)},
      year={2024}
}
```

## Acknowledgments

- IsoBench dataset creators for the comprehensive benchmark
- OpenAI, Google, and Anthropic for providing foundation model APIs
- The open-source community for supporting libraries

## Support

For issues and questions:

1. Check the [troubleshooting](#troubleshooting) section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include log files and configuration details

---

**Happy Evaluating! 🚀**
