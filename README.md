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

6. **Combine multiple options**:
```bash
python eval.py --model claude-3-opus-20240229 --tasks math_parity graph_connectivity --long-prompts --max-samples 100
```

5. **Custom model with verbose output**:
```bash
python eval.py --model claude-3-opus --verbose --output-dir claude_results
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
├── eval.py                 # Main evaluation script
├── models.py              # Model implementations (OpenAI, Gemini, Claude)
├── evaluator.py           # Main evaluator class and result aggregation
├── task_evaluators.py     # Task-specific evaluation logic
├── data_structures.py     # Data classes for results
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── LICENSE               # License information
```

### Module Descriptions

- **`eval.py`**: Main entry point with command-line interface and orchestration
- **`models.py`**: Abstract base class and concrete implementations for each foundation model
- **`evaluator.py`**: Core evaluation logic, result aggregation, and report generation
- **`task_evaluators.py`**: Specialized evaluators for different task categories
- **`data_structures.py`**: Data classes for structured result storage

## Output

The framework generates several output files in the specified output directory:

### Files Generated

1. **`detailed_results.json`**: Complete evaluation results with predictions and ground truth
2. **`table1_report.csv`**: Summary report similar to Table 1 in the original paper
3. **`isobench_evaluation.log`**: Detailed execution log

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
@software{isobench_eval_framework,
  title={IsoBench Evaluation Framework},
  author={AI Assistant},
  year={2025},
  url={https://github.com/your-repo/IsoBench-Eval}
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
