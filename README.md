# pytest-generator : AI-generated unit test cases from Python function signatures and docstrings.
Accelerate test coverage by auto-generating test skeletons that developers can refine. Runs locally on your machine.

<p align="center">
  <img src="logo.png" alt="Pytest Generator" width="450">
</p>

*Testing boilerplate is tedious. Now you can generate it without sending your code anywhere.*

We fine-tuned a specialized 8B language model to generate high-quality pytest test cases from Python function signatures. Since it runs entirely locally, you get instant test generation with zero API costs, no cloud dependencies, and complete privacy.

Your code never leaves your machine.

## Results

| Model | Parameters | LLM-as-a-Judge | Exact Match | Model Link |
| --- | --- | --- | --- | --- |
| Deepseek.v3.1 (teacher) | 671B | 85% | 86% | |
| **Qwen3-8B (tuned)** | **8B** | **72%** | **83%** | [HuggingFace](https://huggingface.co/Priyansu19/pytest-8b) |
| Qwen3-8B Q4 (quantized) | 8B | ~65% | N/A | |
| Qwen3-8B (base) | 8B | 15% | 36% | |

The tuned **Qwen3-8B** model nears the **671B** teacher’s performance while being over **80×** smaller, with major gains over the base model. The Q4 variant preserves most accuracy and enables efficient local execution, making it ideal for private, on-device test generation.

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Priyansurout/pytest-generator.git
cd pytest-generator
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Download the Model

**Auto-download**  
The model is automatically downloaded on first run (~5GB).

**Manual download**
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Priyansu19/pytest-8b-GGUF', filename='pytest-8b-q4_k_m.gguf', local_dir='.')"
```

### 4. Generate Tests
**Single file**

```bash
# Basic usage
python pytest_generator.py calculator.py

# Custom output directory
python pytest_generator.py calculator.py -o ./tests/

# Disable streaming
python pytest_generator.py calculator.py --no-stream
```
**Batch processing (entire directory)**
```bash
# Process all Python files in src/
python pytest_generator.py ./src/ -o ./tests/
```

### Custom configuration
```bash
# Create myconfig.yaml (see Configuration section)
python pytest_generator.py app.py myconfig.yaml
```
---
## Configuration

pytest-generator can be customized using a YAML configuration file. All settings are optional — sensible defaults are provided for local test generation.

### Model Settings

Controls which local GGUF model is used.

- **`model.repo`** – Hugging Face repository containing the model  
  *(default: `Priyansu19/pytest-8b-GGUF`)*
- **`model.file`** – Specific model file to load  
  - `pytest-8b-q4_k_m.gguf` – fast, ~5GB, recommended  
  - `pytest-8b-f16.gguf` – higher quality, ~16GB, slower  

### Generation Parameters

Controls how tests are generated.

- **`generation.n_ctx`** – Context window size (larger functions need more)  
- **`generation.max_tokens`** – Maximum tokens per generated test file  
- **`generation.temperature`** – Randomness (keep low for deterministic tests)  
- **`generation.top_p`** – Nucleus sampling threshold  
- **`generation.repeat_penalty`** – Reduces repetitive output  
- **`generation.batch_size`** – Higher is faster but uses more RAM  

Recommended defaults are tuned for consistent and reliable pytest output.

### Hardware Settings

Controls CPU usage.

- **`hardware.n_threads`** – Number of CPU threads  
  - `-1` auto-detects and leaves cores free for the system  
  - Set a fixed value to limit CPU usage  

### Output Settings

Controls where generated tests are written.

- **`output.default_dir`** – Default directory for generated test files  
  (can be overridden with the `-o` CLI flag)


Use a custom config file by passing it as the last argument:
```bash
python pytest_generator.py app.py myconfig.yaml
```
---
## Usage Examples

pytest-generator analyzes your Python files, extracts functions, and generates structured pytest test cases locally. Tests are written to a `test_<filename>.py` file by default.

### Single File Test Generation

```bash
python testgen.py calculator.py
```
Example output (truncated):

```bash
📄 Processing: calc.py
✅ Found 7 function(s)

[1/7] add()
- Generates parameterized tests
- Adds type and error handling cases
 .......
 .......
[6/7] async_calculate_bulk()
- Detects async functions
- Uses pytest.mark.asyncio
- Mocks external dependencies

✅ Created: ./generated_tests/test_calculator.py
✨ All done! Tests saved to: ./generated_tests/
```
---

## How We Built pytest-generator

### The Problem

Writing unit tests is repetitive and time-consuming. Developers often skip tests or copy boilerplate because setting up pytest scaffolding takes effort, especially for small or fast-moving projects.

Existing AI-based solutions usually rely on cloud APIs, require sending source code externally, or use models that are too large to run locally. This creates friction around privacy, cost, and offline usage.

### Our Approach

We wanted a solution that:

- **Runs locally** – no API calls, works fully offline, and keeps code private  
- **Is practical on developer machines** – runs on CPU with a single GGUF model  
- **Produces usable tests** – structured pytest skeletons that developers can easily refine  
- **Scales down well** – works with quantized models without large quality loss

To achieve this, we fine-tuned a compact **Qwen3-8B** model specifically for generating pytest test cases from Python function signatures and docstrings. The model is optimized for deterministic output and consistent test structure rather than creative text generation.

### Validating the Base Model Fails

We evaluated the base Qwen3-8B model out of the box on our internal test set for pytest generation.

The base model performed poorly, with low exact match and inconsistent structure. Common failure modes included:

- Missing edge cases and error paths
- Incomplete or un-runnable pytest syntax
- Inconsistent test naming and fixture usage
- Overly generic assertions

This confirmed that while the task is clearly learnable, it is **not** reliably handled by a general-purpose base model — making it a strong candidate for targeted fine-tuning.

### Establishing a Teacher Baseline

We evaluated a large teacher model (Deepseek v3.1, 671B) using structured prompts and reference test patterns.

The teacher achieved strong performance across both LLM-as-a-Judge and exact match metrics. This established a clear target:  
**could a much smaller model match this quality while running fully locally?**

### Training Pipeline

**Seed Data**  
We manually created ~25 high-quality examples covering common Python function patterns, edge cases, error handling, async functions, and realistic docstrings.

**Synthetic Expansion**  
Using a large teacher model, we expanded the dataset to ~10,000 training examples via knowledge distillation, ensuring consistent pytest structure and coverage patterns.

**Fine-tuning**  
We fine-tuned the Qwen3-8B model using LoRA for 4 epochs, optimizing for deterministic and structured test generation.

**Quantization**  
The fine-tuned model was converted to Q4_K_M format, enabling efficient CPU-only local inference with minimal quality loss.

## Qualitative Example

### Input Function

```python
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b

    Raises:
        TypeError: If inputs are not numbers
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Inputs must be numbers")
    return a + b
```
### Generated Tests
```python
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (2.5, 3.5, 6.0),
    (0, 0, 0),
])
def test_add(a, b, expected):
    result = add(a, b)
    assert result == expected

def test_add_type_error():
    with pytest.raises(TypeError):
        add("not_number", 5)
    with pytest.raises(TypeError):
        add(1, "not_number")
    with pytest.raises(TypeError):
        add("string", "string")
```

---

## **Train Your Own Model**

### **1. Create a Model**

```bash
distil model create pytest-generator
```

Save the returned `<model-id>` for subsequent commands.

### **2. Upload Training Data**

```bash
distil model upload-data <model-id> --data ./data
```

### **3. Run Teacher Evaluation**

Validate that a large model can solve the task before training:

```bash
distil model run-teacher-evaluation <model-id>
```

Check status:

```bash
distil model teacher-evaluation <model-id>
```

### **4. Train the Model**

Start distillation to create your compact pytest generator:

```bash
distil model run-training <model-id>
```

Monitor progress:

```bash
distil model training <model-id>
```

### **5. Download the Model**

Once training completes, download the Ollama-ready package:

```bash
distil model download <model-id>
```

---
## FAQ ❓

### Q: Why not just use GPT-4 or Claude for this?

Because your code shouldn’t leave your machine.

pytest-generator runs locally, works offline, and keeps everything private.  
No API keys, no rate limits, no usage costs, no data leakage.

---

### Q: Why not use the base Qwen3-8B model directly?

The base model is general-purpose and not optimized for pytest generation.

Out of the box, it produces:
- Incomplete or incorrect mocks
- Weak or missing assertions
- Inconsistent async handling
- Poor coverage of documented exceptions

Fine-tuning is required to reliably generate **structured, runnable pytest code**.

---

### Q: What is `config.yaml` used for?

`config.yaml` allows you to customize how pytest-generator behaves without changing code.

It controls:
- Which model is used
- Generation parameters (tokens, temperature, etc.)
- Output location

If `config.yaml` is missing, pytest-generator runs with sensible defaults.

---

### Q: Is my code sent anywhere?

No.

The model is downloaded once from HuggingFace and cached locally.  
All parsing, inference, and test generation happen on your machine.

Your source code never leaves your computer.

---

### Q: Why does the first run take longer?

Two one-time costs:

- **Model download:** ~5GB GGUF file  
- **Model loading:** Loading weights into RAM  

After that, runs are fast since the model is cached locally.

---

### Q: Can I use pytest-generator offline?

Yes — that’s the point.

- **100% offline** after initial download  
- **Privacy-first** — no data ever leaves your device  
- **Fast** — local inference on CPU  
- **Free** — no API costs or rate limits  

---

### Q: The generated tests aren’t perfect. Is that expected?

Yes. pytest-generator produces **test skeletons**, not final production tests.

The goal is to:
- Cover happy paths and edge cases
- Reflect docstring intent
- Generate runnable pytest code

You’re expected to review and refine the output — but starting from a strong baseline instead of a blank file.

If you see consistent issues, please open an issue with an example.

---

### Q: Can you train a detector for my specific use case?

Yes! Visit [distillabs.ai](https://www.distillabs.ai/) to discuss custom solutions.

## Links

[![Distil Labs Homepage](https://img.shields.io/badge/Distil_Labs-Homepage-blue)](https://www.distillabs.ai/)
[![Documentation](https://img.shields.io/badge/Docs-distillabs.ai-green)](https://docs.distillabs.ai/)
[![GitHub](https://img.shields.io/badge/GitHub-distil--labs-black)](https://github.com/distil-labs)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-distil--labs-yellow)](https://huggingface.co/distil-labs)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Distil_Labs-0077B5)](https://www.linkedin.com/company/distil-labs/)
[![Slack](https://img.shields.io/badge/Slack-Community-4A154B)](https://join.slack.com/t/distil-labs-community/shared_invite/zt-36zqj87le-i3quWUn2bjErRq22xoE58g)

---

*Built with [Distil Labs](https://distillabs.ai) - turn a prompt and a few examples into production-ready small language models.*


















