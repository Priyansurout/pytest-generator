# Fine-tuning a Pytest Generator with Distil CLI

Train a compact model that automatically generates pytest test cases from Python functions using the Distil Labs platform.

---

## 📋 **Prerequisites**

### **Install the Distil CLI:**

```bash
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh
```

### **Authenticate:**

```bash
distil login
```

---

## 📁 **Training Data**

The `data/` folder contains everything needed to train the model:

| File | Description |
|------|-------------|
| `job_description.json` | Task definition for pytest generation |
| `train.jsonl` | 2,127 training examples (function → test code) |
| `test.jsonl` | 1,758 evaluation examples |
| `config.yaml` | Training configuration (Qwen3-8B base model) |
| `unstructured.csv` | Optional domain examples for synthetic data |

---

## 📝 **Example Training Sample**

**Input:**

```python
def calculate_discount(price: float, percentage: float) -> float:
    """Calculate discounted price.
    
    Args:
        price: Original price (must be positive)
        percentage: Discount percentage 0-100
        
    Returns:
        Discounted price
        
    Raises:
        ValueError: If price is negative or percentage invalid
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")
    return price * (1 - percentage / 100)
```

**Output:**

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

@pytest.mark.parametrize(
    "price, percentage, expected",
    [
        (100.0, 0, 100.0),          # no discount
        (200.0, 50, 100.0),         # 50% discount
        (50.0, 100, 0.0),           # full discount
        (99.99, 33.33, 66.65667),   # custom discount
    ],
)
def test_calculate_discount_success(price, percentage, expected):
    """Verify correct discounted prices for valid inputs."""
    result = calculate_discount(price, percentage)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "price, percentage",
    [
        (-10.0, 20),         # negative price
        (0.0, -5),           # negative percentage
        (0.0, 150),          # too high percentage
    ],
)
def test_calculate_discount_invalid_inputs(price, percentage):
    """Check that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        calculate_discount(price, percentage)
```

---

## 🚀 **Training Steps**

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

**Expected metrics:**
- LLM-as-a-Judge: 0.80 (80% quality)
- ROUGE-L: 0.44
- METEOR: 0.43

### **4. Train the Model**

Start distillation to create your compact pytest generator:

```bash
distil model run-training <model-id>
```

Monitor progress:

```bash
distil model training <model-id>
```

**Training time:** ~3-4 hours on Distil Labs infrastructure

**Expected results after training:**
- Student LLM-as-a-Judge: 0.60 (60% quality)
- ROUGE-L: 0.47 (beats teacher!)
- METEOR: 0.47 (beats teacher!)

### **5. Download the Model**

Once training completes, download the Ollama-ready package:

```bash
distil model download <model-id>
```

---

## 🔄 **Convert to GGUF Format**

For efficient local inference, convert the model to GGUF format.

### **Using Google Colab:**

Upload and run `Model_Conversion_GGUF.ipynb`:

```python
# 1. Download model from HuggingFace
from huggingface_hub import snapshot_download
model_path = snapshot_download(repo_id="<your-username>/pytest-generator")

# 2. Convert to GGUF (FP16)
!python convert_hf_to_gguf.py model_path --outfile pytest-gen-fp16.gguf --outtype f16

# 3. Quantize to Q4_K_M (recommended)
!./llama-quantize pytest-gen-fp16.gguf pytest-gen-q4.gguf Q4_K_M

# 4. Download
from google.colab import files
files.download('pytest-gen-q4.gguf')
```

**Model sizes:**
- FP16: ~16GB (full precision)
- Q4_K_M: ~4.5GB (recommended, 95% quality)
- Q3_K_M: ~3.5GB (88% quality, very limited RAM)

---

## 💻 **Local Deployment**

### **Option 1: Using the Optimized CLI**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place model
mkdir -p models
mv pytest-gen-q4.gguf models/

# 3. Run
python testgen_optimized.py calculator.py
```

**Output:**

```
============================================================
🧪 TestGen - Pytest Test Generator (Optimized)
============================================================

Hardware Configuration:
  CPU: 8 cores (using 6 threads)
  RAM: 16.0 GB
  Accelerator: Metal (Apple Silicon)
  Expected Speed: 15-30s per function ⚡ FAST
============================================================

📄 Processing: calculator.py
  [1/7] add... ✓ 18.2s
  [2/7] subtract... ✓ 22.4s
  ...
  ✅ Created: ./generated_tests/test_calculator.py
  Total: 151s (2.5 min)

✨ All done!
```

### **Option 2: Using Ollama**

Run your trained model with Ollama:

```bash
ollama create pytest-gen -f Modelfile
ollama run pytest-gen
```

**Query via Python:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

function_code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''

response = client.chat.completions.create(
    model="pytest-gen",
    messages=[{"role": "user", "content": function_code}]
)

print(response.choices[0].message.content)
```

---

## ⚙️ **Model Configuration**

The training uses:

- **Base model:** Qwen3-8B-Instruct (student)
- **Teacher model:** GPT OSS 120B
- **Task type:** Question-answering
- **Training examples:** 2,127 real + 10,000 synthetic
- **LoRA rank:** 64
- **Training epochs:** 4
- **Temperature:** 0.7 (teacher generation)

**Alternative configurations:**

For faster inference on CPU-only systems:
- **Qwen3-4B:** 60-90s per test (vs 150s for 8B)
- **Qwen3-1.7B:** 25-40s per test (vs 150s for 8B)

Update `config.yaml`:

```yaml
base:
  task: question-answering
  student_model_name: Qwen3-4B-Instruct  # Faster on CPU
  teacher_model_name: openai.gpt-oss-120b
```

---

## 🎯 **Performance Benchmarks**

### **Generation Quality:**

| Metric | Teacher | Base Model | Trained Model | Improvement |
|--------|---------|------------|---------------|-------------|
| LLM-as-a-Judge | 0.80 | 0.13 | **0.60** | **+360%** |
| ROUGE-L | 0.44 | 0.33 | **0.47** | **+42%** |
| METEOR | 0.43 | 0.35 | **0.47** | **+34%** |

### **Inference Speed (8B Q4 model):**

| Hardware | Speed per Test | 7 Functions |
|----------|---------------|-------------|
| Mac M2/M3/M4 | 20-35s | 2-4 min ✅ |
| NVIDIA GPU | 15-25s | 2-3 min ✅ |
| CPU (8 cores) | 90-150s | 10-18 min ⚠️ |
| CPU (4 cores) | 120-180s | 14-21 min ⚠️ |

### **Test Quality Breakdown:**

| Aspect | Score | Status |
|--------|-------|--------|
| Parametrization | 9/10 | ✅ Excellent |
| Async Support | 9.4/10 | ✅ Perfect |
| Mocking | 8.8/10 | ✅ Very Good |
| Error Testing | 8/10 | ✅ Good |
| Edge Cases | 8.5/10 | ✅ Good |
| **Overall** | **8.5/10** | ✅ Production Ready |

---

## 📦 **What Gets Generated**

The model generates comprehensive pytest tests including:

✅ **Parametrized test cases** with `@pytest.mark.parametrize`
✅ **Error handling tests** with `pytest.raises()`
✅ **Async test support** with `@pytest.mark.asyncio`
✅ **Mock setup and verification** using `unittest.mock`
✅ **Edge cases** (infinity, NaN, empty inputs, boundaries)
✅ **Type checking** with `isinstance()` assertions
✅ **Floating-point comparisons** with `pytest.approx()`

---

## 🔧 **Customization**

### **Adjust Generation Parameters:**

Edit `testgen_optimized.py`:

```python
class Config:
    N_CTX = 2048          # Context window (1024-4096)
    MIN_TOKENS = 300      # Minimum per test
    MAX_TOKENS = 1200     # Maximum per test
    TEMPERATURE = 0.2     # Lower = more deterministic (0.1-0.5)
```

### **Optimize for Your Hardware:**

```python
# For 2-core systems
N_CTX = 1024
BATCH_SIZE = 128
n_threads = 2

# For 8+ core systems
N_CTX = 2048
BATCH_SIZE = 512
n_threads = cpu_count - 2
```

---

## 🐛 **Known Limitations**

1. **Import paths:** Generated tests sometimes use `from my_module import` instead of actual module name (easy manual fix)
2. **Type validation:** Occasionally generates tests for type validation that doesn't exist in function
3. **CPU speed:** Still slow on CPU-only systems (recommend GPU or smaller model)
4. **Minor syntax errors:** ~5% of tests may have small issues (comma instead of dot, incomplete lines)

**Overall pass rate:** ~75% of generated tests run without modification

---

## 📊 **Training Domains**

The model is trained across diverse Python testing scenarios:

- **Basic functions:** Math, string operations, list processing
- **Error handling:** Exception testing, edge cases
- **Async functions:** Async/await, AsyncMock
- **Mocking:** API calls, database operations, file I/O
- **Complex scenarios:** Multiple dependencies, retry logic, caching

---

## 🚀 **Quick Start Guide**

```bash
# 1. Train model with Distil CLI
distil model create pytest-generator
distil model upload-data <model-id> --data ./data
distil model run-teacher-evaluation <model-id>
distil model run-training <model-id>
distil model download <model-id>

# 2. Convert to GGUF (Google Colab)
# Run Model_Conversion_GGUF.ipynb

# 3. Deploy locally
pip install -r requirements.txt
python testgen_optimized.py calculator.py

# 4. Enjoy automated pytest generation! 🎉
```

---

## 📄 **Repository Structure**

```
pytest-generator/
├── data/
│   ├── job_description.json
│   ├── train.jsonl           # 2,127 examples
│   ├── test.jsonl            # 1,758 examples
│   ├── config.yaml
│   └── unstructured.csv
├── testgen_optimized.py       # Optimized CLI
├── requirements.txt
├── Model_Conversion_GGUF.ipynb
├── README.md
└── examples/
    └── calculator.py
```

---

## 🙏 **Credits**

- **Training Platform:** [Distil Labs](https://distillabs.ai)
- **Base Model:** Qwen3-8B (Alibaba)
- **Inference Engine:** [llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Quantization:** GGUF format
- **Model:** [Priyansu19/pytest-generator-v4-GGUF](https://huggingface.co/Priyansu19/pytest-generator-v4-GGUF)

---

## 📝 **License**

MIT License - See LICENSE file

---

## 🔗 **Resources**

- [Distil Labs Documentation](https://docs.distillabs.ai)
- [Training Data Guide](data-question-answering.md)
- [Model on HuggingFace](https://huggingface.co/Priyansu19/pytest-generator-v4-GGUF)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)

---

**Generate pytest tests locally, privately, and efficiently!** 🚀

*Training time: 3-4 hours | Inference time: 15-180s per test | Quality: 8.5/10*
