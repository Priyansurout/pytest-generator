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

## **Training Data**

The `data/` folder contains everything needed to train the model:

| File | Description |
|------|-------------|
| `job_description.json` | Task definition for pytest generation |
| `train.jsonl` | training examples (function → test code) |
| `test.jsonl` | evaluation examples |
| `config.yaml` | Training configuration (Qwen3-8B base model) |

---

## **Example Training Sample**

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
        (100.0, 0, 100.0),          
        (200.0, 50, 100.0),         
        (50.0, 100, 0.0),          
        (99.99, 33.33, 66.65667),   
    ],
)
def test_calculate_discount_success(price, percentage, expected):
    result = calculate_discount(price, percentage)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "price, percentage",
    [
        (-10.0, 20),         
        (0.0, -5),           
        (0.0, 150),          
    ],
)
def test_calculate_discount_invalid_inputs(price, percentage):
    """Check that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        calculate_discount(price, percentage)
```

---

## **Training Steps**

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



## **Model Configuration**

The training uses:

- **Base model:** Qwen3-8B-Instruct (student)
- **Teacher model:** Deepseek.v3.1
- **Task type:** Question-answering





