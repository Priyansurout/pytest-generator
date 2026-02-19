# pytest-generator

## Project Overview

CLI tool that generates pytest unit tests from Python source files using a local, CPU-only LLM. Code never leaves the machine. Extracts function signatures + docstrings via AST (body intentionally stripped — model was fine-tuned on signature+docstring inputs), resolves real dependency method names, then produces `test_*.py` files via a fine-tuned Qwen3-8B model.

## Tech Stack

- **Language**: Python 3.x
- **LLM Runtime**: `llama-cpp-python` (GGUF, CPU-only, `n_gpu_layers=0`)
- **Model**: Qwen3-8B fine-tuned on pytest patterns (`pytest-8b-q4_k_m.gguf`, ~5GB, Q4_K_M)
- **Model Hub**: `huggingface-hub` (auto-download on first run)
- **Config**: `pyyaml` | **Hardware detection**: `psutil` | **Code analysis**: stdlib `ast`

## Key Directories

| Path | Purpose |
|------|---------|
| `pytest_generator.py` | Entire tool — single-file CLI (1174 lines) |
| `finetuning/` | Model training artifacts and Distil Labs configs |
| `finetuning/data/` | `train.csv`, `test.csv`, `config.yaml`, `job_description.json` |
| `examples/` | Demo modules (`calculator.py`, `order_service.py`, `http_fetcher.py`, `mixed_service.py`) + pre-generated tests |
| `examples/demo_project_example/` | Multi-module project used for integration testing |
| `myconfig.yaml` | Example YAML showing all overridable settings |
| `.cache/huggingface/` | Model download cache (gitignored) |
| `.claude/docs/architectural_patterns.md` | Design patterns, data flow, extension points |

## Essential Commands

```bash
pip install -r requirements.txt

# Single file (model auto-downloads on first run, ~5GB)
python pytest_generator.py calculator.py

# Write to specific directory
python pytest_generator.py calculator.py -o ./tests/

# Process entire directory
python pytest_generator.py ./src/ -o ./tests/

# Use custom YAML config
python pytest_generator.py app.py myconfig.yaml

# Disable streaming output
python pytest_generator.py app.py --no-stream

# Save codebase index for reuse
python pytest_generator.py app.py --save-index /tmp/index.json

# Load a previously saved index (skip re-scanning)
python pytest_generator.py app.py --load-index /tmp/index.json

# Override which directory is scanned for the class index
python pytest_generator.py app.py --scan-root ./src/

# Disable dependency resolution (pre-v2 behaviour)
python pytest_generator.py app.py --no-index
```

**Config override precedence:** CLI flags > YAML file > hardcoded defaults

## Core Classes (all in `pytest_generator.py`)

| Class | Line | Responsibility |
|-------|------|----------------|
| `Config` | :24 | Dataclass holding all settings; `from_yaml()` at :263 |
| `Colors` | :299 | ANSI terminal color constants |
| `HardwareDetector` | :312 | Auto-detects CPU threads (all cores − 2) |
| `ModelManager` | :340 | Downloads GGUF from HuggingFace; loads model; runs inference |
| `FunctionExtractor` | :474 | AST visitor — extracts signature+docstring (no body), `typed_params`, `attr_calls` |
| `ImportExtractor` | :593 | AST visitor — maps `ClassName → module_path` from import statements |
| `RuntimeInspector` | :625 | `importlib` + `inspect` — gets real method signatures for pip packages |
| `CodebaseIndexer` | :661 | Scans local `.py` files; builds `{ClassName: {methods: [...]}}` AST index |
| `TestWriter` | :746 | Assembles generated tests; writes `test_<module>.py`; replaces module placeholder |

## Key Functions (non-class, `pytest_generator.py`)

| Function | Line | Purpose |
|----------|------|---------|
| `find_project_root()` | :790 | Walk up from source file to find `.git`, `pyproject.toml`, etc. |
| `resolve_dependencies()` | :809 | Match typed params → RuntimeInspector (pip) → CodebaseIndexer (local) |
| `format_dependency_block()` | :848 | Emit `# Dependencies: db.get_order(), db.update_status()` comment |
| `process_file()` | :887 | Extract → resolve deps → generate tests → write for one file |
| `process_directory()` | :962 | Loop `process_file()` with `gc.collect()` between files |
| `main()` | :1004 | CLI entry: arg parsing, config loading, hardware setup, model load, orchestration |

## Data Flow

```
Source file
  → FunctionExtractor (AST): signature + docstring only (no body)
    → typed_params: {param: ClassName}
    → attr_calls: {param: [methods]}
  → ImportExtractor (AST): {ClassName: module_path}
  → resolve_dependencies()
      1st: RuntimeInspector (importlib+inspect) → real method signatures for pip packages
      2nd: CodebaseIndexer (AST scan) → fallback for local project classes
  → format_dependency_block() → "# Dependencies: db.get_order(id), ..."
  → PROMPT_TEMPLATE + function code + dependency comment
  → ModelManager.generate() (Qwen3-8B GGUF, CPU-only, streaming or buffered)
  → strip <think>…</think>, strip ```python fences, trim
  → TestWriter: replace "module" placeholder → write test_<stem>.py
```

## Config Defaults & YAML Keys

| Config attr | Default | YAML key |
|-------------|---------|----------|
| `MODEL_REPO` | `"Priyansu19/pytest-8b-GGUF"` | `model.repo` |
| `MODEL_FILE` | `"pytest-8b-q4_k_m.gguf"` | `model.file` |
| `N_CTX` | `2048` | `generation.n_ctx` |
| `MAX_TOKENS` | `3072` | `generation.max_tokens` |
| `TEMPERATURE` | `0.05` | `generation.temperature` |
| `TOP_P` | `1` | `generation.top_p` |
| `REPEAT_PENALTY` | `1.0` | `generation.repeat_penalty` |
| `BATCH_SIZE` | `128` | `generation.batch_size` |
| `N_THREADS` | `-1` (auto) | `hardware.n_threads` |
| `DEFAULT_OUTPUT_DIR` | `"./generated_tests/"` | `output.default_dir` |
| `PROMPT_TEMPLATE` | (embedded, lines 38–258) | `prompt.template` |

## Architecture Notes

- **Single-file monolith**: all code in `pytest_generator.py` — intentional for easy distribution
- **Body stripping**: fine-tuned model was trained on signature+docstring only; never send body to model
- **Dependency injection format**: `# Dependencies: X.y(), Z()` comment appended to function code — exact format model was trained on; do not change this format
- **Two-tier dep resolution**: RuntimeInspector first (handles pip packages + arg names), CodebaseIndexer fallback (local AST index)
- **Qwen3 chat format**: prompts wrapped in `<|im_start|>user … <|im_end|><|im_start|>assistant\n<think>\n\n</think>\n\n`
- **Module replacement**: generated code uses `importlib.import_module("module")`; TestWriter replaces "module" with real module name post-generation
- **Memory management**: `gc.collect()` after each file in directory mode; model deleted at end of `main()`
- **Silent fallback**: RuntimeInspector and relative imports fail silently → fall back to AST index or skip deps (never crash)
- **Output naming**: `test_<source_stem>.py` in `DEFAULT_OUTPUT_DIR`

## Fine-Tuning (Distil Labs)

```bash
distil model create pytest-generator
distil model upload-data <model-id> --data ./finetuning/data/
distil model run-teacher-evaluation <model-id>
distil model run-training <model-id>
distil model download <model-id>
```

Teacher model: Deepseek-v3.1 (671B). Student: Qwen3-8B with LoRA distillation.
Training data format: CSV with `(input_function_code, expected_test_output)` columns.

## Additional Documentation

- `.claude/docs/architectural_patterns.md` — design patterns, prompt engineering, config schema, extension points
