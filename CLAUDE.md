# pytest-generator

## Project Overview

CLI tool that generates pytest unit tests from Python source files using a local, CPU-only LLM. Code never leaves the machine. Takes Python files/directories as input, extracts function signatures + docstrings via AST parsing (implementation body is intentionally stripped — the model was fine-tuned on signature+docstring inputs), and produces test files via a fine-tuned 8B model.

## Tech Stack

- **Language**: Python 3.x
- **LLM Runtime**: `llama-cpp-python` (GGUF, CPU-only)
- **Model**: Qwen3-8B fine-tuned on pytest patterns (`pytest-8b-q4_k_m.gguf`, ~5GB, Q4_K_M quantization)
- **Model Hub**: `huggingface-hub` (auto-download on first run)
- **Config**: `pyyaml`
- **Hardware detection**: `psutil`
- **Code analysis**: stdlib `ast`

## Key Directories

| Path | Purpose |
|------|---------|
| `pytest_generator.py` | Entire tool — single-file CLI (~965 lines) |
| `finetuning/` | Model training artifacts and Distil Labs configs |
| `finetuning/data/` | Training/test CSVs and job description |
| `examples/` | Demo Python modules and pre-generated test outputs |
| `myconfig.yaml` | Example YAML config showing all overridable settings |
| `.cache/huggingface/` | Model download cache (gitignored) |

## Essential Commands

**Install:**
```bash
pip install -r requirements.txt
```

**Generate tests:**
```bash
# Single file (model auto-downloads on first run, ~5GB)
python pytest_generator.py calculator.py

# Write to specific directory
python pytest_generator.py calculator.py -o ./tests/

# Process entire directory
python pytest_generator.py ./src/ -o ./tests/

# Use custom config
python pytest_generator.py app.py myconfig.yaml

# Disable streaming output
python pytest_generator.py app.py --no-stream

# Save codebase index for inspection / reuse
python pytest_generator.py app.py --save-index /tmp/index.json

# Load a previously saved index (skip re-scanning)
python pytest_generator.py app.py --load-index /tmp/index.json

# Override which directory is scanned for the class index
python pytest_generator.py app.py --scan-root ./src/

# Disable dependency resolution entirely (pre-v2 behaviour)
python pytest_generator.py app.py --no-index
```

**Config override precedence:** CLI flags > YAML file > hardcoded defaults
Config class defaults: `pytest_generator.py:21` | YAML loading: `pytest_generator.py:192`

**Fine-tune the model (Distil Labs):**
```bash
distil model create pytest-generator
distil model upload-data <model-id> --data ./finetuning/data/
distil model run-teacher-evaluation <model-id>
distil model run-training <model-id>
distil model download <model-id>
```

## Core Classes (all in `pytest_generator.py`)

| Class | Line | Responsibility |
|-------|------|----------------|
| `Config` | :21 | Dataclass holding all settings; `from_yaml()` at :192 |
| `HardwareDetector` | :243 | Auto-detects CPU threads (all cores − 2) |
| `ModelManager` | :271 | Downloads and loads GGUF model; runs inference |
| `FunctionExtractor` | :405 | AST visitor — extracts signature+docstring (no body), typed params, attr calls |
| `CodebaseIndexer` | :494 | Scans project `.py` files; builds class/method index |
| `TestWriter` | :579 | Assembles and writes `test_*.py` output files |
| `Colors` | :230 | ANSI terminal color constants |

**Dependency helpers:** `find_project_root()` :608 | `resolve_dependencies()` :627 | `format_dependency_block()` :648

**Orchestration functions:** `process_file()` :678 | `process_directory()` :752 | `main()` :794

## Recent Changes

### v2 — Codebase-Aware Dependency Resolution (2026-02-18)

**Problem solved:** The LLM was hallucinating mock method names (e.g. `mock_db.fetch_order()` when the real method is `get_order()`).

**How it works now:**
1. On startup, `CodebaseIndexer` scans the project with AST and builds a `{ClassName: {methods: [...]}}` map
2. `FunctionExtractor` now also records `typed_params` (`db: DatabaseClient` → `{"db": "DatabaseClient"}`) and `attr_calls` (method calls detected in the function body)
3. `resolve_dependencies()` cross-references typed params against the index
4. `format_dependency_block()` outputs a `    # Dependencies: db.get_order(), db.update_status()` comment — the exact format the fine-tuned model was trained on
5. That comment is appended to the function code before inference, so the model writes mocks with the real method names

**Key design decision:** The dependency info is injected as a `# Dependencies:` comment (not a custom block) because the fine-tuned model was trained exclusively on that format.

**New functions added:**
- `FunctionExtractor._extract_typed_params()` — extracts non-builtin type-annotated params
- `FunctionExtractor._extract_attr_calls()` — extracts `obj.method()` calls from function body
- `CodebaseIndexer` class — scan, save (JSON), load
- `find_project_root()` — walks up to find `.git`, `pyproject.toml`, etc.
- `resolve_dependencies()` — cross-references params against the index
- `format_dependency_block()` — formats as `# Dependencies:` comment

**Bug fixed:** `--no-stream` was parsed but never passed to `process_file()` / `process_directory()`.

**PROMPT_TEMPLATE fixes:**
- Fixed broken intro sentence
- Removed stray `@pytest.mark.asyncio` on a sync test in Example 2
- Rewrote rules to be cleaner and ordered by priority

### v2.1 — Cross-File Dependency Resolution via Runtime Inspection (2026-02-18)

**Problem solved:** `CodebaseIndexer` only scanned local `.py` files, so pip-installed packages (e.g. `httpx`, `stripe`) were invisible — the LLM still hallucinated mock method names for them.

**How it works now:**
1. `ImportExtractor` AST-parses the source file's import statements and builds a `{ClassName: module_path}` map (e.g. `{"AsyncClient": "httpx"}`)
2. `RuntimeInspector` uses `importlib.import_module()` + `inspect.getmembers()` + `inspect.signature()` to get real method signatures for any importable class — including pip packages
3. `resolve_dependencies()` tries `RuntimeInspector` first; falls back to `CodebaseIndexer` for classes not found via import
4. `format_dependency_block()` now includes argument names when the source is runtime-inspected (e.g. `client.get(url)` instead of `client.get()`)

**New classes added:**
- `ImportExtractor` — AST visitor; `from_source(source)` → `{ClassName: module_path}`
- `RuntimeInspector` — static `get_class_methods(class_name, module_path)` → `[{name, args, is_async}]`

**Known limitations:**
- Modules that execute side effects on import (DB connections, config loading, CLI arg parsing) will fail silently; tool falls back to the AST index or skips deps
- Relative imports (`from .module import X`) won't resolve unless the package is installed in the active environment
- If a pip package is missing from the environment, `RuntimeInspector` silently returns `[]` and the tool falls back to the AST index

## Additional Documentation

Check these files when working on relevant areas:

- `.claude/docs/architectural_patterns.md` — design patterns, data flow, prompt engineering, config schema, and extension points
