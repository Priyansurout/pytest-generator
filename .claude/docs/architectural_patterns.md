# Architectural Patterns

## Data Flow

```
Python source file(s)
    → FunctionExtractor (AST visitor)   pytest_generator.py:402
    → list of function dicts (name, args, return type, docstring, is_async)
    → prompt assembly (PROMPT_TEMPLATE)  pytest_generator.py:34
    → ModelManager.generate()           pytest_generator.py:340
    → streaming or buffered token output
    → regex post-processing (strip thinking tags)
    → TestWriter.write_tests()          pytest_generator.py:438
    → test_<filename>.py
```

## Single-File Monolith

The entire tool lives in `pytest_generator.py`. There are no submodules or packages. All classes, helpers, and the `main()` entry point are in one file. This is intentional for distribution simplicity (single-file install).

## Dataclass Config with YAML Override

`Config` (`pytest_generator.py:21`) is a `@dataclass` with class-level defaults. `Config.from_yaml()` (`pytest_generator.py:192`) loads a YAML file and patches matching attributes in-place. CLI arguments then apply on top of the returned instance.

**Override precedence (high → low):**
1. CLI flags (`--output`, `--no-stream`, etc.) — applied in `main()` :574
2. YAML config file (positional `config` arg) — applied via `Config.from_yaml()` :192
3. `@dataclass` field defaults — `Config` :21

**YAML keys map to Config attributes:**
```yaml
model:      → MODEL_REPO, MODEL_FILE
generation: → N_CTX, MAX_TOKENS, TEMPERATURE, TOP_P, REPEAT_PENALTY, BATCH_SIZE
hardware:   → N_THREADS (-1 = auto)
output:     → DEFAULT_OUTPUT_DIR
prompt:     → PROMPT_TEMPLATE
```

## AST Visitor Pattern

`FunctionExtractor(ast.NodeVisitor)` (`pytest_generator.py:402`) walks the parsed AST. `visit_FunctionDef` and `visit_AsyncFunctionDef` (`pytest_generator.py:409`) collect:
- Function name, arguments with type annotations
- Return type annotation
- Docstring (first statement if it is a `Constant`)
- `is_async` flag

The extracted list is the sole input to the prompt builder. No raw source text is passed to the model.

## CPU-Only Inference with Streaming

`ModelManager` (`pytest_generator.py:269`) enforces CPU-only execution:
- `n_gpu_layers=0` hardcoded at `pytest_generator.py:321`
- Thread count from `HardwareDetector` (all physical cores − 2)
- Streaming via `stream=True` on the `Llama.__call__` (`pytest_generator.py:354`)
- Non-streaming path collects the full response string (`pytest_generator.py:373`)

Model is downloaded once via `hf_hub_download` (`pytest_generator.py:291`) and cached under `.cache/huggingface/`.

## Prompt Engineering Convention

The prompt template (`pytest_generator.py:34`) follows these conventions:
- **Qwen3 chat format**: uses `<|im_start|>` / `<|im_end|>` tokens
- **Pre-filled thinking**: assistant turn starts with `<think>\n` to encourage chain-of-thought
- **Structured rules embedded in system prompt**: mock patterns, exception testing, dependency comment syntax
- **Few-shot examples**: shown inline in the template for format consistency
- Template is overridable via YAML `prompt.template` key

## Memory Management Between Files

When processing a directory (`process_directory()` :533), `gc.collect()` is called after each file (`pytest_generator.py:565`) to reclaim memory between generations. The same cleanup runs at the end of `main()` (`pytest_generator.py:696`).

## Output File Naming

`TestWriter` (`pytest_generator.py:438`) derives the output filename by prepending `test_` to the source filename stem. Output directory defaults to `./generated_tests/` (configurable). Files are written UTF-8.

## Fine-tuning Data Format

Training data in `finetuning/data/train.csv` and `test.csv` uses columns that map to `(input_function_code, expected_test_output)`. The teacher model (Deepseek-v3.1, 671B) generated gold-standard test examples; the student (Qwen3-8B) was distilled with LoRA via the Distil Labs platform. Config at `finetuning/data/config.yaml`.

## Extension Points

- **Different model**: change `MODEL_REPO` / `MODEL_FILE` in `Config` or YAML — `ModelManager` is model-agnostic as long as it is a GGUF file
- **Custom prompt**: override `PROMPT_TEMPLATE` in YAML under `prompt.template`
- **Thread tuning**: set `hardware.n_threads` in YAML instead of relying on auto-detection
- **Output location**: `-o` flag or `output.default_dir` in YAML
