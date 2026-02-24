#!/usr/bin/env python3
"""
pytest-generator - CLI tool to generate pytest tests using local AI (GGUF)
CPU-Only Version with Config YAML Support
"""

import argparse
import ast
import importlib
import inspect
import os
import re
import sys
import time
import psutil
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
from dataclasses import dataclass

# DEFAULT CONFIGURATION
@dataclass
class Config:
    """Default application configuration"""
    MODEL_REPO: str = "Priyansu19/pytest-8b-GGUF"
    MODEL_FILE: str = "pytest-8b-q4_k_m.gguf"
    N_CTX: int = 4096
    MAX_TOKENS: int = 3072
    TEMPERATURE: float = 0.05
    TOP_P: float = 1
    REPEAT_PENALTY: float = 1.0
    BATCH_SIZE: int = 128  
    N_THREADS: int = -1  # -1 = auto-detect (all cores - 2)
    DEFAULT_OUTPUT_DIR: str = "./generated_tests/"
    
    # Prompt template
    PROMPT_TEMPLATE: str = """<task>
You are an expert Python test engineer. Generate complete, runnable pytest unit tests from the given function.

<examples>
# Example 1: Simple function WITHOUT dependencies — test logic directly, NO mocks
Input:
def add(a: int, b: int) -> int:
    \"\"\"Add two integers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b

    Raises:
        ValueError: If either argument is negative
    \"\"\"

Output:
import pytest
import importlib

module = importlib.import_module("module")
add = getattr(module, "add")


@pytest.mark.parametrize("a,b,expected", [(2, 3, 5), (0, 0, 0), (100, 1, 101)])
def test_add(a, b, expected):
    result = add(a, b)
    assert result == expected

def test_add_negative_raises():
    with pytest.raises(ValueError):
        add(-1, 5)

---

# Example 2: Function WITH dependencies — @pytest.fixture + MagicMock for each dep
Input:
def save_user(name: str, email: str, db, mailer) -> dict:
    \"\"\"Save user to database and send welcome email.

    Args:
        name: User display name
        email: User email address
        db: Database service
        mailer: Email service

    Returns:
        Created user dict with 'id'

    Raises:
        DatabaseError: If database insert fails

    # Dependencies: db.insert(name, email), mailer.send(to, subject)
    \"\"\"

Output:
import pytest
from unittest.mock import MagicMock
import importlib

module = importlib.import_module("module")
save_user = getattr(module, "save_user")
DatabaseError = getattr(module, "DatabaseError", Exception)


@pytest.fixture
def db():
    return MagicMock()

@pytest.fixture
def mailer():
    return MagicMock()


def test_save_user_success(db, mailer):
    db.insert.return_value = {"id": 1, "name": "Alice"}
    mailer.send.return_value = True

    result = save_user("Alice", "alice@test.com", db, mailer)

    assert result["id"] == 1
    db.insert.assert_called_once_with("Alice", "alice@test.com")
    mailer.send.assert_called_once()

def test_save_user_db_error(db, mailer):
    db.insert.side_effect = DatabaseError("Insert failed")

    with pytest.raises(DatabaseError):
        save_user("Alice", "alice@test.com", db, mailer)

    mailer.send.assert_not_called()

---

# Example 3: Async function WITHOUT dependencies — pytest.mark.asyncio, NO mocks
Input:
async def calculate_total(items: list) -> float:
    \"\"\"Sum the 'price' field of each item dict.

    Args:
        items: List of dicts each containing a 'price' key

    Returns:
        Total as a float; 0.0 for empty list
    \"\"\"

Output:
import pytest
import importlib

module = importlib.import_module("module")
calculate_total = getattr(module, "calculate_total")


@pytest.mark.asyncio
@pytest.mark.parametrize("items,expected", [
    ([{"price": 10}, {"price": 20}], 30.0),
    ([], 0.0),
    ([{"price": 5.5}], 5.5),
])
async def test_calculate_total(items, expected):
    result = await calculate_total(items)
    assert result == expected

---

# Example 4: Async function WITH dependencies — AsyncMock for awaited methods, pytest.mark.asyncio
Input:
async def fetch_user(user_id: int, cache, db) -> dict:
    \"\"\"Fetch user from cache, falling back to DB on miss.

    Args:
        user_id: ID of the user to fetch
        cache: Cache service
        db: Database service

    Returns:
        User dict

    Raises:
        NotFoundError: If user not in cache or DB

    # Dependencies: cache.get(key), db.query(id)
    \"\"\"

Output:
import pytest
from unittest.mock import MagicMock, AsyncMock
import importlib

module = importlib.import_module("module")
fetch_user = getattr(module, "fetch_user")
NotFoundError = getattr(module, "NotFoundError", Exception)


@pytest.fixture
def cache():
    mock = MagicMock()
    mock.get = AsyncMock()
    return mock

@pytest.fixture
def db():
    mock = MagicMock()
    mock.query = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_fetch_user_cache_hit(cache, db):
    cache.get.return_value = {"id": 1, "name": "Alice"}

    result = await fetch_user(1, cache, db)

    assert result["name"] == "Alice"
    cache.get.assert_called_once_with(1)
    db.query.assert_not_called()

@pytest.mark.asyncio
async def test_fetch_user_cache_miss(cache, db):
    cache.get.return_value = None
    db.query.return_value = {"id": 2, "name": "Bob"}

    result = await fetch_user(2, cache, db)

    assert result["name"] == "Bob"
    db.query.assert_called_once_with(2)

@pytest.mark.asyncio
async def test_fetch_user_not_found(cache, db):
    cache.get.return_value = None
    db.query.return_value = None

    with pytest.raises(NotFoundError):
        await fetch_user(999, cache, db)

</examples>

<rules>
CRITICAL — follow exactly:
1. IMPORTS: Always `import importlib` at the top, then `module = importlib.import_module("module")` and `func = getattr(module, "func_name")`
2. EXCEPTIONS: `ErrClass = getattr(module, "ErrClass", Exception)` for every exception in Raises — never define them locally
3. NO DEPS (no `# Dependencies:`) → test logic directly, no fixtures, no mocks
4. HAS DEPS (`# Dependencies:` present) → one `@pytest.fixture` per dependency returning `MagicMock()`; pass as test parameters
5. Use EXACTLY the method names in `# Dependencies:` — never invent or rename them
6. Async methods (called with `await` inside an `async def`) → set as `AsyncMock()` on the fixture: `mock.method = AsyncMock()`
7. Async functions → `@pytest.mark.asyncio` and `async def` on every test
8. Parametrize simple functions with 2-3 meaningful cases covering happy path and edge cases
9. Test every exception listed in the Raises section
10. Assert mock calls: `assert_called_once_with(...)` for single calls, `assert_not_called()` to verify skipped paths
</rules>
</task>

<question>
{code}
</question>
"""


    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        import yaml
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'model' in data:
            config.MODEL_REPO = data['model'].get('repo', config.MODEL_REPO)
            config.MODEL_FILE = data['model'].get('file', config.MODEL_FILE)
        
        if 'generation' in data:
            config.N_CTX = data['generation'].get('n_ctx', config.N_CTX)
            config.MAX_TOKENS = data['generation'].get('max_tokens', config.MAX_TOKENS)
            config.TEMPERATURE = data['generation'].get('temperature', config.TEMPERATURE)
            config.TOP_P = data['generation'].get('top_p', config.TOP_P)
            config.REPEAT_PENALTY = data['generation'].get('repeat_penalty', config.REPEAT_PENALTY)
            config.BATCH_SIZE = data['generation'].get('batch_size', config.BATCH_SIZE)
        
        if 'hardware' in data:
            config.N_THREADS = data['hardware'].get('n_threads', config.N_THREADS)
        
        if 'output' in data:
            config.DEFAULT_OUTPUT_DIR = data['output'].get('default_dir', config.DEFAULT_OUTPUT_DIR)
        
        if 'prompt' in data:
            config.PROMPT_TEMPLATE = data['prompt'].get('template', config.PROMPT_TEMPLATE)
        
        return config



# COLORS FOR TERMINAL OUTPUT

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# HARDWARE DETECTION 

class HardwareDetector:
    """Detect and configure hardware settings"""
    
    @staticmethod
    def get_cpu_info() -> Tuple[int, int]:
        """
        Get CPU information.
        Returns: (total_cores, optimal_threads)
        """
        total_cores = psutil.cpu_count(logical=True)
        # Reserve 2 cores for system, use rest for inference
        optimal_threads = max(1, total_cores - 2)
        return total_cores, optimal_threads
    
    @staticmethod
    def print_config(n_threads: int, total_cores: int):
        """Print hardware configuration."""
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.CYAN}🔧 HARDWARE CONFIGURATION{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"Total CPU Cores: {total_cores}")
        print(f"Using Threads: {n_threads} (reserved 2 for system)")
        print(f"Mode: {Colors.BLUE}💻 CPU ONLY{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}\n")


# MODEL MANAGEMENT

class ModelManager:
    """Handle model downloading and loading"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_path = config.MODEL_FILE
        self.model = None
    
    def download_if_needed(self) -> str:
        """Download model from HuggingFace if not present."""
        if os.path.exists(self.model_path):
            print(f"{Colors.GREEN}✅ Model found: {self.model_path}{Colors.ENDC}")
            return self.model_path
        
        print(f"{Colors.BLUE}📥 Downloading model from HuggingFace...{Colors.ENDC}")
        print(f"   Repository: {self.config.MODEL_REPO}")
        print(f"   File: {self.config.MODEL_FILE}")
        print(f"   Size: ~5 GB (this may take a while...)\n")
        
        try:
            from huggingface_hub import hf_hub_download
            
            downloaded_path = hf_hub_download(
                repo_id=self.config.MODEL_REPO,
                filename=self.config.MODEL_FILE,
                local_dir=".",
                local_dir_use_symlinks=False
            )
            
            print(f"\n{Colors.GREEN}✅ Download complete: {downloaded_path}{Colors.ENDC}\n")
            self.model_path = downloaded_path
            return downloaded_path
            
        except ImportError:
            print(f"{Colors.FAIL}❌ ERROR: huggingface_hub not installed!{Colors.ENDC}")
            print("   Install with: pip install huggingface_hub")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.FAIL}❌ Download failed: {e}{Colors.ENDC}")
            sys.exit(1)
    
    def load(self, n_threads: int) -> 'ModelManager':
        """Load the GGUF model (CPU only)."""
        from llama_cpp import Llama
        
        print(f"{Colors.BLUE}Loading model...{Colors.ENDC}")
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.config.N_CTX,
                n_threads=n_threads,
                n_gpu_layers=0,  
                n_batch=self.config.BATCH_SIZE,
                verbose=False,
                seed=-1,
            )
            
            print(f"{Colors.GREEN}✅ Model loaded{Colors.ENDC}")
            print(f"{Colors.BLUE}💻 Using CPU ({n_threads} threads){Colors.ENDC}\n")
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to load model: {e}{Colors.ENDC}")
            sys.exit(1)
        
        return self
    
    def generate(self, code: str, stream: bool = True, dependencies: str = "") -> Tuple[str, float]:
        """Generate test code for a function."""

        effective_code = (code + "\n" + dependencies) if dependencies else code
        prompt = self.config.PROMPT_TEMPLATE.replace("{code}", effective_code)
        
        # Format with chat template (pre-fill thinking)
        formatted_prompt = (
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n\n</think>\n\n"
        )
        
        start = time.time()
        full_response = ""
        token_count = 0
        
        if stream:
            print(f"{Colors.BLUE}🚀 Generating...{Colors.ENDC} ", end="", flush=True)
            
            stream_output = self.model(
                formatted_prompt,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                repeat_penalty=self.config.REPEAT_PENALTY,
                stop=["<|im_end|>"],
                stream=True
            )
            
            for chunk in stream_output:
                text = chunk["choices"][0]["text"]
                full_response += text
                token_count += 1
                print(text, end="", flush=True)
            
            print()
        else:
            output = self.model(
                formatted_prompt,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                repeat_penalty=self.config.REPEAT_PENALTY,
                stop=["<|im_end|>"],
                echo=False
            )
            full_response = output["choices"][0]["text"]
            token_count = len(full_response.split())
        
        gen_time = time.time() - start
        
        # Clean output
        result = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
        result = re.sub(r'```python\n?', '', result)
        result = re.sub(r'```', '', result)
        result = result.strip()
        
        # Print stats
        tokens_per_sec = token_count / gen_time if gen_time > 0 else 0
        print(f"\n{Colors.CYAN}⏱️  {gen_time:.2f}s | {token_count} tokens | {tokens_per_sec:.1f} tok/s{Colors.ENDC}\n")
        
        return result, gen_time


# FUNCTION EXTRACTION

class FunctionExtractor(ast.NodeVisitor):
    """Extract module-level functions from Python source code (class methods are skipped)."""

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.functions = []
        self._class_depth = 0  # track nesting inside class bodies

    def visit_ClassDef(self, node):
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1

    def visit_FunctionDef(self, node):
        if self._class_depth == 0:
            self._extract_function(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if self._class_depth == 0:
            self._extract_function(node, is_async=True)
        self.generic_visit(node)
    
    def _extract_function(self, node, is_async: bool):
        """Extract function info: signature + docstring only (no implementation body).

        The fine-tuned model was trained on signature+docstring inputs — sending the
        full body would be out-of-distribution and degrade generation quality.
        _extract_attr_calls still walks the full AST node to detect method calls.
        """
        try:
            source_lines = self.source_code.splitlines()
            start = node.lineno - 1  # 0-indexed

            # Decide where to stop:
            #   - has docstring  → include through the closing """ line
            #   - no docstring   → include only the def signature lines (up to first body stmt)
            has_docstring = (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            if has_docstring:
                # end_lineno is 1-indexed inclusive → works as 0-indexed exclusive slice end
                end = node.body[0].end_lineno
            else:
                # body[0].lineno - 1 excludes the first body statement.
                # max() guards single-line functions where body starts on the def line.
                end = max(node.body[0].lineno - 1, node.lineno) if node.body else node.lineno

            func_code = '\n'.join(source_lines[start:end])
            
            self.functions.append({
                'name': node.name,
                'code': func_code,
                'is_async': is_async,
                'line_start': start + 1,
                'line_end': node.end_lineno,
                'typed_params': self._extract_typed_params(node),
                'attr_calls': self._extract_attr_calls(node),
            })
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Could not extract {node.name}: {e}{Colors.ENDC}")

    @staticmethod
    def _extract_typed_params(node) -> Dict[str, str]:
        """Return {param_name: class_name} for non-builtin type-annotated args."""
        _BUILTINS = {
            'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'None', 'Optional', 'List', 'Dict', 'Any', 'Union', 'Tuple',
            'Sequence', 'Iterable', 'object',
        }
        result: Dict[str, str] = {}
        for arg in node.args.args:
            if arg.arg in ('self', 'cls'):
                continue
            if arg.annotation is None:
                continue
            try:
                type_str = ast.unparse(arg.annotation)
            except Exception:
                continue
            # Strip Optional[X] -> X
            if type_str.startswith('Optional[') and type_str.endswith(']'):
                type_str = type_str[len('Optional['):-1]
            if type_str.isidentifier() and type_str not in _BUILTINS:
                result[arg.arg] = type_str
        return result

    @staticmethod
    def _extract_attr_calls(node) -> Dict[str, List[str]]:
        """Return {param_name: [method_names]} from obj.method(...) calls in body."""
        param_names = {
            arg.arg for arg in node.args.args
            if arg.arg not in ('self', 'cls')
        }
        calls: Dict[str, List[str]] = {}
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name):
                continue
            obj_name = func.value.id
            if obj_name not in param_names:
                continue
            method_name = func.attr
            if obj_name not in calls:
                calls[obj_name] = []
            if method_name not in calls[obj_name]:
                calls[obj_name].append(method_name)
        return calls


# IMPORT EXTRACTION AND RUNTIME INSPECTION

class ImportExtractor(ast.NodeVisitor):
    """Parse a source file's import statements and build {ClassName: module_path} map."""

    def __init__(self):
        self.import_map: Dict[str, str] = {}  # {name: "module.path"}

    def visit_ImportFrom(self, node):
        # from module import Class [as alias]  →  {"Class": "module"}
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self.import_map[name] = module
        self.generic_visit(node)

    def visit_Import(self, node):
        # import module [as alias]  →  {"alias": "module"}
        for alias in node.names:
            name = alias.asname or alias.name
            self.import_map[name] = alias.name
        self.generic_visit(node)

    @classmethod
    def from_source(cls, source: str) -> Dict[str, str]:
        try:
            tree = ast.parse(source)
            extractor = cls()
            extractor.visit(tree)
            return extractor.import_map
        except SyntaxError:
            return {}


class RuntimeInspector:
    """Use importlib + inspect to get real method signatures from any importable class."""

    @staticmethod
    def get_class_methods(class_name: str, module_path: str) -> List[Dict]:
        """
        Import module_path at runtime and return public methods of class_name
        as [{name, args, is_async}]. Returns [] on any failure (missing deps,
        import side effects, class not found, etc.).
        """
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            if cls is None or not inspect.isclass(cls):
                return []
            methods = []
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if name.startswith('_'):
                    continue
                try:
                    sig = inspect.signature(method)
                    args = [p for p in sig.parameters if p != 'self']
                except (ValueError, TypeError):
                    args = []
                methods.append({
                    'name': name,
                    'args': args,
                    'is_async': inspect.iscoroutinefunction(method),
                })
            return methods
        except Exception:
            return []


# CODEBASE INDEXING

class CodebaseIndexer:
    """Scan a project directory and build a class/method index for dependency resolution."""

    def __init__(self, scan_root: str):
        self.scan_root = Path(scan_root).resolve()
        self.index: Dict[str, Any] = {}

    def build(self) -> 'CodebaseIndexer':
        """Scan files and build the index. Returns self for chaining."""
        files = self._collect_files()
        print(f"{Colors.CYAN}🔍 Indexing {len(files)} file(s) from {self.scan_root}{Colors.ENDC}")
        for path in files:
            self._index_file(path)
        class_count = len(self.index)
        print(f"{Colors.GREEN}✅ Indexed {class_count} class(es){Colors.ENDC}\n")
        return self

    def _collect_files(self) -> List[Path]:
        _SKIP = ('test_', '__pycache__', 'venv', '.venv', '.git')
        result = []
        for p in self.scan_root.rglob('*.py'):
            rel = str(p)
            if any(skip in rel for skip in _SKIP):
                continue
            result.append(p)
        return result

    def _index_file(self, path: Path):
        try:
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except Exception:
            return
        rel_path = str(path.relative_to(self.scan_root))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = []
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # Skip dunder methods other than __init__
                if item.name.startswith('__') and item.name.endswith('__') and item.name != '__init__':
                    continue
                args = [a.arg for a in item.args.args if a.arg != 'self']
                returns = ''
                if item.returns is not None:
                    try:
                        returns = ast.unparse(item.returns)
                    except Exception:
                        returns = ''
                methods.append({
                    'name': item.name,
                    'args': args,
                    'returns': returns,
                    'is_async': isinstance(item, ast.AsyncFunctionDef),
                })
            self.index[node.name] = {
                'file': rel_path,
                'methods': methods,
            }

    def save(self, json_path: str):
        """Dump the index to a JSON file."""
        data = {
            'scan_root': str(self.scan_root),
            'classes': self.index,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"{Colors.GREEN}✅ Index saved to: {json_path}{Colors.ENDC}")

    @classmethod
    def load(cls, json_path: str) -> 'CodebaseIndexer':
        """Load a previously saved index JSON."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        instance = cls(data.get('scan_root', '.'))
        instance.index = data.get('classes', {})
        print(f"{Colors.GREEN}✅ Loaded index: {len(instance.index)} class(es) from {json_path}{Colors.ENDC}\n")
        return instance


# TEST FILE WRITER

class TestWriter:
    """Write generated tests to file"""
    
    @staticmethod
    def write(tests: List[Dict], output_path: str, source_file: str):
        """Combine all tests and write to file."""

        source_name = Path(source_file).stem

        lines = [
            '"""',
            f'Generated tests for {os.path.basename(source_file)}',
            f'Created by pytest-generator',
            '"""',
        ]

        for i, test in enumerate(tests, 1):
            code = TestWriter._replace_module_name(test['code'], source_name)
            lines.append(f"# Test {i}: {test['function_name']}")
            lines.append(code)
            lines.append('')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return output_path

    @staticmethod
    def _replace_module_name(code: str, module_name: str) -> str:
        """Replace the 'module' placeholder with the real module name in generated code."""
        # importlib pattern: importlib.import_module("module")
        code = code.replace('importlib.import_module("module")', f'importlib.import_module("{module_name}")')
        code = code.replace("importlib.import_module('module')", f"importlib.import_module('{module_name}')")
        # static import pattern: from module import ... / import module
        code = code.replace('from module import', f'from {module_name} import')
        code = re.sub(r'\bimport module\b', f'import {module_name}', code)
        # @patch strings: @patch('module.X') / @patch("module.X")
        code = code.replace("@patch('module.", f"@patch('{module_name}.")
        code = code.replace('@patch("module.', f'@patch("{module_name}.')
        return code


# DEPENDENCY RESOLUTION HELPERS

def find_project_root(start_path: str) -> str:
    """Walk up from start_path looking for project root markers."""
    _MARKERS = ('pyproject.toml', 'setup.py', 'setup.cfg', '.git', 'requirements.txt')
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent
    for _ in range(6):
        for marker in _MARKERS:
            if (current / marker).exists():
                return str(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback to the directory of start_path
    p = Path(start_path).resolve()
    return str(p.parent if p.is_file() else p)


def resolve_dependencies(func_info: Dict, index: Dict, import_map: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Given a function's typed_params and attr_calls, resolve each param's class methods.

    Resolution order:
    1. RuntimeInspector — import the module and use inspect.signature() (covers pip packages)
    2. CodebaseIndexer  — fall back to the pre-built AST index (local project files)
    """
    typed_params: Dict[str, str] = func_info.get('typed_params', {})
    attr_calls: Dict[str, List[str]] = func_info.get('attr_calls', {})
    deps: Dict[str, Any] = {}
    for param_name, class_name in typed_params.items():
        methods = []
        source = None

        # 1. Try runtime inspection (handles pip-installed packages too)
        if import_map and class_name in import_map:
            module_path = import_map[class_name]
            runtime_methods = RuntimeInspector.get_class_methods(class_name, module_path)
            if runtime_methods:
                methods = runtime_methods
                source = "runtime"

        # 2. Fall back to AST-based codebase index (local project scan)
        if not methods and index and class_name in index:
            entry = index[class_name]
            methods = entry['methods']
            source = "index"

        if methods:
            deps[param_name] = {
                'class_name': class_name,
                'methods': methods,
                'called_methods': attr_calls.get(param_name, []),
                'source': source,
            }
    return deps


def format_dependency_block(deps: Dict[str, Any]) -> str:
    """Return a '# Dependencies: ...' comment line compatible with the fine-tuned model.

    When method signatures come from RuntimeInspector, argument names are included
    (e.g. 'db.get_order(order_id)').  Index-sourced methods keep the no-args format
    to preserve parity with the training data.
    """
    if not deps:
        return ""
    parts = []
    for param_name, info in deps.items():
        source = info.get('source', 'index')

        # Build name→args lookup for runtime-sourced methods
        args_map: Dict[str, List[str]] = {}
        if source == "runtime":
            for m in info['methods']:
                args_map[m['name']] = m.get('args', [])

        # Prefer methods actually called in the body; fall back to all public methods
        methods_to_list = info['called_methods'] if info['called_methods'] else [
            m['name'] for m in info['methods']
            if not (m['name'].startswith('__') and m['name'].endswith('__') and m['name'] != '__init__')
        ]

        for method in methods_to_list:
            if source == "runtime" and method in args_map:
                arg_str = ", ".join(args_map[method])
                parts.append(f"{param_name}.{method}({arg_str})")
            else:
                parts.append(f"{param_name}.{method}()")

    if not parts:
        return ""
    return "    # Dependencies: " + ", ".join(parts)


# MAIN PROCESSING

def process_file(file_path: str, model_manager: ModelManager, output_dir: str, config: Config,
                 index=None, stream: bool = True) -> Optional[str]:
    """Process a single Python file."""
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}📄 Processing: {file_path}{Colors.ENDC}\n")
    
    # Read source
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        print(f"{Colors.FAIL}❌ Failed to read {file_path}: {e}{Colors.ENDC}")
        return None
    
    # Extract functions and imports
    try:
        tree = ast.parse(source_code)
        extractor = FunctionExtractor(source_code)
        extractor.visit(tree)
    except SyntaxError as e:
        print(f"{Colors.FAIL}❌ Syntax error: {e}{Colors.ENDC}")
        return None

    import_map = ImportExtractor.from_source(source_code)
    functions = extractor.functions
    
    if not functions:
        print(f"{Colors.WARNING}⚠️  No functions found{Colors.ENDC}")
        return None
    
    print(f"{Colors.GREEN}✅ Found {len(functions)} function(s){Colors.ENDC}\n")
    
    for i, func in enumerate(functions, 1):
        async_marker = "async " if func['is_async'] else ""
        print(f"  {i}. {async_marker}{func['name']}()")
    
    print()
    
    # Generate tests
    generated_tests = []
    total_time = 0
    
    for i, func in enumerate(functions, 1):
        print(f"{Colors.BLUE}[{i}/{len(functions)}] {func['name']}(){Colors.ENDC}")
        
        dep_block = ""
        if index is not None:
            deps = resolve_dependencies(func, index, import_map=import_map)
            if deps:
                dep_block = format_dependency_block(deps)
                print(f"{Colors.CYAN}  ↳ Deps: {', '.join(deps)}{Colors.ENDC}")

        test_code, elapsed = model_manager.generate(func['code'], stream=stream, dependencies=dep_block)
        total_time += elapsed
        
        generated_tests.append({
            'function_name': func['name'],
            'code': test_code,
            'time': elapsed
        })
    
    # Write output
    source_name = Path(file_path).stem
    output_name = f"test_{source_name}.py"
    output_path = os.path.join(output_dir, output_name)
    
    TestWriter.write(generated_tests, output_path, file_path)
    
    avg_time = total_time / len(functions)
    print(f"{Colors.GREEN}{Colors.BOLD}✅ Created: {output_path}{Colors.ENDC}")
    print(f"{Colors.CYAN}   Total: {total_time:.1f}s | Average: {avg_time:.1f}s per function{Colors.ENDC}")
    
    return output_path


def process_directory(dir_path: str, model_manager: ModelManager, output_dir: str, config: Config,
                      index=None, stream: bool = True):
    """Process all Python files in directory."""
    
    py_files = list(Path(dir_path).rglob("*.py"))
    py_files = [
        f for f in py_files 
        if not f.name.startswith('test_') 
        and f.name != '__init__.py'
        and 'venv' not in str(f)
        and '.venv' not in str(f)
        and '__pycache__' not in str(f)
    ]
    
    if not py_files:
        print(f"{Colors.WARNING}No Python files found in {dir_path}{Colors.ENDC}")
        return
    
    print(f"{Colors.CYAN}Found {len(py_files)} Python file(s){Colors.ENDC}\n")
    
    success_count = 0
    
    for i, py_file in enumerate(py_files, 1):
        print(f"\n{Colors.BOLD}{'='*60}")
        print(f"File {i}/{len(py_files)}")
        print(f"{'='*60}{Colors.ENDC}")
        
        result = process_file(str(py_file), model_manager, output_dir, config, index=index, stream=stream)
        if result:
            success_count += 1
        
        # Cleanup memory between files
        if i < len(py_files):
            gc.collect()
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*60}")
    print(f"✨ Complete: {success_count}/{len(py_files)} files processed")
    print(f"{'='*60}{Colors.ENDC}\n")


# CLI MAIN

def main():
    parser = argparse.ArgumentParser(
        description="🧪 pytest-generator - Generate pytest tests using local AI (CPU-Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s calculator.py                    # Use default config (auto-detect cores)
  %(prog)s ./src/ -o ./tests/               # Process directory
  %(prog)s app.py myconfig.yaml             # Use custom config
  %(prog)s app.py --no-stream               # Disable streaming

Hardware:
  By default uses ALL CPU cores minus 2 (reserved for system)
  Example: 8 cores → uses 6 threads
  Example: 4 cores → uses 2 threads

Custom Config YAML:
  Create a myconfig.yaml file to override defaults:
  
  hardware:
    n_threads: 4          # Override auto-detection (use 4 threads)
  
  generation:
    temperature: 0.1      # More creative output
    max_tokens: 2000      # Shorter tests
        """
    )
    
    parser.add_argument(
        'target',
        help='Python file or directory to generate tests for'
    )
    
    parser.add_argument(
        'config',
        nargs='?',  # Optional positional argument
        default=None,
        help='Path to custom config YAML file (optional)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory (overrides config file)'
    )
    
    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Disable streaming output'
    )

    parser.add_argument(
        '--scan-root',
        metavar='DIR',
        default=None,
        help='Root directory to scan for codebase index (default: auto-detect from target)'
    )

    parser.add_argument(
        '--save-index',
        metavar='FILE',
        default=None,
        help='Save class index JSON to this path after scanning'
    )

    parser.add_argument(
        '--load-index',
        metavar='FILE',
        default=None,
        help='Load previously saved index, skip re-scanning'
    )

    parser.add_argument(
        '--no-index',
        action='store_true',
        help='Disable dependency resolution entirely'
    )

    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        if not os.path.exists(args.config):
            print(f"{Colors.FAIL}❌ Config file not found: {args.config}{Colors.ENDC}")
            sys.exit(1)
        
        try:
            config = Config.from_yaml(args.config)
            print(f"{Colors.GREEN}✅ Loaded config from: {args.config}{Colors.ENDC}\n")
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to load config: {e}{Colors.ENDC}")
            sys.exit(1)
    else:
        config = Config()
        print(f"{Colors.BLUE}ℹ️  Using default configuration (auto-detect CPU cores){Colors.ENDC}\n")
    
    # Override output dir if specified via CLI
    if args.output:
        config.DEFAULT_OUTPUT_DIR = args.output
    
    # Print header
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print("🧪 pytest-generator (CPU-Only)")
    print(f"{'='*60}{Colors.ENDC}\n")
    
    # Hardware detection (CPU only)
    total_cores, optimal_threads = HardwareDetector.get_cpu_info()
    
    # Use config threads if specified, otherwise auto-detect
    if config.N_THREADS > 0:
        n_threads = config.N_THREADS
        print(f"{Colors.BLUE}ℹ️  Using {n_threads} threads from config file{Colors.ENDC}")
    else:
        n_threads = optimal_threads
        print(f"{Colors.BLUE}ℹ️  Auto-detected: {total_cores} cores, using {n_threads} threads{Colors.ENDC}")
    
    HardwareDetector.print_config(n_threads, total_cores)
    
    # Setup output directory
    os.makedirs(config.DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    # Download and load model
    model_manager = ModelManager(config)
    model_manager.download_if_needed()
    model_manager.load(n_threads)
    
    # Process target
    target_path = Path(args.target)
    
    if not target_path.exists():
        print(f"{Colors.FAIL}❌ Target not found: {args.target}{Colors.ENDC}")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        # Build or load codebase index
        codebase_index = None
        if not args.no_index:
            if args.load_index:
                codebase_index = CodebaseIndexer.load(args.load_index).index
            else:
                scan_root = args.scan_root or find_project_root(args.target)
                indexer = CodebaseIndexer(scan_root).build()
                codebase_index = indexer.index
                if args.save_index:
                    indexer.save(args.save_index)

        if target_path.is_file():
            process_file(str(target_path), model_manager, config.DEFAULT_OUTPUT_DIR, config,
                         index=codebase_index, stream=not args.no_stream)
        elif target_path.is_dir():
            process_directory(str(target_path), model_manager, config.DEFAULT_OUTPUT_DIR, config,
                              index=codebase_index, stream=not args.no_stream)
        else:
            print(f"{Colors.FAIL}❌ Invalid target: {args.target}{Colors.ENDC}")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}⚠️  Interrupted by user{Colors.ENDC}")
    finally:
        # Cleanup
        del model_manager.model
        gc.collect()
    
    total_time = time.time() - start_time
    print(f"{Colors.GREEN}✨ All done! Total time: {total_time:.1f}s{Colors.ENDC}")
    print(f"{Colors.CYAN}Tests saved to: {config.DEFAULT_OUTPUT_DIR}{Colors.ENDC}\n")


if __name__ == "__main__":
    main()