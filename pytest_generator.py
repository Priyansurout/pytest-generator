#!/usr/bin/env python3
"""
pytest-generator - CLI tool to generate pytest tests using local AI (GGUF)
CPU-Only Version with Config YAML Support
"""

import argparse
import ast
import os
import re
import sys
import time
import psutil
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# DEFAULT CONFIGURATION
@dataclass
class Config:
    """Default application configuration"""
    MODEL_REPO: str = "Priyansu19/pytest-8b-GGUF"
    MODEL_FILE: str = "pytest-8b-q4_k_m.gguf"
    N_CTX: int = 2048
    MAX_TOKENS: int = 3072
    TEMPERATURE: float = 0.05
    TOP_P: float = 1
    REPEAT_PENALTY: float = 1.0
    BATCH_SIZE: int = 128  
    N_THREADS: int = -1  # -1 = auto-detect (all cores - 2)
    DEFAULT_OUTPUT_DIR: str = "./generated_tests/"
    
    # Prompt template
    PROMPT_TEMPLATE: str = """<task>
You are a problem solving model working
Generate pytest unit test cases from function signatures and docstrings.
...
<examples>
# Example 1: Simple function WITHOUT dependencies - NO MOCKS
Input:
def add(a: int, b: int) -> int:
    \"\"\"Add numbers. Raises: ValueError if negative\"\"\"

Output:
import pytest
from module import add

class ValueError(Exception): pass

@pytest.mark.parametrize("a,b,expected", [(2,3,5), (0,0,0), (-1,1,0)])
def test_add(a, b, expected):
    result = add(a, b)
    assert result == expected

def test_add_negative_error():
    with pytest.raises(ValueError):
        add(-5, -3)

---

# Example 2: Function WITH dependencies - USE MOCKS
Input:
def save_user(name: str, email: str) -> dict:
    \"\"\"Save user. Raises: DatabaseError\"\"\"
    # Dependencies: database.insert(), email_service.send()

Output:
import pytest
from unittest.mock import Mock, patch
from module import save_user

class DatabaseError(Exception): pass

@pytest.mark.asyncio
@patch('module.email_service')
@patch('module.database')
def test_save_user(mock_db, mock_email):
    mock_db.insert.return_value = {"id": 1, "name": "Alice"}
    mock_email.send.return_value = True
    
    result = save_user("Alice", "alice@test.com")
    
    assert result["id"] == 1
    mock_db.insert.assert_called_once_with("Alice", "alice@test.com")
    mock_email.send.assert_called_once()

@patch('module.email_service')
@patch('module.database')
def test_save_user_db_error(mock_db, mock_email):
    mock_db.insert.side_effect = DatabaseError("Insert failed")
    
    with pytest.raises(DatabaseError):
        save_user("Alice", "alice@test.com")
    
    mock_email.send.assert_not_called()

---

# Example 3: Async function WITHOUT dependencies - NO MOCKS
Input:
async def calculate_total(items: list[dict]) -> float:
    \"\"\"Calculate total price.\"\"\"

Output:
import pytest
from module import calculate_total

@pytest.mark.asyncio
@pytest.mark.parametrize("items,expected", [
    ([{"price": 10}, {"price": 20}], 30.0),
    ([], 0.0),
])
async def test_calculate_total(items, expected):
    result = await calculate_total(items)
    assert result == expected

---

# Example 4: Async WITH dependencies - USE MOCKS
Input:
async def fetch_user(id: int) -> dict:
    \"\"\"Fetch user. Raises: NotFoundError\"\"\"
    # Dependencies: cache.get(), db.query()

Output:
import pytest
from unittest.mock import AsyncMock, patch
from module import fetch_user

class NotFoundError(Exception): pass

@pytest.mark.asyncio
@patch('module.db')
@patch('module.cache')
async def test_fetch_user_from_cache(mock_cache, mock_db):
    mock_cache.get.return_value = {"id": 1, "name": "Alice"}
    
    result = await fetch_user(1)
    
    assert result["name"] == "Alice"
    mock_cache.get.assert_called_once_with(1)
    mock_db.query.assert_not_called()

@pytest.mark.asyncio
@patch('module.db')
@patch('module.cache')
async def test_fetch_user_cache_miss(mock_cache, mock_db):
    mock_cache.get.return_value = None
    mock_db.query.return_value = {"id": 2, "name": "Bob"}
    
    result = await fetch_user(2)
    
    assert result["name"] == "Bob"
    mock_cache.get.assert_called_once_with(2)
    mock_db.query.assert_called_once_with(2)

@pytest.mark.asyncio
@patch('module.db')
@patch('module.cache')
async def test_fetch_user_not_found(mock_cache, mock_db):
    mock_cache.get.return_value = None
    mock_db.query.return_value = None
    
    with pytest.raises(NotFoundError):
        await fetch_user(999)

</examples>

<rules>
CRITICAL - Read carefully:
1. If function has NO "# Dependencies:" comment → DO NOT use mocks, test logic directly
2. If function HAS "# Dependencies:" → USE @patch for each dependency listed
3. NEVER mock the function being tested (e.g., don't @patch('module.add') when testing add)
4. Import: pytest, Mock/AsyncMock (only if needed), patch (only if needed), function, exceptions
5. Use @pytest.mark.asyncio for async functions
6. Use Mock for sync dependencies, AsyncMock for async dependencies
7. Parametrize tests with 2-3 cases when possible
8. Test all exceptions listed in Raises section
9. Module name is 'module' (will be replaced later)
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
    
    def generate(self, code: str, stream: bool = True) -> Tuple[str, float]:
        """Generate test code for a function."""
        
        prompt = self.config.PROMPT_TEMPLATE.replace("{code}", code)
        
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
    """Extract functions from Python source code"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.functions = []
    
    def visit_FunctionDef(self, node):
        self._extract_function(node, is_async=False)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        self._extract_function(node, is_async=True)
        self.generic_visit(node)
    
    def _extract_function(self, node, is_async: bool):
        """Extract function info from AST node."""
        try:
            start_line = node.lineno - 1
            end_line = node.end_lineno
            source_lines = self.source_code.splitlines()[start_line:end_line]
            func_code = '\n'.join(source_lines)
            
            self.functions.append({
                'name': node.name,
                'code': func_code,
                'is_async': is_async,
                'line_start': start_line + 1,
                'line_end': end_line,
            })
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Could not extract {node.name}: {e}{Colors.ENDC}")


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
            lines.append(f"# Test {i}: {test['function_name']}")
            lines.append(test['code'])
            lines.append('')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path


# MAIN PROCESSING

def process_file(file_path: str, model_manager: ModelManager, output_dir: str, config: Config) -> Optional[str]:
    """Process a single Python file."""
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}📄 Processing: {file_path}{Colors.ENDC}\n")
    
    # Read source
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        print(f"{Colors.FAIL}❌ Failed to read {file_path}: {e}{Colors.ENDC}")
        return None
    
    # Extract functions
    try:
        tree = ast.parse(source_code)
        extractor = FunctionExtractor(source_code)
        extractor.visit(tree)
    except SyntaxError as e:
        print(f"{Colors.FAIL}❌ Syntax error: {e}{Colors.ENDC}")
        return None
    
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
        
        test_code, elapsed = model_manager.generate(func['code'], stream=True)
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


def process_directory(dir_path: str, model_manager: ModelManager, output_dir: str, config: Config):
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
        
        result = process_file(str(py_file), model_manager, output_dir, config)
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
        if target_path.is_file():
            process_file(str(target_path), model_manager, config.DEFAULT_OUTPUT_DIR, config)
        elif target_path.is_dir():
            process_directory(str(target_path), model_manager, config.DEFAULT_OUTPUT_DIR, config)
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