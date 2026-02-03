"""
TestGen - Optimized Pytest Test Case Generator CLI
Generates pytest tests from Python functions using local AI model

Performance Optimizations:
- Smart GPU/CPU detection
- Dynamic token allocation
- Optimized thread usage
- Aggressive stop sequences
- Efficient batch processing
"""

import argparse
import ast
import os
import re
import sys
import time
import platform
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional
from llama_cpp import Llama


class Colors:
    """Terminal colors for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class HardwareDetector:
    """Detect and optimize for available hardware"""
    
    @staticmethod
    def get_cpu_threads():
        """Get optimal CPU thread count"""
        try:
            cpu_count = multiprocessing.cpu_count()
            # Use all cores except 2 (leave for system)
            optimal = max(4, cpu_count - 2)
            # Cap at 16 (diminishing returns beyond this)
            return min(optimal, 16)
        except:
            return 8  # Safe default
    
    @staticmethod
    def has_nvidia_gpu():
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                timeout=2,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def is_apple_silicon():
        """Check if running on Apple Silicon (M1/M2/M3/M4)"""
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    
    @staticmethod
    def get_gpu_config():
        """
        Determine optimal GPU configuration
        Returns: (n_gpu_layers, device_type)
        """
        if HardwareDetector.is_apple_silicon():
            return 35, "Metal (Apple Silicon)"
        elif HardwareDetector.has_nvidia_gpu():
            return 35, "CUDA (NVIDIA)"
        else:
            return 0, "CPU only"
    
    @staticmethod
    def get_ram_gb():
        """Get total system RAM in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 8  # Conservative default
    
    @staticmethod
    def print_system_info():
        """Print detected hardware configuration"""
        cpu_threads = HardwareDetector.get_cpu_threads()
        gpu_layers, device = HardwareDetector.get_gpu_config()
        ram_gb = HardwareDetector.get_ram_gb()
        
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.CYAN}Hardware Configuration:{Colors.ENDC}")
        print(f"  CPU: {multiprocessing.cpu_count()} cores (using {cpu_threads} threads)")
        print(f"  RAM: {ram_gb:.1f} GB")
        print(f"  Accelerator: {device}")
        
        # Performance estimate
        if gpu_layers > 0:
            est_time = "15-30s per function"
            rating = "⚡ FAST"
        else:
            est_time = "60-120s per function"
            rating = "⚠️  SLOW (consider GPU or smaller model)"
        
        print(f"  Expected Speed: {est_time} {rating}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}\n")


class Config:
    """Optimized configuration settings"""
    MODEL_REPO = "Priyansu19/pytest-generator-v4-GGUF"
    MODEL_FILE_8B = "pytest-v4-q4_k_m.gguf"
    MODEL_FILE_FP16 = "pytest-v4-f16.gguf"
    
    # Context and generation settings
    N_CTX = 2048          # Context window
    MIN_TOKENS = 300      # Minimum tokens per test
    MAX_TOKENS = 1200     # Maximum tokens per test
    TEMPERATURE = 0.2     # Lower = faster + more deterministic
    BATCH_SIZE = 512      # Larger batches = faster processing


class TokenEstimator:
    """Estimate optimal token count based on function complexity"""
    
    @staticmethod
    def estimate(function_code: str) -> int:
        """
        Estimate tokens needed based on function complexity
        
        Simple function (5 lines, 2 params) → ~400 tokens
        Complex function (20 lines, 5 params, async, errors) → ~1000 tokens
        """
        lines = len(function_code.split('\n'))
        params = function_code.count(',') + 1
        
        # Complexity indicators
        is_async = 'async def' in function_code
        has_dependencies = '# Dependencies:' in function_code
        raises_count = function_code.lower().count('raises:')
        has_typing = 'Union' in function_code or 'Optional' in function_code
        
        # Base allocation
        tokens = Config.MIN_TOKENS
        
        # Add based on complexity
        tokens += lines * 8                    # ~8 tokens per line of code
        tokens += params * 25                  # ~25 tokens per parameter
        tokens += 100 if is_async else 0       # Async tests need more
        tokens += raises_count * 80            # Each exception needs test case
        tokens += 50 if has_dependencies else 0
        tokens += 30 if has_typing else 0
        
        # Clamp to range
        return min(max(tokens, Config.MIN_TOKENS), Config.MAX_TOKENS)


class FunctionExtractor(ast.NodeVisitor):
    """Extracts function definitions from Python source code"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.functions = []
    
    def visit_FunctionDef(self, node):
        """Extract regular function"""
        func_info = self._extract_function(node)
        if func_info:
            self.functions.append(func_info)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Extract async function"""
        func_info = self._extract_function(node, is_async=True)
        if func_info:
            self.functions.append(func_info)
        self.generic_visit(node)
    
    def _extract_function(self, node, is_async=False) -> Optional[Dict]:
        """Extract function info from AST node"""
        try:
            # Get source lines
            start_line = node.lineno - 1
            end_line = node.end_lineno
            source_lines = self.source_code.splitlines()[start_line:end_line]
            func_code = '\n'.join(source_lines)
            
            # Get docstring
            docstring = ast.get_docstring(node) or ""
            
            # Detect dependencies
            has_deps = "# Dependencies:" in func_code
            
            # Count complexity
            param_count = len(node.args.args)
            
            return {
                'name': node.name,
                'code': func_code,
                'docstring': docstring,
                'is_async': is_async,
                'has_deps': has_deps,
                'param_count': param_count,
                'line_start': start_line + 1,
                'line_end': end_line,
                'lines': len(source_lines)
            }
        except Exception as e:
            print(f"{Colors.WARNING}Warning: Could not extract function {node.name}: {e}{Colors.ENDC}")
            return None


class TestGenerator:
    """Generates pytest tests using optimized local GGUF model"""
    
    def __init__(self, model_path: str, verbose: bool = False):
        self.model = None
        self.model_path = model_path
        self.verbose = verbose
        self._load_model()
    
    def _load_model(self):
        """Load the GGUF model with optimal settings"""
        print(f"{Colors.BLUE}Loading model from {os.path.basename(self.model_path)}...{Colors.ENDC}")
        
        # Get hardware config
        n_threads = HardwareDetector.get_cpu_threads()
        n_gpu_layers, device = HardwareDetector.get_gpu_config()
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=Config.N_CTX,
                n_batch=Config.BATCH_SIZE,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=self.verbose
            )
            print(f"{Colors.GREEN}✅ Model loaded ({device}){Colors.ENDC}\n")
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to load model: {e}{Colors.ENDC}")
            sys.exit(1)
    
    def generate(self, function_info: Dict) -> str:
        """
        Generate pytest tests for a single function
        Optimized with dynamic token allocation and aggressive stopping
        """
        function_code = function_info['code']
        function_name = function_info['name']
        
        # Estimate optimal token count
        max_tokens = TokenEstimator.estimate(function_code)
        
        # Build optimized prompt
        prompt = self._build_prompt(function_code)
        
        # Aggressive stop sequences to prevent over-generation
        stop_sequences = [
            "```",                    # End code block
            "<|im_end|>",            # Chat template end
            "\n\n\n\n",              # Multiple blank lines
            "# " + "="*50,           # Separator
            "\ndef test_test_",      # Duplicate test prefix
            "\nclass ",              # New class definition
            "\n\nHere's",            # Explanation starter
            "\n\nThis test",         # Explanation starter
            "\n\nNote:",             # Note starter
            "\n\n# Explanation:",   # Explanation section
            "# TODO:",               # TODO comment
            "\n\nif __name__",       # Main block
        ]
        
        if self.verbose:
            print(f"      Max tokens: {max_tokens}")
        
        try:
            start_time = time.time()
            
            output = self.model.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=Config.TEMPERATURE,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=stop_sequences,
                echo=False
            )
            
            elapsed = time.time() - start_time
            
            generated = output["choices"][0]["text"]
            tokens_generated = output["usage"]["completion_tokens"]
            
            if self.verbose:
                print(f"      Generated: {tokens_generated} tokens in {elapsed:.1f}s")
            
            # Clean and validate
            code = self._clean_code(generated, function_name)
            
            return code, elapsed
            
        except Exception as e:
            print(f"{Colors.WARNING}Warning: Generation failed for {function_name}: {e}{Colors.ENDC}")
            return f"# TODO: Generate tests for {function_name}\n", 0
    
    def _build_prompt(self, function_code: str) -> str:
        """Build optimized prompt for model"""
        # Minimal, focused prompt
        return f"""<|im_start|>system
You are a pytest expert. Generate complete test code only. No explanations.<|im_end|>
<|im_start|>user
Generate pytest tests for this function:

{function_code}<|im_end|>
<|im_start|>assistant
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch"""
    
    def _clean_code(self, code: str, function_name: str) -> str:
        """Clean generated code aggressively"""
        
        # Extract from code block if present
        if "```python" in code:
            code = code.split("```python")[-1].split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            code = parts[0] if len(parts) == 2 else parts[-2]
        
        # Remove non-ASCII characters (Chinese, etc.)
        code = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', code)
        code = re.sub(r'[\u3040-\u309f\u30a0-\u30ff]+', '', code)
        
        # Cut at explanation markers
        explanation_markers = [
            "\n\nHere's",
            "\n\nThis code",
            "\n\nThis test",
            "\n\nThe test",
            "\n\nNote:",
            "\n\nExplanation:",
            "\n\n# Explanation",
            "\n\nThese tests",
            "\n\nLet me know",
            "\n\nFeel free",
        ]
        
        for marker in explanation_markers:
            if marker in code:
                code = code.split(marker)[0]
        
        # Remove incomplete last line if it doesn't end properly
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # Check if last line is incomplete
            if last_line and not any(last_line.endswith(x) for x in [':', ')', ']', '}', '"', "'"]):
                if not last_line.startswith(('#', 'assert', 'with', 'def')):
                    lines = lines[:-1]
        
        code = '\n'.join(lines)
        
        # Ensure proper imports
        if 'import pytest' not in code:
            code = 'import pytest\n' + code
        
        return code.strip()


class TestFileWriter:
    """Writes combined test file with all generated tests"""
    
    @staticmethod
    def write(tests: List[Dict], output_path: str, source_file: str):
        """Combine all tests and write to file"""
        
        source_name = Path(source_file).stem
        
        # Build header
        imports = [
            '"""',
            f'Generated tests for {os.path.basename(source_file)}',
            f'Created by TestGen - Pytest Generator',
            '"""',
            '',
            'import pytest',
            'from unittest.mock import Mock, AsyncMock, patch',
            'from typing import Any',
            '',
            f'# Import functions from {source_name}',
            f'from {source_name} import *',
            '',
            ''
        ]
        
        content_lines = imports
        
        # Add each test
        for i, test in enumerate(tests, 1):
            if test['code']:
                content_lines.append(f"{'='*60}")
                content_lines.append(f"# Test {i}/{len(tests)}: {test['function_name']}")
                content_lines.append(f"{'='*60}")
                content_lines.append(test['code'])
                content_lines.append('')
                content_lines.append('')
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        
        return output_path


def process_file(file_path: str, generator: TestGenerator, output_dir: str, verbose: bool = False) -> str:
    """Process a single Python file"""
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}📄 Processing: {file_path}{Colors.ENDC}")
    
    # Read source
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        print(f"{Colors.FAIL}❌ Failed to read {file_path}: {e}{Colors.ENDC}")
        return None
    
    # Extract functions
    print(f"  {Colors.BLUE}Analyzing...{Colors.ENDC}")
    
    try:
        tree = ast.parse(source_code)
        extractor = FunctionExtractor(source_code)
        extractor.visit(tree)
    except SyntaxError as e:
        print(f"{Colors.FAIL}❌ Syntax error in {file_path}: {e}{Colors.ENDC}")
        return None
    
    functions = extractor.functions
    
    if not functions:
        print(f"{Colors.WARNING}  ⚠️  No functions found{Colors.ENDC}")
        return None
    
    print(f"  {Colors.GREEN}✅ Found {len(functions)} function(s){Colors.ENDC}\n")
    
    # Show function summary
    for i, func in enumerate(functions, 1):
        async_marker = "async " if func['is_async'] else ""
        deps_marker = "with deps" if func['has_deps'] else ""
        print(f"    {i}. {async_marker}{func['name']}({func['param_count']} params) {deps_marker}")
    
    print()
    
    # Generate tests for each function
    generated_tests = []
    total_time = 0
    
    for i, func in enumerate(functions, 1):
        print(f"  {Colors.BLUE}[{i}/{len(functions)}] {func['name']}...{Colors.ENDC}", end=" ")
        sys.stdout.flush()
        
        test_code, elapsed = generator.generate(func)
        total_time += elapsed
        
        generated_tests.append({
            'function_name': func['name'],
            'code': test_code,
            'time': elapsed,
            'is_async': func['is_async']
        })
        
        print(f"{Colors.GREEN}✓ {elapsed:.1f}s{Colors.ENDC}")
    
    # Write combined test file
    source_name = Path(file_path).stem
    output_name = f"test_{source_name}.py"
    output_path = os.path.join(output_dir, output_name)
    
    TestFileWriter.write(generated_tests, output_path, file_path)
    
    # Summary
    avg_time = total_time / len(functions)
    print(f"\n  {Colors.GREEN}{Colors.BOLD}✅ Created: {output_path}{Colors.ENDC}")
    print(f"  {Colors.CYAN}Total: {total_time:.1f}s | Average: {avg_time:.1f}s per function{Colors.ENDC}")
    
    return output_path


def process_directory(dir_path: str, generator: TestGenerator, output_dir: str, verbose: bool = False):
    """Batch process all Python files in directory"""
    
    py_files = list(Path(dir_path).rglob("*.py"))
    
    # Exclude test files and __init__.py
    py_files = [
        f for f in py_files 
        if not f.name.startswith('test_') 
        and f.name != '__init__.py'
        and 'venv' not in str(f)
        and '.venv' not in str(f)
    ]
    
    if not py_files:
        print(f"{Colors.WARNING}No Python files found in {dir_path}{Colors.ENDC}")
        return
    
    print(f"{Colors.CYAN}Found {len(py_files)} Python file(s) to process{Colors.ENDC}\n")
    
    success_count = 0
    
    for i, py_file in enumerate(py_files, 1):
        print(f"{Colors.BOLD}{'='*60}")
        print(f"File {i}/{len(py_files)}")
        print(f"{'='*60}{Colors.ENDC}")
        
        result = process_file(str(py_file), generator, output_dir, verbose)
        
        if result:
            success_count += 1
    
    # Final summary
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*60}")
    print(f"✨ Batch Complete: {success_count}/{len(py_files)} files processed")
    print(f"{'='*60}{Colors.ENDC}\n")


def download_model_if_needed(model_file: str):
    """Download GGUF model if not present locally"""
    
    if os.path.exists(model_file):
        return model_file
    
    print(f"{Colors.BLUE}Model not found locally. Downloading from HuggingFace...{Colors.ENDC}")
    
    try:
        from huggingface_hub import hf_hub_download
        
        model_path = hf_hub_download(
            repo_id=Config.MODEL_REPO,
            filename=model_file,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        
        print(f"{Colors.GREEN}✅ Model downloaded: {model_path}{Colors.ENDC}\n")
        return model_path
        
    except ImportError:
        print(f"{Colors.FAIL}❌ huggingface_hub not installed{Colors.ENDC}")
        print(f"Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Download failed: {e}{Colors.ENDC}")
        print(f"\n{Colors.WARNING}Please download manually from:{Colors.ENDC}")
        print(f"https://huggingface.co/{Config.MODEL_REPO}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="🧪 TestGen - Generate pytest tests using local AI (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s calculator.py                    # Generate tests for single file
  %(prog)s ./src/ -o ./tests/               # Process all files in src/
  %(prog)s math.py --model custom.gguf      # Use custom model
  %(prog)s app.py -v                        # Verbose mode

Performance Tips:
  • Mac M2/M3/M4: Automatically uses Metal GPU (fast!)
  • NVIDIA GPU: Automatically detected (fast!)
  • CPU only: Consider using smaller 3B model for speed
        """
    )
    
    parser.add_argument(
        'target',
        help='Python file or directory to generate tests for'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='./generated_tests/',
        help='Output directory for test files (default: ./generated_tests/)'
    )
    
    parser.add_argument(
        '--model',
        default=None,
        help='Path to GGUF model file (default: auto-download 8B Q4)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (show token counts, timings)'
    )
    
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 model instead of Q4 (higher quality, slower)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("="*60)
    print("🧪 TestGen - Pytest Test Generator (Optimized)")
    print("="*60)
    print(f"{Colors.ENDC}\n")
    
    # Show system info
    HardwareDetector.print_system_info()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get model path
    if args.model:
        model_path = args.model
    else:
        model_file = Config.MODEL_FILE_FP16 if args.fp16 else Config.MODEL_FILE_8B
        model_path = download_model_if_needed(model_file)
    
    if not os.path.exists(model_path):
        print(f"{Colors.FAIL}❌ Model file not found: {model_path}{Colors.ENDC}")
        sys.exit(1)
    
    # Initialize generator
    generator = TestGenerator(model_path, verbose=args.verbose)
    
    # Process target
    target_path = Path(args.target)
    
    if not target_path.exists():
        print(f"{Colors.FAIL}❌ Target not found: {args.target}{Colors.ENDC}")
        sys.exit(1)
    
    start_time = time.time()
    
    if target_path.is_file():
        process_file(str(target_path), generator, args.output, args.verbose)
    elif target_path.is_dir():
        process_directory(str(target_path), generator, args.output, args.verbose)
    else:
        print(f"{Colors.FAIL}❌ Invalid target: {args.target}{Colors.ENDC}")
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    # Final message
    print(f"\n{Colors.GREEN}{Colors.BOLD}✨ All done!{Colors.ENDC}")
    print(f"{Colors.GREEN}Tests saved to: {args.output}{Colors.ENDC}")
    print(f"{Colors.CYAN}Total time: {total_time:.1f}s{Colors.ENDC}\n")


if __name__ == "__main__":
    main()