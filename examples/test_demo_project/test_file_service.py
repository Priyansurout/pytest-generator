"""
Generated tests for file_service.py
Created by pytest-generator

NOTE: This is unedited raw model output. It contains known 8B model
limitations (wrong function signatures, incorrect method kwargs) that
require developer review and refinement before use.
"""
# Test 1: list_log_files
import pytest
import importlib
from unittest.mock import Mock
from pathlib import Path


module = importlib.import_module("file_service")
list_log_files = getattr(module, "list_log_files")
FileNotFoundError = getattr(module, "FileNotFoundError", Exception)


@pytest.fixture
def mock_directory():
    mock = Mock()
    mock.glob.return_value = ["file1.log", "file2.log"]
    return mock


@pytest.mark.parametrize("directory,expected_files", [
    (Path("/var/log"), ["file1.log", "file2.log"]),
    (Path("/tmp/logs"), ["app.log", "error.log"]),
])
def test_list_log_files_success(directory, expected_files):
    mock = Mock()
    mock.glob.return_value = expected_files
    mock.exists.return_value = True

    result = list_log_files(directory, mock)
    assert result == expected_files
    mock.glob.assert_called_once_with("*.log")
    mock.exists.assert_called_once_with(follow_symlinks=True)


def test_list_log_files_directory_does_not_exist():
    mock = Mock()
    mock.exists.return_value = False

    with pytest.raises(FileNotFoundError):
        list_log_files(Path("/nonexistent"), mock)
    mock.exists.assert_called_once_with(follow_symlinks=True)

# Test 2: read_config_file
import pytest
import importlib
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import os


module = importlib.import_module("file_service")
read_config_file = getattr(module, "read_config_file")
Path = getattr(module, "Path", Path)
os = getattr(module, "os", os)


@pytest.fixture
def config_path():
    return Mock(spec=Path)

@pytest.fixture
def os_mock():
    mock = Mock()
    mock.path = Mock()
    mock.path.exists = Mock(return_value=True)
    return mock

@pytest.mark.parametrize("config_path_content,expected", [
    ("config_content", "config_content"),
    ("", None),
    (None, None),
])
def test_read_config_file(config_path, os_mock, config_path_content, expected):
    os_mock.path.exists.return_value = True
    config_path.read_text.return_value = config_path_content
    
    result = read_config_file(config_path, os_mock)
    assert result == expected
    config_path.read_text.assert_called_once_with(encoding='utf-8', errors='strict')
    os_mock.path.exists.assert_called_once_with(config_path)

def test_read_config_file_file_does_not_exist(config_path, os_mock):
    os_mock.path.exists.return_value = False
    
    result = read_config_file(config_path, os_mock)
    assert result is None
    os_mock.path.exists.assert_called_once_with(config_path)

# Test 3: create_backup
import pytest
import importlib
from unittest.mock import MagicMock
import os
from pathlib import Path


module = importlib.import_module("file_service")
create_backup = getattr(module, "create_backup")
FileNotFoundError = getattr(module, "FileNotFoundError", Exception)


@pytest.fixture
def source():
    return MagicMock(spec=Path)

@pytest.fixture
def backup_dir():
    return MagicMock(spec=Path)


def test_create_backup_success(source, backup_dir):
    source.read_text.return_value = "content"
    source.exists.return_value = True
    backup_dir.exists.return_value = True
    backup_dir.mkdir.return_value = None

    expected_backup_path = backup_dir / "backup.txt"
    result = create_backup(source, backup_dir)

    assert result == expected_backup_path
    source.read_text.assert_called_once_with(encoding="utf-8", errors="strict")
    source.exists.assert_called_once_with(follow_symlinks=True)
    backup_dir.exists.assert_called_once_with(follow_symlinks=True)
    backup_dir.mkdir.assert_called_once_with(mode=0o777, parents=True, exist_ok=True)

def test_create_backup_source_does_not_exist(source, backup_dir):
    source.exists.return_value = False

    with pytest.raises(FileNotFoundError):
        create_backup(source, backup_dir)
