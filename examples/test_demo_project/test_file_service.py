"""
Generated tests for file_service.py
Created by pytest-generator
"""
import pytest
from unittest.mock import MagicMock
import importlib

module = importlib.import_module("file_service")
list_log_files = getattr(module, "list_log_files")
read_config_file = getattr(module, "read_config_file")
create_backup = getattr(module, "create_backup")


# Test 1: list_log_files
def test_list_log_files_success():
    directory = MagicMock()
    directory.exists.return_value = True
    mock_file1 = MagicMock()
    mock_file2 = MagicMock()
    mock_file1.__str__ = lambda self: "/logs/app.log"
    mock_file2.__str__ = lambda self: "/logs/error.log"
    directory.glob.return_value = [mock_file1, mock_file2]

    result = list_log_files(directory)
    assert result == ["/logs/app.log", "/logs/error.log"]
    directory.exists.assert_called_once()
    directory.glob.assert_called_once_with("*.log")

def test_list_log_files_empty():
    directory = MagicMock()
    directory.exists.return_value = True
    directory.glob.return_value = []

    result = list_log_files(directory)
    assert result == []

def test_list_log_files_directory_not_found():
    directory = MagicMock()
    directory.exists.return_value = False

    with pytest.raises(FileNotFoundError):
        list_log_files(directory)


# Test 2: read_config_file
def test_read_config_file_success():
    config_path = MagicMock()
    config_path.exists.return_value = True
    config_path.read_text.return_value = "key: value"

    result = read_config_file(config_path)
    assert result == "key: value"
    config_path.read_text.assert_called_once_with(encoding="utf-8")

def test_read_config_file_not_found():
    config_path = MagicMock()
    config_path.exists.return_value = False

    result = read_config_file(config_path)
    assert result is None


# Test 3: create_backup
def test_create_backup_success():
    source = MagicMock()
    source.exists.return_value = True
    source.name = "data.txt"
    source.read_text.return_value = "file content"

    backup_dir = MagicMock()
    backup_dir.exists.return_value = True

    backup_file = MagicMock()
    backup_dir.__truediv__ = MagicMock(return_value=backup_file)

    result = create_backup(source, backup_dir)

    backup_dir.__truediv__.assert_called_once_with("data.txt.backup")
    source.read_text.assert_called_once()
    backup_file.write_text.assert_called_once_with("file content")
    assert result == backup_file

def test_create_backup_creates_dir():
    source = MagicMock()
    source.exists.return_value = True
    source.name = "report.csv"
    source.read_text.return_value = "csv data"

    backup_dir = MagicMock()
    backup_dir.exists.return_value = False

    backup_file = MagicMock()
    backup_dir.__truediv__ = MagicMock(return_value=backup_file)

    create_backup(source, backup_dir)

    backup_dir.mkdir.assert_called_once_with(parents=True)

def test_create_backup_source_not_found():
    source = MagicMock()
    source.exists.return_value = False
    backup_dir = MagicMock()

    with pytest.raises(FileNotFoundError):
        create_backup(source, backup_dir)
