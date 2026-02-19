"""
File service for handling file operations.
This file demonstrates runtime inspection with stdlib modules.
"""

from pathlib import Path
from typing import List, Optional


def list_log_files(directory: Path) -> List[str]:
    """
    List all log files in a directory.
    
    Args:
        directory: Path to search
        
    Returns:
        List of log file paths
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} not found")
    
    log_files = directory.glob("*.log")
    return [str(f) for f in log_files]


def read_config_file(config_path: Path) -> Optional[str]:
    """
    Read configuration from a file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config content or None if file doesn't exist
    """
    if not config_path.exists():
        return None
    
    return config_path.read_text(encoding='utf-8')


def create_backup(source: Path, backup_dir: Path) -> Path:
    """
    Create a backup of a file.
    
    Args:
        source: Source file path
        backup_dir: Directory for backups
        
    Returns:
        Path to backup file
        
    Raises:
        FileNotFoundError: If source doesn't exist
    """
    if not source.exists():
        raise FileNotFoundError(f"Source file {source} not found")
    
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True)
    
    backup_path = backup_dir / f"{source.name}.backup"
    content = source.read_text()
    backup_path.write_text(content)
    
    return backup_path
