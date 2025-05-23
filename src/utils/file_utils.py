"""
File Management Utilities for SEO Competitive Intelligence

Advanced file operations, backup management, and data export capabilities
"""

import os
import shutil
import json
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator
import logging
from datetime import datetime, timedelta
import zipfile
import gzip
import pandas as pd
from dataclasses import dataclass, asdict
import hashlib
import tempfile
from contextlib import contextmanager

@dataclass
class FileInfo:
    """File information structure"""
    path: Path
    size_bytes: int
    created_time: datetime
    modified_time: datetime
    checksum: str
    file_type: str

@dataclass
class BackupInfo:
    """Backup information structure"""
    backup_id: str
    source_path: Path
    backup_path: Path
    created_time: datetime
    size_bytes: int
    checksum: str
    compression_ratio: float

class FileManager:
    """
    Advanced file management for SEO competitive intelligence.
    Provides comprehensive file operations including safe read/write,
    atomic operations, and file system monitoring.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.gettempdir()) / "seo_intelligence"
        self.temp_dir.mkdir(exist_ok=True)

    def safe_read_file(
        self,
        file_path: Union[str, Path],
        encoding: str = 'utf-8',
        fallback_encodings: List[str] = None
    ) -> str:
        """
        Safely read text file with encoding detection and fallbacks.
        
        Args:
            file_path: Path to file
            encoding: Primary encoding to try
            fallback_encodings: List of fallback encodings
            
        Returns:
            File content as string
        """
        try:
            if fallback_encodings is None:
                fallback_encodings = ['utf-8', 'latin-1', 'cp1252']
            
            path_obj = Path(file_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Try primary encoding first
            try:
                with open(path_obj, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                pass

            # Try fallback encodings
            for fallback_encoding in fallback_encodings:
                try:
                    with open(path_obj, 'r', encoding=fallback_encoding) as f:
                        content = f.read()
                        self.logger.warning(f"Used fallback encoding {fallback_encoding} for {file_path}")
                        return content
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, read as binary and decode with errors='replace'
            with open(path_obj, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')
                self.logger.warning(f"Read {file_path} with error replacement")
                return content

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def safe_write_file(
        self,
        file_path: Union[str, Path],
        content: str,
        encoding: str = 'utf-8',
        atomic: bool = True,
        backup: bool = False
    ) -> bool:
        """
        Safely write text file with atomic operations.
        
        Args:
            file_path: Path to file
            content: Content to write
            encoding: Text encoding
            atomic: Whether to use atomic write operation
            backup: Whether to create backup before writing
            
        Returns:
            True if write successful
        """
        try:
            path_obj = Path(file_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if requested and file exists
            if backup and path_obj.exists():
                backup_path = self._create_backup(path_obj)
                self.logger.info(f"Created backup: {backup_path}")

            if atomic:
                # Atomic write using temporary file
                temp_path = path_obj.with_suffix(path_obj.suffix + '.tmp')
                try:
                    with open(temp_path, 'w', encoding=encoding) as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk

                    # Atomic move
                    temp_path.replace(path_obj)
                except Exception as e:
                    # Clean up temp file on error
                    if temp_path.exists():
                        temp_path.unlink()
                    raise
            else:
                # Direct write
                with open(path_obj, 'w', encoding=encoding) as f:
                    f.write(content)

            self.logger.debug(f"Successfully wrote file: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {str(e)}")
            return False

    def safe_read_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Safely read JSON file with error handling.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        try:
            content = self.safe_read_file(file_path)
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading JSON file {file_path}: {str(e)}")
            raise

    def safe_write_json(
        self,
        file_path: Union[str, Path],
        data: Dict[str, Any],
        indent: int = 2,
        atomic: bool = True
    ) -> bool:
        """
        Safely write JSON file.
        
        Args:
            file_path: Path to JSON file
            data: Data to write
            indent: JSON indentation
            atomic: Whether to use atomic write
            
        Returns:
            True if write successful
        """
        try:
            json_content = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
            return self.safe_write_file(file_path, json_content, atomic=atomic)
        except Exception as e:
            self.logger.error(f"Error writing JSON file {file_path}: {str(e)}")
            return False

    def get_file_info(self, file_path: Union[str, Path]) -> Optional[FileInfo]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileInfo object or None if error
        """
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return None

            stat = path_obj.stat()

            # Calculate checksum
            checksum = self._calculate_file_checksum(path_obj)

            # Determine file type
            file_type = self._determine_file_type(path_obj)

            return FileInfo(
                path=path_obj,
                size_bytes=stat.st_size,
                created_time=datetime.fromtimestamp(stat.st_ctime),
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                checksum=checksum,
                file_type=file_type
            )

        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return None

    def find_files(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
        file_type: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        modified_after: Optional[datetime] = None,
        modified_before: Optional[datetime] = None
    ) -> List[FileInfo]:
        """
        Find files matching criteria.
        
        Args:
            directory: Directory to search
            pattern: File pattern (glob)
            recursive: Whether to search recursively
            file_type: Filter by file type
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            modified_after: Modified after this date
            modified_before: Modified before this date
            
        Returns:
            List of matching FileInfo objects
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return []

            # Find files using glob
            if recursive:
                file_paths = directory_path.rglob(pattern)
            else:
                file_paths = directory_path.glob(pattern)

            matching_files = []

            for file_path in file_paths:
                if not file_path.is_file():
                    continue

                file_info = self.get_file_info(file_path)
                if not file_info:
                    continue

                # Apply filters
                if file_type and file_info.file_type != file_type:
                    continue

                if min_size and file_info.size_bytes < min_size:
                    continue

                if max_size and file_info.size_bytes > max_size:
                    continue

                if modified_after and file_info.modified_time < modified_after:
                    continue

                if modified_before and file_info.modified_time > modified_before:
                    continue

                matching_files.append(file_info)

            return matching_files

        except Exception as e:
            self.logger.error(f"Error finding files in {directory}: {str(e)}")
            return []

    # Methods from paste file merged below
    def ensure_directory(self, path: Union[str, Path]) -> bool:
        """Ensure directory exists (from paste file)"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create directory {path}: {str(e)}")
            return False

    def copy_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path]
    ) -> bool:
        """Copy file to destination (from paste file)"""
        try:
            source = Path(source)
            destination = Path(destination)
            
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, destination)
            self.logger.info(f"Copied {source} to {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy file: {str(e)}")
            return False

    def move_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path]
    ) -> bool:
        """Move file to destination (from paste file)"""
        try:
            source = Path(source)
            destination = Path(destination)
            
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(destination))
            self.logger.info(f"Moved {source} to {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move file: {str(e)}")
            return False

    def delete_file(self, path: Union[str, Path]) -> bool:
        """Delete file (from paste file)"""
        try:
            Path(path).unlink(missing_ok=True)
            self.logger.debug(f"Deleted file: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete file: {str(e)}")
            return False

    def clean_directory(
        self,
        directory: Union[str, Path],
        keep_days: int = 30
    ) -> int:
        """Clean old files from directory (enhanced from paste file)"""
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return 0

            cutoff_time = datetime.now() - timedelta(days=keep_days)
            deleted_count = 0

            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    file_info = self.get_file_info(file_path)
                    if file_info and file_info.modified_time < cutoff_time:
                        if self.delete_file(file_path):
                            deleted_count += 1

            self.logger.info(f"Cleaned {deleted_count} old files from {directory}")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error cleaning directory {directory}: {str(e)}")
            return 0

    @contextmanager
    def temp_file(self, suffix: str = "", prefix: str = "temp_", content: str = ""):
        """
        Context manager for temporary file operations.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            content: Initial content
            
        Yields:
            Path to temporary file
        """
        temp_path = None
        try:
            # Create temporary file
            temp_path = self.temp_dir / f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
            
            # Write initial content
            if content:
                with open(temp_path, 'w') as f:
                    f.write(content)
            
            yield temp_path

        finally:
            # Clean up
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not remove temp file {temp_path}: {str(e)}")

    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_name(f"{file_path.stem}_{timestamp}_backup{file_path.suffix}")
        shutil.copy2(file_path, backup_path)
        return backup_path

    def _calculate_file_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum."""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {str(e)}")
            return ""

    def _determine_file_type(self, file_path: Path) -> str:
        """Determine file type based on extension and content."""
        extension = file_path.suffix.lower()
        type_mappings = {
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.csv': 'csv',
            '.json': 'json',
            '.txt': 'text',
            '.log': 'log',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.html': 'html',
            '.pdf': 'pdf',
            '.zip': 'archive',
            '.gz': 'compressed'
        }
        return type_mappings.get(extension, 'unknown')


class ExportManager:
    """
    Advanced export management for SEO competitive intelligence.
    Handles various export formats and provides compression,
    encryption, and batch export capabilities.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.file_manager = FileManager(logger)

    def export_dataframe(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        format: str = 'auto',
        compression: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Export DataFrame to various formats.
        
        Args:
            df: DataFrame to export
            file_path: Output file path
            format: Export format ('auto', 'csv', 'excel', 'json', 'parquet')
            compression: Compression method
            **kwargs: Additional format-specific arguments
            
        Returns:
            True if export successful
        """
        try:
            path_obj = Path(file_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Auto-detect format from extension
            if format == 'auto':
                format = self._detect_format_from_extension(path_obj)

            # Export based on format
            if format == 'csv':
                df.to_csv(path_obj, index=False, compression=compression, **kwargs)
            elif format == 'excel':
                df.to_excel(path_obj, index=False, **kwargs)
            elif format == 'json':
                df.to_json(path_obj, orient='records', indent=2, **kwargs)
            elif format == 'parquet':
                df.to_parquet(path_obj, compression=compression, **kwargs)
            elif format == 'pickle':
                df.to_pickle(path_obj, compression=compression, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Exported DataFrame to {file_path} (format: {format})")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting DataFrame: {str(e)}")
            return False

    def export_multiple_dataframes(
        self,
        dataframes: Dict[str, pd.DataFrame],
        output_directory: Union[str, Path],
        format: str = 'csv',
        compression: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Export multiple DataFrames to files.
        
        Args:
            dataframes: Dictionary of DataFrames with names as keys
            output_directory: Output directory
            format: Export format
            compression: Compression method
            
        Returns:
            Dictionary of export results
        """
        try:
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)

            results = {}

            for name, df in dataframes.items():
                file_name = f"{name}.{format}"
                file_path = output_path / file_name
                success = self.export_dataframe(
                    df, file_path, format=format, compression=compression
                )
                results[name] = success

            successful_exports = sum(1 for success in results.values() if success)
            self.logger.info(f"Exported {successful_exports}/{len(dataframes)} DataFrames")

            return results

        except Exception as e:
            self.logger.error(f"Error exporting multiple DataFrames: {str(e)}")
            return {}

    def create_export_archive(
        self,
        source_files: List[Union[str, Path]],
        archive_path: Union[str, Path],
        compression_level: int = 6
    ) -> bool:
        """
        Create compressed archive of export files.
        
        Args:
            source_files: List of files to archive
            archive_path: Output archive path
            compression_level: Compression level (0-9)
            
        Returns:
            True if archive created successfully
        """
        try:
            archive_path_obj = Path(archive_path)
            archive_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(
                archive_path_obj,
                'w',
                zipfile.ZIP_DEFLATED,
                compresslevel=compression_level
            ) as zipf:
                for source_file in source_files:
                    source_path = Path(source_file)
                    if source_path.exists():
                        # Add file to archive with relative name
                        arcname = source_path.name
                        zipf.write(source_path, arcname)
                        self.logger.debug(f"Added {source_path} to archive as {arcname}")
                    else:
                        self.logger.warning(f"Source file not found: {source_file}")

            # Calculate compression ratio
            original_size = sum(Path(f).stat().st_size for f in source_files if Path(f).exists())
            compressed_size = archive_path_obj.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            self.logger.info(
                f"Created archive {archive_path} with {compression_ratio:.1f}% compression"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error creating export archive: {str(e)}")
            return False

    def _detect_format_from_extension(self, file_path: Path) -> str:
        """Detect export format from file extension."""
        extension = file_path.suffix.lower()
        format_mappings = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.parquet': 'parquet',
            '.pickle': 'pickle',
            '.pkl': 'pickle'
        }
        return format_mappings.get(extension, 'csv')


class BackupManager:
    """
    Advanced backup management for SEO competitive intelligence.
    Provides automated backup scheduling, retention policies,
    and backup verification capabilities.
    """

    def __init__(self, backup_directory: Union[str, Path], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        self.file_manager = FileManager(logger)

    def create_backup(
        self,
        source_path: Union[str, Path],
        backup_name: Optional[str] = None,
        compression: bool = True
    ) -> Optional[BackupInfo]:
        """
        Create backup of file or directory.
        
        Args:
            source_path: Path to backup
            backup_name: Custom backup name
            compression: Whether to compress backup
            
        Returns:
            BackupInfo object or None if failed
        """
        try:
            source_path_obj = Path(source_path)
            if not source_path_obj.exists():
                raise FileNotFoundError(f"Source path not found: {source_path}")

            # Generate backup name
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{source_path_obj.name}_{timestamp}"

            # Determine backup path
            if compression:
                backup_path = self.backup_directory / f"{backup_name}.zip"
            else:
                backup_path = self.backup_directory / backup_name

            # Create backup
            original_size = self._calculate_size(source_path_obj)
            
            if source_path_obj.is_file():
                if compression:
                    self._compress_file(source_path_obj, backup_path)
                else:
                    shutil.copy2(source_path_obj, backup_path)
            else:
                if compression:
                    self._compress_directory(source_path_obj, backup_path)
                else:
                    shutil.copytree(source_path_obj, backup_path)

            # Calculate backup info
            backup_size = self._calculate_size(backup_path)
            compression_ratio = (1 - backup_size / original_size) * 100 if original_size > 0 else 0
            checksum = self.file_manager._calculate_file_checksum(backup_path) if backup_path.is_file() else ""

            backup_info = BackupInfo(
                backup_id=backup_name,
                source_path=source_path_obj,
                backup_path=backup_path,
                created_time=datetime.now(),
                size_bytes=backup_size,
                checksum=checksum,
                compression_ratio=compression_ratio
            )

            self.logger.info(f"Created backup: {backup_path}")
            return backup_info

        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return None

    def _compress_file(self, source_file: Path, backup_path: Path):
        """Compress single file."""
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(source_file, source_file.name)

    def _compress_directory(self, source_dir: Path, backup_path: Path):
        """Compress directory."""
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)

    def _calculate_size(self, path: Path) -> int:
        """Calculate total size of file or directory."""
        if path.is_file():
            return path.stat().st_size
        else:
            total_size = 0
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
