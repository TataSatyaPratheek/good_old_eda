"""
File Management Module for SEO Competitive Intelligence
Advanced file handling, data loading, and export management using utility framework
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging

# Import our utilities to eliminate redundancy
from src.utils.file_utils import FileManager as BaseFileManager, ExportManager, BackupManager
from src.utils.common_helpers import StringHelper, DateHelper, memoize, timing_decorator, ensure_list
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager, PathManager
from src.utils.validation_utils import SchemaValidator
from src.utils.export_utils import ReportExporter, DataExporter

class FileManager:
    """
    Advanced file management for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide file operations,
    data loading, validation, and export capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities instead of custom implementations."""
        self.logger = logger or LoggerFactory.get_logger("file_manager")
        self.config = config_manager or ConfigManager()
        
        # Initialize utility classes - no more redundant implementations
        self.base_file_manager = BaseFileManager(self.logger)
        self.export_manager = ExportManager(self.logger)
        self.backup_manager = BackupManager(
            backup_directory=self._get_backup_directory(),
            logger=self.logger
        )
        self.data_processor = DataProcessor(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.schema_validator = SchemaValidator(self.logger)
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        self.path_manager = PathManager(config_manager=self.config)
        self.report_exporter = ReportExporter(self.logger)
        self.data_exporter = DataExporter(self.logger)
        
        # Load file type configurations from config
        analysis_config = self.config.get_analysis_config()
        self.max_file_size = getattr(analysis_config, 'max_file_size_mb', 100) * 1024 * 1024

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def load_seo_data_file(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        validate_data: bool = True,
        clean_data: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load SEO data file with comprehensive processing using utilities.
        
        Args:
            file_path: Path to the data file
            file_type: Optional file type specification
            validate_data: Whether to validate loaded data
            clean_data: Whether to clean loaded data
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        try:
            with self.performance_tracker.track_block("load_seo_data_file"):
                # Audit log the file access
                self.audit_logger.log_data_access(
                    user_id="system",
                    resource=str(file_path),
                    action="load_seo_data_file"
                )
                
                # Get file info using BaseFileManager
                file_info = self.base_file_manager.get_file_info(file_path)
                if not file_info:
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Check file size
                if file_info.size_bytes > self.max_file_size:
                    self.logger.warning(f"Large file detected: {file_info.size_bytes / (1024*1024):.1f} MB")
                
                # Load data using appropriate method
                df = self._load_data_by_type(file_path, file_info.file_type)
                
                # Process data using our utilities
                metadata = {
                    'source_file': str(file_path),
                    'file_type': file_info.file_type,
                    'file_size': file_info.size_bytes,
                    'load_timestamp': datetime.now(),
                    'original_shape': df.shape
                }
                
                # Validate data using SchemaValidator
                if validate_data:
                    schema_type = self._detect_seo_schema_type(df)
                    validation_report = self._validate_seo_data(df, schema_type)
                    metadata['validation_report'] = validation_report
                    
                    if validation_report.critical_issues > 0:
                        self.logger.warning(f"Critical data quality issues found: {validation_report.critical_issues}")
                
                # Clean data using DataProcessor
                if clean_data:
                    original_count = len(df)
                    df = self.data_processor.clean_seo_data(df)
                    cleaned_count = len(df)
                    metadata['rows_removed_during_cleaning'] = original_count - cleaned_count
                    metadata['final_shape'] = df.shape
                
                self.logger.info(f"Loaded SEO data: {len(df)} records from {file_path}")
                return df, metadata
                
        except Exception as e:
            self.logger.error(f"Error loading SEO data file: {str(e)}")
            self.audit_logger.log_data_access(
                user_id="system",
                resource=str(file_path),
                action="load_seo_data_file",
                result="failure",
                details={"error": str(e)}
            )
            raise

    @timing_decorator()
    def save_processed_data(
        self,
        data: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = 'csv',
        include_metadata: bool = True,
        create_backup: bool = False
    ) -> bool:
        """
        Save processed data with metadata using utility framework.
        
        Args:
            data: DataFrame to save
            output_path: Output file path
            format: Export format
            include_metadata: Whether to include metadata
            create_backup: Whether to create backup
            
        Returns:
            True if save successful
        """
        try:
            with self.performance_tracker.track_block("save_processed_data"):
                # Create backup if requested
                if create_backup and Path(output_path).exists():
                    backup_info = self.backup_manager.create_backup(output_path)
                    if backup_info:
                        self.logger.info(f"Created backup: {backup_info.backup_path}")
                
                # Save using ExportManager
                success = self.export_manager.export_dataframe(
                    data, output_path, format=format
                )
                
                # Save metadata if requested
                if include_metadata and success:
                    metadata = {
                        'export_timestamp': datetime.now().isoformat(),
                        'record_count': len(data),
                        'column_count': len(data.columns),
                        'columns': data.columns.tolist(),
                        'format': format,
                        'data_types': data.dtypes.astype(str).to_dict()
                    }
                    
                    metadata_path = Path(output_path).with_suffix('.metadata.json')
                    self.base_file_manager.safe_write_json(metadata_path, metadata)
                
                # Audit log the export
                self.audit_logger.log_export_event(
                    user_id="system",
                    export_type=format,
                    data_scope=f"{len(data)} records",
                    destination=str(output_path),
                    result="success" if success else "failure"
                )
                
                return success
                
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            return False

    @timing_decorator()
    def batch_process_files(
        self,
        file_paths: List[Union[str, Path]],
        output_directory: Union[str, Path],
        processing_function: Optional[callable] = None
    ) -> Dict[str, bool]:
        """
        Batch process multiple files using utility framework.
        
        Args:
            file_paths: List of file paths to process
            output_directory: Output directory
            processing_function: Optional custom processing function
            
        Returns:
            Dictionary of processing results by file
        """
        try:
            with self.performance_tracker.track_block("batch_process_files"):
                # Ensure output directory exists using PathManager
                output_path = self.path_manager.ensure_directory_exists(output_directory)
                
                results = {}
                
                for file_path in file_paths:
                    file_name = Path(file_path).stem
                    
                    try:
                        self.logger.info(f"Processing file: {file_name}")
                        
                        # Load file
                        df, metadata = self.load_seo_data_file(file_path)
                        
                        # Apply custom processing if provided
                        if processing_function:
                            df = processing_function(df)
                        
                        # Save processed file
                        output_file = Path(output_directory) / f"{file_name}_processed.csv"
                        success = self.save_processed_data(df, output_file)
                        
                        results[file_name] = success
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {file_name}: {str(e)}")
                        results[file_name] = False
                
                successful_count = sum(1 for success in results.values() if success)
                self.logger.info(f"Batch processing completed: {successful_count}/{len(file_paths)} files successful")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error in batch file processing: {str(e)}")
            return {}

    def create_file_archive(
        self,
        source_paths: List[Union[str, Path]],
        archive_path: Union[str, Path],
        compression_level: int = 6
    ) -> bool:
        """
        Create file archive using ExportManager.
        
        Args:
            source_paths: List of source file paths
            archive_path: Output archive path
            compression_level: Compression level
            
        Returns:
            True if archive created successfully
        """
        try:
            return self.export_manager.create_export_archive(
                source_paths, archive_path, compression_level
            )
        except Exception as e:
            self.logger.error(f"Error creating file archive: {str(e)}")
            return False

    def organize_files_by_date(
        self,
        source_directory: Union[str, Path],
        target_directory: Union[str, Path],
        date_pattern: str = "%Y%m%d"
    ) -> Dict[str, int]:
        """
        Organize files by date using PathManager and DateHelper.
        
        Args:
            source_directory: Source directory containing files
            target_directory: Target directory for organized files
            date_pattern: Date pattern for organization
            
        Returns:
            Dictionary of organization results
        """
        try:
            with self.performance_tracker.track_block("organize_files_by_date"):
                # Find files using BaseFileManager
                files = self.base_file_manager.find_files(
                    source_directory, recursive=False
                )
                
                organization_results = {}
                
                for file_info in files:
                    try:
                        # Extract date from filename or modification date
                        file_date = self._extract_date_from_filename(
                            file_info.path.name, date_pattern
                        ) or file_info.modified_time
                        
                        # Create dated folder using PathManager
                        dated_folder = self.path_manager.create_dated_folder(
                            str(target_directory), file_date
                        )
                        
                        # Move file
                        target_path = dated_folder / file_info.path.name
                        file_info.path.rename(target_path)
                        
                        date_key = file_date.strftime("%Y-%m-%d")
                        organization_results[date_key] = organization_results.get(date_key, 0) + 1
                        
                    except Exception as e:
                        self.logger.warning(f"Could not organize file {file_info.path}: {str(e)}")
                
                self.logger.info(f"Organized {len(files)} files by date")
                return organization_results
                
        except Exception as e:
            self.logger.error(f"Error organizing files by date: {str(e)}")
            return {}

    def clean_old_files(
        self,
        directory: Union[str, Path],
        days_old: int = 30,
        file_pattern: str = "*",
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Clean old files using utility framework.
        
        Args:
            directory: Directory to clean
            days_old: Files older than this many days
            file_pattern: File pattern to match
            dry_run: Whether to perform dry run
            
        Returns:
            Cleanup results
        """
        try:
            with self.performance_tracker.track_block("clean_old_files"):
                cutoff_date = datetime.now() - timedelta(days=days_old)
                
                # Find old files using BaseFileManager
                old_files = self.base_file_manager.find_files(
                    directory,
                    pattern=file_pattern,
                    modified_before=cutoff_date
                )
                
                cleanup_results = {
                    'files_found': len(old_files),
                    'total_size_mb': sum(f.size_bytes for f in old_files) / (1024 * 1024),
                    'files_removed': 0,
                    'space_freed_mb': 0,
                    'dry_run': dry_run
                }
                
                if not dry_run:
                    for file_info in old_files:
                        try:
                            file_info.path.unlink()
                            cleanup_results['files_removed'] += 1
                            cleanup_results['space_freed_mb'] += file_info.size_bytes / (1024 * 1024)
                        except Exception as e:
                            self.logger.warning(f"Could not remove {file_info.path}: {str(e)}")
                
                action = "Would remove" if dry_run else "Removed"
                self.logger.info(
                    f"{action} {cleanup_results['files_removed']} files, "
                    f"freed {cleanup_results['space_freed_mb']:.2f} MB"
                )
                
                return cleanup_results
                
        except Exception as e:
            self.logger.error(f"Error cleaning old files: {str(e)}")
            return {}

    def _load_data_by_type(self, file_path: Union[str, Path], file_type: str) -> pd.DataFrame:
        """Load data based on detected file type."""
        try:
            if file_type in ['csv', 'text']:
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    for sep in [',', ';', '\t']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if len(df.columns) > 1:  # Successful parsing
                                return df
                        except:
                            continue
                # Fallback
                return pd.read_csv(file_path)
                
            elif file_type == 'excel':
                return pd.read_excel(file_path)
                
            elif file_type == 'json':
                return pd.read_json(file_path)
                
            else:
                # Default to CSV
                return pd.read_csv(file_path)
                
        except Exception as e:
            self.logger.error(f"Error loading data by type: {str(e)}")
            raise

    def _detect_seo_schema_type(self, df: pd.DataFrame) -> str:
        """Detect SEO schema type from DataFrame columns."""
        try:
            columns = set(df.columns)
            
            # Check for different SEO data types
            if {'Keyword', 'Position'}.issubset(columns):
                return 'positions'
            elif {'Domain', 'Organic Keywords'}.issubset(columns):
                return 'competitors'
            elif {'Keyword', 'Volume'}.issubset(columns) and any('.' in col for col in columns):
                return 'gap_keywords'
            else:
                return 'generic'
                
        except Exception:
            return 'generic'

    def _validate_seo_data(self, df: pd.DataFrame, schema_type: str):
        """Validate SEO data using SchemaValidator."""
        try:
            # Use the data validator for SEO-specific validation
            return self.data_validator.validate_seo_dataset(df, schema_type)
        except Exception as e:
            self.logger.error(f"Error validating SEO data: {str(e)}")
            return None

    def _extract_date_from_filename(self, filename: str, date_pattern: str) -> Optional[datetime]:
        """Extract date from filename using DateHelper."""
        try:
            # Look for date patterns in filename
            import re
            
            # Common date patterns
            patterns = [
                r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
                r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
                r'(\d{2})-(\d{2})-(\d{4})',  # MM-DD-YYYY
                r'(\d{2})(\d{2})(\d{4})',   # MMDDYYYY
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    groups = match.groups()
                    if len(groups) == 3:
                        # Try different date formats
                        date_string = f"{groups[0]}-{groups[1]}-{groups[2]}"
                        parsed_date = DateHelper.parse_flexible_date(date_string)
                        if parsed_date:
                            return parsed_date
            
            return None
            
        except Exception:
            return None

    def _get_backup_directory(self) -> Path:
        """Get backup directory using PathManager."""
        try:
            return self.path_manager.get_data_path("backups")
        except Exception:
            return Path("backups")

    def get_file_statistics(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file statistics using BaseFileManager."""
        try:
            files = self.base_file_manager.find_files(directory, recursive=True)
            
            if not files:
                return {}
            
            # Calculate statistics using our utilities
            sizes = [f.size_bytes for f in files]
            total_size = sum(sizes)
            
            # Use StringHelper for file type analysis
            file_types = {}
            for file_info in files:
                file_type = file_info.file_type
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            statistics = {
                'total_files': len(files),
                'total_size_mb': total_size / (1024 * 1024),
                'average_size_mb': (total_size / len(files)) / (1024 * 1024),
                'largest_file_mb': max(sizes) / (1024 * 1024),
                'smallest_file_mb': min(sizes) / (1024 * 1024),
                'file_types': file_types,
                'oldest_file': min(files, key=lambda f: f.modified_time).modified_time,
                'newest_file': max(files, key=lambda f: f.modified_time).modified_time
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error getting file statistics: {str(e)}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for file operations."""
        return self.audit_logger.get_audit_trail(time_window_hours=hours)
