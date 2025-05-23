"""
Main Data Loader Module for SEO Competitive Intelligence
Orchestrates file loading, validation, and merging operations using utility framework
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import date, datetime, timedelta
from dataclasses import dataclass

# Import our utilities to eliminate redundancy
from src.utils.common_helpers import DateHelper, memoize, timing_decorator, ensure_list
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager, PathManager
from src.utils.file_utils import FileManager as BaseFileManager

# Import refactored modules
from .file_manager import FileManager
from .schema_validator import SchemaValidator, SEOSchemaValidationResult
from .merge_strategy import MergeStrategy, MergeResult, GapAnalysis

@dataclass
class DataLoadSummary:
    """Summary of data loading operation"""
    total_files_found: int
    files_loaded_successfully: int
    files_failed: int
    total_records: int
    data_types_loaded: List[str]
    date_range: Tuple[Optional[date], Optional[date]]
    average_quality_score: float
    loading_time_seconds: float
    critical_issues_found: int

class SEMrushDataLoader:
    """
    Main orchestrator for SEMrush data loading operations.
    
    Leverages the comprehensive utility framework and refactored components
    to provide high-level interface for loading, validating, and merging data.
    """
    
    def __init__(
        self,
        base_data_path: Union[str, Path],
        logger=None,
        config_manager=None
    ):
        """Initialize the data loader with utilities and refactored components."""
        self.logger = logger or LoggerFactory.get_logger("semrush_data_loader")
        self.config = config_manager or ConfigManager()
        
        # Initialize path management
        self.path_manager = PathManager(config_manager=self.config)
        self.base_path = Path(base_data_path)
        
        # Initialize utility classes
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        self.data_processor = DataProcessor(self.logger)
        self.data_validator = DataValidator(self.logger)
        
        # Initialize refactored components
        self.file_manager = FileManager(self.logger, self.config)
        self.schema_validator = SchemaValidator(self.logger, self.config)
        self.merge_strategy = MergeStrategy(self.logger, self.config)
        
        # Data cache using utilities
        self._data_cache = {}
        self._quality_reports = {}
        
        # Load configuration
        analysis_config = self.config.get_analysis_config()
        self.max_cache_size = getattr(analysis_config, 'max_cache_size', 1000)
        self.cache_ttl = getattr(analysis_config, 'cache_ttl_minutes', 60) * 60

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def load_all_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        validate_schema: bool = True,
        cache_data: bool = True,
        clean_data: bool = True
    ) -> Dict[str, Dict[date, pd.DataFrame]]:
        """
        Load all available data organized by type and date using utility framework.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            validate_schema: Whether to perform schema validation
            cache_data: Whether to cache loaded data
            clean_data: Whether to clean loaded data
            
        Returns:
            Nested dict: {data_type: {date: DataFrame}}
        """
        try:
            with self.performance_tracker.track_block("load_all_data"):
                # Audit log the operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="data_loading",
                    parameters={
                        "start_date": str(start_date) if start_date else None,
                        "end_date": str(end_date) if end_date else None,
                        "validate_schema": validate_schema,
                        "cache_data": cache_data
                    }
                )
                
                # Discover files using utility framework
                files_by_date = self._discover_data_files(start_date, end_date)
                
                if not files_by_date:
                    self.logger.warning("No data files found in specified date range")
                    return {}
                
                # Organize by type
                files_by_type = self._organize_files_by_type(files_by_date)
                
                # Load data for each type and date
                loaded_data = {}
                total_files = 0
                successful_loads = 0
                
                for file_type, date_files in files_by_type.items():
                    loaded_data[file_type] = {}
                    
                    for file_date, file_path in date_files.items():
                        total_files += 1
                        
                        # Check cache first
                        cache_key = f"{file_type}_{file_date}"
                        if cache_data and cache_key in self._data_cache:
                            loaded_data[file_type][file_date] = self._data_cache[cache_key]
                            successful_loads += 1
                            continue
                        
                        # Load data using refactored FileManager
                        try:
                            df, metadata = self.file_manager.load_seo_data_file(
                                file_path,
                                validate_data=validate_schema,
                                clean_data=clean_data
                            )
                            
                            if not df.empty:
                                loaded_data[file_type][file_date] = df
                                successful_loads += 1
                                
                                # Store quality report
                                if 'validation_report' in metadata:
                                    self._quality_reports[str(file_path)] = metadata['validation_report']
                                
                                # Cache if requested
                                if cache_data:
                                    self._cache_data(cache_key, df)
                                
                        except Exception as e:
                            self.logger.error(f"Failed to load {file_path}: {str(e)}")
                            continue
                
                self.logger.info(f"Successfully loaded {successful_loads}/{total_files} files across {len(loaded_data)} data types")
                
                # Audit log success
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="data_loading",
                    result="success",
                    details={
                        "files_loaded": successful_loads,
                        "total_files": total_files,
                        "data_types": list(loaded_data.keys())
                    }
                )
                
                return loaded_data
                
        except Exception as e:
            self.logger.error(f"Error loading all data: {str(e)}")
            self.audit_logger.log_analysis_execution(
                user_id="system",
                analysis_type="data_loading",
                result="failure",
                details={"error": str(e)}
            )
            return {}

    @timing_decorator()
    def load_competitor_positions(
        self,
        competitors: List[str] = ['lenovo', 'hp', 'dell'],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        merge_temporal_data: bool = True,
        merge_strategy: str = 'intelligent'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load and merge position data for specified competitors using MergeStrategy.
        
        Args:
            competitors: List of competitor domains to load
            start_date: Optional start date filter
            end_date: Optional end date filter
            merge_temporal_data: Whether to merge temporal data
            merge_strategy: Strategy for merging ('intelligent', 'union', 'intersection')
            
        Returns:
            Dict mapping competitor names to merged DataFrames
        """
        try:
            with self.performance_tracker.track_block("load_competitor_positions"):
                # Load all data first
                all_data = self.load_all_data(start_date, end_date)
                competitor_data = {}
                
                for competitor in competitors:
                    type_key = f"positions_{competitor}"
                    
                    if type_key in all_data and all_data[type_key]:
                        # Collect temporal data for this competitor
                        temporal_datasets = all_data[type_key]
                        
                        if merge_temporal_data and len(temporal_datasets) > 1:
                            # Use MergeStrategy to merge temporal data
                            primary_date = max(temporal_datasets.keys())
                            primary_dataset = temporal_datasets[primary_date]
                            
                            # Prepare secondary datasets (older dates)
                            secondary_datasets = {
                                str(date_key): df for date_key, df in temporal_datasets.items()
                                if date_key != primary_date
                            }
                            
                            if secondary_datasets:
                                merge_result = self.merge_strategy.intelligent_data_merge(
                                    primary_dataset,
                                    secondary_datasets,
                                    merge_strategy=merge_strategy,
                                    conflict_resolution='prioritize_quality'
                                )
                                merged_df = merge_result.merged_data
                            else:
                                merged_df = primary_dataset
                        else:
                            # Use most recent data if no merging needed
                            latest_date = max(temporal_datasets.keys())
                            merged_df = temporal_datasets[latest_date]
                        
                        if not merged_df.empty:
                            competitor_data[competitor] = merged_df
                            self.logger.info(f"Loaded {len(merged_df)} rows for {competitor}")
                
                return competitor_data
                
        except Exception as e:
            self.logger.error(f"Error loading competitor positions: {str(e)}")
            return {}

    @timing_decorator()
    def get_comprehensive_data_quality_summary(self) -> pd.DataFrame:
        """
        Get comprehensive data quality summary using SchemaValidator.
        
        Returns:
            DataFrame with enhanced quality metrics by file
        """
        try:
            if not self._quality_reports:
                self.logger.warning("No quality reports available - load data first")
                return pd.DataFrame()
            
            quality_summary = []
            
            for file_name, report in self._quality_reports.items():
                # Handle both old and new validation report formats
                if hasattr(report, 'base_validation_report'):
                    # New SEOSchemaValidationResult format
                    base_report = report.base_validation_report
                    summary = {
                        'file_name': file_name,
                        'total_rows': base_report.total_records,
                        'quality_score': base_report.overall_score,
                        'critical_issues': base_report.critical_issues,
                        'warnings': base_report.warnings,
                        'seo_specific_issues': len(report.seo_specific_issues),
                        'business_rule_violations': len(report.business_rule_violations),
                        'data_quality_metrics': report.data_quality_metrics,
                        'overall_data_quality': report.data_quality_metrics.get('overall_quality', 0),
                        'has_critical_issues': base_report.critical_issues > 0,
                        'validation_timestamp': base_report.validation_timestamp
                    }
                else:
                    # Old format compatibility
                    summary = {
                        'file_name': file_name,
                        'total_rows': getattr(report, 'total_records', 0),
                        'quality_score': getattr(report, 'quality_score', 0),
                        'critical_issues': getattr(report, 'critical_issues', 0),
                        'warnings': getattr(report, 'warnings', 0),
                        'seo_specific_issues': 0,
                        'business_rule_violations': 0,
                        'data_quality_metrics': {},
                        'overall_data_quality': getattr(report, 'quality_score', 0),
                        'has_critical_issues': getattr(report, 'critical_issues', 0) > 0,
                        'validation_timestamp': datetime.now()
                    }
                
                quality_summary.append(summary)
            
            summary_df = pd.DataFrame(quality_summary)
            summary_df = summary_df.sort_values('quality_score', ascending=False)
            
            # Add summary statistics
            if not summary_df.empty:
                avg_quality = summary_df['quality_score'].mean()
                critical_files = summary_df['has_critical_issues'].sum()
                
                self.logger.info(
                    f"Data quality summary: {avg_quality:.2f} average quality, "
                    f"{critical_files} files with critical issues"
                )
            
            return summary_df
            
        except Exception as e:
            self.logger.error(f"Error creating quality summary: {str(e)}")
            return pd.DataFrame()

    @timing_decorator()
    def perform_comprehensive_gap_analysis(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        analysis_depth: str = 'comprehensive'
    ) -> GapAnalysis:
        """
        Perform comprehensive gap analysis using MergeStrategy.
        
        Args:
            lenovo_data: Lenovo's position data
            competitor_data: Dictionary of competitor position data
            analysis_depth: Analysis depth level
            
        Returns:
            GapAnalysis with comprehensive findings
        """
        try:
            return self.merge_strategy.comprehensive_gap_analysis(
                lenovo_data, competitor_data, analysis_depth
            )
            
        except Exception as e:
            self.logger.error(f"Error in gap analysis: {str(e)}")
            return GapAnalysis(pd.DataFrame(), {}, pd.DataFrame(), [], [], {})

    @timing_decorator()
    def get_data_load_summary(self) -> DataLoadSummary:
        """
        Get comprehensive data loading summary.
        
        Returns:
            DataLoadSummary with detailed loading metrics
        """
        try:
            # Get performance metrics
            perf_summary = self.performance_tracker.get_performance_summary()
            
            # Calculate summary metrics
            total_files = len(self._quality_reports)
            successful_files = sum(
                1 for report in self._quality_reports.values()
                if getattr(report, 'quality_score', 0) > 0
            )
            failed_files = total_files - successful_files
            
            # Calculate total records
            total_records = sum(
                getattr(report, 'total_records', 0)
                for report in self._quality_reports.values()
            )
            
            # Calculate average quality score
            quality_scores = [
                getattr(report, 'quality_score', 0)
                for report in self._quality_reports.values()
                if getattr(report, 'quality_score', 0) > 0
            ]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Count critical issues
            critical_issues = sum(
                getattr(report, 'critical_issues', 0)
                for report in self._quality_reports.values()
            )
            
            # Determine date range
            date_range = self.get_available_date_range()
            
            # Get data types loaded
            data_types = list(set(
                self._extract_data_type_from_filename(filename)
                for filename in self._quality_reports.keys()
            ))
            
            summary = DataLoadSummary(
                total_files_found=total_files,
                files_loaded_successfully=successful_files,
                files_failed=failed_files,
                total_records=total_records,
                data_types_loaded=data_types,
                date_range=date_range,
                average_quality_score=avg_quality,
                loading_time_seconds=perf_summary.get('total_execution_time', 0),
                critical_issues_found=critical_issues
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating data load summary: {str(e)}")
            return DataLoadSummary(0, 0, 0, 0, [], (None, None), 0.0, 0.0, 0)

    def export_loaded_data(
        self,
        export_directory: Union[str, Path],
        include_quality_reports: bool = True,
        export_format: str = 'xlsx'
    ) -> Dict[str, bool]:
        """
        Export all loaded data using FileManager export capabilities.
        
        Args:
            export_directory: Directory to export data
            include_quality_reports: Whether to include quality reports
            export_format: Export format
            
        Returns:
            Dictionary of export results
        """
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare data for export
            export_results = {}
            
            # Export cached data
            if self._data_cache:
                datasets_to_export = {}
                for cache_key, df in self._data_cache.items():
                    datasets_to_export[cache_key] = df
                
                # Use FileManager to export
                export_success = self.file_manager.save_processed_data(
                    pd.concat(datasets_to_export.values(), keys=datasets_to_export.keys()),
                    export_path / f"loaded_data.{export_format}",
                    format=export_format,
                    include_metadata=True
                )
                export_results['loaded_data'] = export_success
            
            # Export quality reports if requested
            if include_quality_reports:
                quality_summary = self.get_comprehensive_data_quality_summary()
                if not quality_summary.empty:
                    quality_export_success = self.file_manager.save_processed_data(
                        quality_summary,
                        export_path / f"quality_summary.{export_format}",
                        format=export_format
                    )
                    export_results['quality_summary'] = quality_export_success
            
            return export_results
            
        except Exception as e:
            self.logger.error(f"Error exporting loaded data: {str(e)}")
            return {}

    def _discover_data_files(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[date, List[Path]]:
        """Discover data files using utility framework."""
        try:
            # Use BaseFileManager to find files
            base_file_manager = BaseFileManager(self.logger)
            files = base_file_manager.find_files(
                self.base_path,
                pattern="*.xlsx",
                recursive=True
            )
            
            files_by_date = {}
            
            for file_info in files:
                # Extract date from filename using DateHelper
                file_date = self._extract_date_from_filename(file_info.path.name)
                
                if file_date:
                    file_date_obj = file_date.date()
                    
                    # Apply date filters
                    if start_date and file_date_obj < start_date:
                        continue
                    if end_date and file_date_obj > end_date:
                        continue
                    
                    if file_date_obj not in files_by_date:
                        files_by_date[file_date_obj] = []
                    
                    files_by_date[file_date_obj].append(file_info.path)
            
            return files_by_date
            
        except Exception as e:
            self.logger.error(f"Error discovering data files: {str(e)}")
            return {}

    def _organize_files_by_type(
        self,
        files_by_date: Dict[date, List[Path]]
    ) -> Dict[str, Dict[date, Path]]:
        """Organize files by type and date."""
        try:
            files_by_type = {}
            
            for file_date, file_paths in files_by_date.items():
                for file_path in file_paths:
                    file_type = self._extract_data_type_from_filename(file_path.name)
                    
                    if file_type not in files_by_type:
                        files_by_type[file_type] = {}
                    
                    files_by_type[file_type][file_date] = file_path
            
            return files_by_type
            
        except Exception as e:
            self.logger.error(f"Error organizing files by type: {str(e)}")
            return {}

    def _extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from filename using DateHelper."""
        return DateHelper.parse_flexible_date(filename)

    def _extract_data_type_from_filename(self, filename: str) -> str:
        """Extract data type from filename."""
        filename_lower = filename.lower()
        
        if 'position' in filename_lower:
            if 'lenovo' in filename_lower:
                return 'positions_lenovo'
            elif 'hp' in filename_lower:
                return 'positions_hp'
            elif 'dell' in filename_lower:
                return 'positions_dell'
            else:
                return 'positions'
        elif 'competitor' in filename_lower:
            return 'competitors'
        elif 'gap' in filename_lower or 'keyword' in filename_lower:
            return 'gap_keywords'
        else:
            return 'unknown'

    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with size management."""
        try:
            # Simple cache size management
            if len(self._data_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._data_cache))
                del self._data_cache[oldest_key]
            
            self._data_cache[cache_key] = data
            
        except Exception as e:
            self.logger.error(f"Error caching data: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the data cache and quality reports."""
        self._data_cache.clear()
        self._quality_reports.clear()
        self.logger.info("Data cache and quality reports cleared")

    def get_available_date_range(self) -> Tuple[Optional[date], Optional[date]]:
        """Get the available date range in the data using utility framework."""
        try:
            files_by_date = self._discover_data_files()
            
            if not files_by_date:
                return None, None
            
            dates = list(files_by_date.keys())
            return min(dates), max(dates)
            
        except Exception as e:
            self.logger.error(f"Error getting date range: {str(e)}")
            return None, None

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for data loading operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    def validate_all_loaded_data(self) -> Dict[str, SEOSchemaValidationResult]:
        """
        Validate all loaded data using SchemaValidator.
        
        Returns:
            Dictionary of validation results by dataset
        """
        try:
            validation_results = {}
            
            for cache_key, df in self._data_cache.items():
                # Determine dataset type from cache key
                dataset_type = 'positions'  # Default
                if 'competitor' in cache_key:
                    dataset_type = 'competitors'
                elif 'gap' in cache_key:
                    dataset_type = 'gap_keywords'
                
                # Validate using SchemaValidator
                validation_result = self.schema_validator.validate_seo_dataset(
                    df,
                    dataset_type=dataset_type,
                    include_business_rules=True,
                    include_quality_metrics=True
                )
                
                validation_results[cache_key] = validation_result
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating loaded data: {str(e)}")
            return {}
