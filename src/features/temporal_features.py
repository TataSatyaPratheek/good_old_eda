"""
Temporal Features Module for SEO Competitive Intelligence
Advanced temporal feature engineering leveraging the comprehensive utility framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Import our utilities to eliminate ALL redundancy
from src.utils.common_helpers import StringHelper, DateHelper, memoize, timing_decorator, safe_divide, ensure_list
from src.utils.data_utils import DataProcessor, DataValidator, DataTransformer
from src.utils.math_utils import StatisticalCalculator, OptimizationHelper, TimeSeriesAnalyzer
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager, PathManager
from src.utils.validation_utils import SchemaValidator, BusinessRuleValidator
from src.utils.export_utils import ReportExporter, DataExporter
from src.utils.file_utils import FileManager

@dataclass
class TemporalFeatureConfig:
    """Configuration for temporal feature engineering"""
    lag_periods: List[int] = None
    rolling_windows: List[int] = None
    seasonal_periods: List[int] = None
    trend_analysis: bool = True
    cyclical_analysis: bool = True
    change_point_detection: bool = True
    fourier_features: bool = False
    holiday_features: bool = False
    business_calendar_features: bool = True

@dataclass
class TemporalAnalysisResult:
    """Result of temporal analysis"""
    trend_components: pd.DataFrame
    seasonal_components: pd.DataFrame
    cyclical_patterns: Dict[str, Any]
    change_points: List[datetime]
    volatility_metrics: Dict[str, float]
    forecasting_features: pd.DataFrame
    temporal_statistics: Dict[str, Any]

@dataclass
class TemporalFeatureResult:
    """Result of temporal feature engineering"""
    engineered_features: pd.DataFrame
    feature_metadata: Dict[str, Any]
    temporal_analysis: TemporalAnalysisResult
    feature_importance: Dict[str, float]
    processing_summary: Dict[str, Any]

class TemporalFeatureEngineer:
    """
    Advanced temporal feature engineering for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    temporal feature engineering capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("temporal_features")
        self.config = config_manager or ConfigManager()
        
        # Initialize utility classes - eliminate ALL redundancy
        self.data_processor = DataProcessor(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.data_transformer = DataTransformer(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.optimization_helper = OptimizationHelper(self.logger)
        self.time_series_analyzer = TimeSeriesAnalyzer(self.logger)
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        self.schema_validator = SchemaValidator(self.logger)
        self.business_rule_validator = BusinessRuleValidator(self.logger)
        self.report_exporter = ReportExporter(self.logger)
        self.data_exporter = DataExporter(self.logger)
        self.file_manager = FileManager(self.logger)
        self.path_manager = PathManager(config_manager=self.config)
        
        # Load temporal feature configurations
        analysis_config = self.config.get_analysis_config()
        self.default_config = TemporalFeatureConfig(
            lag_periods=[1, 7, 14, 30],
            rolling_windows=[7, 14, 30, 90],
            seasonal_periods=[7, 30, 90, 365]
        )

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def comprehensive_temporal_feature_engineering(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        config: Optional[TemporalFeatureConfig] = None,
        entity_column: Optional[str] = None
    ) -> TemporalFeatureResult:
        """
        Perform comprehensive temporal feature engineering using utility framework.
        
        Args:
            df: Input DataFrame with temporal data
            date_column: Column containing date/time information
            value_columns: Columns to create temporal features for
            config: Temporal feature configuration
            entity_column: Column for grouping entities (e.g., keywords, competitors)
            
        Returns:
            TemporalFeatureResult with comprehensive temporal features
        """
        try:
            with self.performance_tracker.track_block("comprehensive_temporal_feature_engineering"):
                # Audit log the operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="temporal_feature_engineering",
                    parameters={
                        "rows": len(df),
                        "date_column": date_column,
                        "value_columns":
