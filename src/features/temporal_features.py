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
                        "value_columns": value_columns,
                        "entity_column": entity_column
                    }
                )

                self.logger.info(f"Starting temporal feature engineering for {len(df)} records")

                # Use config or defaults
                feature_config = config or self.default_config

                # Phase 1: Data Preparation and Validation
                prepared_data = self._prepare_temporal_data(df, date_column, value_columns, entity_column)

                # Phase 2: Create Basic Temporal Features
                basic_features = self._create_basic_temporal_features(
                    prepared_data, date_column, value_columns, feature_config
                )

                # Phase 3: Create Lag Features
                lag_features = self._create_lag_features(
                    basic_features, value_columns, feature_config.lag_periods, entity_column
                )

                # Phase 4: Create Rolling Window Features
                rolling_features = self._create_rolling_window_features(
                    lag_features, value_columns, feature_config.rolling_windows, entity_column
                )

                # Phase 5: Create Seasonal Features
                seasonal_features = self._create_seasonal_features(
                    rolling_features, date_column, value_columns, feature_config.seasonal_periods, entity_column
                )

                # Phase 6: Trend Analysis
                trend_analysis = None
                if feature_config.trend_analysis:
                    trend_analysis = self._perform_trend_analysis(
                        seasonal_features, date_column, value_columns, entity_column
                    )

                # Phase 7: Change Point Detection
                change_points = []
                if feature_config.change_point_detection:
                    change_points = self._detect_change_points(
                        seasonal_features, date_column, value_columns, entity_column
                    )

                # Phase 8: Cyclical Pattern Analysis
                cyclical_patterns = {}
                if feature_config.cyclical_analysis:
                    cyclical_patterns = self._analyze_cyclical_patterns(
                        seasonal_features, date_column, value_columns, entity_column
                    )

                # Phase 9: Create Advanced Features
                advanced_features = self._create_advanced_temporal_features(
                    seasonal_features, trend_analysis, change_points, cyclical_patterns, feature_config
                )

                # Phase 10: Feature Importance Analysis
                feature_importance = self._analyze_feature_importance(
                    advanced_features, value_columns
                )

                # Phase 11: Create Temporal Analysis Result
                temporal_analysis = self._create_temporal_analysis_result(
                    trend_analysis, cyclical_patterns, change_points, advanced_features
                )

                # Phase 12: Export and Validation
                processing_summary = self._create_processing_summary(
                    df, advanced_features, feature_config, temporal_analysis
                )

                # Create comprehensive result
                result = TemporalFeatureResult(
                    engineered_features=advanced_features,
                    feature_metadata=self._create_feature_metadata(advanced_features, feature_config),
                    temporal_analysis=temporal_analysis,
                    feature_importance=feature_importance,
                    processing_summary=processing_summary
                )

                self.logger.info(f"Temporal feature engineering completed: {advanced_features.shape}")
                return result

        except Exception as e:
            self.logger.error(f"Error in temporal feature engineering: {str(e)}")
            return TemporalFeatureResult(
                engineered_features=pd.DataFrame(),
                feature_metadata={},
                temporal_analysis=TemporalAnalysisResult(
                    pd.DataFrame(), pd.DataFrame(), {}, [], {}, pd.DataFrame(), {}
                ),
                feature_importance={},
                processing_summary={"error": str(e)}
            )

    def _prepare_temporal_data(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        entity_column: Optional[str]
    ) -> pd.DataFrame:
        """Prepare and validate temporal data using utility framework"""
        try:
            # Clean data using data processor
            cleaned_df = self.data_processor.clean_seo_data(df)

            # Validate data quality
            validation_result = self.data_validator.validate_seo_dataset(cleaned_df, 'positions')
            if validation_result.quality_score < 0.7:
                self.logger.warning(f"Low data quality detected: {validation_result.quality_score:.3f}")

            # Convert date column using DateHelper
            cleaned_df[date_column] = cleaned_df[date_column].apply(
                lambda x: DateHelper.parse_flexible_date(x)
            )

            # Remove rows with invalid dates
            cleaned_df = cleaned_df.dropna(subset=[date_column])

            # Sort by date and entity if applicable
            sort_columns = [date_column]
            if entity_column and entity_column in cleaned_df.columns:
                sort_columns.insert(0, entity_column)

            cleaned_df = cleaned_df.sort_values(sort_columns).reset_index(drop=True)

            self.logger.info(f"Data preparation completed: {len(cleaned_df)} records")
            return cleaned_df

        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            return df

    def _create_basic_temporal_features(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        config: TemporalFeatureConfig
    ) -> pd.DataFrame:
        """Create basic temporal features using DataTransformer"""
        try:
            # Use DataTransformer to create time features
            features_df = self.data_transformer.create_time_features(df, date_column)

            # Add day of year, week of year, etc.
            features_df[f'{date_column}_dayofyear'] = features_df[date_column].dt.dayofyear
            features_df[f'{date_column}_weekofyear'] = features_df[date_column].dt.isocalendar().week
            features_df[f'{date_column}_is_weekend'] = features_df[date_column].dt.weekday >= 5
            features_df[f'{date_column}_is_month_start'] = features_df[date_column].dt.is_month_start
            features_df[f'{date_column}_is_month_end'] = features_df[date_column].dt.is_month_end

            # Business calendar features
            if config.business_calendar_features:
                features_df = self._add_business_calendar_features(features_df, date_column)

            self.logger.info("Basic temporal features created")
            return features_df

        except Exception as e:
            self.logger.error(f"Error creating basic temporal features: {str(e)}")
            return df

    def _create_lag_features(
        self,
        df: pd.DataFrame,
        value_columns: List[str],
        lag_periods: List[int],
        entity_column: Optional[str]
    ) -> pd.DataFrame:
        """Create lag features for specified periods"""
        try:
            features_df = df.copy()

            for value_col in value_columns:
                if value_col not in df.columns:
                    continue

                for lag in lag_periods:
                    lag_col_name = f"{value_col}_lag_{lag}"

                    if entity_column:
                        # Create lags within each entity group
                        features_df[lag_col_name] = features_df.groupby(entity_column)[value_col].shift(lag)
                    else:
                        # Create lags across entire dataset
                        features_df[lag_col_name] = features_df[value_col].shift(lag)

            self.logger.info(f"Lag features created for {len(lag_periods)} periods")
            return features_df

        except Exception as e:
            self.logger.error(f"Error creating lag features: {str(e)}")
            return df

    def _create_rolling_window_features(
        self,
        df: pd.DataFrame,
        value_columns: List[str],
        rolling_windows: List[int],
        entity_column: Optional[str]
    ) -> pd.DataFrame:
        """Create rolling window statistical features"""
        try:
            features_df = df.copy()

            for value_col in value_columns:
                if value_col not in df.columns:
                    continue

                for window in rolling_windows:
                    base_name = f"{value_col}_rolling_{window}"

                    if entity_column:
                        # Rolling features within each entity
                        rolling_group = features_df.groupby(entity_column)[value_col].rolling(window=window, min_periods=1)
                        features_df[f"{base_name}_mean"] = rolling_group.mean().reset_index(0, drop=True)
                        features_df[f"{base_name}_std"] = rolling_group.std().reset_index(0, drop=True)
                        features_df[f"{base_name}_min"] = rolling_group.min().reset_index(0, drop=True)
                        features_df[f"{base_name}_max"] = rolling_group.max().reset_index(0, drop=True)
                        features_df[f"{base_name}_median"] = rolling_group.median().reset_index(0, drop=True)
                    else:
                        # Rolling features across entire dataset
                        rolling_series = features_df[value_col].rolling(window=window, min_periods=1)
                        features_df[f"{base_name}_mean"] = rolling_series.mean()
                        features_df[f"{base_name}_std"] = rolling_series.std()
                        features_df[f"{base_name}_min"] = rolling_series.min()
                        features_df[f"{base_name}_max"] = rolling_series.max()
                        features_df[f"{base_name}_median"] = rolling_series.median()

                    # Add rate of change features
                    mean_col = f"{base_name}_mean"
                    if mean_col in features_df.columns:
                        features_df[f"{base_name}_change"] = features_df[mean_col].pct_change()

            self.logger.info(f"Rolling window features created for {len(rolling_windows)} windows")
            return features_df

        except Exception as e:
            self.logger.error(f"Error creating rolling window features: {str(e)}")
            return df

    def _create_seasonal_features(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        seasonal_periods: List[int],
        entity_column: Optional[str]
    ) -> pd.DataFrame:
        """Create seasonal decomposition features using TimeSeriesAnalyzer"""
        try:
            features_df = df.copy()

            for value_col in value_columns:
                if value_col not in df.columns:
                    continue

                for period in seasonal_periods:
                    # Use TimeSeriesAnalyzer for seasonal decomposition
                    if entity_column:
                        # Process each entity separately
                        seasonal_features = []
                        for entity, group in features_df.groupby(entity_column):
                            if len(group) >= period * 2:  # Need sufficient data
                                decomposition = self.time_series_analyzer.decompose_time_series(
                                    group.set_index(date_column)[value_col], period=period
                                )
                                group_features = group.copy()
                                if 'trend' in decomposition:
                                    group_features[f"{value_col}_trend_{period}"] = decomposition['trend'].reindex(group_features.index)
                                if 'seasonal' in decomposition:
                                    group_features[f"{value_col}_seasonal_{period}"] = decomposition['seasonal'].reindex(group_features.index)
                                seasonal_features.append(group_features)
                        
                        if seasonal_features:
                            features_df = pd.concat(seasonal_features, ignore_index=True)
                    else:
                        # Process entire dataset
                        if len(features_df) >= period * 2:
                            decomposition = self.time_series_analyzer.decompose_time_series(
                                features_df.set_index(date_column)[value_col], period=period
                            )
                            if 'trend' in decomposition:
                                features_df[f"{value_col}_trend_{period}"] = decomposition['trend'].values
                            if 'seasonal' in decomposition:
                                features_df[f"{value_col}_seasonal_{period}"] = decomposition['seasonal'].values

            self.logger.info(f"Seasonal features created for {len(seasonal_periods)} periods")
            return features_df

        except Exception as e:
            self.logger.error(f"Error creating seasonal features: {str(e)}")
            return df

    def _perform_trend_analysis(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        entity_column: Optional[str]
    ) -> Dict[str, Any]:
        """Perform comprehensive trend analysis using TimeSeriesAnalyzer"""
        try:
            trend_results = {}

            for value_col in value_columns:
                if value_col not in df.columns:
                    continue

                if entity_column:
                    # Analyze trends for each entity
                    entity_trends = {}
                    for entity, group in df.groupby(entity_column):
                        if len(group) > 10:  # Need sufficient data
                            trend_model = self.time_series_analyzer.fit_trend_model(
                                group.set_index(date_column)[value_col], model_type='linear'
                            )
                            entity_trends[entity] = trend_model
                    trend_results[value_col] = entity_trends
                else:
                    # Analyze trend for entire dataset
                    trend_model = self.time_series_analyzer.fit_trend_model(
                        df.set_index(date_column)[value_col], model_type='linear'
                    )
                    trend_results[value_col] = trend_model

            self.logger.info("Trend analysis completed")
            return trend_results

        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return {}

    def _detect_change_points(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        entity_column: Optional[str]
    ) -> List[datetime]:
        """Detect change points using TimeSeriesAnalyzer"""
        try:
            all_change_points = []

            for value_col in value_columns:
                if value_col not in df.columns:
                    continue

                if entity_column:
                    # Detect change points for each entity
                    for entity, group in df.groupby(entity_column):
                        if len(group) > 20:  # Need sufficient data
                            change_points = self.time_series_analyzer.detect_changepoints(
                                group.set_index(date_column)[value_col]
                            )
                            # Convert indices to dates
                            if change_points:
                                dates = group[date_column].iloc[change_points].tolist()
                                all_change_points.extend(dates)
                else:
                    # Detect change points for entire dataset
                    change_points = self.time_series_analyzer.detect_changepoints(
                        df.set_index(date_column)[value_col]
                    )
                    if change_points:
                        dates = df[date_column].iloc[change_points].tolist()
                        all_change_points.extend(dates)

            self.logger.info(f"Change point detection completed: {len(all_change_points)} points found")
            return list(set(all_change_points))  # Remove duplicates

        except Exception as e:
            self.logger.error(f"Error in change point detection: {str(e)}")
            return []

    def _analyze_cyclical_patterns(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        entity_column: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze cyclical patterns using TimeSeriesAnalyzer"""
        try:
            cyclical_patterns = {}

            for value_col in value_columns:
                if value_col not in df.columns:
                    continue

                patterns = {}
                
                if entity_column:
                    # Analyze patterns for each entity
                    for entity, group in df.groupby(entity_column):
                        if len(group) > 30:  # Need sufficient data
                            autocorr, lags = self.time_series_analyzer.calculate_autocorrelation(
                                group.set_index(date_column)[value_col]
                            )
                            patterns[entity] = {
                                'autocorrelation': autocorr,
                                'lags': lags,
                                'dominant_cycle': lags[np.argmax(autocorr[1:])] + 1 if len(autocorr) > 1 else None
                            }
                else:
                    # Analyze patterns for entire dataset
                    autocorr, lags = self.time_series_analyzer.calculate_autocorrelation(
                        df.set_index(date_column)[value_col]
                    )
                    patterns['overall'] = {
                        'autocorrelation': autocorr,
                        'lags': lags,
                        'dominant_cycle': lags[np.argmax(autocorr[1:])] + 1 if len(autocorr) > 1 else None
                    }

                cyclical_patterns[value_col] = patterns

            self.logger.info("Cyclical pattern analysis completed")
            return cyclical_patterns

        except Exception as e:
            self.logger.error(f"Error in cyclical pattern analysis: {str(e)}")
            return {}

    def _create_advanced_temporal_features(
        self,
        df: pd.DataFrame,
        trend_analysis: Optional[Dict[str, Any]],
        change_points: List[datetime],
        cyclical_patterns: Dict[str, Any],
        config: TemporalFeatureConfig
    ) -> pd.DataFrame:
        """Create advanced temporal features based on analysis results"""
        try:
            features_df = df.copy()

            # Add change point indicators
            if change_points:
                features_df['days_since_change_point'] = features_df.apply(
                    lambda row: min([(row.iloc[0] - cp).days for cp in change_points if cp <= row.iloc[0]] + [999]),
                    axis=1
                )
                features_df['is_change_point_period'] = (features_df['days_since_change_point'] <= 7).astype(int)

            # Add trend strength indicators
            if trend_analysis:
                for col, trend_data in trend_analysis.items():
                    if isinstance(trend_data, dict) and 'r_squared' in trend_data:
                        features_df[f'{col}_trend_strength'] = trend_data['r_squared']

            # Add cyclical strength indicators
            for col, patterns in cyclical_patterns.items():
                if 'overall' in patterns:
                    pattern_data = patterns['overall']
                    if 'autocorrelation' in pattern_data and len(pattern_data['autocorrelation']) > 1:
                        max_autocorr = max(pattern_data['autocorrelation'][1:])
                        features_df[f'{col}_cyclical_strength'] = max_autocorr

            # Add volatility features using StatisticalCalculator
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in features_df.columns:
                    continue
                
                # Rolling volatility
                rolling_std = features_df[col].rolling(window=30, min_periods=5).std()
                features_df[f'{col}_volatility_30d'] = rolling_std

                # Coefficient of variation
                rolling_mean = features_df[col].rolling(window=30, min_periods=5).mean()
                features_df[f'{col}_cv_30d'] = safe_divide(rolling_std, rolling_mean)

            self.logger.info("Advanced temporal features created")
            return features_df

        except Exception as e:
            self.logger.error(f"Error creating advanced temporal features: {str(e)}")
            return df

    def _analyze_feature_importance(
        self,
        df: pd.DataFrame,
        value_columns: List[str]
    ) -> Dict[str, float]:
        """Analyze feature importance using statistical methods"""
        try:
            feature_importance = {}
            
            # Get all temporal features (exclude original value columns and date columns)
            temporal_features = [col for col in df.columns 
                               if any(keyword in col.lower() for keyword in 
                                     ['lag', 'rolling', 'trend', 'seasonal', 'cyclical', 'volatility', 'cv'])
                               and col not in value_columns]

            for target_col in value_columns:
                if target_col not in df.columns:
                    continue
                
                # Calculate correlation-based importance
                correlations = df[temporal_features + [target_col]].corr()[target_col].abs()
                correlations = correlations.drop(target_col, errors='ignore')
                
                # Normalize to 0-1 range
                if len(correlations) > 0:
                    max_corr = correlations.max()
                    if max_corr > 0:
                        normalized_correlations = correlations / max_corr
                        feature_importance[target_col] = normalized_correlations.to_dict()

            self.logger.info("Feature importance analysis completed")
            return feature_importance

        except Exception as e:
            self.logger.error(f"Error in feature importance analysis: {str(e)}")
            return {}

    def _create_temporal_analysis_result(
        self,
        trend_analysis: Optional[Dict[str, Any]],
        cyclical_patterns: Dict[str, Any],
        change_points: List[datetime],
        features_df: pd.DataFrame
    ) -> TemporalAnalysisResult:
        """Create comprehensive temporal analysis result"""
        try:
            # Extract trend components
            trend_components = pd.DataFrame()
            if trend_analysis:
                trend_data = []
                for col, trends in trend_analysis.items():
                    if isinstance(trends, dict):
                        trend_data.append({
                            'variable': col,
                            'slope': trends.get('slope', 0),
                            'r_squared': trends.get('r_squared', 0),
                            'trend_direction': 'increasing' if trends.get('slope', 0) > 0 else 'decreasing'
                        })
                trend_components = pd.DataFrame(trend_data)

            # Extract seasonal components
            seasonal_components = pd.DataFrame()
            seasonal_cols = [col for col in features_df.columns if 'seasonal' in col.lower()]
            if seasonal_cols:
                seasonal_components = features_df[seasonal_cols].describe()

            # Calculate volatility metrics
            volatility_metrics = {}
            for col in features_df.select_dtypes(include=[np.number]).columns:
                if 'volatility' in col.lower():
                    volatility_metrics[col] = features_df[col].mean()

            # Create forecasting features
            forecasting_features = features_df[[col for col in features_df.columns 
                                              if any(keyword in col.lower() for keyword in 
                                                    ['trend', 'seasonal', 'lag', 'rolling'])]]

            # Calculate temporal statistics using StatisticalCalculator
            temporal_statistics = {}
            for col in features_df.select_dtypes(include=[np.number]).columns:
                if col in features_df.columns:
                    stats = self.stats_calculator.calculate_descriptive_statistics(
                        features_df[col].dropna()
                    )
                    temporal_statistics[col] = stats

            return TemporalAnalysisResult(
                trend_components=trend_components,
                seasonal_components=seasonal_components,
                cyclical_patterns=cyclical_patterns,
                change_points=change_points,
                volatility_metrics=volatility_metrics,
                forecasting_features=forecasting_features,
                temporal_statistics=temporal_statistics
            )

        except Exception as e:
            self.logger.error(f"Error creating temporal analysis result: {str(e)}")
            return TemporalAnalysisResult(
                pd.DataFrame(), pd.DataFrame(), {}, [], {}, pd.DataFrame(), {}
            )

    def _create_feature_metadata(
        self,
        features_df: pd.DataFrame,
        config: TemporalFeatureConfig
    ) -> Dict[str, Any]:
        """Create metadata for engineered features"""
        try:
            # Categorize features
            feature_categories = {
                'lag_features': [col for col in features_df.columns if 'lag' in col.lower()],
                'rolling_features': [col for col in features_df.columns if 'rolling' in col.lower()],
                'trend_features': [col for col in features_df.columns if 'trend' in col.lower()],
                'seasonal_features': [col for col in features_df.columns if 'seasonal' in col.lower()],
                'cyclical_features': [col for col in features_df.columns if 'cyclical' in col.lower()],
                'volatility_features': [col for col in features_df.columns if 'volatility' in col.lower() or 'cv' in col.lower()],
                'calendar_features': [col for col in features_df.columns if any(keyword in col.lower() for keyword in 
                                                                              ['day', 'week', 'month', 'year', 'weekend'])]
            }

            metadata = {
                'total_features': len(features_df.columns),
                'feature_categories': feature_categories,
                'category_counts': {cat: len(features) for cat, features in feature_categories.items()},
                'configuration': {
                    'lag_periods': config.lag_periods,
                    'rolling_windows': config.rolling_windows,
                    'seasonal_periods': config.seasonal_periods,
                    'trend_analysis': config.trend_analysis,
                    'cyclical_analysis': config.cyclical_analysis,
                    'change_point_detection': config.change_point_detection
                },
                'data_shape': features_df.shape,
                'creation_timestamp': datetime.now(),
                'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024 / 1024
            }

            return metadata

        except Exception as e:
            self.logger.error(f"Error creating feature metadata: {str(e)}")
            return {}

    def _create_processing_summary(
        self,
        original_df: pd.DataFrame,
        features_df: pd.DataFrame,
        config: TemporalFeatureConfig,
        temporal_analysis: TemporalAnalysisResult
    ) -> Dict[str, Any]:
        """Create processing summary"""
        try:
            return {
                'input_records': len(original_df),
                'output_records': len(features_df),
                'input_features': len(original_df.columns),
                'output_features': len(features_df.columns),
                'features_created': len(features_df.columns) - len(original_df.columns),
                'change_points_detected': len(temporal_analysis.change_points),
                'trend_components_found': len(temporal_analysis.trend_components),
                'processing_time': datetime.now(),
                'performance_metrics': self.performance_tracker.get_performance_summary(),
                'configuration_used': {
                    'lag_periods': config.lag_periods,
                    'rolling_windows': config.rolling_windows,
                    'seasonal_periods': config.seasonal_periods
                }
            }

        except Exception as e:
            self.logger.error(f"Error creating processing summary: {str(e)}")
            return {'error': str(e)}

    def _add_business_calendar_features(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """Add business calendar features"""
        try:
            features_df = df.copy()
            
            # Add quarter start/end indicators
            features_df[f'{date_column}_is_quarter_start'] = features_df[date_column].dt.is_quarter_start
            features_df[f'{date_column}_is_quarter_end'] = features_df[date_column].dt.is_quarter_end
            
            # Add year start/end indicators
            features_df[f'{date_column}_is_year_start'] = features_df[date_column].dt.is_year_start
            features_df[f'{date_column}_is_year_end'] = features_df[date_column].dt.is_year_end
            
            # Add business day indicator
            features_df[f'{date_column}_is_business_day'] = (features_df[date_column].dt.weekday < 5).astype(int)
            
            return features_df

        except Exception as e:
            self.logger.error(f"Error adding business calendar features: {str(e)}")
            return df

    @timing_decorator()
    def export_temporal_features(
        self,
        result: TemporalFeatureResult,
        export_path: Optional[str] = None
    ) -> bool:
        """Export temporal features and analysis results"""
        try:
            if export_path is None:
                export_path = self.path_manager.get_exports_path("temporal_features")

            # Export engineered features
            features_export = self.data_exporter.export_with_metadata(
                result.engineered_features,
                result.feature_metadata,
                f"{export_path}/temporal_features.xlsx"
            )

            # Export temporal analysis
            analysis_export = self.report_exporter.export_analysis_report(
                {
                    'temporal_analysis': result.temporal_analysis.__dict__,
                    'feature_importance': result.feature_importance,
                    'processing_summary': result.processing_summary
                },
                f"{export_path}/temporal_analysis_report.json"
            )

            self.logger.info(f"Temporal features exported to {export_path}")
            return features_export and analysis_export

        except Exception as e:
            self.logger.error(f"Error exporting temporal features: {str(e)}")
            return False

    def get_feature_engineering_recommendations(
        self,
        result: TemporalFeatureResult
    ) -> List[str]:
        """Get recommendations for temporal feature engineering"""
        try:
            recommendations = []

            # Check feature importance
            if result.feature_importance:
                low_importance_features = []
                for target, features in result.feature_importance.items():
                    low_features = [f for f, imp in features.items() if imp < 0.1]
                    low_importance_features.extend(low_features)

                if low_importance_features:
                    recommendations.append(f"Consider removing {len(low_importance_features)} low-importance features")

            # Check for missing trend components
            if result.temporal_analysis.trend_components.empty:
                recommendations.append("No trend components detected - consider longer time periods")

            # Check change points
            if not result.temporal_analysis.change_points:
                recommendations.append("No change points detected - data may be stable or need different detection method")

            # Check volatility
            if result.temporal_analysis.volatility_metrics:
                high_volatility = [k for k, v in result.temporal_analysis.volatility_metrics.items() if v > 0.5]
                if high_volatility:
                    recommendations.append(f"High volatility detected in {len(high_volatility)} features - consider smoothing")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations"]
