"""
Anomaly Detection Module for SEO Competitive Intelligence
Advanced anomaly detection leveraging the comprehensive utility framework
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
class AnomalyDetectionConfig:
    """Configuration for anomaly detection operations"""
    detection_methods: List[str] = None
    contamination_rate: float = 0.1
    time_window_days: int = 30
    min_data_points: int = 20
    sensitivity_level: str = 'medium'
    include_seasonal_adjustment: bool = True
    context_aware_detection: bool = True

@dataclass
class AnomalyAlert:
    """Individual anomaly alert"""
    timestamp: datetime
    metric: str
    anomaly_type: str
    severity: str
    current_value: float
    expected_value: float
    deviation_score: float
    confidence: float
    context: Dict[str, Any]
    recommended_actions: List[str]

@dataclass
class AnomalyReport:
    """Comprehensive anomaly detection report"""
    detection_timestamp: datetime
    total_anomalies: int
    anomaly_alerts: List[AnomalyAlert]
    trend_analysis: Dict[str, Any]
    pattern_insights: Dict[str, Any]
    severity_distribution: Dict[str, int]
    recommendations: List[str]
    detection_metadata: Dict[str, Any]

class AnomalyDetector:
    """
    Advanced anomaly detection for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    anomaly detection capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("anomaly_detector")
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
        
        # Load anomaly detection configurations
        analysis_config = self.config.get_analysis_config()
        self.default_config = AnomalyDetectionConfig(
            detection_methods=['isolation_forest', 'statistical', 'time_series'],
            contamination_rate=getattr(analysis_config, 'traffic_anomaly_threshold', 0.1),
            sensitivity_level='medium'
        )

    @timing_decorator()
    @memoize(ttl=1800)  # Cache for 30 minutes
    def detect_comprehensive_anomalies(
        self,
        data: pd.DataFrame,
        target_columns: List[str] = None,
        config: Optional[AnomalyDetectionConfig] = None,
        historical_context: Optional[pd.DataFrame] = None
    ) -> AnomalyReport:
        """
        Perform comprehensive anomaly detection using multiple methods.
        
        Args:
            data: Input DataFrame with SEO metrics
            target_columns: Columns to analyze for anomalies
            config: Anomaly detection configuration
            historical_context: Historical data for context-aware detection
            
        Returns:
            AnomalyReport with comprehensive anomaly analysis
        """
        try:
            with self.performance_tracker.track_block("detect_comprehensive_anomalies"):
                # Audit log the anomaly detection operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="anomaly_detection",
                    parameters={
                        "rows": len(data),
                        "target_columns": target_columns,
                        "detection_methods": config.detection_methods if config else "default"
                    }
                )
                
                if config is None:
                    config = self.default_config
                
                if target_columns is None:
                    target_columns = ['Position', 'Traffic (%)', 'Search Volume', 'CPC']
                
                # Clean and validate data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(data)
                
                # Validate data quality using DataValidator
                validation_report = self.data_validator.validate_seo_dataset(cleaned_data, 'positions')
                if validation_report.quality_score < 0.7:
                    self.logger.warning(f"Low data quality for anomaly detection: {validation_report.quality_score:.3f}")
                
                # Initialize anomaly alerts list
                all_anomaly_alerts = []
                
                # 1. Statistical Anomaly Detection
                if 'statistical' in config.detection_methods:
                    statistical_anomalies = self._detect_statistical_anomalies(
                        cleaned_data, target_columns, config
                    )
                    all_anomaly_alerts.extend(statistical_anomalies)
                
                # 2. Machine Learning Anomaly Detection
                if 'isolation_forest' in config.detection_methods:
                    ml_anomalies = self._detect_ml_anomalies(
                        cleaned_data, target_columns, config
                    )
                    all_anomaly_alerts.extend(ml_anomalies)
                
                # 3. Time Series Anomaly Detection
                if 'time_series' in config.detection_methods and 'date' in cleaned_data.columns:
                    ts_anomalies = self._detect_time_series_anomalies(
                        cleaned_data, target_columns, config
                    )
                    all_anomaly_alerts.extend(ts_anomalies)
                
                # 4. Business Rule Anomaly Detection
                if 'business_rules' in config.detection_methods:
                    business_anomalies = self._detect_business_rule_anomalies(
                        cleaned_data, target_columns, config
                    )
                    all_anomaly_alerts.extend(business_anomalies)
                
                # 5. Context-Aware Anomaly Detection
                if config.context_aware_detection and historical_context is not None:
                    context_anomalies = self._detect_context_aware_anomalies(
                        cleaned_data, historical_context, target_columns, config
                    )
                    all_anomaly_alerts.extend(context_anomalies)
                
                # Analyze trends using TimeSeriesAnalyzer
                trend_analysis = self._analyze_anomaly_trends(cleaned_data, all_anomaly_alerts)
                
                # Extract patterns and insights
                pattern_insights = self._extract_anomaly_patterns(all_anomaly_alerts, cleaned_data)
                
                # Calculate severity distribution
                severity_distribution = self._calculate_severity_distribution(all_anomaly_alerts)
                
                # Generate comprehensive recommendations
                recommendations = self._generate_anomaly_recommendations(
                    all_anomaly_alerts, trend_analysis, pattern_insights
                )
                
                # Create detection metadata
                detection_metadata = {
                    'detection_methods_used': config.detection_methods,
                    'data_quality_score': validation_report.quality_score,
                    'total_data_points': len(cleaned_data),
                    'analysis_time_window': config.time_window_days,
                    'contamination_rate': config.contamination_rate,
                    'sensitivity_level': config.sensitivity_level
                }
                
                # Create comprehensive report
                report = AnomalyReport(
                    detection_timestamp=datetime.now(),
                    total_anomalies=len(all_anomaly_alerts),
                    anomaly_alerts=all_anomaly_alerts,
                    trend_analysis=trend_analysis,
                    pattern_insights=pattern_insights,
                    severity_distribution=severity_distribution,
                    recommendations=recommendations,
                    detection_metadata=detection_metadata
                )
                
                self.logger.info(f"Comprehensive anomaly detection completed: {len(all_anomaly_alerts)} anomalies detected")
                return report
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive anomaly detection: {str(e)}")
            return AnomalyReport(
                datetime.now(), 0, [], {}, {}, {}, [f"Detection failed: {str(e)}"], {}
            )

    @timing_decorator()
    def detect_position_anomalies(
        self,
        keyword_positions: pd.DataFrame,
        lookback_days: int = 30,
        sensitivity: str = 'medium'
    ) -> List[AnomalyAlert]:
        """
        Detect position-specific anomalies using SEO domain knowledge.
        
        Args:
            keyword_positions: DataFrame with keyword position data
            lookback_days: Days to look back for baseline
            sensitivity: Detection sensitivity level
            
        Returns:
            List of position-related anomaly alerts
        """
        try:
            with self.performance_tracker.track_block("detect_position_anomalies"):
                # Clean data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(keyword_positions)
                
                position_anomalies = []
                
                if 'Position' not in cleaned_data.columns:
                    return position_anomalies
                
                # Group by keyword for individual analysis
                if 'Keyword' in cleaned_data.columns:
                    keyword_groups = cleaned_data.groupby('Keyword')
                else:
                    keyword_groups = [('all_keywords', cleaned_data)]
                
                for keyword, keyword_data in keyword_groups:
                    if len(keyword_data) < 5:  # Need minimum data points
                        continue
                    
                    # Use TimeSeriesAnalyzer for position trend analysis
                    if 'date' in keyword_data.columns:
                        position_series = keyword_data.set_index('date')['Position'].sort_index()
                        
                        # Detect sudden position drops/gains using changepoint detection
                        changepoints = self.time_series_analyzer.detect_changepoints(
                            position_series, method='variance', min_size=3
                        )
                        
                        for changepoint in changepoints:
                            if changepoint < len(position_series) - 1:
                                # Analyze the magnitude of change
                                before_avg = position_series.iloc[max(0, changepoint-5):changepoint].mean()
                                after_avg = position_series.iloc[changepoint:changepoint+5].mean()
                                
                                position_change = after_avg - before_avg
                                
                                # Determine severity based on position change
                                if abs(position_change) > 20:
                                    severity = 'critical'
                                elif abs(position_change) > 10:
                                    severity = 'high'
                                elif abs(position_change) > 5:
                                    severity = 'medium'
                                else:
                                    continue
                                
                                # Create anomaly alert
                                anomaly_type = 'position_drop' if position_change > 0 else 'position_gain'
                                
                                alert = AnomalyAlert(
                                    timestamp=position_series.index[changepoint],
                                    metric='Position',
                                    anomaly_type=anomaly_type,
                                    severity=severity,
                                    current_value=after_avg,
                                    expected_value=before_avg,
                                    deviation_score=abs(position_change),
                                    confidence=min(abs(position_change) / 30, 1.0),
                                    context={
                                        'keyword': keyword,
                                        'position_change': position_change,
                                        'changepoint_index': changepoint
                                    },
                                    recommended_actions=self._get_position_anomaly_actions(
                                        anomaly_type, position_change, keyword
                                    )
                                )
                                
                                position_anomalies.append(alert)
                    
                    else:
                        # Without date, use statistical outlier detection
                        position_outliers = self._detect_statistical_outliers(
                            keyword_data['Position'], 'Position', keyword
                        )
                        position_anomalies.extend(position_outliers)
                
                self.logger.info(f"Detected {len(position_anomalies)} position anomalies")
                return position_anomalies
                
        except Exception as e:
            self.logger.error(f"Error detecting position anomalies: {str(e)}")
            return []

    @timing_decorator()
    def detect_traffic_anomalies(
        self,
        traffic_data: pd.DataFrame,
        competitive_context: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[AnomalyAlert]:
        """
        Detect traffic anomalies with competitive context analysis.
        
        Args:
            traffic_data: DataFrame with traffic data
            competitive_context: Optional competitor data for context
            
        Returns:
            List of traffic-related anomaly alerts
        """
        try:
            with self.performance_tracker.track_block("detect_traffic_anomalies"):
                # Clean data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(traffic_data)
                
                traffic_anomalies = []
                
                traffic_columns = ['Traffic (%)', 'Traffic', 'Organic Traffic']
                available_traffic_columns = [col for col in traffic_columns if col in cleaned_data.columns]
                
                if not available_traffic_columns:
                    return traffic_anomalies
                
                for traffic_col in available_traffic_columns:
                    # Use statistical methods for traffic anomaly detection
                    traffic_series = cleaned_data[traffic_col].dropna()
                    
                    if len(traffic_series) < 10:
                        continue
                    
                    # Calculate statistical bounds using StatisticalCalculator
                    stats_dict = self.stats_calculator.calculate_descriptive_statistics(
                        traffic_series, include_advanced=True
                    )
                    
                    # Define thresholds based on statistical measures
                    mean_traffic = stats_dict.get('mean', 0)
                    std_traffic = stats_dict.get('std', 0)
                    q25 = stats_dict.get('q25', 0)
                    q75 = stats_dict.get('q75', 0)
                    iqr = stats_dict.get('iqr', 0)
                    
                    # IQR-based anomaly detection
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    # Find anomalies
                    anomaly_mask = (traffic_series < lower_bound) | (traffic_series > upper_bound)
                    anomaly_indices = cleaned_data.index[anomaly_mask]
                    
                    for idx in anomaly_indices:
                        current_value = traffic_series.loc[idx]
                        expected_value = mean_traffic
                        deviation_score = abs(current_value - expected_value) / max(std_traffic, 1)
                        
                        # Determine anomaly type and severity
                        if current_value < lower_bound:
                            anomaly_type = 'traffic_drop'
                            severity = 'critical' if current_value < mean_traffic * 0.5 else 'high'
                        else:
                            anomaly_type = 'traffic_spike'
                            severity = 'medium' if current_value < mean_traffic * 2 else 'high'
                        
                        # Add competitive context if available
                        context = {'traffic_column': traffic_col}
                        if competitive_context:
                            context.update(self._analyze_competitive_traffic_context(
                                current_value, competitive_context, traffic_col
                            ))
                        
                        alert = AnomalyAlert(
                            timestamp=datetime.now(),
                            metric=traffic_col,
                            anomaly_type=anomaly_type,
                            severity=severity,
                            current_value=current_value,
                            expected_value=expected_value,
                            deviation_score=deviation_score,
                            confidence=min(deviation_score / 3, 1.0),
                            context=context,
                            recommended_actions=self._get_traffic_anomaly_actions(
                                anomaly_type, deviation_score, competitive_context is not None
                            )
                        )
                        
                        traffic_anomalies.append(alert)
                
                self.logger.info(f"Detected {len(traffic_anomalies)} traffic anomalies")
                return traffic_anomalies
                
        except Exception as e:
            self.logger.error(f"Error detecting traffic anomalies: {str(e)}")
            return []

    def _detect_statistical_anomalies(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        config: AnomalyDetectionConfig
    ) -> List[AnomalyAlert]:
        """Detect anomalies using statistical methods."""
        try:
            statistical_anomalies = []
            
            for column in target_columns:
                if column not in data.columns:
                    continue
                
                column_data = data[column].dropna()
                if len(column_data) < config.min_data_points:
                    continue
                
                # Use StatisticalCalculator for robust statistical analysis
                stats_dict = self.stats_calculator.calculate_descriptive_statistics(
                    column_data, include_advanced=True
                )
                
                # Multiple statistical anomaly detection methods
                
                # 1. Z-score based detection
                z_threshold = self._get_z_threshold(config.sensitivity_level)
                mean_val = stats_dict.get('mean', 0)
                std_val = stats_dict.get('std', 1)
                
                if std_val > 0:
                    z_scores = np.abs((column_data - mean_val) / std_val)
                    z_anomalies = z_scores > z_threshold
                    
                    for idx in column_data.index[z_anomalies]:
                        value = column_data.loc[idx]
                        z_score = z_scores.loc[idx]
                        
                        alert = AnomalyAlert(
                            timestamp=datetime.now(),
                            metric=column,
                            anomaly_type='statistical_outlier',
                            severity=self._calculate_severity_from_z_score(z_score),
                            current_value=value,
                            expected_value=mean_val,
                            deviation_score=z_score,
                            confidence=min(z_score / 5, 1.0),
                            context={'detection_method': 'z_score', 'threshold': z_threshold},
                            recommended_actions=['Investigate data quality', 'Verify measurement accuracy']
                        )
                        statistical_anomalies.append(alert)
                
                # 2. IQR based detection
                q25 = stats_dict.get('q25', 0)
                q75 = stats_dict.get('q75', 0)
                iqr = stats_dict.get('iqr', 0)
                
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    iqr_anomalies = (column_data < lower_bound) | (column_data > upper_bound)
                    
                    for idx in column_data.index[iqr_anomalies]:
                        value = column_data.loc[idx]
                        
                        if idx not in [alert.timestamp for alert in statistical_anomalies]:  # Avoid duplicates
                            alert = AnomalyAlert(
                                timestamp=datetime.now(),
                                metric=column,
                                anomaly_type='iqr_outlier',
                                severity='medium',
                                current_value=value,
                                expected_value=(q25 + q75) / 2,
                                deviation_score=abs(value - (q25 + q75) / 2) / iqr,
                                confidence=0.75,
                                context={'detection_method': 'iqr', 'bounds': [lower_bound, upper_bound]},
                                recommended_actions=['Review recent changes', 'Check for external factors']
                            )
                            statistical_anomalies.append(alert)
            
            return statistical_anomalies
            
        except Exception as e:
            self.logger.error(f"Error in statistical anomaly detection: {str(e)}")
            return []

    def _detect_ml_anomalies(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        config: AnomalyDetectionConfig
    ) -> List[AnomalyAlert]:
        """Detect anomalies using machine learning methods."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            ml_anomalies = []
            
            # Prepare feature matrix
            feature_data = data[target_columns].select_dtypes(include=[np.number]).fillna(0)
            
            if feature_data.empty or len(feature_data) < config.min_data_points:
                return ml_anomalies
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(
                contamination=config.contamination_rate,
                random_state=42,
                n_estimators=100
            )
            
            anomaly_labels = iso_forest.fit_predict(scaled_features)
            anomaly_scores = iso_forest.decision_function(scaled_features)
            
            # Process results
            anomaly_mask = anomaly_labels == -1
            anomaly_indices = data.index[anomaly_mask]
            
            for idx in anomaly_indices:
                anomaly_score = anomaly_scores[data.index.get_loc(idx)]
                
                # Find the most anomalous feature for this instance
                instance_features = scaled_features[data.index.get_loc(idx)]
                most_anomalous_feature_idx = np.argmax(np.abs(instance_features))
                most_anomalous_feature = feature_data.columns[most_anomalous_feature_idx]
                
                alert = AnomalyAlert(
                    timestamp=datetime.now(),
                    metric=most_anomalous_feature,
                    anomaly_type='ml_detected_anomaly',
                    severity=self._calculate_severity_from_anomaly_score(anomaly_score),
                    current_value=feature_data.iloc[data.index.get_loc(idx)][most_anomalous_feature],
                    expected_value=feature_data[most_anomalous_feature].mean(),
                    deviation_score=abs(anomaly_score),
                    confidence=min(abs(anomaly_score) * 2, 1.0),
                    context={
                        'detection_method': 'isolation_forest',
                        'anomaly_score': anomaly_score,
                        'features_analyzed': list(feature_data.columns)
                    },
                    recommended_actions=['Deep dive analysis', 'Cross-reference with external events']
                )
                ml_anomalies.append(alert)
            
            return ml_anomalies
            
        except Exception as e:
            self.logger.error(f"Error in ML anomaly detection: {str(e)}")
            return []

    def _detect_time_series_anomalies(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        config: AnomalyDetectionConfig
    ) -> List[AnomalyAlert]:
        """Detect time series anomalies using TimeSeriesAnalyzer."""
        try:
            ts_anomalies = []
            
            if 'date' not in data.columns:
                return ts_anomalies
            
            # Prepare time series data
            data_sorted = data.sort_values('date')
            
            for column in target_columns:
                if column not in data_sorted.columns:
                    continue
                
                # Create time series
                ts_data = data_sorted.set_index('date')[column].dropna()
                
                if len(ts_data) < config.min_data_points:
                    continue
                
                # Use TimeSeriesAnalyzer for anomaly detection
                anomaly_mask, anomaly_scores = self.time_series_analyzer.detect_anomalies_in_series(
                    ts_data,
                    method='isolation_forest',
                    contamination=config.contamination_rate
                )
                
                # Process detected anomalies
                anomaly_dates = ts_data.index[anomaly_mask]
                
                for i, anomaly_date in enumerate(anomaly_dates):
                    current_value = ts_data.loc[anomaly_date]
                    
                    # Calculate expected value using surrounding data
                    window_start = max(0, np.where(ts_data.index == anomaly_date)[0][0] - 5)
                    window_end = min(len(ts_data), np.where(ts_data.index == anomaly_date)[0][0] + 5)
                    surrounding_data = ts_data.iloc[window_start:window_end]
                    expected_value = surrounding_data[surrounding_data.index != anomaly_date].mean()
                    
                    anomaly_score = anomaly_scores[anomaly_mask][i] if len(anomaly_scores) > 0 else 0
                    
                    alert = AnomalyAlert(
                        timestamp=anomaly_date,
                        metric=column,
                        anomaly_type='time_series_anomaly',
                        severity=self._calculate_severity_from_anomaly_score(anomaly_score),
                        current_value=current_value,
                        expected_value=expected_value,
                        deviation_score=abs(anomaly_score),
                        confidence=min(abs(anomaly_score) * 1.5, 1.0),
                        context={
                            'detection_method': 'time_series',
                            'anomaly_date': anomaly_date.isoformat(),
                            'time_window': config.time_window_days
                        },
                        recommended_actions=self._get_time_series_anomaly_actions(column, anomaly_date)
                    )
                    ts_anomalies.append(alert)
            
            return ts_anomalies
            
        except Exception as e:
            self.logger.error(f"Error in time series anomaly detection: {str(e)}")
            return []

    def export_anomaly_report(
        self,
        anomaly_report: AnomalyReport,
        export_directory: str,
        include_detailed_analysis: bool = True
    ) -> Dict[str, bool]:
        """Export comprehensive anomaly detection report."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'anomaly_summary': {
                    'detection_timestamp': anomaly_report.detection_timestamp.isoformat(),
                    'total_anomalies': anomaly_report.total_anomalies,
                    'severity_distribution': anomaly_report.severity_distribution,
                    'detection_methods': anomaly_report.detection_metadata.get('detection_methods_used', [])
                },
                'anomaly_alerts': [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'metric': alert.metric,
                        'anomaly_type': alert.anomaly_type,
                        'severity': alert.severity,
                        'current_value': alert.current_value,
                        'expected_value': alert.expected_value,
                        'deviation_score': alert.deviation_score,
                        'confidence': alert.confidence,
                        'context': alert.context,
                        'recommended_actions': alert.recommended_actions
                    }
                    for alert in anomaly_report.anomaly_alerts
                ],
                'trend_analysis': anomaly_report.trend_analysis,
                'pattern_insights': anomaly_report.pattern_insights,
                'recommendations': anomaly_report.recommendations,
                'detection_metadata': anomaly_report.detection_metadata
            }
            
            # Export detailed data using DataExporter
            data_export_success = self.data_exporter.export_analysis_dataset(
                {'anomaly_detection_report': pd.DataFrame([export_data])},
                export_path / "anomaly_detection_detailed.xlsx"
            )
            
            # Export anomaly alerts as separate dataset
            if anomaly_report.anomaly_alerts:
                alerts_df = pd.DataFrame([
                    {
                        'timestamp': alert.timestamp,
                        'metric': alert.metric,
                        'anomaly_type': alert.anomaly_type,
                        'severity': alert.severity,
                        'current_value': alert.current_value,
                        'expected_value': alert.expected_value,
                        'deviation_score': alert.deviation_score,
                        'confidence': alert.confidence
                    }
                    for alert in anomaly_report.anomaly_alerts
                ])
                
                alerts_export_success = self.data_exporter.export_with_metadata(
                    alerts_df,
                    metadata={'analysis_type': 'anomaly_alerts', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "anomaly_alerts.xlsx"
                )
            else:
                alerts_export_success = True
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_executive_report(
                export_data,
                export_path / "anomaly_detection_executive_report.html",
                format='html',
                include_charts=True
            )
            
            return {
                'detailed_report': data_export_success,
                'anomaly_alerts': alerts_export_success,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting anomaly detection report: {str(e)}")
            return {}

    # Helper methods using utility framework
    def _get_z_threshold(self, sensitivity_level: str) -> float:
        """Get Z-score threshold based on sensitivity level."""
        thresholds = {'low': 3.5, 'medium': 3.0, 'high': 2.5}
        return thresholds.get(sensitivity_level, 3.0)

    def _calculate_severity_from_z_score(self, z_score: float) -> str:
        """Calculate severity based on Z-score."""
        if z_score > 4:
            return 'critical'
        elif z_score > 3:
            return 'high'
        elif z_score > 2:
            return 'medium'
        else:
            return 'low'

    def _calculate_severity_from_anomaly_score(self, anomaly_score: float) -> str:
        """Calculate severity based on anomaly score."""
        abs_score = abs(anomaly_score)
        if abs_score > 0.5:
            return 'critical'
        elif abs_score > 0.3:
            return 'high'
        elif abs_score > 0.1:
            return 'medium'
        else:
            return 'low'

    def _get_position_anomaly_actions(self, anomaly_type: str, position_change: float, keyword: str) -> List[str]:
        """Get recommended actions for position anomalies."""
        actions = []
        
        if anomaly_type == 'position_drop':
            actions.extend([
                f"Investigate content quality for keyword: {keyword}",
                "Check for algorithm updates",
                "Analyze competitor content improvements",
                "Review technical SEO factors"
            ])
        elif anomaly_type == 'position_gain':
            actions.extend([
                f"Document successful changes for keyword: {keyword}",
                "Scale successful strategies to similar keywords",
                "Monitor for sustainability"
            ])
        
        return actions

    def _get_traffic_anomaly_actions(self, anomaly_type: str, deviation_score: float, has_competitive_context: bool) -> List[str]:
        """Get recommended actions for traffic anomalies."""
        actions = []
        
        if anomaly_type == 'traffic_drop':
            actions.extend([
                "Check for technical issues",
                "Analyze position changes",
                "Review recent content modifications"
            ])
            if has_competitive_context:
                actions.append("Compare with competitor traffic patterns")
        
        elif anomaly_type == 'traffic_spike':
            actions.extend([
                "Identify driving factors",
                "Document successful elements",
                "Prepare to maintain momentum"
            ])
        
        return actions

    def _get_time_series_anomaly_actions(self, column: str, anomaly_date: datetime) -> List[str]:
        """Get recommended actions for time series anomalies."""
        return [
            f"Investigate events around {anomaly_date.strftime('%Y-%m-%d')}",
            f"Analyze {column} metric drivers",
            "Check for external factors (holidays, campaigns, etc.)",
            "Review measurement methodology"
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for anomaly detection operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods would be implemented here...
    def _analyze_anomaly_trends(self, data: pd.DataFrame, anomalies: List[AnomalyAlert]) -> Dict[str, Any]:
        """Analyze trends in detected anomalies."""
        try:
            if not anomalies:
                return {}
            
            # Group anomalies by type and metric
            anomaly_trends = {
                'anomalies_by_metric': {},
                'anomalies_by_type': {},
                'severity_trends': {},
                'temporal_patterns': {}
            }
            
            for anomaly in anomalies:
                # By metric
                if anomaly.metric not in anomaly_trends['anomalies_by_metric']:
                    anomaly_trends['anomalies_by_metric'][anomaly.metric] = 0
                anomaly_trends['anomalies_by_metric'][anomaly.metric] += 1
                
                # By type
                if anomaly.anomaly_type not in anomaly_trends['anomalies_by_type']:
                    anomaly_trends['anomalies_by_type'][anomaly.anomaly_type] = 0
                anomaly_trends['anomalies_by_type'][anomaly.anomaly_type] += 1
                
                # By severity
                if anomaly.severity not in anomaly_trends['severity_trends']:
                    anomaly_trends['severity_trends'][anomaly.severity] = 0
                anomaly_trends['severity_trends'][anomaly.severity] += 1
            
            return anomaly_trends
            
        except Exception:
            return {}

    def _extract_anomaly_patterns(self, anomalies: List[AnomalyAlert], data: pd.DataFrame) -> Dict[str, Any]:
        """Extract patterns from detected anomalies."""
        try:
            patterns = {
                'common_contexts': {},
                'metric_correlations': {},
                'timing_patterns': {}
            }
            
            # Analyze common contexts
            context_keys = set()
            for anomaly in anomalies:
                context_keys.update(anomaly.context.keys())
            
            for key in context_keys:
                values = [anomaly.context.get(key) for anomaly in anomalies if key in anomaly.context]
                patterns['common_contexts'][key] = list(set(values))
            
            return patterns
            
        except Exception:
            return {}

    def _calculate_severity_distribution(self, anomalies: List[AnomalyAlert]) -> Dict[str, int]:
        """Calculate distribution of anomaly severities."""
        try:
            distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
            for anomaly in anomalies:
                if anomaly.severity in distribution:
                    distribution[anomaly.severity] += 1
            
            return distribution
            
        except Exception:
            return {}

    def _generate_anomaly_recommendations(
        self,
        anomalies: List[AnomalyAlert],
        trend_analysis: Dict[str, Any],
        pattern_insights: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive recommendations based on anomaly analysis."""
        try:
            recommendations = []
            
            # Overall recommendations based on anomaly count and severity
            total_anomalies = len(anomalies)
            critical_count = sum(1 for a in anomalies if a.severity == 'critical')
            
            if critical_count > 0:
                recommendations.append(f"Immediate attention required: {critical_count} critical anomalies detected")
            
            if total_anomalies > 10:
                recommendations.append("High anomaly volume detected - consider reviewing data collection processes")
            
            # Metric-specific recommendations
            metric_counts = trend_analysis.get('anomalies_by_metric', {})
            for metric, count in metric_counts.items():
                if count > 3:
                    recommendations.append(f"Focus on {metric} metric - {count} anomalies detected")
            
            # General recommendations
            recommendations.extend([
                "Establish regular anomaly monitoring schedule",
                "Create automated alerts for critical anomalies",
                "Document investigation procedures for common anomaly types"
            ])
            
            return recommendations
            
        except Exception:
            return ["Review anomaly detection results and create action plan"]
