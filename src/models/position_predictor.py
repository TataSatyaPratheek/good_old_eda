"""
Position Predictor Module for SEO Competitive Intelligence
Advanced position prediction leveraging the comprehensive utility framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
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
class PredictionConfig:
    """Configuration for position prediction"""
    prediction_horizon_days: int = 30
    model_types: List[str] = None
    include_seasonal_factors: bool = True
    include_competitive_factors: bool = True
    confidence_intervals: List[float] = None
    feature_engineering: bool = True
    cross_validation_folds: int = 5

@dataclass
class PredictionResult:
    """Individual prediction result"""
    keyword: str
    current_position: float
    predicted_position: float
    prediction_date: datetime
    confidence_interval_lower: float
    confidence_interval_upper: float
    prediction_confidence: float
    contributing_factors: Dict[str, float]
    trend_direction: str
    volatility_score: float

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    mape: float  # Mean Absolute Percentage Error
    r2_score: float
    directional_accuracy: float
    confidence_calibration: float

@dataclass
class PredictionReport:
    """Comprehensive prediction report"""
    predictions: List[PredictionResult]
    model_performance: Dict[str, ModelPerformance]
    feature_importance: Dict[str, float]
    prediction_summary: Dict[str, Any]
    recommendations: List[str]
    uncertainty_analysis: Dict[str, Any]

class PositionPredictor:
    """
    Advanced position prediction for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    position prediction capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("position_predictor")
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
        
        # Load prediction configurations
        analysis_config = self.config.get_analysis_config()
        self.default_config = PredictionConfig(
            model_types=['linear_regression', 'random_forest', 'time_series', 'ensemble'],
            confidence_intervals=[0.8, 0.95]
        )
        
        # Initialize trained models storage
        self.trained_models = {}
        self.model_metadata = {}

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def predict_positions(
        self,
        historical_data: pd.DataFrame,
        keywords_to_predict: Optional[List[str]] = None,
        config: Optional[PredictionConfig] = None,
        competitive_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> PredictionReport:
        """
        Predict future positions using comprehensive modeling approach.
        
        Args:
            historical_data: Historical position data
            keywords_to_predict: Specific keywords to predict
            config: Prediction configuration
            competitive_data: Competitive data for context
            
        Returns:
            PredictionReport with comprehensive predictions
        """
        try:
            with self.performance_tracker.track_block("predict_positions"):
                # Audit log the prediction operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="position_prediction",
                    parameters={
                        "historical_records": len(historical_data),
                        "keywords_count": len(keywords_to_predict) if keywords_to_predict else "all",
                        "prediction_horizon": config.prediction_horizon_days if config else "default"
                    }
                )
                
                if config is None:
                    config = self.default_config
                
                # Clean and validate data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(historical_data)
                
                # Validate data quality using DataValidator
                validation_report = self.data_validator.validate_seo_dataset(cleaned_data, 'positions')
                if validation_report.quality_score < 0.7:
                    self.logger.warning(f"Low data quality for prediction: {validation_report.quality_score:.3f}")
                
                # Prepare keywords list
                if keywords_to_predict is None:
                    keywords_to_predict = cleaned_data['Keyword'].unique().tolist()
                
                # Feature engineering using DataTransformer
                if config.feature_engineering:
                    engineered_data = self._engineer_prediction_features(
                        cleaned_data, competitive_data, config
                    )
                else:
                    engineered_data = cleaned_data
                
                # Train prediction models
                trained_models = self._train_prediction_models(
                    engineered_data, config
                )
                
                # Generate predictions for each keyword
                all_predictions = []
                for keyword in keywords_to_predict:
                    keyword_predictions = self._predict_keyword_position(
                        keyword, engineered_data, trained_models, config
                    )
                    if keyword_predictions:
                        all_predictions.extend(keyword_predictions)
                
                # Evaluate model performance
                model_performance = self._evaluate_model_performance(
                    engineered_data, trained_models, config
                )
                
                # Calculate feature importance using statistical methods
                feature_importance = self._calculate_feature_importance(
                    engineered_data, trained_models
                )
                
                # Generate prediction summary
                prediction_summary = self._generate_prediction_summary(
                    all_predictions, model_performance
                )
                
                # Perform uncertainty analysis
                uncertainty_analysis = self._analyze_prediction_uncertainty(
                    all_predictions, engineered_data
                )
                
                # Generate recommendations using optimization
                recommendations = self._generate_prediction_recommendations(
                    all_predictions, model_performance, uncertainty_analysis
                )
                
                report = PredictionReport(
                    predictions=all_predictions,
                    model_performance=model_performance,
                    feature_importance=feature_importance,
                    prediction_summary=prediction_summary,
                    recommendations=recommendations,
                    uncertainty_analysis=uncertainty_analysis
                )
                
                self.logger.info(f"Position prediction completed for {len(all_predictions)} predictions")
                return report
                
        except Exception as e:
            self.logger.error(f"Error in position prediction: {str(e)}")
            return PredictionReport([], {}, {}, {}, [], {})

    @timing_decorator()
    def predict_single_keyword(
        self,
        keyword: str,
        historical_data: pd.DataFrame,
        prediction_horizon_days: int = 30,
        include_uncertainty: bool = True
    ) -> Optional[PredictionResult]:
        """
        Predict position for a single keyword with detailed analysis.
        
        Args:
            keyword: Keyword to predict
            historical_data: Historical data
            prediction_horizon_days: Prediction horizon
            include_uncertainty: Whether to include uncertainty analysis
            
        Returns:
            PredictionResult for the keyword
        """
        try:
            with self.performance_tracker.track_block("predict_single_keyword"):
                # Clean data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(historical_data)
                
                # Filter data for specific keyword
                keyword_data = cleaned_data[cleaned_data['Keyword'].str.lower() == keyword.lower()]
                
                if keyword_data.empty:
                    self.logger.warning(f"No data found for keyword: {keyword}")
                    return None
                
                # Sort by date if available
                if 'date' in keyword_data.columns:
                    keyword_data = keyword_data.sort_values('date')
                
                # Prepare time series data using TimeSeriesAnalyzer
                position_series = keyword_data['Position']
                
                # Trend analysis using TimeSeriesAnalyzer
                trend_model = self.time_series_analyzer.fit_trend_model(
                    position_series, 'linear'
                )
                
                # Seasonal decomposition if enough data
                seasonal_components = {}
                if len(position_series) >= 14:
                    seasonal_components = self.time_series_analyzer.decompose_time_series(
                        position_series, period=7, model='additive'
                    )
                
                # Generate prediction using multiple methods
                predictions = []
                
                # Method 1: Linear trend extrapolation
                if trend_model and 'slope' in trend_model:
                    linear_pred = position_series.iloc[-1] + (trend_model['slope'] * prediction_horizon_days)
                    predictions.append(linear_pred)
                
                # Method 2: Moving average with trend
                if len(position_series) >= 7:
                    recent_avg = position_series.tail(7).mean()
                    trend_factor = trend_model.get('slope', 0) * prediction_horizon_days
                    ma_pred = recent_avg + trend_factor
                    predictions.append(ma_pred)
                
                # Method 3: Statistical forecast using ARIMA-like approach
                if len(position_series) >= 20:
                    statistical_pred = self._statistical_forecast(position_series, prediction_horizon_days)
                    if statistical_pred is not None:
                        predictions.append(statistical_pred)
                
                # Ensemble prediction
                if predictions:
                    final_prediction = np.mean(predictions)
                    prediction_std = np.std(predictions) if len(predictions) > 1 else 0
                else:
                    final_prediction = position_series.iloc[-1]
                    prediction_std = position_series.std()
                
                # Calculate confidence intervals
                confidence_lower = final_prediction - 1.96 * prediction_std
                confidence_upper = final_prediction + 1.96 * prediction_std
                
                # Ensure position bounds
                final_prediction = max(1, min(100, final_prediction))
                confidence_lower = max(1, min(100, confidence_lower))
                confidence_upper = max(1, min(100, confidence_upper))
                
                # Calculate contributing factors using statistical analysis
                contributing_factors = self._analyze_contributing_factors(
                    keyword_data, trend_model, seasonal_components
                )
                
                # Determine trend direction
                trend_direction = self._determine_trend_direction(trend_model)
                
                # Calculate volatility score using statistical methods
                volatility_score = self._calculate_volatility_score(position_series)
                
                # Calculate prediction confidence
                prediction_confidence = self._calculate_prediction_confidence(
                    position_series, predictions, trend_model
                )
                
                result = PredictionResult(
                    keyword=keyword,
                    current_position=float(position_series.iloc[-1]),
                    predicted_position=final_prediction,
                    prediction_date=datetime.now() + timedelta(days=prediction_horizon_days),
                    confidence_interval_lower=confidence_lower,
                    confidence_interval_upper=confidence_upper,
                    prediction_confidence=prediction_confidence,
                    contributing_factors=contributing_factors,
                    trend_direction=trend_direction,
                    volatility_score=volatility_score
                )
                
                self.logger.info(f"Single keyword prediction completed for: {keyword}")
                return result
                
        except Exception as e:
            self.logger.error(f"Error predicting single keyword {keyword}: {str(e)}")
            return None

    @timing_decorator()
    def analyze_position_volatility(
        self,
        data: pd.DataFrame,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Analyze position volatility patterns using statistical methods.
        
        Args:
            data: Historical position data
            lookback_days: Days to analyze
            
        Returns:
            Volatility analysis results
        """
        try:
            with self.performance_tracker.track_block("analyze_position_volatility"):
                # Clean data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(data)
                
                # Filter by date if available
                if 'date' in cleaned_data.columns:
                    cutoff_date = datetime.now() - timedelta(days=lookback_days)
                    cleaned_data = cleaned_data[pd.to_datetime(cleaned_data['date']) >= cutoff_date]
                
                volatility_analysis = {}
                
                # Analyze volatility by keyword
                for keyword in cleaned_data['Keyword'].unique():
                    keyword_data = cleaned_data[cleaned_data['Keyword'] == keyword]
                    position_series = keyword_data['Position']
                    
                    if len(position_series) < 5:
                        continue
                    
                    # Calculate volatility metrics using StatisticalCalculator
                    stats_dict = self.stats_calculator.calculate_descriptive_statistics(
                        position_series, include_advanced=True
                    )
                    
                    # Position volatility analysis
                    position_changes = position_series.diff().dropna()
                    
                    volatility_metrics = {
                        'standard_deviation': stats_dict.get('std', 0),
                        'coefficient_of_variation': stats_dict.get('coefficient_of_variation', 0),
                        'average_absolute_change': np.mean(np.abs(position_changes)),
                        'max_single_day_change': np.max(np.abs(position_changes)) if len(position_changes) > 0 else 0,
                        'volatility_trend': self._calculate_volatility_trend(position_series),
                        'stability_score': self._calculate_stability_score(position_series),
                        'change_frequency': self._calculate_change_frequency(position_changes)
                    }
                    
                    # Change point detection using TimeSeriesAnalyzer
                    changepoints = self.time_series_analyzer.detect_changepoints(
                        position_series, method='variance'
                    )
                    
                    volatility_metrics['changepoints_count'] = len(changepoints)
                    volatility_metrics['changepoints'] = changepoints
                    
                    volatility_analysis[keyword] = volatility_metrics
                
                # Calculate overall volatility statistics
                overall_volatility = self._calculate_overall_volatility_stats(volatility_analysis)
                
                return {
                    'keyword_volatility': volatility_analysis,
                    'overall_statistics': overall_volatility,
                    'high_volatility_keywords': self._identify_high_volatility_keywords(volatility_analysis),
                    'volatility_recommendations': self._generate_volatility_recommendations(volatility_analysis)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing position volatility: {str(e)}")
            return {}

    def _engineer_prediction_features(
        self,
        data: pd.DataFrame,
        competitive_data: Optional[Dict[str, pd.DataFrame]],
        config: PredictionConfig
    ) -> pd.DataFrame:
        """Engineer features for prediction using DataTransformer."""
        try:
            # Create temporal features using DataTransformer
            engineered_data = self.data_transformer.create_temporal_features(
                data, date_column='date', value_columns=['Position', 'Traffic (%)', 'Search Volume']
            )
            
            # Add SEO-specific features
            if 'Keyword' in engineered_data.columns:
                # Keyword characteristics using StringHelper
                engineered_data['keyword_length'] = engineered_data['Keyword'].apply(
                    lambda x: len(StringHelper.clean_keyword(str(x))) if pd.notna(x) else 0
                )
                
                engineered_data['keyword_word_count'] = engineered_data['Keyword'].apply(
                    lambda x: len(StringHelper.clean_keyword(str(x)).split()) if pd.notna(x) else 0
                )
            
            # Position-based features
            if 'Position' in engineered_data.columns:
                # Position tier features
                engineered_data['position_tier'] = pd.cut(
                    engineered_data['Position'], 
                    bins=[0, 3, 10, 20, 50, 100], 
                    labels=['top_3', 'top_10', 'top_20', 'top_50', 'beyond_50']
                )
                
                # Distance from page 1
                engineered_data['distance_from_page1'] = np.maximum(0, engineered_data['Position'] - 10)
            
            # Seasonal features if configured
            if config.include_seasonal_factors and 'date' in engineered_data.columns:
                engineered_data = self._add_seasonal_features(engineered_data)
            
            # Competitive features if data available
            if config.include_competitive_factors and competitive_data:
                engineered_data = self._add_competitive_features(engineered_data, competitive_data)
            
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"Error engineering prediction features: {str(e)}")
            return data

    def _train_prediction_models(
        self,
        data: pd.DataFrame,
        config: PredictionConfig
    ) -> Dict[str, Any]:
        """Train multiple prediction models."""
        try:
            trained_models = {}
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in ['Keyword', 'Position', 'date']]
            X = data[feature_columns].select_dtypes(include=[np.number]).fillna(0)
            y = data['Position']
            
            if X.empty or len(X) < 10:
                return trained_models
            
            # Train different model types
            for model_type in config.model_types:
                try:
                    if model_type == 'linear_regression':
                        model = self._train_linear_regression(X, y)
                    elif model_type == 'random_forest':
                        model = self._train_random_forest(X, y)
                    elif model_type == 'time_series':
                        model = self._train_time_series_model(data)
                    elif model_type == 'ensemble':
                        model = self._train_ensemble_model(X, y)
                    else:
                        continue
                    
                    if model is not None:
                        trained_models[model_type] = model
                        
                except Exception as e:
                    self.logger.warning(f"Failed to train {model_type} model: {str(e)}")
            
            return trained_models
            
        except Exception as e:
            self.logger.error(f"Error training prediction models: {str(e)}")
            return {}

    def _train_linear_regression(self, X: pd.DataFrame, y: pd.Series) -> Optional[Any]:
        """Train linear regression model."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_names': X.columns.tolist(),
                'model_type': 'linear_regression'
            }
            
        except Exception as e:
            self.logger.error(f"Error training linear regression: {str(e)}")
            return None

    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Optional[Any]:
        """Train random forest model."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            return {
                'model': model,
                'feature_names': X.columns.tolist(),
                'model_type': 'random_forest',
                'feature_importances': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            self.logger.error(f"Error training random forest: {str(e)}")
            return None

    def _statistical_forecast(self, series: pd.Series, horizon: int) -> Optional[float]:
        """Simple statistical forecasting using moving averages and trends."""
        try:
            # Use last 30 days or all available data
            recent_data = series.tail(min(30, len(series)))
            
            # Simple trend calculation
            x = np.arange(len(recent_data))
            y = recent_data.values
            
            # Linear regression for trend
            slope, intercept = np.polyfit(x, y, 1)
            
            # Forecast
            forecast = intercept + slope * (len(recent_data) + horizon)
            
            return float(forecast)
            
        except Exception:
            return None

    def export_prediction_results(
        self,
        prediction_report: PredictionReport,
        export_directory: str,
        include_detailed_analysis: bool = True
    ) -> Dict[str, bool]:
        """Export comprehensive prediction results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'prediction_summary': {
                    'total_predictions': len(prediction_report.predictions),
                    'average_confidence': np.mean([p.prediction_confidence for p in prediction_report.predictions]),
                    'prediction_timestamp': datetime.now().isoformat(),
                    'model_performance': {
                        name: {
                            'mae': perf.mae,
                            'rmse': perf.rmse,
                            'r2_score': perf.r2_score,
                            'directional_accuracy': perf.directional_accuracy
                        }
                        for name, perf in prediction_report.model_performance.items()
                    }
                },
                'predictions': [
                    {
                        'keyword': pred.keyword,
                        'current_position': pred.current_position,
                        'predicted_position': pred.predicted_position,
                        'prediction_date': pred.prediction_date.isoformat(),
                        'confidence_interval_lower': pred.confidence_interval_lower,
                        'confidence_interval_upper': pred.confidence_interval_upper,
                        'prediction_confidence': pred.prediction_confidence,
                        'trend_direction': pred.trend_direction,
                        'volatility_score': pred.volatility_score
                    }
                    for pred in prediction_report.predictions
                ],
                'feature_importance': prediction_report.feature_importance,
                'recommendations': prediction_report.recommendations,
                'uncertainty_analysis': prediction_report.uncertainty_analysis
            }
            
            # Export detailed data using DataExporter
            summary_export_success = self.data_exporter.export_analysis_dataset(
                {'prediction_results': pd.DataFrame([export_data])},
                export_path / "position_predictions_summary.xlsx"
            )
            
            # Export predictions as DataFrame
            if prediction_report.predictions:
                predictions_df = pd.DataFrame([
                    {
                        'keyword': pred.keyword,
                        'current_position': pred.current_position,
                        'predicted_position': pred.predicted_position,
                        'prediction_date': pred.prediction_date,
                        'confidence_lower': pred.confidence_interval_lower,
                        'confidence_upper': pred.confidence_interval_upper,
                        'confidence': pred.prediction_confidence,
                        'trend_direction': pred.trend_direction,
                        'volatility_score': pred.volatility_score
                    }
                    for pred in prediction_report.predictions
                ])
                
                predictions_export_success = self.data_exporter.export_with_metadata(
                    predictions_df,
                    metadata={'analysis_type': 'position_predictions', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "detailed_position_predictions.xlsx"
                )
            else:
                predictions_export_success = True
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_executive_report(
                export_data,
                export_path / "position_prediction_executive_report.html",
                format='html',
                include_charts=True
            )
            
            return {
                'summary_export': summary_export_success,
                'detailed_predictions': predictions_export_success,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting prediction results: {str(e)}")
            return {}

    # Helper methods using utility framework
    def _determine_trend_direction(self, trend_model: Dict[str, Any]) -> str:
        """Determine trend direction from model."""
        try:
            slope = trend_model.get('slope', 0)
            if slope > 0.1:
                return 'improving'  # Position getting better (lower numbers)
            elif slope < -0.1:
                return 'declining'  # Position getting worse (higher numbers)
            else:
                return 'stable'
        except Exception:
            return 'unknown'

    def _calculate_volatility_score(self, position_series: pd.Series) -> float:
        """Calculate volatility score using statistical methods."""
        try:
            # Use StatisticalCalculator for robust analysis
            stats_dict = self.stats_calculator.calculate_descriptive_statistics(position_series)
            
            # Normalized volatility score (0-1)
            cv = stats_dict.get('coefficient_of_variation', 0)
            volatility_score = min(cv / 2, 1.0)  # Normalize to 0-1 range
            
            return volatility_score
        except Exception:
            return 0.0

    def _calculate_prediction_confidence(
        self,
        position_series: pd.Series,
        predictions: List[float],
        trend_model: Dict[str, Any]
    ) -> float:
        """Calculate prediction confidence score."""
        try:
            confidence_factors = []
            
            # Data quality factor
            data_quality = min(len(position_series) / 30, 1.0)  # More data = higher confidence
            confidence_factors.append(data_quality)
            
            # Model agreement factor
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                prediction_mean = np.mean(predictions)
                agreement = 1 - (prediction_std / max(prediction_mean, 1))
                confidence_factors.append(max(0, agreement))
            
            # Trend stability factor
            r_squared = trend_model.get('r_squared', 0)
            confidence_factors.append(r_squared)
            
            # Historical volatility factor (inverse)
            volatility = self._calculate_volatility_score(position_series)
            stability_factor = 1 - volatility
            confidence_factors.append(stability_factor)
            
            # Overall confidence
            overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            
            return min(max(overall_confidence, 0.0), 1.0)
            
        except Exception:
            return 0.5

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for prediction operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods would be implemented here...
    def _analyze_contributing_factors(
        self,
        keyword_data: pd.DataFrame,
        trend_model: Dict[str, Any],
        seasonal_components: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze factors contributing to position changes."""
        try:
            factors = {}
            
            # Trend contribution
            if trend_model and 'slope' in trend_model:
                trend_strength = abs(trend_model['slope'])
                factors['trend'] = min(trend_strength / 10, 1.0)
            
            # Seasonal contribution
            if seasonal_components and 'seasonal_strength' in seasonal_components:
                seasonal_strength = seasonal_components['seasonal_strength'].iloc[0] if len(seasonal_components['seasonal_strength']) > 0 else 0
                factors['seasonality'] = seasonal_strength
            
            # Volume factor
            if 'Search Volume' in keyword_data.columns:
                volume_stats = self.stats_calculator.calculate_descriptive_statistics(
                    keyword_data['Search Volume'].dropna()
                )
                volume_cv = volume_stats.get('coefficient_of_variation', 0)
                factors['search_volume_variability'] = min(volume_cv, 1.0)
            
            # Normalize factors to sum to 1
            total = sum(factors.values())
            if total > 0:
                factors = {k: v/total for k, v in factors.items()}
            
            return factors
        except Exception:
            return {}

    def _calculate_stability_score(self, position_series: pd.Series) -> float:
        """Calculate position stability score."""
        try:
            # Calculate consecutive periods with minimal change
            changes = np.abs(position_series.diff()).fillna(0)
            stable_periods = (changes <= 2).sum()  # Changes of 2 positions or less
            stability_score = stable_periods / len(position_series)
            
            return stability_score
        except Exception:
            return 0.0

    def _calculate_change_frequency(self, position_changes: pd.Series) -> float:
        """Calculate frequency of position changes."""
        try:
            significant_changes = (np.abs(position_changes) > 2).sum()
            change_frequency = significant_changes / len(position_changes) if len(position_changes) > 0 else 0
            
            return change_frequency
        except Exception:
            return 0.0

    def _generate_prediction_recommendations(
        self,
        predictions: List[PredictionResult],
        model_performance: Dict[str, ModelPerformance],
        uncertainty_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on predictions."""
        try:
            recommendations = []
            
            if not predictions:
                return ["No predictions available for analysis"]
            
            # Analyze prediction patterns
            improving_keywords = [p for p in predictions if p.trend_direction == 'improving']
            declining_keywords = [p for p in predictions if p.trend_direction == 'declining']
            high_confidence_predictions = [p for p in predictions if p.prediction_confidence > 0.8]
            volatile_keywords = [p for p in predictions if p.volatility_score > 0.5]
            
            # Generate specific recommendations
            if len(improving_keywords) > len(predictions) * 0.3:
                recommendations.append(f"Positive trend detected: {len(improving_keywords)} keywords showing improvement")
            
            if len(declining_keywords) > len(predictions) * 0.3:
                recommendations.append(f"Attention needed: {len(declining_keywords)} keywords showing decline")
            
            if len(high_confidence_predictions) > len(predictions) * 0.7:
                recommendations.append("High prediction confidence - consider strategic planning based on forecasts")
            
            if len(volatile_keywords) > len(predictions) * 0.2:
                recommendations.append(f"Monitor {len(volatile_keywords)} highly volatile keywords for sudden changes")
            
            # Model performance recommendations
            best_model = max(model_performance.items(), key=lambda x: x[1].r2_score) if model_performance else None
            if best_model:
                recommendations.append(f"Best performing model: {best_model[0]} (R² = {best_model[1].r2_score:.3f})")
            
            return recommendations
        except Exception:
            return ["Review prediction results and develop action plan based on trends"]
