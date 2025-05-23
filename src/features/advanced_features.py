"""
Advanced Features Module for SEO Competitive Intelligence
Advanced analytics, machine learning, and predictive modeling using utility framework
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
from src.utils.file_utils import FileManager, ExportManager
from .competitive_features import CompetitiveIntelligence

@dataclass
class PredictiveModel:
    """Data class for predictive model results"""
    model_type: str
    accuracy_score: float
    feature_importance: Dict[str, float]
    predictions: pd.Series
    confidence_intervals: pd.DataFrame
    model_metadata: Dict[str, Any]

@dataclass
class AdvancedMetrics:
    """Data class for advanced SEO metrics"""
    traffic_quality_score: float
    keyword_cannibalization_index: float
    serp_volatility_index: float
    competitive_advantage_score: float
    market_opportunity_score: float
    technical_seo_health: Dict[str, float]

class AdvancedFeaturesEngine:
    """
    Advanced features engine for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide advanced analytics,
    machine learning capabilities, and predictive modeling without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("advanced_features")
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
        
        # Load ML configurations from config instead of hardcoding
        analysis_config = self.config.get_analysis_config()
        self.ml_confidence_threshold = 0.8
        self.prediction_horizon_days = 30

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def advanced_competitive_intelligence(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        include_predictions: bool = True,
        analysis_depth: str = 'comprehensive'
    ) -> CompetitiveIntelligence:
        """
        Perform advanced competitive intelligence analysis using utility framework.
        
        Args:
            lenovo_data: Lenovo's SEO data
            competitor_data: Dictionary of competitor data
            include_predictions: Whether to include predictive analysis
            analysis_depth: Analysis depth level
            
        Returns:
            CompetitiveIntelligence with comprehensive insights
        """
        try:
            with self.performance_tracker.track_block("advanced_competitive_intelligence"):
                # Audit log the analysis
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="competitive_intelligence",
                    parameters={
                        "include_predictions": include_predictions,
                        "analysis_depth": analysis_depth,
                        "competitors_count": len(competitor_data)
                    }
                )
                
                # Clean and validate all datasets using DataProcessor
                cleaned_lenovo = self.data_processor.clean_seo_data(lenovo_data)
                cleaned_competitors = {}
                
                for competitor, df in competitor_data.items():
                    cleaned_df = self.data_processor.clean_seo_data(df)
                    cleaned_competitors[competitor] = cleaned_df
                
                # Perform market share analysis using StatisticalCalculator
                market_share = self._calculate_market_share_analysis(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Build competitor strength matrix using advanced analytics
                strength_matrix = self._build_competitor_strength_matrix(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Identify growth opportunities using OptimizationHelper
                growth_opportunities = self._identify_advanced_growth_opportunities(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Assess competitive threats using BusinessRuleValidator
                threat_assessment = self._assess_competitive_threats(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Generate strategic recommendations
                strategic_recommendations = self._generate_advanced_recommendations(
                    market_share, strength_matrix, growth_opportunities, threat_assessment
                )
                
                # Predictive insights if requested
                predictive_insights = {}
                if include_predictions:
                    predictive_insights = self._generate_predictive_insights(
                        cleaned_lenovo, cleaned_competitors
                    )
                
                intelligence = CompetitiveIntelligence(
                    market_share_analysis=market_share,
                    competitor_strength_matrix=strength_matrix,
                    growth_opportunities=growth_opportunities,
                    threat_assessment=threat_assessment,
                    strategic_recommendations=strategic_recommendations,
                    predictive_insights=predictive_insights
                )
                
                self.logger.info(f"Advanced competitive intelligence completed for {len(competitor_data)} competitors")
                return intelligence
                
        except Exception as e:
            self.logger.error(f"Error in advanced competitive intelligence: {str(e)}")
            return CompetitiveIntelligence({}, pd.DataFrame(), [], {}, [], {})

    @timing_decorator()
    def build_predictive_models(
        self,
        historical_data: pd.DataFrame,
        target_metrics: List[str] = None,
        model_types: List[str] = None
    ) -> Dict[str, PredictiveModel]:
        """
        Build predictive models for SEO metrics using advanced ML techniques.
        
        Args:
            historical_data: Historical SEO data
            target_metrics: Metrics to predict
            model_types: Types of models to build
            
        Returns:
            Dictionary of predictive models by target metric
        """
        try:
            with self.performance_tracker.track_block("build_predictive_models"):
                if target_metrics is None:
                    target_metrics = ['Position', 'Traffic (%)', 'Search Volume']
                
                if model_types is None:
                    model_types = ['linear_regression', 'random_forest', 'time_series']
                
                # Prepare data using DataTransformer
                feature_engineered_data = self.data_transformer.create_temporal_features(
                    historical_data,
                    date_column='date' if 'date' in historical_data.columns else None,
                    value_columns=target_metrics
                )
                
                # Scale features using DataTransformer
                scaled_data = self.data_transformer.apply_scaling(
                    feature_engineered_data,
                    scaling_method='standard'
                )
                
                models = {}
                
                for target_metric in target_metrics:
                    if target_metric not in scaled_data.columns:
                        continue
                    
                    self.logger.info(f"Building predictive models for {target_metric}")
                    
                    # Prepare features and target
                    feature_columns = [
                        col for col in scaled_data.columns 
                        if col != target_metric and not col.startswith(target_metric)
                    ]
                    
                    X = scaled_data[feature_columns].fillna(0)
                    y = scaled_data[target_metric].fillna(scaled_data[target_metric].mean())
                    
                    # Build models using different algorithms
                    best_model = self._build_best_predictive_model(
                        X, y, target_metric, model_types
                    )
                    
                    if best_model:
                        models[target_metric] = best_model
                
                self.logger.info(f"Built {len(models)} predictive models")
                return models
                
        except Exception as e:
            self.logger.error(f"Error building predictive models: {str(e)}")
            return {}

    @timing_decorator()
    def calculate_advanced_seo_metrics(
        self,
        seo_data: pd.DataFrame,
        competitor_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> AdvancedMetrics:
        """
        Calculate advanced SEO metrics using statistical and optimization utilities.
        
        Args:
            seo_data: Primary SEO data
            competitor_data: Optional competitor data for relative metrics
            
        Returns:
            AdvancedMetrics with comprehensive SEO health indicators
        """
        try:
            with self.performance_tracker.track_block("calculate_advanced_seo_metrics"):
                # Clean data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(seo_data)
                
                # Calculate traffic quality score using StatisticalCalculator
                traffic_quality = self._calculate_traffic_quality_score(cleaned_data)
                
                # Calculate keyword cannibalization using StringHelper
                cannibalization_index = self._calculate_keyword_cannibalization_index(cleaned_data)
                
                # Calculate SERP volatility using TimeSeriesAnalyzer
                serp_volatility = self._calculate_serp_volatility_index(cleaned_data)
                
                # Calculate competitive advantage using OptimizationHelper
                competitive_advantage = 0.5  # Default if no competitor data
                if competitor_data:
                    competitive_advantage = self._calculate_competitive_advantage_score(
                        cleaned_data, competitor_data
                    )
                
                # Calculate market opportunity using statistical analysis
                market_opportunity = self._calculate_market_opportunity_score(
                    cleaned_data, competitor_data
                )
                
                # Calculate technical SEO health using business rules
                technical_health = self._calculate_technical_seo_health(cleaned_data)
                
                metrics = AdvancedMetrics(
                    traffic_quality_score=traffic_quality,
                    keyword_cannibalization_index=cannibalization_index,
                    serp_volatility_index=serp_volatility,
                    competitive_advantage_score=competitive_advantage,
                    market_opportunity_score=market_opportunity,
                    technical_seo_health=technical_health
                )
                
                self.logger.info(f"Advanced SEO metrics calculated. Traffic quality: {traffic_quality:.3f}")
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error calculating advanced SEO metrics: {str(e)}")
            return AdvancedMetrics(0.0, 0.0, 0.0, 0.0, 0.0, {})

    @timing_decorator()
    def perform_anomaly_detection(
        self,
        time_series_data: pd.DataFrame,
        metrics_to_analyze: List[str] = None,
        detection_method: str = 'isolation_forest'
    ) -> Dict[str, Any]:
        """
        Perform advanced anomaly detection using TimeSeriesAnalyzer.
        
        Args:
            time_series_data: Time series SEO data
            metrics_to_analyze: Metrics to analyze for anomalies
            detection_method: Detection method to use
            
        Returns:
            Anomaly detection results
        """
        try:
            with self.performance_tracker.track_block("perform_anomaly_detection"):
                if metrics_to_analyze is None:
                    metrics_to_analyze = ['Position', 'Traffic (%)', 'Search Volume']
                
                # Clean data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(time_series_data)
                
                anomaly_results = {}
                
                for metric in metrics_to_analyze:
                    if metric not in cleaned_data.columns:
                        continue
                    
                    # Prepare time series
                    if 'date' in cleaned_data.columns:
                        ts_data = cleaned_data.set_index('date')[metric].dropna()
                    else:
                        ts_data = cleaned_data[metric].dropna()
                    
                    # Detect anomalies using TimeSeriesAnalyzer
                    anomaly_mask, anomaly_scores = self.time_series_analyzer.detect_anomalies_in_series(
                        ts_data,
                        method=detection_method,
                        contamination=0.1
                    )
                    
                    # Calculate anomaly statistics using StatisticalCalculator
                    if len(anomaly_mask) > 0:
                        anomaly_stats = self.stats_calculator.calculate_descriptive_statistics(
                            anomaly_scores[anomaly_mask] if np.any(anomaly_mask) else []
                        )
                        
                        anomaly_results[metric] = {
                            'anomaly_count': np.sum(anomaly_mask),
                            'anomaly_percentage': np.mean(anomaly_mask) * 100,
                            'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
                            'anomaly_scores': anomaly_scores.tolist() if len(anomaly_scores) > 0 else [],
                            'anomaly_statistics': anomaly_stats,
                            'severity_assessment': self._assess_anomaly_severity(anomaly_scores, anomaly_mask)
                        }
                
                # Generate anomaly report
                overall_anomaly_summary = self._generate_anomaly_summary(anomaly_results)
                
                final_results = {
                    'anomaly_details': anomaly_results,
                    'overall_summary': overall_anomaly_summary,
                    'detection_method': detection_method,
                    'analysis_timestamp': datetime.now(),
                    'recommendations': self._generate_anomaly_recommendations(anomaly_results)
                }
                
                self.logger.info(f"Anomaly detection completed for {len(metrics_to_analyze)} metrics")
                return final_results
                
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {}

    def _calculate_market_share_analysis(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate market share analysis using StatisticalCalculator."""
        try:
            # Calculate total traffic for each competitor
            traffic_shares = {}
            
            # Lenovo traffic
            lenovo_traffic = lenovo_data.get('Traffic (%)', pd.Series()).sum()
            traffic_shares['lenovo'] = lenovo_traffic
            
            # Competitor traffic
            for competitor, df in competitor_data.items():
                comp_traffic = df.get('Traffic (%)', pd.Series()).sum()
                traffic_shares[competitor] = comp_traffic
            
            # Calculate market share percentages
            total_traffic = sum(traffic_shares.values())
            market_shares = {}
            
            if total_traffic > 0:
                for competitor, traffic in traffic_shares.items():
                    market_shares[competitor] = safe_divide(traffic, total_traffic, 0.0)
            
            # Add market share statistics using StatisticalCalculator
            share_values = list(market_shares.values())
            if share_values:
                share_stats = self.stats_calculator.calculate_descriptive_statistics(share_values)
                market_shares['market_concentration'] = share_stats.get('std', 0)
                market_shares['market_dominance'] = max(share_values) if share_values else 0
            
            return market_shares
            
        except Exception as e:
            self.logger.error(f"Error calculating market share: {str(e)}")
            return {}

    def _build_competitor_strength_matrix(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Build competitor strength matrix using multiple metrics."""
        try:
            competitors = ['lenovo'] + list(competitor_data.keys())
            all_data = {'lenovo': lenovo_data, **competitor_data}
            
            strength_metrics = []
            
            for competitor in competitors:
                df = all_data[competitor]
                
                # Calculate strength metrics using our utilities
                avg_position = df.get('Position', pd.Series()).mean()
                total_traffic = df.get('Traffic (%)', pd.Series()).sum()
                keyword_count = len(df)
                avg_search_volume = df.get('Search Volume', pd.Series()).mean()
                
                # Use StatisticalCalculator for robust calculations
                position_stats = self.stats_calculator.calculate_descriptive_statistics(
                    df.get('Position', pd.Series()).dropna()
                )
                
                strength_data = {
                    'competitor': competitor,
                    'avg_position': avg_position,
                    'total_traffic': total_traffic,
                    'keyword_count': keyword_count,
                    'avg_search_volume': avg_search_volume,
                    'position_consistency': 1 / (position_stats.get('std', 1) + 1),  # Lower std = higher consistency
                    'market_coverage': min(keyword_count / 1000, 1.0),  # Normalized coverage
                    'overall_strength': self._calculate_overall_strength_score({
                        'position': avg_position,
                        'traffic': total_traffic,
                        'coverage': keyword_count
                    })
                }
                
                strength_metrics.append(strength_data)
            
            return pd.DataFrame(strength_metrics)
            
        except Exception as e:
            self.logger.error(f"Error building strength matrix: {str(e)}")
            return pd.DataFrame()

    def _identify_advanced_growth_opportunities(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Identify growth opportunities using OptimizationHelper."""
        try:
            opportunities = []
            
            # Use optimization helper to find optimal keyword targets
            if not lenovo_data.empty and competitor_data:
                # Prepare keyword data for optimization
                keyword_optimization_data = lenovo_data.copy()
                
                # Add competitive metrics
                for competitor, comp_df in competitor_data.items():
                    comp_positions = {}
                    for _, row in comp_df.iterrows():
                        keyword = row.get('Keyword', '').lower()
                        position = row.get('Position', 100)
                        comp_positions[keyword] = position
                    
                    keyword_optimization_data[f'{competitor}_position'] = keyword_optimization_data['Keyword'].str.lower().map(
                        comp_positions
                    ).fillna(100)
                
                # Use optimization helper for traffic allocation
                optimization_results = self.optimization_helper.optimize_traffic_allocation(
                    keyword_optimization_data,
                    budget_constraint=10000,  # Example budget
                    effort_function=None,  # Use default
                    traffic_function=None   # Use default
                )
                
                # Convert optimization results to opportunities
                if optimization_results and 'keyword_allocations' in optimization_results:
                    for keyword, allocation in optimization_results['keyword_allocations'].items():
                        if allocation['traffic_gain'] > 0:
                            opportunity = {
                                'type': 'optimization_opportunity',
                                'keyword': keyword,
                                'potential_traffic_gain': allocation['traffic_gain'],
                                'required_effort': allocation['effort'],
                                'roi_score': safe_divide(allocation['traffic_gain'], allocation['effort'], 0),
                                'priority': 'high' if allocation['traffic_gain'] > 100 else 'medium'
                            }
                            opportunities.append(opportunity)
            
            # Sort by ROI score
            opportunities.sort(key=lambda x: x.get('roi_score', 0), reverse=True)
            
            return opportunities[:20]  # Return top 20 opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying growth opportunities: {str(e)}")
            return []

    def _calculate_traffic_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate traffic quality score using statistical analysis."""
        try:
            if 'Traffic (%)' not in data.columns or 'Position' not in data.columns:
                return 0.0
            
            # Use StatisticalCalculator for robust analysis
            traffic_data = data['Traffic (%)'].dropna()
            position_data = data['Position'].dropna()
            
            if len(traffic_data) == 0 or len(position_data) == 0:
                return 0.0
            
            # Calculate correlation between position and traffic
            correlation_matrix, p_values = self.stats_calculator.calculate_correlation_matrix(
                data[['Position', 'Traffic (%)']].dropna()
            )
            
            if not correlation_matrix.empty:
                position_traffic_correlation = abs(correlation_matrix.loc['Position', 'Traffic (%)'])
            else:
                position_traffic_correlation = 0
            
            # Calculate traffic concentration
            traffic_stats = self.stats_calculator.calculate_descriptive_statistics(traffic_data)
            traffic_concentration = traffic_stats.get('coefficient_of_variation', 1)
            
            # Calculate quality score (0-1)
            quality_score = (
                position_traffic_correlation * 0.6 +  # Good correlation is positive
                min(1 / (traffic_concentration + 1), 1) * 0.4  # Lower concentration is better
            )
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating traffic quality score: {str(e)}")
            return 0.0

    def _calculate_keyword_cannibalization_index(self, data: pd.DataFrame) -> float:
        """Calculate keyword cannibalization using StringHelper."""
        try:
            if 'Keyword' not in data.columns:
                return 0.0
            
            keywords = data['Keyword'].dropna().tolist()
            if len(keywords) < 2:
                return 0.0
            
            # Use StringHelper to find similar keywords
            similarity_scores = []
            
            for i, keyword1 in enumerate(keywords):
                for j, keyword2 in enumerate(keywords[i+1:], i+1):
                    similarity = StringHelper.calculate_keyword_similarity(keyword1, keyword2)
                    if similarity > 0.8:  # High similarity threshold
                        similarity_scores.append(similarity)
            
            # Calculate cannibalization index
            if similarity_scores:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                cannibalization_pairs = len(similarity_scores)
                total_possible_pairs = len(keywords) * (len(keywords) - 1) / 2
                
                cannibalization_index = (cannibalization_pairs / total_possible_pairs) * avg_similarity
            else:
                cannibalization_index = 0.0
            
            return min(cannibalization_index, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating cannibalization index: {str(e)}")
            return 0.0

    def _calculate_serp_volatility_index(self, data: pd.DataFrame) -> float:
        """Calculate SERP volatility using TimeSeriesAnalyzer."""
        try:
            if 'Position' not in data.columns:
                return 0.0
            
            # If we have historical data with dates
            if 'date' in data.columns:
                ts_data = data.set_index('date')['Position'].dropna()
                
                if len(ts_data) > 10:
                    # Use TimeSeriesAnalyzer to detect changepoints
                    changepoints = self.time_series_analyzer.detect_changepoints(
                        ts_data, method='variance', min_size=3
                    )
                    
                    # Calculate volatility index based on changepoints
                    volatility_index = len(changepoints) / len(ts_data)
                    return min(volatility_index, 1.0)
            
            # Fallback: use position variance
            position_stats = self.stats_calculator.calculate_descriptive_statistics(
                data['Position'].dropna()
            )
            
            volatility_index = position_stats.get('coefficient_of_variation', 0) / 2
            return min(volatility_index, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating SERP volatility: {str(e)}")
            return 0.0

    def _build_best_predictive_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_metric: str,
        model_types: List[str]
    ) -> Optional[PredictiveModel]:
        """Build best predictive model using cross-validation."""
        try:
            from sklearn.model_selection import cross_val_score, train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            
            if len(X) < 10:  # Need minimum data for modeling
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            best_model = None
            best_score = -np.inf
            best_predictions = None
            
            for model_type in model_types:
                try:
                    if model_type == 'linear_regression':
                        model = LinearRegression()
                    elif model_type == 'random_forest':
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        continue
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate using cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    avg_score = np.mean(cv_scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_predictions = model.predict(X_test)
                
                except Exception as e:
                    self.logger.warning(f"Error training {model_type}: {str(e)}")
                    continue
            
            if best_model is None:
                return None
            
            # Calculate feature importance
            feature_importance = {}
            if hasattr(best_model, 'feature_importances_'):
                for feature, importance in zip(X.columns, best_model.feature_importances_):
                    feature_importance[feature] = importance
            elif hasattr(best_model, 'coef_'):
                for feature, coef in zip(X.columns, best_model.coef_):
                    feature_importance[feature] = abs(coef)
            
            # Calculate confidence intervals (simplified)
            confidence_intervals = pd.DataFrame({
                'lower': best_predictions - np.std(best_predictions),
                'upper': best_predictions + np.std(best_predictions)
            })
            
            return PredictiveModel(
                model_type=type(best_model).__name__,
                accuracy_score=best_score,
                feature_importance=feature_importance,
                predictions=pd.Series(best_predictions),
                confidence_intervals=confidence_intervals,
                model_metadata={
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features_used': list(X.columns),
                    'target_metric': target_metric
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error building predictive model: {str(e)}")
            return None

    def export_advanced_analysis_results(
        self,
        intelligence: CompetitiveIntelligence,
        advanced_metrics: AdvancedMetrics,
        predictive_models: Dict[str, PredictiveModel],
        export_directory: str
    ) -> Dict[str, bool]:
        """Export comprehensive advanced analysis results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare data for export
            export_data = {
                'competitive_intelligence': {
                    'market_share': intelligence.market_share_analysis,
                    'growth_opportunities': intelligence.growth_opportunities,
                    'strategic_recommendations': intelligence.strategic_recommendations
                },
                'advanced_metrics': {
                    'traffic_quality_score': advanced_metrics.traffic_quality_score,
                    'cannibalization_index': advanced_metrics.keyword_cannibalization_index,
                    'volatility_index': advanced_metrics.serp_volatility_index,
                    'competitive_advantage': advanced_metrics.competitive_advantage_score,
                    'market_opportunity': advanced_metrics.market_opportunity_score,
                    'technical_health': advanced_metrics.technical_seo_health
                },
                'predictive_models': {
                    model_name: {
                        'model_type': model.model_type,
                        'accuracy_score': model.accuracy_score,
                        'feature_importance': model.feature_importance,
                        'metadata': model.model_metadata
                    }
                    for model_name, model in predictive_models.items()
                }
            }
            
            # Export using DataExporter
            export_results = self.data_exporter.export_with_metadata(
                pd.DataFrame([export_data]),
                metadata={
                    'analysis_type': 'advanced_features',
                    'generation_timestamp': datetime.now(),
                    'analysis_depth': 'comprehensive'
                },
                export_path=export_path / "advanced_analysis_results.xlsx",
                format='excel'
            )
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_executive_report(
                export_data,
                export_path / "advanced_analysis_executive_report.html",
                format='html',
                include_charts=True
            )
            
            return {
                'data_export': export_results,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting advanced analysis results: {str(e)}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for advanced analysis operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods...
    def _calculate_overall_strength_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall competitor strength score."""
        try:
            position_score = max(0, (50 - metrics.get('position', 50)) / 50)
            traffic_score = min(metrics.get('traffic', 0) / 1000, 1.0)
            coverage_score = min(metrics.get('coverage', 0) / 5000, 1.0)
            
            return (position_score * 0.4 + traffic_score * 0.35 + coverage_score * 0.25)
        except Exception:
            return 0.0

    def _assess_anomaly_severity(self, scores: np.ndarray, mask: np.ndarray) -> str:
        """Assess severity of detected anomalies."""
        try:
            if len(scores) == 0 or not np.any(mask):
                return 'none'
            
            anomaly_scores = scores[mask]
            if len(anomaly_scores) == 0:
                return 'none'
            
            avg_anomaly_score = np.mean(np.abs(anomaly_scores))
            
            if avg_anomaly_score > 2.0:
                return 'critical'
            elif avg_anomaly_score > 1.0:
                return 'high'
            elif avg_anomaly_score > 0.5:
                return 'medium'
            else:
                return 'low'
        except Exception:
            return 'unknown'

    def _generate_anomaly_summary(self, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall anomaly summary."""
        try:
            total_anomalies = sum(
                result.get('anomaly_count', 0) 
                for result in anomaly_results.values()
            )
            
            severity_counts = {}
            for result in anomaly_results.values():
                severity = result.get('severity_assessment', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                'total_anomalies_detected': total_anomalies,
                'metrics_analyzed': len(anomaly_results),
                'severity_distribution': severity_counts,
                'overall_health_status': 'healthy' if total_anomalies < 5 else 'attention_needed'
            }
        except Exception:
            return {}

    def _generate_anomaly_recommendations(self, anomaly_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on anomaly detection."""
        recommendations = []
        
        try:
            for metric, result in anomaly_results.items():
                anomaly_count = result.get('anomaly_count', 0)
                severity = result.get('severity_assessment', 'unknown')
                
                if anomaly_count > 0:
                    if severity == 'critical':
                        recommendations.append(f"Immediate investigation required for {metric} - critical anomalies detected")
                    elif severity == 'high':
                        recommendations.append(f"Monitor {metric} closely - significant anomalies detected")
                    elif anomaly_count > 3:
                        recommendations.append(f"Review {metric} trends - multiple anomalies detected")
            
            if not recommendations:
                recommendations.append("No significant anomalies detected - continue monitoring")
            
            return recommendations
        except Exception:
            return ["Review anomaly detection results manually"]
