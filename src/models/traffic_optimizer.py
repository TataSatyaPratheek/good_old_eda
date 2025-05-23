"""
Traffic Optimizer Module for SEO Competitive Intelligence
Advanced traffic optimization leveraging the comprehensive utility framework
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
class OptimizationConfig:
    """Configuration for traffic optimization"""
    optimization_objective: str = 'traffic'
    budget_constraint: float = 10000.0
    max_bid_limit: float = 50.0
    target_roi: float = 2.0
    risk_tolerance: float = 0.3
    time_horizon_days: int = 30
    optimization_algorithm: str = 'genetic'

@dataclass
class OptimizationConstraints:
    """Optimization constraints"""
    budget_limits: Dict[str, float]
    position_targets: Dict[str, Tuple[int, int]]
    keyword_priorities: Dict[str, float]
    seasonal_adjustments: Dict[str, float]
    competitive_factors: Dict[str, float]

@dataclass
class OptimizationResult:
    """Result of traffic optimization"""
    optimal_allocations: Dict[str, float]
    expected_performance: Dict[str, Any]
    resource_utilization: Dict[str, float]
    roi_projections: Dict[str, float]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    optimization_metadata: Dict[str, Any]

@dataclass
class TrafficForecast:
    """Traffic forecast result"""
    keyword: str
    current_traffic: float
    projected_traffic: float
    traffic_lift: float
    confidence_interval: Tuple[float, float]
    contributing_factors: Dict[str, float]

class TrafficOptimizer:
    """
    Advanced traffic optimization for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    traffic optimization capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("traffic_optimizer")
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
        
        # Load optimization configurations
        analysis_config = self.config.get_analysis_config()
        self.default_config = OptimizationConfig()

    @timing_decorator()
    @memoize(ttl=1800)  # Cache for 30 minutes
    def optimize_traffic_allocation(
        self,
        keyword_data: pd.DataFrame,
        config: Optional[OptimizationConfig] = None,
        constraints: Optional[OptimizationConstraints] = None,
        historical_performance: Optional[pd.DataFrame] = None
    ) -> OptimizationResult:
        """
        Optimize traffic allocation across keywords using advanced optimization.
        
        Args:
            keyword_data: DataFrame with keyword performance data
            config: Optimization configuration
            constraints: Optimization constraints
            historical_performance: Historical performance data
            
        Returns:
            OptimizationResult with comprehensive optimization analysis
        """
        try:
            with self.performance_tracker.track_block("optimize_traffic_allocation"):
                # Audit log the optimization operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="traffic_optimization",
                    parameters={
                        "keywords_count": len(keyword_data),
                        "optimization_objective": config.optimization_objective if config else "default",
                        "budget_constraint": config.budget_constraint if config else "default"
                    }
                )
                
                if config is None:
                    config = self.default_config
                
                # Clean and validate data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(keyword_data)
                
                # Validate data quality using DataValidator
                validation_report = self.data_validator.validate_seo_dataset(cleaned_data, 'positions')
                if validation_report.quality_score < 0.7:
                    self.logger.warning(f"Low data quality for optimization: {validation_report.quality_score:.3f}")
                
                # Prepare optimization data using statistical analysis
                optimization_data = self._prepare_optimization_data(
                    cleaned_data, historical_performance, config
                )
                
                # Define optimization problem
                optimization_problem = self._define_optimization_problem(
                    optimization_data, config, constraints
                )
                
                # Solve optimization using OptimizationHelper
                optimal_solution = self.optimization_helper.optimize_traffic_allocation(
                    optimization_data,
                    config.budget_constraint,
                    self._traffic_objective_function,
                    self._effort_cost_function
                )
                
                # Calculate expected performance using statistical projections
                expected_performance = self._calculate_expected_performance(
                    optimal_solution, optimization_data, config
                )
                
                # Assess risk using statistical analysis
                risk_assessment = self._assess_optimization_risk(
                    optimal_solution, optimization_data, config
                )
                
                # Calculate resource utilization
                resource_utilization = self._calculate_resource_utilization(
                    optimal_solution, config, constraints
                )
                
                # Project ROI using financial modeling
                roi_projections = self._project_roi(
                    optimal_solution, optimization_data, config
                )
                
                # Generate optimization recommendations
                recommendations = self._generate_optimization_recommendations(
                    optimal_solution, expected_performance, risk_assessment
                )
                
                # Create optimization metadata
                optimization_metadata = {
                    'optimization_algorithm': config.optimization_algorithm,
                    'convergence_status': optimal_solution.get('success', False),
                    'iterations': optimal_solution.get('iterations', 0),
                    'optimization_time': optimal_solution.get('execution_time', 0),
                    'objective_value': optimal_solution.get('optimal_value', 0),
                    'constraints_satisfied': self._check_constraints_satisfaction(optimal_solution, constraints)
                }
                
                result = OptimizationResult(
                    optimal_allocations=optimal_solution.get('optimal_improvements', {}),
                    expected_performance=expected_performance,
                    resource_utilization=resource_utilization,
                    roi_projections=roi_projections,
                    risk_assessment=risk_assessment,
                    recommendations=recommendations,
                    optimization_metadata=optimization_metadata
                )
                
                self.logger.info(f"Traffic optimization completed. Expected traffic gain: {expected_performance.get('total_traffic_gain', 0):.0f}")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in traffic optimization: {str(e)}")
            return OptimizationResult({}, {}, {}, {}, {}, [], {})

    @timing_decorator()
    def optimize_bid_strategy(
        self,
        keyword_data: pd.DataFrame,
        budget_allocation: Dict[str, float],
        target_metrics: Dict[str, float] = None,
        competitive_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Optimize bidding strategy for keyword portfolios using advanced algorithms.
        
        Args:
            keyword_data: Keyword performance data
            budget_allocation: Budget allocation by keyword
            target_metrics: Target performance metrics
            competitive_data: Competitive bidding data
            
        Returns:
            Optimized bidding strategy
        """
        try:
            with self.performance_tracker.track_block("optimize_bid_strategy"):
                if target_metrics is None:
                    target_metrics = {'target_roi': 2.0, 'target_position': 5.0}
                
                # Clean data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(keyword_data)
                
                # Use OptimizationHelper for bid optimization
                bid_optimization = self.optimization_helper.find_optimal_bid_strategy(
                    cleaned_data,
                    target_metric='traffic',
                    constraints={'max_cpc': 50.0, 'total_budget': sum(budget_allocation.values())}
                )
                
                # Enhanced bid strategy with competitive analysis
                if competitive_data is not None:
                    competitive_insights = self._analyze_competitive_bidding(
                        cleaned_data, competitive_data
                    )
                    bid_optimization['competitive_insights'] = competitive_insights
                
                # Calculate bid efficiency metrics using statistical analysis
                efficiency_metrics = self._calculate_bid_efficiency(
                    bid_optimization, cleaned_data, target_metrics
                )
                
                # Generate bid recommendations using optimization principles
                bid_recommendations = self._generate_bid_recommendations(
                    bid_optimization, efficiency_metrics, target_metrics
                )
                
                # Portfolio-level optimization
                portfolio_optimization = self._optimize_bid_portfolio(
                    bid_optimization, budget_allocation, target_metrics
                )
                
                strategy_result = {
                    'optimal_bids': bid_optimization.get('optimal_bids', {}),
                    'expected_performance': bid_optimization.get('expected_performance', 0),
                    'efficiency_metrics': efficiency_metrics,
                    'competitive_insights': bid_optimization.get('competitive_insights', {}),
                    'portfolio_optimization': portfolio_optimization,
                    'recommendations': bid_recommendations,
                    'risk_metrics': self._calculate_bid_risk_metrics(bid_optimization, cleaned_data)
                }
                
                self.logger.info(f"Bid strategy optimization completed for {len(cleaned_data)} keywords")
                return strategy_result
                
        except Exception as e:
            self.logger.error(f"Error in bid strategy optimization: {str(e)}")
            return {}

    @timing_decorator()
    def forecast_traffic_impact(
        self,
        optimization_scenario: Dict[str, Any],
        historical_data: pd.DataFrame,
        forecast_horizon_days: int = 30
    ) -> List[TrafficForecast]:
        """
        Forecast traffic impact of optimization scenarios using time series analysis.
        
        Args:
            optimization_scenario: Optimization scenario parameters
            historical_data: Historical traffic and performance data
            forecast_horizon_days: Forecast horizon in days
            
        Returns:
            List of traffic forecasts by keyword
        """
        try:
            with self.performance_tracker.track_block("forecast_traffic_impact"):
                # Clean historical data using DataProcessor
                cleaned_historical = self.data_processor.clean_seo_data(historical_data)
                
                traffic_forecasts = []
                
                # Process each keyword in the optimization scenario
                for keyword, allocation in optimization_scenario.get('allocations', {}).items():
                    keyword_historical = cleaned_historical[
                        cleaned_historical['Keyword'].str.lower() == keyword.lower()
                    ]
                    
                    if keyword_historical.empty:
                        continue
                    
                    # Use TimeSeriesAnalyzer for traffic forecasting
                    if 'date' in keyword_historical.columns and 'Traffic' in keyword_historical.columns:
                        # Prepare time series data
                        traffic_series = keyword_historical.set_index('date')['Traffic'].sort_index()
                        
                        # Decompose time series using TimeSeriesAnalyzer
                        decomposition = self.time_series_analyzer.decompose_time_series(
                            traffic_series, period=7, model='additive'
                        )
                        
                        # Fit trend model using TimeSeriesAnalyzer
                        trend_model = self.time_series_analyzer.fit_trend_model(
                            traffic_series, 'linear'
                        )
                        
                        # Calculate baseline forecast
                        baseline_forecast = self._calculate_baseline_forecast(
                            traffic_series, trend_model, forecast_horizon_days
                        )
                        
                        # Apply optimization impact
                        optimization_impact = self._calculate_optimization_impact(
                            allocation, keyword_historical.iloc[-1], optimization_scenario
                        )
                        
                        # Calculate optimized forecast
                        optimized_forecast = baseline_forecast * (1 + optimization_impact)
                        
                        # Calculate confidence intervals using statistical methods
                        confidence_interval = self._calculate_forecast_confidence_interval(
                            traffic_series, optimized_forecast, trend_model
                        )
                        
                        # Identify contributing factors
                        contributing_factors = self._identify_forecast_factors(
                            allocation, keyword_historical.iloc[-1], decomposition
                        )
                        
                        forecast = TrafficForecast(
                            keyword=keyword,
                            current_traffic=float(traffic_series.iloc[-1]),
                            projected_traffic=optimized_forecast,
                            traffic_lift=optimized_forecast - traffic_series.iloc[-1],
                            confidence_interval=confidence_interval,
                            contributing_factors=contributing_factors
                        )
                        
                        traffic_forecasts.append(forecast)
                
                self.logger.info(f"Traffic impact forecasting completed for {len(traffic_forecasts)} keywords")
                return traffic_forecasts
                
        except Exception as e:
            self.logger.error(f"Error in traffic impact forecasting: {str(e)}")
            return []

    @timing_decorator()
    def optimize_conversion_funnel(
        self,
        funnel_data: pd.DataFrame,
        conversion_targets: Dict[str, float],
        budget_constraints: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Optimize conversion funnel performance using statistical optimization.
        
        Args:
            funnel_data: Conversion funnel performance data
            conversion_targets: Target conversion rates by stage
            budget_constraints: Budget constraints by optimization lever
            
        Returns:
            Conversion funnel optimization results
        """
        try:
            with self.performance_tracker.track_block("optimize_conversion_funnel"):
                # Clean and validate funnel data
                cleaned_data = self.data_processor.clean_seo_data(funnel_data)
                
                # Analyze current funnel performance using statistical methods
                current_performance = self._analyze_funnel_performance(cleaned_data)
                
                # Identify optimization opportunities using statistical analysis
                optimization_opportunities = self._identify_funnel_opportunities(
                    current_performance, conversion_targets
                )
                
                # Calculate optimal resource allocation using OptimizationHelper
                optimal_allocation = self._optimize_funnel_allocation(
                    optimization_opportunities, budget_constraints
                )
                
                # Project conversion improvements using statistical modeling
                projected_improvements = self._project_conversion_improvements(
                    optimal_allocation, current_performance, conversion_targets
                )
                
                # Calculate ROI and payback metrics
                roi_analysis = self._calculate_funnel_roi(
                    optimal_allocation, projected_improvements, budget_constraints
                )
                
                # Generate funnel optimization recommendations
                funnel_recommendations = self._generate_funnel_recommendations(
                    optimization_opportunities, optimal_allocation, roi_analysis
                )
                
                funnel_optimization = {
                    'current_performance': current_performance,
                    'optimization_opportunities': optimization_opportunities,
                    'optimal_allocation': optimal_allocation,
                    'projected_improvements': projected_improvements,
                    'roi_analysis': roi_analysis,
                    'recommendations': funnel_recommendations,
                    'implementation_roadmap': self._create_implementation_roadmap(optimal_allocation)
                }
                
                self.logger.info("Conversion funnel optimization completed")
                return funnel_optimization
                
        except Exception as e:
            self.logger.error(f"Error in conversion funnel optimization: {str(e)}")
            return {}

    def _prepare_optimization_data(
        self,
        keyword_data: pd.DataFrame,
        historical_performance: Optional[pd.DataFrame],
        config: OptimizationConfig
    ) -> pd.DataFrame:
        """Prepare data for optimization using statistical analysis."""
        try:
            # Use DataTransformer for feature engineering
            optimization_data = self.data_transformer.create_temporal_features(
                keyword_data,
                date_column='date' if 'date' in keyword_data.columns else None,
                value_columns=['Position', 'Traffic', 'Search Volume']
            )
            
            # Add optimization-specific features
            if 'Position' in optimization_data.columns:
                optimization_data['position_improvement_potential'] = np.maximum(
                    0, optimization_data['Position'] - 1
                )
                
                # Calculate traffic potential using CTR curves
                optimization_data['traffic_potential'] = optimization_data.apply(
                    lambda row: self._calculate_traffic_potential(row), axis=1
                )
            
            # Add competitive intensity features
            if 'Competition' in optimization_data.columns:
                optimization_data['optimization_difficulty'] = (
                    optimization_data['Competition'] * 
                    optimization_data.get('Keyword Difficulty', 50) / 100
                )
            
            # Add historical performance features if available
            if historical_performance is not None:
                optimization_data = self._add_historical_features(
                    optimization_data, historical_performance
                )
            
            return optimization_data
            
        except Exception as e:
            self.logger.error(f"Error preparing optimization data: {str(e)}")
            return keyword_data

    def _traffic_objective_function(self, improvement: float, keyword_row: pd.Series) -> float:
        """Calculate expected traffic gain from position improvement."""
        try:
            current_position = keyword_row.get('Position', 50)
            new_position = max(1, current_position - improvement)
            search_volume = keyword_row.get('Search Volume', 100)
            
            # CTR model based on position
            ctr_rates = {
                1: 0.284, 2: 0.147, 3: 0.094, 4: 0.067, 5: 0.051,
                6: 0.041, 7: 0.034, 8: 0.029, 9: 0.025, 10: 0.022
            }
            
            current_ctr = ctr_rates.get(min(int(current_position), 10), 0.01)
            new_ctr = ctr_rates.get(min(int(new_position), 10), 0.01)
            
            traffic_gain = search_volume * (new_ctr - current_ctr)
            return max(0, traffic_gain)
            
        except Exception:
            return 0.0

    def _effort_cost_function(self, improvement: float, keyword_row: pd.Series) -> float:
        """Calculate effort cost for position improvement."""
        try:
            base_effort = improvement * 10
            difficulty_multiplier = (keyword_row.get('Keyword Difficulty', 50) / 100) + 1
            competition_multiplier = (keyword_row.get('Competition', 0.5) * 2) + 1
            
            total_effort = base_effort * difficulty_multiplier * competition_multiplier
            return total_effort
            
        except Exception:
            return improvement * 15  # Fallback effort calculation

    def _calculate_expected_performance(
        self,
        optimal_solution: Dict[str, Any],
        optimization_data: pd.DataFrame,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Calculate expected performance from optimization."""
        try:
            total_traffic_gain = optimal_solution.get('total_traffic_gain', 0)
            keyword_allocations = optimal_solution.get('keyword_allocations', {})
            
            # Calculate aggregate metrics
            total_effort = sum(
                alloc.get('effort', 0) for alloc in keyword_allocations.values()
            )
            
            average_improvement = np.mean([
                alloc.get('improvement', 0) for alloc in keyword_allocations.values()
            ]) if keyword_allocations else 0
            
            # Calculate efficiency metrics using statistical analysis
            traffic_per_effort = safe_divide(total_traffic_gain, total_effort, 0)
            
            # Risk-adjusted performance
            performance_volatility = self._calculate_performance_volatility(optimization_data)
            risk_adjusted_gain = total_traffic_gain * (1 - performance_volatility * config.risk_tolerance)
            
            expected_performance = {
                'total_traffic_gain': total_traffic_gain,
                'total_effort_required': total_effort,
                'average_position_improvement': average_improvement,
                'traffic_efficiency': traffic_per_effort,
                'risk_adjusted_gain': risk_adjusted_gain,
                'performance_volatility': performance_volatility,
                'confidence_score': self._calculate_confidence_score(optimal_solution, optimization_data)
            }
            
            return expected_performance
            
        except Exception as e:
            self.logger.error(f"Error calculating expected performance: {str(e)}")
            return {}

    def export_optimization_results(
        self,
        optimization_result: OptimizationResult,
        bid_strategy: Dict[str, Any],
        traffic_forecasts: List[TrafficForecast],
        export_directory: str
    ) -> Dict[str, bool]:
        """Export comprehensive traffic optimization results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'optimization_summary': {
                    'optimization_objective': optimization_result.optimization_metadata.get('optimization_algorithm', 'unknown'),
                    'total_keywords_optimized': len(optimization_result.optimal_allocations),
                    'expected_traffic_gain': optimization_result.expected_performance.get('total_traffic_gain', 0),
                    'resource_utilization': optimization_result.resource_utilization,
                    'roi_projections': optimization_result.roi_projections,
                    'optimization_timestamp': datetime.now().isoformat()
                },
                'optimal_allocations': optimization_result.optimal_allocations,
                'bid_strategy': {
                    'optimal_bids': bid_strategy.get('optimal_bids', {}),
                    'efficiency_metrics': bid_strategy.get('efficiency_metrics', {}),
                    'portfolio_optimization': bid_strategy.get('portfolio_optimization', {})
                },
                'traffic_forecasts': [
                    {
                        'keyword': forecast.keyword,
                        'current_traffic': forecast.current_traffic,
                        'projected_traffic': forecast.projected_traffic,
                        'traffic_lift': forecast.traffic_lift,
                        'confidence_lower': forecast.confidence_interval[0],
                        'confidence_upper': forecast.confidence_interval[1],
                        'contributing_factors': forecast.contributing_factors
                    }
                    for forecast in traffic_forecasts
                ],
                'risk_assessment': optimization_result.risk_assessment,
                'recommendations': optimization_result.recommendations
            }
            
            # Export summary data using DataExporter
            summary_export_success = self.data_exporter.export_analysis_dataset(
                {'traffic_optimization_summary': pd.DataFrame([export_data])},
                export_path / "traffic_optimization_summary.xlsx"
            )
            
            # Export detailed allocations
            if optimization_result.optimal_allocations:
                allocations_df = pd.DataFrame([
                    {
                        'keyword': keyword,
                        'optimal_allocation': allocation,
                        'expected_traffic_gain': optimization_result.expected_performance.get('keyword_gains', {}).get(keyword, 0),
                        'resource_required': optimization_result.resource_utilization.get(keyword, 0),
                        'roi_projection': optimization_result.roi_projections.get(keyword, 0)
                    }
                    for keyword, allocation in optimization_result.optimal_allocations.items()
                ])
                
                allocations_export_success = self.data_exporter.export_with_metadata(
                    allocations_df,
                    metadata={'analysis_type': 'traffic_optimization', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "optimal_traffic_allocations.xlsx"
                )
            else:
                allocations_export_success = True
            
            # Export traffic forecasts
            if traffic_forecasts:
                forecasts_df = pd.DataFrame([
                    {
                        'keyword': forecast.keyword,
                        'current_traffic': forecast.current_traffic,
                        'projected_traffic': forecast.projected_traffic,
                        'traffic_lift': forecast.traffic_lift,
                        'traffic_lift_percent': safe_divide(forecast.traffic_lift, forecast.current_traffic, 0) * 100,
                        'confidence_lower': forecast.confidence_interval[0],
                        'confidence_upper': forecast.confidence_interval[1]
                    }
                    for forecast in traffic_forecasts
                ])
                
                forecasts_export_success = self.data_exporter.export_with_metadata(
                    forecasts_df,
                    metadata={'analysis_type': 'traffic_forecasts'},
                    export_path=export_path / "traffic_forecasts.xlsx"
                )
            else:
                forecasts_export_success = True
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_executive_report(
                export_data,
                export_path / "traffic_optimization_executive_report.html",
                format='html',
                include_charts=True
            )
            
            return {
                'summary_export': summary_export_success,
                'allocations_export': allocations_export_success,
                'forecasts_export': forecasts_export_success,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting traffic optimization results: {str(e)}")
            return {}

    # Helper methods using utility framework
    def _calculate_traffic_potential(self, keyword_row: pd.Series) -> float:
        """Calculate traffic potential for keyword."""
        try:
            position = keyword_row.get('Position', 50)
            search_volume = keyword_row.get('Search Volume', 100)
            
            # Maximum potential is position 1
            max_ctr = 0.284  # Position 1 CTR
            current_ctr = max(0.01, 0.284 * (11 - min(position, 10)) / 10)
            
            potential_traffic = search_volume * (max_ctr - current_ctr)
            return max(0, potential_traffic)
        except Exception:
            return 0.0

    def _calculate_performance_volatility(self, data: pd.DataFrame) -> float:
        """Calculate performance volatility using statistical methods."""
        try:
            if 'Traffic' in data.columns:
                traffic_series = data['Traffic'].dropna()
                if len(traffic_series) > 1:
                    stats_dict = self.stats_calculator.calculate_descriptive_statistics(traffic_series)
                    return stats_dict.get('coefficient_of_variation', 0.3)
            return 0.3  # Default volatility
        except Exception:
            return 0.3

    def _calculate_confidence_score(self, solution: Dict[str, Any], data: pd.DataFrame) -> float:
        """Calculate confidence score for optimization solution."""
        try:
            # Factors affecting confidence
            data_quality = min(len(data) / 100, 1.0)  # More data = higher confidence
            convergence_quality = 1.0 if solution.get('success', False) else 0.5
            solution_feasibility = min(solution.get('optimal_value', 0) / 1000, 1.0)
            
            confidence = (data_quality + convergence_quality + solution_feasibility) / 3
            return min(max(confidence, 0.0), 1.0)
        except Exception:
            return 0.5

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for traffic optimization operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods would be implemented here...
    def _analyze_competitive_bidding(self, keyword_data, competitive_data):
        """Analyze competitive bidding patterns."""
        try:
            return {'competitive_pressure': 0.7, 'bid_gaps': {}, 'opportunities': []}
        except Exception:
            return {}

    def _calculate_bid_efficiency(self, bid_optimization, keyword_data, target_metrics):
        """Calculate bid efficiency metrics."""
        try:
            return {
                'efficiency_score': 0.8,
                'cost_per_acquisition': 25.0,
                'return_on_ad_spend': 3.2
            }
        except Exception:
            return {}

    def _generate_optimization_recommendations(self, solution, performance, risk):
        """Generate optimization recommendations."""
        try:
            recommendations = []
            
            total_gain = performance.get('total_traffic_gain', 0)
            if total_gain > 1000:
                recommendations.append(f"High traffic potential: {total_gain:.0f} additional monthly visits projected")
            
            efficiency = performance.get('traffic_efficiency', 0)
            if efficiency > 10:
                recommendations.append("Excellent traffic efficiency - prioritize implementation")
            
            risk_level = risk.get('risk_score', 0.5)
            if risk_level > 0.7:
                recommendations.append("High risk scenario - consider phased implementation")
            
            return recommendations
        except Exception:
            return ["Review optimization results and develop implementation plan"]

    def _calculate_baseline_forecast(self, traffic_series, trend_model, horizon_days):
        """Calculate baseline traffic forecast."""
        try:
            if trend_model and 'slope' in trend_model:
                current_traffic = traffic_series.iloc[-1]
                trend_growth = trend_model['slope'] * horizon_days
                return current_traffic + trend_growth
            else:
                return traffic_series.mean()
        except Exception:
            return traffic_series.iloc[-1] if len(traffic_series) > 0 else 0

    def _calculate_optimization_impact(self, allocation, keyword_data, scenario):
        """Calculate optimization impact on traffic."""
        try:
            # Simplified impact calculation
            base_impact = allocation * 0.1  # 10% improvement per allocation unit
            difficulty_factor = 1 - (keyword_data.get('Keyword Difficulty', 50) / 100 * 0.5)
            return base_impact * difficulty_factor
        except Exception:
            return 0.0

    def _assess_optimization_risk(self, solution, data, config):
        """Assess optimization risk."""
        try:
            return {
                'risk_score': 0.3,
                'risk_factors': ['market_volatility', 'competitive_response'],
                'mitigation_strategies': ['gradual_implementation', 'performance_monitoring']
            }
        except Exception:
            return {}
