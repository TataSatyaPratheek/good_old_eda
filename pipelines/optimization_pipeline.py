"""
Optimization Pipeline
Comprehensive optimization pipeline leveraging refactored modules and src/utils
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import field

# Import refactored modules
from src.models.traffic_optimizer import TrafficOptimizer, OptimizationResult, TrafficForecast
from src.models.position_predictor import PositionPredictor, PredictionReport
from src.models.ensemble_models import EnsembleModelManager
from src.features.feature_selector import FeatureSelector

# Import utils framework
from src.utils.common_helpers import timing_decorator, memoize
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.export_utils import ReportExporter, DataExporter
from src.utils.math_utils import StatisticalCalculator, OptimizationHelper

# Import pipeline configuration
from .pipeline_config import PipelineConfigManager

class OptimizationConfig:
    """Optimization configuration class"""
    def __init__(self):
        self.target_roi = 2.0  # Add this missing attribute
        self.max_budget = 10000.0
        self.risk_tolerance = 0.3
        self.optimization_objectives: List[str] = field(default_factory=lambda: ['traffic', 'positions'])
        self.budget_constraints: Dict[str, float] = field(default_factory=dict)
        self.risk_tolerance: float = 0.3
        self.prediction_horizon_days: int = 30

class OptimizationPipeline:
    """
    Advanced Optimization Pipeline
    
    Orchestrates comprehensive optimization using all refactored modules
    """
    
    def __init__(self, config_manager: Optional[PipelineConfigManager] = None):
        """Initialize optimization pipeline with comprehensive utilities"""
        self.logger = LoggerFactory.get_logger("optimization_pipeline")
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        
        # Configuration management
        self.config_manager = config_manager or PipelineConfigManager()
        self.pipeline_config = self.config_manager.get_pipeline_config('optimization_pipeline')
        self.optimization_config = self.config_manager.optimization_config
        
        # Initialize refactored optimization modules
        self.traffic_optimizer = TrafficOptimizer(logger=self.logger)
        self.position_predictor = PositionPredictor(logger=self.logger)
        self.ensemble_manager = EnsembleModelManager(logger=self.logger)
        self.feature_selector = FeatureSelector(logger=self.logger)
        
        # Utilities
        self.data_processor = DataProcessor(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.optimization_helper = OptimizationHelper(self.logger)
        self.report_exporter = ReportExporter(self.logger)
        self.data_exporter = DataExporter(self.logger)
        
        # Pipeline state
        self.pipeline_results = {}
        self.optimization_models = {}

    @timing_decorator()
    async def run_comprehensive_optimization(
        self,
        primary_data: pd.DataFrame,
        competitive_data: Optional[Dict[str, pd.DataFrame]] = None,
        historical_performance: Optional[pd.DataFrame] = None,
        optimization_objectives: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive optimization pipeline
        
        Args:
            primary_data: Primary dataset for optimization
            competitive_data: Competitive context data
            historical_performance: Historical performance data
            optimization_objectives: Specific optimization objectives
            constraints: Optimization constraints
            
        Returns:
            Comprehensive optimization results
        """
        try:
            with self.performance_tracker.track_block("comprehensive_optimization"):
                # Audit log pipeline execution
                self.audit_logger.log_analysis_execution(
                    user_id="pipeline_system",
                    analysis_type="comprehensive_optimization",
                    parameters={
                        "primary_data_rows": len(primary_data),
                        "competitive_datasets": len(competitive_data) if competitive_data else 0,
                        "optimization_objectives": optimization_objectives,
                        "has_constraints": bool(constraints)
                    }
                )
                
                self.logger.info("Starting comprehensive optimization pipeline")
                
                # Phase 1: Optimization Data Preparation
                prepared_data = await self._prepare_optimization_data(
                    primary_data, competitive_data, historical_performance
                )
                
                # Phase 2: Traffic Optimization
                traffic_optimization = await self._execute_traffic_optimization(
                    prepared_data, constraints
                )
                
                # Phase 3: Position Optimization
                position_optimization = await self._execute_position_optimization(
                    prepared_data, traffic_optimization
                )
                
                # Phase 4: Bid Strategy Optimization
                bid_optimization = await self._execute_bid_optimization(
                    prepared_data, traffic_optimization
                )
                
                # Phase 5: Content Optimization
                content_optimization = await self._execute_content_optimization(
                    prepared_data, position_optimization
                )
                
                # Phase 6: Resource Allocation Optimization
                resource_optimization = await self._execute_resource_optimization(
                    prepared_data, traffic_optimization, position_optimization
                )
                
                # Phase 7: Portfolio Optimization
                portfolio_optimization = await self._execute_portfolio_optimization({
                    'traffic_optimization': traffic_optimization,
                    'position_optimization': position_optimization,
                    'bid_optimization': bid_optimization,
                    'content_optimization': content_optimization,
                    'resource_optimization': resource_optimization
                })
                
                # Phase 8: Optimization Integration and Validation
                integrated_optimization = await self._integrate_optimization_results({
                    'prepared_data': prepared_data,
                    'traffic_optimization': traffic_optimization,
                    'position_optimization': position_optimization,
                    'bid_optimization': bid_optimization,
                    'content_optimization': content_optimization,
                    'resource_optimization': resource_optimization,
                    'portfolio_optimization': portfolio_optimization
                })
                
                # Phase 9: Implementation Planning
                implementation_plan = await self._create_implementation_plan(
                    integrated_optimization
                )
                
                # Export comprehensive results
                export_results = await self._export_optimization_results(integrated_optimization)
                integrated_optimization['export_results'] = export_results
                integrated_optimization['implementation_plan'] = implementation_plan
                
                self.pipeline_results = integrated_optimization
                self.logger.info("Comprehensive optimization pipeline completed")
                return integrated_optimization
                
        except Exception as e:
            self.logger.error(f"Error in optimization pipeline: {str(e)}")
            await self._handle_pipeline_error(e)
            return {}

    async def _prepare_optimization_data(
        self,
        primary_data: pd.DataFrame,
        competitive_data: Optional[Dict[str, pd.DataFrame]],
        historical_performance: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Prepare data for optimization"""
        try:
            with self.performance_tracker.track_block("optimization_data_preparation"):
                self.logger.info("Preparing data for optimization")
                
                # Clean primary data
                cleaned_primary = self.data_processor.clean_seo_data(primary_data)
                
                # Prepare competitive context
                cleaned_competitive = {}
                if competitive_data:
                    for competitor, data in competitive_data.items():
                        cleaned_competitive[competitor] = self.data_processor.clean_seo_data(data)
                
                # Prepare historical performance data
                cleaned_historical = None
                if historical_performance is not None:
                    cleaned_historical = self.data_processor.clean_seo_data(historical_performance)
                
                # Feature selection for optimization
                optimization_features = self._select_optimization_features(
                    cleaned_primary, cleaned_competitive
                )
                
                # Create optimization baseline
                optimization_baseline = self._create_optimization_baseline(
                    cleaned_primary, cleaned_historical
                )
                
                # Define optimization scope
                optimization_scope = await self._define_optimization_scope(
                    cleaned_primary, self.optimization_config
                )
                
                prepared_data = {
                    'primary_data': cleaned_primary,
                    'competitive_data': cleaned_competitive,
                    'historical_performance': cleaned_historical,
                    'optimization_features': optimization_features,
                    'optimization_baseline': optimization_baseline,
                    'optimization_scope': optimization_scope,
                    'data_summary': {
                        'total_keywords': len(cleaned_primary),
                        'competitors_included': len(cleaned_competitive),
                        'optimization_features_count': len(optimization_features),
                        'historical_data_points': len(cleaned_historical) if cleaned_historical is not None else 0,
                        'preparation_timestamp': datetime.now()
                    }
                }
                
                self.logger.info(f"Optimization data preparation completed: {len(cleaned_primary)} keywords, {len(optimization_features)} features")
                return prepared_data
                
        except Exception as e:
            self.logger.error(f"Error in optimization data preparation: {str(e)}")
            return {}

    async def _execute_traffic_optimization(
        self,
        prepared_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> OptimizationResult:
        """Execute comprehensive traffic optimization"""
        try:
            with self.performance_tracker.track_block("traffic_optimization"):
                self.logger.info("Executing traffic optimization")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitive_data = prepared_data.get('competitive_data', {})
                historical_performance = prepared_data.get('historical_performance')
                
                # Use traffic optimizer from refactored module
                traffic_result = self.traffic_optimizer.optimize_traffic_allocation(
                    keyword_data=primary_data,
                    config=None,  # Use default config
                    constraints=None,  # Will be enhanced
                    historical_performance=historical_performance
                )
                
                # Enhanced traffic forecasting
                traffic_forecasts = await self._generate_enhanced_traffic_forecasts(
                    primary_data, traffic_result, competitive_data
                )
                
                # Traffic optimization validation
                optimization_validation = self._validate_traffic_optimization(
                    traffic_result, primary_data
                )
                
                # Enhanced traffic optimization result
                enhanced_result = OptimizationResult(
                    optimal_allocations=traffic_result.optimal_allocations,
                    expected_performance=traffic_result.expected_performance,
                    resource_utilization=traffic_result.resource_utilization,
                    roi_projections=traffic_result.roi_projections,
                    risk_assessment=traffic_result.risk_assessment,
                    recommendations=traffic_result.recommendations + self._generate_additional_traffic_recommendations(traffic_result),
                    optimization_metadata={
                        **traffic_result.optimization_metadata,
                        'traffic_forecasts': traffic_forecasts,
                        'optimization_validation': optimization_validation,
                        'enhancement_timestamp': datetime.now()
                    }
                )
                
                self.logger.info("Traffic optimization completed")
                return enhanced_result
                
        except Exception as e:
            self.logger.error(f"Error in traffic optimization: {str(e)}")
            return OptimizationResult({}, {}, {}, {}, {}, [], {})

    async def _execute_position_optimization(
        self,
        prepared_data: Dict[str, Any],
        traffic_optimization: OptimizationResult
    ) -> Dict[str, Any]:
        """Execute position optimization"""
        try:
            with self.performance_tracker.track_block("position_optimization"):
                self.logger.info("Executing position optimization")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                
                # Use position predictor for optimization
                position_predictions = self.position_predictor.predict_positions(
                    historical_data=primary_data,
                    keywords_to_predict=None,
                    config=None,
                    competitive_data=prepared_data.get('competitive_data')
                )
                
                # Position improvement optimization
                position_improvements = await self._optimize_position_improvements(
                    position_predictions, traffic_optimization
                )
                
                # SERP feature optimization
                serp_optimization = self._optimize_serp_features(
                    primary_data, position_predictions
                )
                
                # Content positioning optimization
                content_positioning = self._optimize_content_positioning(
                    primary_data, position_predictions
                )
                
                position_optimization = {
                    'position_predictions': position_predictions,
                    'position_improvements': position_improvements,
                    'serp_optimization': serp_optimization,
                    'content_positioning': content_positioning,
                    'optimization_impact': self._calculate_position_optimization_impact(
                        position_predictions, position_improvements
                    ),
                    'implementation_priority': self._prioritize_position_optimizations(
                        position_improvements, serp_optimization
                    )
                }
                
                self.logger.info("Position optimization completed")
                return position_optimization
                
        except Exception as e:
            self.logger.error(f"Error in position optimization: {str(e)}")
            return {}

    async def _execute_bid_optimization(
        self,
        prepared_data: Dict[str, Any],
        traffic_optimization: OptimizationResult
    ) -> Dict[str, Any]:
        """Execute bid strategy optimization"""
        try:
            with self.performance_tracker.track_block("bid_optimization"):
                self.logger.info("Executing bid strategy optimization")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitive_data = prepared_data.get('competitive_data', {})
                
                # Extract budget allocation from traffic optimization
                budget_allocation = traffic_optimization.optimal_allocations
                
                # Use traffic optimizer for bid strategy
                bid_strategy = self.traffic_optimizer.optimize_bid_strategy(
                    keyword_data=primary_data,
                    budget_allocation=budget_allocation,
                    target_metrics={'target_roi': self.optimization_config.target_roi},
                    competitive_data=competitive_data.get('dell', pd.DataFrame()) if competitive_data else None
                )
                
                # Portfolio-level bid optimization
                portfolio_bidding = self._optimize_portfolio_bidding(
                    bid_strategy, budget_allocation, primary_data
                )
                
                # Dynamic bid adjustment optimization
                dynamic_bidding = self._optimize_dynamic_bidding(
                    bid_strategy, primary_data, competitive_data
                )
                
                # Bid performance forecasting
                bid_performance_forecast = self._forecast_bid_performance(
                    bid_strategy, primary_data
                )
                
                bid_optimization = {
                    'bid_strategy': bid_strategy,
                    'portfolio_bidding': portfolio_bidding,
                    'dynamic_bidding': dynamic_bidding,
                    'performance_forecast': bid_performance_forecast,
                    'optimization_efficiency': self._calculate_bid_optimization_efficiency(bid_strategy),
                    'risk_assessment': self._assess_bid_optimization_risk(bid_strategy, competitive_data)
                }
                
                self.logger.info("Bid strategy optimization completed")
                return bid_optimization
                
        except Exception as e:
            self.logger.error(f"Error in bid optimization: {str(e)}")
            return {}

    async def _execute_content_optimization(
        self,
        prepared_data: Dict[str, Any],
        position_optimization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute content optimization"""
        try:
            with self.performance_tracker.track_block("content_optimization"):
                self.logger.info("Executing content optimization")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitive_data = prepared_data.get('competitive_data', {})
                
                # Content gap optimization
                content_gaps = await self._optimize_content_gaps(
                    primary_data, competitive_data
                )
                
                # Semantic optimization
                semantic_optimization = self._optimize_semantic_content(
                    primary_data, position_optimization
                )
                
                # Content performance optimization
                content_performance = self._optimize_content_performance(
                    primary_data, competitive_data
                )
                
                # Content distribution optimization
                content_distribution = self._optimize_content_distribution(
                    primary_data, position_optimization
                )
                
                content_optimization = {
                    'content_gaps': content_gaps,
                    'semantic_optimization': semantic_optimization,
                    'content_performance': content_performance,
                    'content_distribution': content_distribution,
                    'optimization_roadmap': self._create_content_optimization_roadmap(
                        content_gaps, semantic_optimization, content_performance
                    ),
                    'success_metrics': self._define_content_optimization_metrics()
                }
                
                self.logger.info("Content optimization completed")
                return content_optimization
                
        except Exception as e:
            self.logger.error(f"Error in content optimization: {str(e)}")
            return {}

    async def _execute_resource_optimization(
        self,
        prepared_data: Dict[str, Any],
        traffic_optimization: OptimizationResult,
        position_optimization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute resource allocation optimization"""
        try:
            with self.performance_tracker.track_block("resource_optimization"):
                self.logger.info("Executing resource allocation optimization")
                
                # Budget allocation optimization
                budget_optimization = await self._optimize_budget_allocation(
                    traffic_optimization, position_optimization
                )
                
                # Team resource optimization
                team_optimization = self._optimize_team_resources(
                    traffic_optimization, position_optimization
                )
                
                # Technology resource optimization
                technology_optimization = self._optimize_technology_resources(
                    prepared_data, traffic_optimization
                )
                
                # Time allocation optimization
                time_optimization = self._optimize_time_allocation(
                    traffic_optimization, position_optimization
                )
                
                resource_optimization = {
                    'budget_optimization': budget_optimization,
                    'team_optimization': team_optimization,
                    'technology_optimization': technology_optimization,
                    'time_optimization': time_optimization,
                    'resource_efficiency': self._calculate_resource_efficiency(
                        budget_optimization, team_optimization, time_optimization
                    ),
                    'allocation_recommendations': self._generate_allocation_recommendations(
                        budget_optimization, team_optimization, technology_optimization
                    )
                }
                
                self.logger.info("Resource allocation optimization completed")
                return resource_optimization
                
        except Exception as e:
            self.logger.error(f"Error in resource optimization: {str(e)}")
            return {}

    async def _execute_portfolio_optimization(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio-level optimization"""
        try:
            with self.performance_tracker.track_block("portfolio_optimization"):
                self.logger.info("Executing portfolio optimization")
                
                # Portfolio risk optimization
                risk_optimization = await self._optimize_portfolio_risk(optimization_results)
                
                # Portfolio diversification optimization
                diversification_optimization = self._optimize_portfolio_diversification(
                    optimization_results
                )
                
                # Portfolio performance optimization
                performance_optimization = self._optimize_portfolio_performance(
                    optimization_results
                )
                
                # Portfolio correlation optimization
                correlation_optimization = self._optimize_portfolio_correlations(
                    optimization_results
                )
                
                portfolio_optimization = {
                    'risk_optimization': risk_optimization,
                    'diversification_optimization': diversification_optimization,
                    'performance_optimization': performance_optimization,
                    'correlation_optimization': correlation_optimization,
                    'portfolio_efficiency': self._calculate_portfolio_efficiency(
                        risk_optimization, performance_optimization
                    ),
                    'rebalancing_recommendations': self._generate_rebalancing_recommendations(
                        risk_optimization, diversification_optimization, performance_optimization
                    )
                }
                
                self.logger.info("Portfolio optimization completed")
                return portfolio_optimization
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            return {}

    async def _integrate_optimization_results(self, all_optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate all optimization results"""
        try:
            with self.performance_tracker.track_block("optimization_integration"):
                self.logger.info("Integrating optimization results")
                
                # Create optimization executive summary
                executive_summary = self._create_optimization_executive_summary(all_optimizations)
                
                # Optimization performance summary
                performance_summary = await self._create_optimization_performance_summary(all_optimizations)
                
                # Optimization recommendations synthesis
                recommendations_synthesis = self._synthesize_optimization_recommendations(all_optimizations)
                
                # Optimization risk assessment
                risk_assessment = self._assess_integrated_optimization_risk(all_optimizations)
                
                # Optimization implementation roadmap
                implementation_roadmap = self._create_optimization_implementation_roadmap(all_optimizations)
                
                # Optimization monitoring framework
                monitoring_framework = self._create_optimization_monitoring_framework(all_optimizations)
                
                integrated_optimization = {
                    'executive_summary': executive_summary,
                    'performance_summary': performance_summary,
                    'recommendations_synthesis': recommendations_synthesis,
                    'risk_assessment': risk_assessment,
                    'implementation_roadmap': implementation_roadmap,
                    'monitoring_framework': monitoring_framework,
                    'detailed_optimizations': all_optimizations,
                    'integration_metadata': {
                        'optimization_components': list(all_optimizations.keys()),
                        'integration_timestamp': datetime.now(),
                        'overall_optimization_score': self._calculate_overall_optimization_score(all_optimizations),
                        'implementation_complexity': self._assess_implementation_complexity(all_optimizations),
                        'expected_impact': self._calculate_expected_optimization_impact(all_optimizations)
                    }
                }
                
                self.logger.info("Optimization results integration completed")
                return integrated_optimization
                
        except Exception as e:
            self.logger.error(f"Error integrating optimization results: {str(e)}")
            return all_optimizations

    async def _create_implementation_plan(self, integrated_optimization: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive implementation plan"""
        try:
            with self.performance_tracker.track_block("implementation_planning"):
                self.logger.info("Creating optimization implementation plan")
                
                # Phase-based implementation plan
                phased_plan = await self._create_phased_implementation_plan(integrated_optimization)
                
                # Resource requirements planning
                resource_planning = self._plan_implementation_resources(integrated_optimization)
                
                # Timeline and milestones
                timeline_planning = self._create_implementation_timeline(integrated_optimization)
                
                # Risk mitigation planning
                risk_mitigation = self._plan_implementation_risk_mitigation(integrated_optimization)
                
                # Success measurement planning
                measurement_planning = self._plan_success_measurement(integrated_optimization)
                
                # Change management planning
                change_management = self._plan_change_management(integrated_optimization)
                
                implementation_plan = {
                    'phased_plan': phased_plan,
                    'resource_planning': resource_planning,
                    'timeline_planning': timeline_planning,
                    'risk_mitigation': risk_mitigation,
                    'measurement_planning': measurement_planning,
                    'change_management': change_management,
                    'implementation_readiness': self._assess_implementation_readiness(integrated_optimization),
                    'success_probability': self._calculate_implementation_success_probability(integrated_optimization)
                }
                
                self.logger.info("Implementation plan creation completed")
                return implementation_plan
                
        except Exception as e:
            self.logger.error(f"Error creating implementation plan: {str(e)}")
            return {}

    async def _export_optimization_results(self, integrated_optimization: Dict[str, Any]) -> Dict[str, bool]:
        """Export comprehensive optimization results"""
        try:
            with self.performance_tracker.track_block("optimization_export"):
                self.logger.info("Exporting optimization results")
                
                export_results = {}
                
                # Export executive summary
                executive_export = self.report_exporter.export_executive_report(
                    integrated_optimization.get('executive_summary', {}),
                    f"{self.config_manager.data_config.output_directory}/optimization_executive_summary.html",
                    format='html',
                    include_charts=True
                )
                export_results['executive_summary'] = executive_export
                
                # Export detailed optimization results
                traffic_optimization = integrated_optimization.get('detailed_optimizations', {}).get('traffic_optimization')
                if traffic_optimization and hasattr(traffic_optimization, 'optimal_allocations'):
                    optimization_df = pd.DataFrame([
                        {
                            'keyword': keyword,
                            'optimal_allocation': allocation,
                            'expected_performance': traffic_optimization.expected_performance.get('keyword_gains', {}).get(keyword, 0),
                            'roi_projection': traffic_optimization.roi_projections.get(keyword, 0)
                        }
                        for keyword, allocation in traffic_optimization.optimal_allocations.items()
                    ])
                    
                    optimization_export = self.data_exporter.export_with_metadata(
                        optimization_df,
                        metadata={'analysis_type': 'traffic_optimization', 'generation_timestamp': datetime.now()},
                        export_path=f"{self.config_manager.data_config.output_directory}/traffic_optimization_results.xlsx"
                    )
                    export_results['traffic_optimization'] = optimization_export
                
                # Export implementation plan
                implementation_plan = integrated_optimization.get('implementation_plan', {})
                if implementation_plan:
                    plan_export = self.data_exporter.export_analysis_dataset(
                        {'implementation_plan': pd.DataFrame([implementation_plan])},
                        f"{self.config_manager.data_config.output_directory}/optimization_implementation_plan.xlsx"
                    )
                    export_results['implementation_plan'] = plan_export
                
                self.logger.info("Optimization results export completed")
                return export_results
                
        except Exception as e:
            self.logger.error(f"Error exporting optimization results: {str(e)}")
            return {}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_name': 'optimization_pipeline',
            'status': 'completed' if self.pipeline_results else 'not_started',
            'optimization_models_count': len(self.optimization_models),
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'results_available': bool(self.pipeline_results)
        }

    async def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline execution errors"""
        self.logger.error(f"Optimization pipeline error: {str(error)}")
        self.audit_logger.log_analysis_execution(
            user_id="pipeline_system",
            analysis_type="optimization_pipeline_error",
            result="failure",
            details={"error": str(error)}
        )

    # Helper methods (simplified implementations for brevity)
    def _select_optimization_features(self, primary_data, competitive_data):
        """Select relevant features for optimization"""
        try:
            optimization_features = []
            
            # Core SEO features
            core_features = ['Position', 'Traffic (%)', 'Search Volume', 'CPC', 'Keyword Difficulty']
            for feature in core_features:
                if feature in primary_data.columns:
                    optimization_features.append(feature)
            
            return optimization_features
        except Exception:
            return []

    def _create_optimization_baseline(self, primary_data, historical_data):
        """Create optimization baseline metrics"""
        try:
            baseline = {
                'current_traffic': primary_data.get('Traffic (%)', pd.Series()).sum(),
                'average_position': primary_data.get('Position', pd.Series()).mean(),
                'total_keywords': len(primary_data),
                'baseline_timestamp': datetime.now()
            }
            
            if historical_data is not None:
                baseline['historical_traffic_trend'] = self._calculate_traffic_trend(historical_data)
            
            return baseline
        except Exception:
            return {}

    def _calculate_traffic_trend(self, historical_data):
        """Calculate traffic trend from historical data"""
        try:
            if 'Traffic (%)' in historical_data.columns and 'date' in historical_data.columns:
                # Simple trend calculation
                traffic_by_date = historical_data.groupby('date')['Traffic (%)'].sum().sort_index()
                if len(traffic_by_date) > 1:
                    return (traffic_by_date.iloc[-1] - traffic_by_date.iloc[0]) / len(traffic_by_date)
            return 0.0
        except Exception:
            return 0.0

    def _create_optimization_executive_summary(self, all_optimizations):
        """Create executive summary for optimization"""
        return {
            'optimization_scope': 'comprehensive',
            'optimization_components': len(all_optimizations),
            'expected_impact': self._calculate_expected_optimization_impact(all_optimizations),
            'implementation_complexity': 'medium',
            'optimization_timestamp': datetime.now()
        }

    def _calculate_expected_optimization_impact(self, all_optimizations):
        """Calculate expected impact from all optimizations"""
        try:
            traffic_optimization = all_optimizations.get('traffic_optimization')
            if traffic_optimization and hasattr(traffic_optimization, 'expected_performance'):
                return traffic_optimization.expected_performance.get('total_traffic_gain', 0)
            return 0
        except Exception:
            return 0

    def _calculate_overall_optimization_score(self, all_optimizations):
        """Calculate overall optimization score"""
        try:
            # Simplified scoring based on available optimizations
            score = 0
            max_score = len(all_optimizations) * 20  # 20 points per optimization
            
            for optimization in all_optimizations.values():
                if optimization:
                    score += 20
            
            return score / max_score if max_score > 0 else 0
        except Exception:
            return 0.0
    def _define_optimization_scope(self, data):
        """Define optimization scope and parameters"""
        try:
            self.logger.info("Defining optimization scope")
            
            # Determine data characteristics
            data_shape = data.shape if hasattr(data, 'shape') else (0, 0)
            
            # Define optimization objectives
            objectives = ['traffic_optimization', 'position_improvement', 'roi_maximization']
            
            # Define constraints based on data
            constraints = {
                'budget_constraint': self.config.budget_constraints.get('total_budget', 10000.0),
                'time_constraint': 90,  # days
                'resource_constraint': 100,  # arbitrary units
                'risk_constraint': self.config.risk_tolerance,
                'quality_constraint': 0.8
            }
            
            # Define optimization variables
            variables = {
                'keyword_focus': list(range(min(100, data_shape[0]))),  # Keywords to focus on
                'content_optimization': [0.0, 1.0],  # Optimization intensity
                'budget_allocation': [0.0, constraints['budget_constraint']],
                'timeline_allocation': [1, constraints['time_constraint']]
            }
            
            # Define success metrics
            success_metrics = [
                'traffic_increase_percentage',
                'position_improvement_average',
                'roi_achievement',
                'cost_efficiency',
                'time_to_results'
            ]
            
            optimization_scope = {
                'objectives': objectives,
                'constraints': constraints,
                'variables': variables,
                'success_metrics': success_metrics,
                'data_characteristics': {
                    'total_keywords': data_shape[0],
                    'total_features': data_shape[1],
                    'optimization_potential': min(data_shape[0] / 100, 1.0)
                },
                'scope_definition': {
                    'primary_focus': 'traffic_and_positions',
                    'secondary_focus': 'roi_optimization',
                    'optimization_method': 'multi_objective',
                    'optimization_horizon': f"{constraints['time_constraint']} days"
                }
            }
            
            self.logger.info(f"Optimization scope defined: {len(objectives)} objectives, {len(constraints)} constraints")
            return optimization_scope
            
        except Exception as e:
            self.logger.error(f"Error defining optimization scope: {str(e)}")
            return {
                'objectives': ['traffic_optimization'],
                'constraints': {'budget_constraint': 10000.0},
                'variables': {'keyword_focus': []},
                'success_metrics': ['traffic_increase']
            }

    def _optimize_position_improvements(self, data):
        """Optimize position improvements"""
        try:
            self.logger.info("Optimizing position improvements")
            
            if data.empty:
                return {'position_recommendations': [], 'expected_improvement': 0}
            
            # Analyze current positions
            if 'Position' in data.columns:
                current_positions = data['Position'].dropna()
                avg_position = current_positions.mean()
                
                # Identify improvement opportunities
                improvable_keywords = data[data['Position'] > 10] if 'Position' in data.columns else pd.DataFrame()
                
                # Generate position optimization recommendations
                recommendations = []
                for idx, row in improvable_keywords.head(20).iterrows():  # Top 20 opportunities
                    current_pos = row.get('Position', 50)
                    target_pos = max(1, current_pos - 5)  # Improve by 5 positions
                    
                    recommendations.append({
                        'keyword': row.get('Keyword', f'keyword_{idx}'),
                        'current_position': current_pos,
                        'target_position': target_pos,
                        'improvement_potential': current_pos - target_pos,
                        'estimated_effort': 'medium' if current_pos > 20 else 'high',
                        'priority': 'high' if current_pos > 30 else 'medium',
                        'optimization_tactics': [
                            'content_optimization',
                            'internal_linking',
                            'technical_seo'
                        ]
                    })
                
                # Calculate expected overall improvement
                total_improvement = sum(rec['improvement_potential'] for rec in recommendations)
                expected_improvement = total_improvement / len(data) if len(data) > 0 else 0
                
            else:
                recommendations = []
                expected_improvement = 0
            
            return {
                'position_recommendations': recommendations,
                'expected_improvement': expected_improvement,
                'total_opportunities': len(recommendations),
                'high_priority_count': len([r for r in recommendations if r['priority'] == 'high']),
                'optimization_strategy': 'focused_improvement',
                'timeline_estimate': '3-6 months'
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing position improvements: {str(e)}")
            return {'position_recommendations': [], 'expected_improvement': 0}

    def _optimize_content_gaps(self, data):
        """Optimize content gaps"""
        try:
            self.logger.info("Optimizing content gaps")
            
            if data.empty:
                return {'content_recommendations': [], 'priority_score': 0}
            
            content_recommendations = []
            
            # Analyze keyword themes for content gaps
            if 'Keyword' in data.columns:
                keywords = data['Keyword'].dropna().tolist()
                
                # Extract topics from keywords (simplified approach)
                topics = {}
                for keyword in keywords[:100]:  # Limit analysis
                    words = keyword.lower().split()
                    if words:
                        topic = words[0]  # Use first word as topic proxy
                        if topic not in topics:
                            topics[topic] = []
                        topics[topic].append(keyword)
                
                # Identify content opportunities
                for topic, topic_keywords in topics.items():
                    if len(topic_keywords) >= 3:  # Minimum threshold for content opportunity
                        avg_volume = 0
                        if 'Volume' in data.columns:
                            topic_data = data[data['Keyword'].isin(topic_keywords)]
                            avg_volume = topic_data['Volume'].mean() if not topic_data.empty else 0
                        
                        content_recommendations.append({
                            'topic': topic,
                            'keyword_count': len(topic_keywords),
                            'content_type': 'comprehensive_guide',
                            'priority': 'high' if len(topic_keywords) > 5 else 'medium',
                            'estimated_traffic_potential': avg_volume * len(topic_keywords) * 0.1,
                            'content_tactics': [
                                'pillar_page_creation',
                                'topic_clustering',
                                'internal_linking_optimization'
                            ],
                            'timeline': '4-8 weeks'
                        })
            
            # Calculate priority score
            high_priority_count = len([r for r in content_recommendations if r['priority'] == 'high'])
            priority_score = min(high_priority_count / 10, 1.0)  # Normalize to max 10 high priority items
            
            return {
                'content_recommendations': content_recommendations,
                'priority_score': priority_score,
                'total_opportunities': len(content_recommendations),
                'high_priority_topics': [r['topic'] for r in content_recommendations if r['priority'] == 'high'],
                'estimated_total_impact': sum(r.get('estimated_traffic_potential', 0) for r in content_recommendations),
                'content_strategy': 'topic_cluster_approach'
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing content gaps: {str(e)}")
            return {'content_recommendations': [], 'priority_score': 0}

    def _optimize_budget_allocation(self, data):
        """Optimize budget allocation"""
        try:
            self.logger.info("Optimizing budget allocation")
            
            total_budget = self.config.budget_constraints.get('total_budget', 10000.0)
            
            # Define budget categories
            budget_categories = {
                'content_optimization': 0.4,    # 40% for content
                'technical_seo': 0.25,          # 25% for technical
                'link_building': 0.20,          # 20% for links
                'paid_promotion': 0.10,         # 10% for paid
                'tools_and_analytics': 0.05     # 5% for tools
            }
            
            # Calculate actual allocations
            budget_allocation = {}
            for category, percentage in budget_categories.items():
                allocated_amount = total_budget * percentage
                budget_allocation[category] = {
                    'allocated_amount': allocated_amount,
                    'percentage': percentage,
                    'priority': 'high' if percentage > 0.3 else 'medium' if percentage > 0.15 else 'low',
                    'expected_roi': self._estimate_category_roi(category),
                    'risk_level': self._assess_category_risk(category)
                }
            
            # Calculate expected overall ROI
            expected_roi = sum(
                alloc['allocated_amount'] * alloc['expected_roi'] 
                for alloc in budget_allocation.values()
            ) / total_budget
            
            # Generate allocation recommendations
            recommendations = []
            for category, allocation in budget_allocation.items():
                if allocation['expected_roi'] > 2.0:
                    recommendations.append({
                        'category': category,
                        'recommendation': 'increase_allocation',
                        'reason': 'high_expected_roi',
                        'suggested_change': '+10%'
                    })
                elif allocation['expected_roi'] < 1.5:
                    recommendations.append({
                        'category': category,
                        'recommendation': 'decrease_allocation',
                        'reason': 'low_expected_roi',
                        'suggested_change': '-5%'
                    })
            
            return {
                'budget_allocation': budget_allocation,
                'total_budget': total_budget,
                'expected_roi': expected_roi,
                'allocation_recommendations': recommendations,
                'budget_efficiency_score': expected_roi / 2.0,  # Normalize against target ROI of 2.0
                'risk_adjusted_return': expected_roi * 0.8,  # Conservative estimate
                'optimization_strategy': 'roi_based_allocation'
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing budget allocation: {str(e)}")
            return {'budget_allocation': {}, 'expected_roi': 0}

    def _estimate_category_roi(self, category):
        """Estimate ROI for budget category"""
        roi_estimates = {
            'content_optimization': 3.0,
            'technical_seo': 2.5,
            'link_building': 2.0,
            'paid_promotion': 1.5,
            'tools_and_analytics': 1.8
        }
        return roi_estimates.get(category, 2.0)

    def _assess_category_risk(self, category):
        """Assess risk level for budget category"""
        risk_levels = {
            'content_optimization': 'low',
            'technical_seo': 'low',
            'link_building': 'medium',
            'paid_promotion': 'high',
            'tools_and_analytics': 'low'
        }
        return risk_levels.get(category, 'medium')

    def _optimize_portfolio_risk(self, data):
        """Optimize portfolio risk"""
        try:
            self.logger.info("Optimizing portfolio risk")
            
            if data.empty:
                return {'risk_distribution': {}, 'risk_score': 0.5}
            
            # Analyze keyword portfolio risk
            risk_factors = {
                'keyword_concentration': self._calculate_keyword_concentration_risk(data),
                'position_volatility': self._calculate_position_volatility_risk(data),
                'traffic_dependence': self._calculate_traffic_dependence_risk(data),
                'competitive_pressure': self._calculate_competitive_pressure_risk(data),
                'search_volume_risk': self._calculate_search_volume_risk(data)
            }
            
            # Calculate overall risk score
            risk_weights = {
                'keyword_concentration': 0.25,
                'position_volatility': 0.20,
                'traffic_dependence': 0.20,
                'competitive_pressure': 0.20,
                'search_volume_risk': 0.15
            }
            
            overall_risk_score = sum(
                risk_factors[factor] * risk_weights[factor] 
                for factor in risk_factors
            )
            
            # Generate risk mitigation recommendations
            risk_mitigation = []
            for factor, risk_level in risk_factors.items():
                if risk_level > 0.7:
                    risk_mitigation.append({
                        'risk_factor': factor,
                        'risk_level': risk_level,
                        'mitigation_strategy': self._get_risk_mitigation_strategy(factor),
                        'priority': 'high',
                        'implementation_timeline': '1-3 months'
                    })
            
            # Create risk distribution
            risk_distribution = {
                'low_risk_keywords': len(data) * (1 - overall_risk_score) * 0.6,
                'medium_risk_keywords': len(data) * overall_risk_score * 0.7,
                'high_risk_keywords': len(data) * overall_risk_score * 0.3,
                'total_keywords': len(data)
            }
            
            return {
                'risk_distribution': risk_distribution,
                'risk_score': overall_risk_score,
                'risk_factors': risk_factors,
                'risk_mitigation': risk_mitigation,
                'portfolio_health': 'healthy' if overall_risk_score < 0.4 else 'moderate' if overall_risk_score < 0.7 else 'high_risk',
                'diversification_score': 1 - risk_factors.get('keyword_concentration', 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio risk: {str(e)}")
            return {'risk_distribution': {}, 'risk_score': 0.5}

    def _calculate_keyword_concentration_risk(self, data):
        """Calculate keyword concentration risk"""
        try:
            if 'Traffic (%)' in data.columns:
                traffic_values = data['Traffic (%)'].dropna()
                if len(traffic_values) > 0:
                    # Calculate concentration using Gini coefficient approach
                    sorted_traffic = np.sort(traffic_values)
                    n = len(sorted_traffic)
                    index = np.arange(1, n + 1)
                    concentration = (2 * np.sum(index * sorted_traffic)) / (n * np.sum(sorted_traffic)) - (n + 1) / n
                    return min(concentration, 1.0)
            return 0.5
        except Exception:
            return 0.5

    def _calculate_position_volatility_risk(self, data):
        """Calculate position volatility risk"""
        try:
            if 'Position' in data.columns:
                positions = data['Position'].dropna()
                if len(positions) > 1:
                    volatility = np.std(positions) / np.mean(positions) if np.mean(positions) > 0 else 0
                    return min(volatility / 2, 1.0)  # Normalize
            return 0.3
        except Exception:
            return 0.3

    def _calculate_traffic_dependence_risk(self, data):
        """Calculate traffic dependence risk"""
        try:
            if 'Traffic (%)' in data.columns:
                traffic_values = data['Traffic (%)'].dropna()
                if len(traffic_values) > 0:
                    # Check if top 20% of keywords drive 80% of traffic (Pareto principle)
                    sorted_traffic = traffic_values.sort_values(ascending=False)
                    top_20_percent_count = max(1, int(len(sorted_traffic) * 0.2))
                    top_20_percent_traffic = sorted_traffic.head(top_20_percent_count).sum()
                    total_traffic = sorted_traffic.sum()
                    
                    if total_traffic > 0:
                        dependence_ratio = top_20_percent_traffic / total_traffic
                        return min(dependence_ratio, 1.0)
            return 0.4
        except Exception:
            return 0.4

    def _calculate_competitive_pressure_risk(self, data):
        """Calculate competitive pressure risk"""
        try:
            if 'Keyword Difficulty' in data.columns:
                difficulty_values = data['Keyword Difficulty'].dropna()
                if len(difficulty_values) > 0:
                    avg_difficulty = difficulty_values.mean()
                    return min(avg_difficulty / 100, 1.0)  # Normalize to 0-1 scale
            return 0.5
        except Exception:
            return 0.5

    def _calculate_search_volume_risk(self, data):
        """Calculate search volume risk"""
        try:
            if 'Volume' in data.columns:
                volumes = data['Volume'].dropna()
                if len(volumes) > 0:
                    # Low volume keywords are riskier
                    avg_volume = volumes.mean()
                    # Risk decreases as volume increases, using log scale
                    risk = 1 - min(np.log10(avg_volume + 1) / 5, 1.0)  # Normalize
                    return max(0, risk)
            return 0.6
        except Exception:
            return 0.6

    def _get_risk_mitigation_strategy(self, risk_factor):
        """Get risk mitigation strategy for specific risk factor"""
        strategies = {
            'keyword_concentration': 'Diversify keyword portfolio across multiple themes and long-tail variations',
            'position_volatility': 'Strengthen content quality and technical SEO foundation',
            'traffic_dependence': 'Develop secondary traffic sources and reduce dependency on top keywords',
            'competitive_pressure': 'Focus on differentiation and unique value propositions',
            'search_volume_risk': 'Balance portfolio with mix of high and medium volume keywords'
        }
        return strategies.get(risk_factor, 'Monitor and adjust strategy based on performance data')

    def _create_optimization_performance_summary(self, results):
        """Create optimization performance summary"""
        try:
            self.logger.info("Creating optimization performance summary")
            
            # Extract performance metrics from various optimization results
            position_results = results.get('position_optimization', {})
            content_results = results.get('content_optimization', {})
            budget_results = results.get('budget_optimization', {})
            portfolio_results = results.get('portfolio_optimization', {})
            
            # Calculate overall performance scores
            performance_metrics = {
                'position_improvement_score': self._calculate_position_performance_score(position_results),
                'content_optimization_score': self._calculate_content_performance_score(content_results),
                'budget_efficiency_score': budget_results.get('budget_efficiency_score', 0.5),
                'portfolio_health_score': 1 - portfolio_results.get('risk_score', 0.5),
                'overall_optimization_score': 0.0
            }
            
            # Calculate weighted overall score
            weights = {
                'position_improvement_score': 0.3,
                'content_optimization_score': 0.25,
                'budget_efficiency_score': 0.25,
                'portfolio_health_score': 0.2
            }
            
            performance_metrics['overall_optimization_score'] = sum(
                performance_metrics[metric] * weights[metric] 
                for metric in weights
            )
            
            # Generate performance insights
            performance_insights = []
            if performance_metrics['overall_optimization_score'] > 0.7:
                performance_insights.append("Excellent optimization performance across all areas")
            elif performance_metrics['overall_optimization_score'] > 0.5:
                performance_insights.append("Good optimization performance with room for improvement")
            else:
                performance_insights.append("Optimization performance needs significant improvement")
            
            # Identify best and worst performing areas
            best_area = max(performance_metrics, key=performance_metrics.get)
            worst_area = min(performance_metrics, key=performance_metrics.get)
            
            performance_insights.append(f"Strongest area: {best_area}")
            performance_insights.append(f"Area needing attention: {worst_area}")
            
            performance_summary = {
                'performance_metrics': performance_metrics,
                'performance_insights': performance_insights,
                'optimization_effectiveness': 'high' if performance_metrics['overall_optimization_score'] > 0.7 else 'medium',
                'key_achievements': self._identify_key_achievements(results),
                'improvement_areas': self._identify_improvement_areas(performance_metrics),
                'next_steps': self._generate_next_steps(performance_metrics),
                'summary_timestamp': datetime.now()
            }
            
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"Error creating optimization performance summary: {str(e)}")
            return {
                'performance_metrics': {'overall_optimization_score': 0.5},
                'performance_insights': ['Performance summary generation failed'],
                'optimization_effectiveness': 'unknown'
            }

    def _calculate_position_performance_score(self, position_results):
        """Calculate position optimization performance score"""
        try:
            recommendations = position_results.get('position_recommendations', [])
            if not recommendations:
                return 0.3
            
            # Score based on number and quality of recommendations
            total_improvement = sum(rec.get('improvement_potential', 0) for rec in recommendations)
            high_priority_count = len([rec for rec in recommendations if rec.get('priority') == 'high'])
            
            score = min((total_improvement / 100) * 0.6 + (high_priority_count / 10) * 0.4, 1.0)
            return score
        except Exception:
            return 0.3

    def _calculate_content_performance_score(self, content_results):
        """Calculate content optimization performance score"""
        try:
            recommendations = content_results.get('content_recommendations', [])
            priority_score = content_results.get('priority_score', 0)
            
            if not recommendations:
                return 0.3
            
            # Score based on number of opportunities and priority
            opportunity_score = min(len(recommendations) / 20, 1.0)  # Normalize to max 20 opportunities
            combined_score = (opportunity_score * 0.6) + (priority_score * 0.4)
            
            return combined_score
        except Exception:
            return 0.3

    def _identify_key_achievements(self, results):
        """Identify key achievements from optimization results"""
        try:
            achievements = []
            
            # Position optimization achievements
            position_results = results.get('position_optimization', {})
            if position_results.get('expected_improvement', 0) > 5:
                achievements.append(f"Identified {position_results.get('expected_improvement', 0):.1f} average position improvement potential")
            
            # Content optimization achievements
            content_results = results.get('content_optimization', {})
            content_count = len(content_results.get('content_recommendations', []))
            if content_count > 5:
                achievements.append(f"Discovered {content_count} content optimization opportunities")
            
            # Budget optimization achievements
            budget_results = results.get('budget_optimization', {})
            expected_roi = budget_results.get('expected_roi', 0)
            if expected_roi > 2.0:
                achievements.append(f"Optimized budget allocation with {expected_roi:.1f}x expected ROI")
            
            return achievements if achievements else ["Optimization analysis completed successfully"]
        except Exception:
            return ["Optimization analysis completed"]

    def _identify_improvement_areas(self, performance_metrics):
        """Identify areas needing improvement"""
        try:
            improvement_areas = []
            
            threshold = 0.5
            for metric, score in performance_metrics.items():
                if metric != 'overall_optimization_score' and score < threshold:
                    area_name = metric.replace('_score', '').replace('_', ' ').title()
                    improvement_areas.append({
                        'area': area_name,
                        'current_score': score,
                        'target_score': 0.7,
                        'improvement_needed': 0.7 - score,
                        'priority': 'high' if score < 0.3 else 'medium'
                    })
            
            return improvement_areas
        except Exception:
            return []

    def _generate_next_steps(self, performance_metrics):
        """Generate next steps based on performance"""
        try:
            next_steps = []
            
            overall_score = performance_metrics.get('overall_optimization_score', 0.5)
            
            if overall_score < 0.4:
                next_steps.extend([
                    "Conduct comprehensive audit of current optimization strategies",
                    "Prioritize quick wins with highest impact potential",
                    "Establish baseline performance metrics"
                ])
            elif overall_score < 0.7:
                next_steps.extend([
                    "Implement high-priority optimization recommendations",
                    "Monitor performance improvements closely",
                    "Refine optimization strategies based on results"
                ])
            else:
                next_steps.extend([
                    "Maintain current optimization momentum",
                    "Explore advanced optimization techniques",
                    "Scale successful strategies across portfolio"
                ])
            
            # Add specific next steps based on weak areas
            worst_score = min(performance_metrics.values())
            worst_area = min(performance_metrics, key=performance_metrics.get)
            
            if worst_score < 0.5:
                area_name = worst_area.replace('_score', '').replace('_', ' ')
                next_steps.append(f"Focus immediate attention on {area_name} improvements")
            
            return next_steps[:5]  # Limit to top 5 next steps
        except Exception:
            return ["Continue optimization efforts", "Monitor performance regularly"]

    def _create_phased_implementation_plan(self, recommendations):
        """Create phased implementation plan"""
        try:
            self.logger.info("Creating phased implementation plan")
            
            # Collect all recommendations from different optimization areas
            all_recommendations = []
            
            if isinstance(recommendations, dict):
                for optimization_type, rec_data in recommendations.items():
                    if isinstance(rec_data, dict):
                        if 'position_recommendations' in rec_data:
                            for rec in rec_data['position_recommendations']:
                                rec['optimization_type'] = 'position'
                                all_recommendations.append(rec)
                        if 'content_recommendations' in rec_data:
                            for rec in rec_data['content_recommendations']:
                                rec['optimization_type'] = 'content'
                                all_recommendations.append(rec)
                        if 'allocation_recommendations' in rec_data:
                            for rec in rec_data['allocation_recommendations']:
                                rec['optimization_type'] = 'budget'
                                all_recommendations.append(rec)
            
            # Phase recommendations based on priority and effort
            phases = {
                'phase_1_quick_wins': {
                    'timeline': '0-1 months',
                    'focus': 'Low effort, high impact optimizations',
                    'recommendations': []
                },
                'phase_2_medium_term': {
                    'timeline': '1-3 months',
                    'focus': 'Medium effort optimizations with good ROI',
                    'recommendations': []
                },
                'phase_3_long_term': {
                    'timeline': '3-6 months',
                    'focus': 'High effort, strategic optimizations',
                    'recommendations': []
                }
            }
            
            # Categorize recommendations into phases
            for rec in all_recommendations[:30]:  # Limit to top 30 recommendations
                priority = rec.get('priority', 'medium')
                effort = rec.get('estimated_effort', 'medium')
                
                if priority == 'high' and effort in ['low', 'easy']:
                    phases['phase_1_quick_wins']['recommendations'].append(rec)
                elif priority in ['high', 'medium'] and effort == 'medium':
                    phases['phase_2_medium_term']['recommendations'].append(rec)
                else:
                    phases['phase_3_long_term']['recommendations'].append(rec)
            
            # Create implementation timeline
            implementation_timeline = []
            for phase_name, phase_data in phases.items():
                if phase_data['recommendations']:
                    implementation_timeline.append({
                        'phase': phase_name.replace('_', ' ').title(),
                        'timeline': phase_data['timeline'],
                        'focus': phase_data['focus'],
                        'recommendation_count': len(phase_data['recommendations']),
                        'key_activities': [rec.get('keyword', rec.get('topic', rec.get('category', 'optimization'))) 
                                        for rec in phase_data['recommendations'][:3]],
                        'expected_impact': self._estimate_phase_impact(phase_data['recommendations'])
                    })
            
            # Generate resource requirements
            resource_requirements = {
                'content_team': sum(1 for rec in all_recommendations if rec.get('optimization_type') == 'content'),
                'technical_team': sum(1 for rec in all_recommendations if rec.get('optimization_type') == 'position'),
                'analytics_team': sum(1 for rec in all_recommendations if rec.get('optimization_type') == 'budget'),
                'estimated_hours': len(all_recommendations) * 8,  # 8 hours per recommendation
                'budget_required': len(all_recommendations) * 500  # $500 per recommendation
            }
            
            implementation_plan = {
                'phases': phases,
                'implementation_timeline': implementation_timeline,
                'resource_requirements': resource_requirements,
                'total_recommendations': len(all_recommendations),
                'implementation_duration': '6 months',
                'success_metrics': [
                    'recommendations_implemented',
                    'performance_improvement',
                    'roi_achievement',
                    'timeline_adherence'
                ],
                'risk_factors': [
                    'Resource availability',
                    'Technical complexity',
                    'Market changes',
                    'Competitive responses'
                ]
            }
            
            return implementation_plan
            
        except Exception as e:
            self.logger.error(f"Error creating implementation plan: {str(e)}")
            return {
                'phases': {'phase_1': {'timeline': '1-3 months', 'recommendations': []}},
                'implementation_timeline': [],
                'total_recommendations': 0
            }

    def _estimate_phase_impact(self, recommendations):
        """Estimate impact of a phase of recommendations"""
        try:
            if not recommendations:
                return 'low'
            
            # Count high priority recommendations
            high_priority_count = len([rec for rec in recommendations if rec.get('priority') == 'high'])
            total_count = len(recommendations)
            
            high_priority_ratio = high_priority_count / total_count if total_count > 0 else 0
            
            if high_priority_ratio > 0.6:
                return 'high'
            elif high_priority_ratio > 0.3:
                return 'medium'
            else:
                return 'low'
        except Exception:
            return 'low'
