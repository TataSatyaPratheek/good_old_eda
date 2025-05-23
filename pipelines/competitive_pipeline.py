"""
Competitive Intelligence Pipeline
Comprehensive competitive analysis pipeline leveraging refactored modules and src/utils
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio

# Import refactored modules
from src.models.competitive_analyzer import CompetitiveAnalyzer, CompetitiveAnalysisResult, GapAnalysisResult
from src.features.competitive_features import CompetitiveFeatures, CompetitiveIntelligence
from src.analysis.traffic_comparator import TrafficComparator
from src.analysis.position_analyzer import PositionAnalyzer
from src.models.anomaly_detector import AnomalyDetector

# Import utils framework
from src.utils.common_helpers import timing_decorator, memoize
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.export_utils import ReportExporter, DataExporter
from src.utils.math_utils import StatisticalCalculator, TimeSeriesAnalyzer
from src.utils.visualization_utils import VisualizationEngine

# Import pipeline configuration
from .pipeline_config import PipelineConfigManager

class CompetitivePipeline:
    """
    Advanced Competitive Intelligence Pipeline
    
    Orchestrates comprehensive competitive analysis using all refactored modules
    """
    
    def __init__(self, config_manager: Optional[PipelineConfigManager] = None):
        """Initialize competitive pipeline with comprehensive utilities"""
        self.logger = LoggerFactory.get_logger("competitive_pipeline")
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        
        # Configuration management
        self.config_manager = config_manager or PipelineConfigManager()
        self.pipeline_config = self.config_manager.get_pipeline_config('competitive_pipeline')
        self.data_config = self.config_manager.data_config
        self.analysis_config = self.config_manager.analysis_config
        
        # Initialize refactored competitive modules
        self.competitive_analyzer = CompetitiveAnalyzer(logger=self.logger)
        self.competitive_features = CompetitiveFeatures(logger=self.logger)
        self.traffic_comparator = TrafficComparator(logger=self.logger)
        self.position_analyzer = PositionAnalyzer(logger=self.logger)
        self.anomaly_detector = AnomalyDetector(logger=self.logger)
        
        # Utilities
        self.data_processor = DataProcessor(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.time_series_analyzer = TimeSeriesAnalyzer(self.logger)
        self.report_exporter = ReportExporter(self.logger)
        self.data_exporter = DataExporter(self.logger)
        self.viz_engine = VisualizationEngine(self.logger)
        
        # Pipeline state
        self.pipeline_results = {}
        self.competitive_intelligence = {}

    @timing_decorator()
    async def run_comprehensive_competitive_analysis(
        self,
        primary_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        analysis_scope: Optional[List[str]] = None,
        competitive_objectives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive competitive intelligence pipeline
        
        Args:
            primary_data: Primary company data (e.g., Lenovo)
            competitor_data: Dictionary of competitor datasets
            analysis_scope: Scope of competitive analysis
            competitive_objectives: Specific competitive objectives
            
        Returns:
            Comprehensive competitive intelligence results
        """
        try:
            with self.performance_tracker.track_block("comprehensive_competitive_analysis"):
                # Audit log pipeline execution
                self.audit_logger.log_analysis_execution(
                    user_id="pipeline_system",
                    analysis_type="comprehensive_competitive_analysis",
                    parameters={
                        "primary_data_rows": len(primary_data),
                        "competitors_count": len(competitor_data),
                        "competitors": list(competitor_data.keys()),
                        "analysis_scope": analysis_scope,
                        "competitive_objectives": competitive_objectives
                    }
                )
                
                self.logger.info("Starting comprehensive competitive intelligence pipeline")
                
                # Phase 1: Data Preparation and Validation
                prepared_data = await self._prepare_competitive_data(
                    primary_data, competitor_data
                )
                
                # Phase 2: Market Landscape Analysis
                market_analysis = await self._execute_market_landscape_analysis(
                    prepared_data
                )
                
                # Phase 3: Competitive Positioning Analysis
                positioning_analysis = await self._execute_competitive_positioning(
                    prepared_data
                )
                
                # Phase 4: Gap Analysis and Opportunity Assessment
                gap_analysis = await self._execute_gap_analysis(
                    prepared_data
                )
                
                # Phase 5: Competitive Trend Analysis
                trend_analysis = await self._execute_competitive_trend_analysis(
                    prepared_data
                )
                
                # Phase 6: Threat Assessment and Risk Analysis
                threat_analysis = await self._execute_threat_assessment(
                    prepared_data, positioning_analysis
                )
                
                # Phase 7: Competitive Intelligence Synthesis
                intelligence_synthesis = await self._synthesize_competitive_intelligence({
                    'prepared_data': prepared_data,
                    'market_analysis': market_analysis,
                    'positioning_analysis': positioning_analysis,
                    'gap_analysis': gap_analysis,
                    'trend_analysis': trend_analysis,
                    'threat_analysis': threat_analysis
                })
                
                # Phase 8: Strategic Recommendations Generation
                strategic_recommendations = await self._generate_strategic_recommendations(
                    intelligence_synthesis
                )
                
                # Phase 9: Competitive Intelligence Dashboard
                intelligence_dashboard = await self._create_intelligence_dashboard(
                    intelligence_synthesis, strategic_recommendations
                )
                
                # Export comprehensive results
                export_results = await self._export_competitive_results(intelligence_synthesis)
                intelligence_synthesis['export_results'] = export_results
                intelligence_synthesis['intelligence_dashboard'] = intelligence_dashboard
                
                self.pipeline_results = intelligence_synthesis
                self.competitive_intelligence = intelligence_dashboard
                
                self.logger.info("Comprehensive competitive intelligence pipeline completed")
                return intelligence_synthesis
                
        except Exception as e:
            self.logger.error(f"Error in competitive intelligence pipeline: {str(e)}")
            await self._handle_pipeline_error(e)
            return {}

    async def _prepare_competitive_data(
        self,
        primary_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Prepare and validate competitive data"""
        try:
            with self.performance_tracker.track_block("competitive_data_preparation"):
                self.logger.info("Preparing competitive data")
                
                # Clean primary data
                cleaned_primary = self.data_processor.clean_seo_data(primary_data)
                
                # Validate primary data quality
                primary_validation = self.data_processor.validate_data_quality(cleaned_primary)
                
                # Clean and validate competitor data
                cleaned_competitors = {}
                competitor_validations = {}
                
                for competitor, data in competitor_data.items():
                    cleaned_data = self.data_processor.clean_seo_data(data)
                    validation = self.data_processor.validate_data_quality(cleaned_data)
                    
                    cleaned_competitors[competitor] = cleaned_data
                    competitor_validations[competitor] = validation
                    
                    self.logger.info(f"Processed {competitor}: {cleaned_data.shape}, quality: {validation.quality_score:.3f}")
                
                # Create competitive data summary
                competitive_summary = {
                    'primary_company': 'lenovo',
                    'primary_data_shape': cleaned_primary.shape,
                    'competitors': list(cleaned_competitors.keys()),
                    'competitor_data_shapes': {comp: data.shape for comp, data in cleaned_competitors.items()},
                    'data_quality_scores': {
                        'primary': primary_validation.quality_score,
                        **{comp: val.quality_score for comp, val in competitor_validations.items()}
                    },
                    'analysis_timeframe': self._determine_analysis_timeframe(cleaned_primary, cleaned_competitors),
                    'preparation_timestamp': datetime.now()
                }
                
                # Identify common keywords for analysis
                common_keywords = self._identify_common_keywords(cleaned_primary, cleaned_competitors)
                
                prepared_data = {
                    'primary_data': cleaned_primary,
                    'competitor_data': cleaned_competitors,
                    'primary_validation': primary_validation,
                    'competitor_validations': competitor_validations,
                    'competitive_summary': competitive_summary,
                    'common_keywords': common_keywords,
                    'analysis_metadata': {
                        'total_keywords_primary': len(cleaned_primary),
                        'total_keywords_competitors': {comp: len(data) for comp, data in cleaned_competitors.items()},
                        'common_keywords_count': len(common_keywords),
                        'data_coverage_overlap': self._calculate_data_coverage_overlap(cleaned_primary, cleaned_competitors)
                    }
                }
                
                self.logger.info(f"Competitive data preparation completed: {len(cleaned_competitors)} competitors, {len(common_keywords)} common keywords")
                return prepared_data
                
        except Exception as e:
            self.logger.error(f"Error in competitive data preparation: {str(e)}")
            return {}

    async def _execute_market_landscape_analysis(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive market landscape analysis"""
        try:
            with self.performance_tracker.track_block("market_landscape_analysis"):
                self.logger.info("Executing market landscape analysis")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitor_data = prepared_data.get('competitor_data', {})
                
                # Use competitive analyzer for comprehensive analysis
                market_analysis = self.competitive_analyzer.perform_comprehensive_competitive_analysis(
                    lenovo_data=primary_data,
                    competitor_data=competitor_data,
                    analysis_config={'include_trends': True, 'include_forecasting': True}
                )
                
                # Market size and share analysis
                market_metrics = self._calculate_market_metrics(primary_data, competitor_data)
                
                # Competitive intensity analysis
                competitive_intensity = self._analyze_competitive_intensity(primary_data, competitor_data)
                
                # Market opportunity analysis
                market_opportunities = self._identify_market_opportunities(
                    market_analysis, market_metrics
                )
                
                # Market evolution trends using time series analysis
                market_evolution = await self._analyze_market_evolution(primary_data, competitor_data)
                
                landscape_analysis = {
                    'comprehensive_analysis': market_analysis,
                    'market_metrics': market_metrics,
                    'competitive_intensity': competitive_intensity,
                    'market_opportunities': market_opportunities,
                    'market_evolution': market_evolution,
                    'landscape_insights': self._extract_landscape_insights(
                        market_analysis, market_metrics, competitive_intensity
                    ),
                    'analysis_metadata': {
                        'analysis_scope': 'comprehensive_market_landscape',
                        'competitors_analyzed': len(competitor_data),
                        'market_segments_identified': len(market_opportunities.get('segments', [])),
                        'analysis_timestamp': datetime.now()
                    }
                }
                
                self.logger.info("Market landscape analysis completed")
                return landscape_analysis
                
        except Exception as e:
            self.logger.error(f"Error in market landscape analysis: {str(e)}")
            return {}

    async def _execute_competitive_positioning(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute competitive positioning analysis"""
        try:
            with self.performance_tracker.track_block("competitive_positioning"):
                self.logger.info("Executing competitive positioning analysis")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitor_data = prepared_data.get('competitor_data', {})
                
                # Position comparison analysis using position analyzer
                position_comparison = self.position_analyzer.compare_competitive_positions(
                    lenovo_data=primary_data,
                    competitor_data=competitor_data,
                    analysis_depth='comprehensive'
                )
                
                # Traffic performance comparison using traffic comparator
                traffic_comparison = self.traffic_comparator.compare_traffic_performance(
                    lenovo_data=primary_data,
                    competitor_data=competitor_data,
                    comparison_metrics=['total_traffic', 'traffic_efficiency', 'growth_rate']
                )
                
                # Competitive strengths and weaknesses analysis
                strengths_weaknesses = self._analyze_competitive_strengths_weaknesses(
                    position_comparison, traffic_comparison
                )
                
                # Market position matrix
                position_matrix = self._create_competitive_position_matrix(
                    primary_data, competitor_data
                )
                
                # Competitive differentiation analysis
                differentiation_analysis = self._analyze_competitive_differentiation(
                    primary_data, competitor_data
                )
                
                positioning_analysis = {
                    'position_comparison': position_comparison,
                    'traffic_comparison': traffic_comparison,
                    'strengths_weaknesses': strengths_weaknesses,
                    'position_matrix': position_matrix,
                    'differentiation_analysis': differentiation_analysis,
                    'positioning_insights': self._extract_positioning_insights(
                        position_comparison, traffic_comparison, strengths_weaknesses
                    ),
                    'competitive_rankings': self._calculate_competitive_rankings(
                        position_comparison, traffic_comparison
                    )
                }
                
                self.logger.info("Competitive positioning analysis completed")
                return positioning_analysis
                
        except Exception as e:
            self.logger.error(f"Error in competitive positioning analysis: {str(e)}")
            return {}

    async def _execute_gap_analysis(self, prepared_data: Dict[str, Any]) -> GapAnalysisResult:
        """Execute comprehensive gap analysis"""
        try:
            with self.performance_tracker.track_block("gap_analysis"):
                self.logger.info("Executing gap analysis")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitor_data = prepared_data.get('competitor_data', {})
                
                # Use competitive analyzer for gap analysis
                gap_analysis = self.competitive_analyzer.perform_keyword_gap_analysis(
                    lenovo_data=primary_data,
                    competitor_data=competitor_data,
                    gap_threshold=10,
                    volume_threshold=100
                )
                
                # Content gap analysis
                content_gaps = self._analyze_content_gaps(primary_data, competitor_data)
                
                # SERP feature gaps
                serp_gaps = self._analyze_serp_feature_gaps(primary_data, competitor_data)
                
                # Performance gaps
                performance_gaps = self._analyze_performance_gaps(primary_data, competitor_data)
                
                # Enhance gap analysis with additional insights
                enhanced_gap_analysis = GapAnalysisResult(
                    keyword_gaps=gap_analysis.keyword_gaps,
                    content_gaps=gap_analysis.content_gaps + content_gaps,
                    opportunity_score=gap_analysis.opportunity_score,
                    priority_keywords=gap_analysis.priority_keywords,
                    competitive_disadvantages=gap_analysis.competitive_disadvantages,
                    actionable_insights=gap_analysis.actionable_insights + self._generate_additional_gap_insights(
                        content_gaps, serp_gaps, performance_gaps
                    )
                )
                
                self.logger.info(f"Gap analysis completed: {len(enhanced_gap_analysis.keyword_gaps)} keyword gaps identified")
                return enhanced_gap_analysis
                
        except Exception as e:
            self.logger.error(f"Error in gap analysis: {str(e)}")
            return GapAnalysisResult(pd.DataFrame(), [], 0.0, [], [], [])

    async def _execute_competitive_trend_analysis(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute competitive trend analysis"""
        try:
            with self.performance_tracker.track_block("competitive_trend_analysis"):
                self.logger.info("Executing competitive trend analysis")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitor_data = prepared_data.get('competitor_data', {})
                
                # Use competitive analyzer for trend analysis
                trend_analysis = self.competitive_analyzer.analyze_competitive_trends(
                    competitor_data=competitor_data,
                    time_period_days=90,
                    trend_analysis_method='comprehensive'
                )
                
                # Growth trajectory analysis
                growth_trajectories = self._analyze_growth_trajectories(primary_data, competitor_data)
                
                # Market momentum analysis
                market_momentum = self._analyze_market_momentum(trend_analysis, growth_trajectories)
                
                # Seasonal trend patterns
                seasonal_patterns = self._analyze_seasonal_patterns(primary_data, competitor_data)
                
                # Competitive velocity analysis
                competitive_velocity = self._analyze_competitive_velocity(trend_analysis)
                
                trend_analysis_result = {
                    'trend_analysis': trend_analysis,
                    'growth_trajectories': growth_trajectories,
                    'market_momentum': market_momentum,
                    'seasonal_patterns': seasonal_patterns,
                    'competitive_velocity': competitive_velocity,
                    'trend_insights': self._extract_trend_insights(
                        trend_analysis, growth_trajectories, market_momentum
                    ),
                    'trend_forecasts': self._generate_trend_forecasts(
                        trend_analysis, growth_trajectories
                    )
                }
                
                self.logger.info("Competitive trend analysis completed")
                return trend_analysis_result
                
        except Exception as e:
            self.logger.error(f"Error in competitive trend analysis: {str(e)}")
            return {}

    async def _execute_threat_assessment(
        self, 
        prepared_data: Dict[str, Any], 
        positioning_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute comprehensive threat assessment"""
        try:
            with self.performance_tracker.track_block("threat_assessment"):
                self.logger.info("Executing threat assessment")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitor_data = prepared_data.get('competitor_data', {})
                
                # Competitive threat scoring
                threat_scores = self._calculate_threat_scores(
                    positioning_analysis, prepared_data
                )
                
                # Market disruption analysis
                disruption_analysis = self._analyze_market_disruption_potential(
                    competitor_data, positioning_analysis
                )
                
                # Competitive response analysis
                response_analysis = self._analyze_competitive_responses(
                    primary_data, competitor_data
                )
                
                # Emerging competitor identification
                emerging_threats = self._identify_emerging_threats(
                    competitor_data, positioning_analysis
                )
                
                # Risk assessment matrix
                risk_matrix = self._create_competitive_risk_matrix(
                    threat_scores, disruption_analysis, emerging_threats
                )
                
                # Early warning indicators
                warning_indicators = self._establish_early_warning_indicators(
                    prepared_data, positioning_analysis
                )
                
                threat_assessment = {
                    'threat_scores': threat_scores,
                    'disruption_analysis': disruption_analysis,
                    'response_analysis': response_analysis,
                    'emerging_threats': emerging_threats,
                    'risk_matrix': risk_matrix,
                    'warning_indicators': warning_indicators,
                    'threat_mitigation_strategies': self._develop_threat_mitigation_strategies(
                        threat_scores, disruption_analysis, emerging_threats
                    ),
                    'threat_monitoring_plan': self._create_threat_monitoring_plan(
                        warning_indicators, emerging_threats
                    )
                }
                
                self.logger.info("Threat assessment completed")
                return threat_assessment
                
        except Exception as e:
            self.logger.error(f"Error in threat assessment: {str(e)}")
            return {}

    async def _synthesize_competitive_intelligence(self, all_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize comprehensive competitive intelligence"""
        try:
            with self.performance_tracker.track_block("intelligence_synthesis"):
                self.logger.info("Synthesizing competitive intelligence")
                
                # Create executive intelligence summary
                executive_summary = self._create_competitive_executive_summary(all_analysis)
                
                # Key competitive insights
                key_insights = self._extract_key_competitive_insights(all_analysis)
                
                # Strategic implications
                strategic_implications = self._analyze_strategic_implications(all_analysis)
                
                # Competitive intelligence scorecard
                intelligence_scorecard = self._create_intelligence_scorecard(all_analysis)
                
                # Action priority matrix
                action_priorities = self._create_action_priority_matrix(all_analysis)
                
                # Competitive intelligence timeline
                intelligence_timeline = self._create_intelligence_timeline(all_analysis)
                
                # Integration quality assessment
                integration_quality = self._assess_integration_quality(all_analysis)
                
                synthesized_intelligence = {
                    'executive_summary': executive_summary,
                    'key_insights': key_insights,
                    'strategic_implications': strategic_implications,
                    'intelligence_scorecard': intelligence_scorecard,
                    'action_priorities': action_priorities,
                    'intelligence_timeline': intelligence_timeline,
                    'detailed_analysis': all_analysis,
                    'synthesis_metadata': {
                        'synthesis_timestamp': datetime.now(),
                        'analysis_components': list(all_analysis.keys()),
                        'integration_quality_score': integration_quality,
                        'intelligence_confidence': self._calculate_intelligence_confidence(all_analysis),
                        'pipeline_execution_time': self.performance_tracker.get_performance_summary()
                    }
                }
                
                self.logger.info("Competitive intelligence synthesis completed")
                return synthesized_intelligence
                
        except Exception as e:
            self.logger.error(f"Error in intelligence synthesis: {str(e)}")
            return all_analysis

    async def _generate_strategic_recommendations(self, intelligence_synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive strategic recommendations"""
        try:
            with self.performance_tracker.track_block("strategic_recommendations"):
                self.logger.info("Generating strategic recommendations")
                
                # Short-term tactical recommendations (0-3 months)
                short_term_recommendations = self._generate_short_term_recommendations(
                    intelligence_synthesis
                )
                
                # Medium-term strategic recommendations (3-12 months)
                medium_term_recommendations = self._generate_medium_term_recommendations(
                    intelligence_synthesis
                )
                
                # Long-term strategic recommendations (12+ months)
                long_term_recommendations = self._generate_long_term_recommendations(
                    intelligence_synthesis
                )
                
                # Quick wins identification
                quick_wins = self._identify_quick_wins(intelligence_synthesis)
                
                # Resource allocation recommendations
                resource_allocation = self._recommend_resource_allocation(
                    intelligence_synthesis
                )
                
                # Risk mitigation recommendations
                risk_mitigation = self._recommend_risk_mitigation(intelligence_synthesis)
                
                # Competitive response strategies
                response_strategies = self._develop_competitive_response_strategies(
                    intelligence_synthesis
                )
                
                strategic_recommendations = {
                    'short_term_recommendations': short_term_recommendations,
                    'medium_term_recommendations': medium_term_recommendations,
                    'long_term_recommendations': long_term_recommendations,
                    'quick_wins': quick_wins,
                    'resource_allocation': resource_allocation,
                    'risk_mitigation': risk_mitigation,
                    'response_strategies': response_strategies,
                    'implementation_roadmap': self._create_implementation_roadmap(
                        short_term_recommendations, medium_term_recommendations, long_term_recommendations
                    ),
                    'success_metrics': self._define_success_metrics(intelligence_synthesis),
                    'recommendations_metadata': {
                        'total_recommendations': len(short_term_recommendations) + len(medium_term_recommendations) + len(long_term_recommendations),
                        'priority_distribution': self._analyze_priority_distribution(
                            short_term_recommendations, medium_term_recommendations, long_term_recommendations
                        ),
                        'generation_timestamp': datetime.now()
                    }
                }
                
                self.logger.info("Strategic recommendations generation completed")
                return strategic_recommendations
                
        except Exception as e:
            self.logger.error(f"Error generating strategic recommendations: {str(e)}")
            return {}

    async def _create_intelligence_dashboard(
        self, 
        intelligence_synthesis: Dict[str, Any], 
        strategic_recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive competitive intelligence dashboard"""
        try:
            with self.performance_tracker.track_block("intelligence_dashboard"):
                self.logger.info("Creating competitive intelligence dashboard")
                
                # Executive dashboard metrics
                executive_metrics = self._create_executive_dashboard_metrics(
                    intelligence_synthesis
                )
                
                # Competitive landscape overview
                landscape_overview = self._create_landscape_overview_dashboard(
                    intelligence_synthesis
                )
                
                # Threat monitoring dashboard
                threat_dashboard = self._create_threat_monitoring_dashboard(
                    intelligence_synthesis
                )
                
                # Opportunity tracking dashboard
                opportunity_dashboard = self._create_opportunity_tracking_dashboard(
                    intelligence_synthesis
                )
                
                # Performance benchmarking dashboard
                benchmarking_dashboard = self._create_benchmarking_dashboard(
                    intelligence_synthesis
                )
                
                # Action items dashboard
                action_dashboard = self._create_action_items_dashboard(
                    strategic_recommendations
                )
                
                # Intelligence alerts and notifications
                intelligence_alerts = self._create_intelligence_alerts(
                    intelligence_synthesis
                )
                
                intelligence_dashboard = {
                    'executive_metrics': executive_metrics,
                    'landscape_overview': landscape_overview,
                    'threat_dashboard': threat_dashboard,
                    'opportunity_dashboard': opportunity_dashboard,
                    'benchmarking_dashboard': benchmarking_dashboard,
                    'action_dashboard': action_dashboard,
                    'intelligence_alerts': intelligence_alerts,
                    'dashboard_metadata': {
                        'last_updated': datetime.now(),
                        'data_freshness': self._calculate_data_freshness(intelligence_synthesis),
                        'dashboard_completeness': self._assess_dashboard_completeness(
                            executive_metrics, landscape_overview, threat_dashboard
                        ),
                        'refresh_schedule': 'daily',
                        'alert_thresholds': self._define_alert_thresholds()
                    }
                }
                
                self.logger.info("Competitive intelligence dashboard creation completed")
                return intelligence_dashboard
                
        except Exception as e:
            self.logger.error(f"Error creating intelligence dashboard: {str(e)}")
            return {}

    async def _export_competitive_results(self, intelligence_synthesis: Dict[str, Any]) -> Dict[str, bool]:
        """Export comprehensive competitive intelligence results"""
        try:
            with self.performance_tracker.track_block("competitive_results_export"):
                self.logger.info("Exporting competitive intelligence results")
                
                export_results = {}
                
                # Export executive summary
                executive_export = self.report_exporter.export_executive_report(
                    intelligence_synthesis.get('executive_summary', {}),
                    f"{self.data_config.output_directory}/competitive_intelligence_executive_summary.html",
                    format='html',
                    include_charts=True
                )
                export_results['executive_summary'] = executive_export
                
                # Export detailed competitive analysis
                detailed_analysis = intelligence_synthesis.get('detailed_analysis', {})
                if detailed_analysis:
                    detailed_export = self.data_exporter.export_analysis_dataset(
                        {'competitive_analysis': pd.DataFrame([detailed_analysis])},
                        f"{self.data_config.output_directory}/detailed_competitive_analysis.xlsx"
                    )
                    export_results['detailed_analysis'] = detailed_export
                
                # Export gap analysis
                gap_analysis = detailed_analysis.get('gap_analysis')
                if gap_analysis and hasattr(gap_analysis, 'keyword_gaps') and not gap_analysis.keyword_gaps.empty:
                    gap_export = self.data_exporter.export_with_metadata(
                        gap_analysis.keyword_gaps,
                        metadata={'analysis_type': 'competitive_gaps', 'generation_timestamp': datetime.now()},
                        export_path=f"{self.data_config.output_directory}/competitive_gap_analysis.xlsx"
                    )
                    export_results['gap_analysis'] = gap_export
                
                # Export competitive visualizations
                viz_export = await self._export_competitive_visualizations(intelligence_synthesis)
                export_results['visualizations'] = viz_export
                
                # Export intelligence scorecard
                scorecard = intelligence_synthesis.get('intelligence_scorecard', {})
                if scorecard:
                    scorecard_export = self.data_exporter.export_with_metadata(
                        pd.DataFrame([scorecard]),
                        metadata={'analysis_type': 'intelligence_scorecard'},
                        export_path=f"{self.data_config.output_directory}/competitive_intelligence_scorecard.xlsx"
                    )
                    export_results['intelligence_scorecard'] = scorecard_export
                
                self.logger.info("Competitive intelligence results export completed")
                return export_results
                
        except Exception as e:
            self.logger.error(f"Error exporting competitive results: {str(e)}")
            return {}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_name': 'competitive_pipeline',
            'status': 'completed' if self.pipeline_results else 'not_started',
            'competitive_intelligence_available': bool(self.competitive_intelligence),
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'results_available': bool(self.pipeline_results)
        }

    async def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline execution errors"""
        self.logger.error(f"Competitive pipeline error: {str(error)}")
        self.audit_logger.log_analysis_execution(
            user_id="pipeline_system",
            analysis_type="competitive_pipeline_error",
            result="failure",
            details={"error": str(error)}
        )

    # New feature methods (stubs)
    def _analyze_competitive_intensity(self, primary_data: pd.DataFrame, competitor_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze competitive intensity"""
        try:
            # primary_data is available if needed for more complex analysis
            return {
                'intensity_score': 0.5,  # Example value
                'key_competitors': list(competitor_data.keys()),
                'competitive_pressure': 'medium'  # Example value
            }
        except Exception as e:
            self.logger.error(f"Error analyzing competitive intensity: {str(e)}")
            return {}

    def _analyze_content_gaps(self, primary_data: pd.DataFrame, competitor_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Analyze content gaps"""
        try:
            # primary_data and competitor_data are available for analysis
            # Should return a list of dictionaries, each representing a content gap
            return []  # Placeholder for List[Dict[str, Any]]
        except Exception as e:
            self.logger.error(f"Error analyzing content gaps: {str(e)}")
            return []

    def _analyze_growth_trajectories(self, primary_data: pd.DataFrame, competitor_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze growth trajectories"""
        try:
            # primary_data and competitor_data are available for analysis
            return {
                'growth_patterns': {},
                'trajectory_analysis': {},
                'future_projections': {}
            }
        except Exception as e:
            self.logger.error(f"Error analyzing growth trajectories: {str(e)}")
            return {}

    def _calculate_threat_scores(self, positioning_analysis: Dict[str, Any], prepared_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate threat scores for competitors"""
        try:
            # positioning_analysis and prepared_data (which contains competitor_data) are available
            threat_scores = {}
            competitor_names = prepared_data.get('competitor_data', {}).keys()
            for competitor in competitor_names:
                threat_scores[competitor] = 0.5  # Default medium threat
            return threat_scores
        except Exception as e:
            self.logger.error(f"Error calculating threat scores: {str(e)}")
            return {}

    def _extract_key_competitive_insights(self, all_analysis: Dict[str, Any]) -> List[str]:
        """Extract key competitive insights"""
        try:
            # all_analysis contains all processed stages
            return [
                "Competitive landscape analysis completed (stub insight)",
                "Key opportunities identified (stub insight)",
                "Threat assessment performed (stub insight)"
            ]
        except Exception as e:
            self.logger.error(f"Error extracting insights: {str(e)}")
            return []

    def _generate_short_term_recommendations(self, intelligence_synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate short-term recommendations"""
        try:
            # intelligence_synthesis is available for generating recommendations
            return [
                {
                    'recommendation': 'Monitor competitor movements (stub recommendation)',
                    'priority': 'high',
                    'timeframe': 'short-term'
                }
            ]
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def _create_executive_dashboard_metrics(self, intelligence_synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive dashboard metrics"""
        try:
            # intelligence_synthesis is available for dashboard creation
            return {
                'dashboard_data_summary': { # Changed key to avoid direct full data copy
                    'executive_summary_points': intelligence_synthesis.get('executive_summary', {}).get('points', []),
                    'key_insight_count': len(intelligence_synthesis.get('key_insights', []))
                },
                'key_metrics': {
                     'overall_opportunity_score': intelligence_synthesis.get('detailed_analysis', {}).get('gap_analysis', {}).get('opportunity_score', 0.0),
                     'market_position_estimate': intelligence_synthesis.get('executive_summary', {}).get('market_position', 'unknown')
                },
                'visualizations_needed': ["Market Share Pie Chart", "Competitive Positioning Matrix"] # Placeholder names
            }
        except Exception as e:
            self.logger.error(f"Error creating dashboard metrics: {str(e)}")
            return {}

    # Helper methods
    def _determine_analysis_timeframe(self, primary_data, competitor_data):
        """Determine analysis timeframe from data"""
        try:
            all_dates = []
            if 'date' in primary_data.columns:
                all_dates.extend(pd.to_datetime(primary_data['date']).tolist())
            
            for data in competitor_data.values():
                if 'date' in data.columns:
                    all_dates.extend(pd.to_datetime(data['date']).tolist())
            
            if all_dates:
                return {
                    'start_date': min(all_dates),
                    'end_date': max(all_dates),
                    'total_days': (max(all_dates) - min(all_dates)).days
                }
            
            return {'start_date': None, 'end_date': None, 'total_days': 0}
        except Exception:
            return {'start_date': None, 'end_date': None, 'total_days': 0}

    def _identify_common_keywords(self, primary_data, competitor_data):
        """Identify keywords common across datasets"""
        try:
            primary_keywords = set(primary_data['Keyword'].str.lower().tolist())
            
            common_keywords = primary_keywords
            for data in competitor_data.values():
                competitor_keywords = set(data['Keyword'].str.lower().tolist())
                common_keywords = common_keywords.intersection(competitor_keywords)
            
            return list(common_keywords)
        except Exception:
            return []

    def _calculate_data_coverage_overlap(self, primary_data, competitor_data):
        """Calculate data coverage overlap percentage"""
        try:
            primary_keywords = set(primary_data['Keyword'].str.lower().tolist())
            total_unique_keywords = primary_keywords.copy()
            
            for data in competitor_data.values():
                competitor_keywords = set(data['Keyword'].str.lower().tolist())
                total_unique_keywords.update(competitor_keywords)
            
            overlap_percentage = len(primary_keywords) / len(total_unique_keywords) if total_unique_keywords else 0
            return overlap_percentage
        except Exception:
            return 0.0

    def _calculate_market_metrics(self, primary_data, competitor_data):
        """Calculate comprehensive market metrics"""
        try:
            # Calculate total market size
            total_traffic = primary_data.get('Traffic (%)', pd.Series()).sum()
            competitor_traffic = sum(data.get('Traffic (%)', pd.Series()).sum() for data in competitor_data.values())
            total_market_traffic = total_traffic + competitor_traffic
            
            # Market share calculation
            market_share = total_traffic / total_market_traffic if total_market_traffic > 0 else 0
            
            # Keyword universe size
            total_keywords = len(primary_data)
            competitor_keywords = sum(len(data) for data in competitor_data.values())
            total_keyword_universe = total_keywords + competitor_keywords
            
            return {
                'total_market_traffic': total_market_traffic,
                'primary_market_share': market_share,
                'primary_traffic': total_traffic,
                'competitor_traffic': competitor_traffic,
                'keyword_universe_size': total_keyword_universe,
                'primary_keyword_coverage': total_keywords / total_keyword_universe if total_keyword_universe > 0 else 0,
                'market_concentration_ratio': self._calculate_market_concentration(primary_data, competitor_data)
            }
        except Exception:
            return {}

    def _calculate_market_concentration(self, primary_data, competitor_data):
        """Calculate market concentration ratio"""
        try:
            traffic_values = [primary_data.get('Traffic (%)', pd.Series()).sum()]
            traffic_values.extend([data.get('Traffic (%)', pd.Series()).sum() for data in competitor_data.values()])
            
            total_traffic = sum(traffic_values)
            if total_traffic == 0:
                return 0
            
            # Calculate HHI (Herfindahl-Hirschman Index)
            market_shares = [traffic / total_traffic for traffic in traffic_values]
            hhi = sum(share ** 2 for share in market_shares)
            
            return hhi
        except Exception:
            return 0

    def _create_competitive_executive_summary(self, all_analysis):
        """Create executive summary for competitive analysis"""
        return {
            'analysis_scope': 'comprehensive_competitive_intelligence',
            'competitors_analyzed': len(all_analysis.get('prepared_data', {}).get('competitor_data', {})),
            'market_position': self._assess_market_position(all_analysis),
            'competitive_threats': self._count_competitive_threats(all_analysis),
            'opportunities_identified': self._count_opportunities(all_analysis),
            'strategic_priority': 'high',  # Based on analysis
            'analysis_timestamp': datetime.now()
        }

    def _assess_market_position(self, all_analysis):
        """Assess overall market position"""
        try:
            positioning = all_analysis.get('positioning_analysis', {})
            if positioning and 'competitive_rankings' in positioning:
                rankings = positioning['competitive_rankings']
                # Simplified assessment
                return 'leading' if rankings.get('overall_rank', 5) <= 2 else 'competitive'
            return 'unknown'
        except Exception:
            return 'unknown'

    def _count_competitive_threats(self, all_analysis):
        """Count identified competitive threats"""
        try:
            threat_analysis = all_analysis.get('threat_analysis', {})
            threat_scores = threat_analysis.get('threat_scores', {})
            high_threats = len([score for score in threat_scores.values() if score > 0.7])
            return high_threats
        except Exception:
            return 0

    def _count_opportunities(self, all_analysis):
        """Count identified opportunities"""
        try:
            gap_analysis = all_analysis.get('gap_analysis')
            if gap_analysis and hasattr(gap_analysis, 'priority_keywords'):
                return len(gap_analysis.priority_keywords)
            return 0
        except Exception:
            return 0

    async def _export_competitive_visualizations(self, intelligence_synthesis):
        """Export competitive visualizations"""
        try:
            # Create competitive landscape visualization
            landscape_viz = self.viz_engine.create_competitive_landscape_chart(
                intelligence_synthesis.get('detailed_analysis', {}),
                output_path=f"{self.data_config.output_directory}/visuals/competitive_landscape/"
            )
            
            # Create market share visualization
            market_viz = self.viz_engine.create_market_share_charts(
                intelligence_synthesis.get('detailed_analysis', {}),
                output_path=f"{self.data_config.output_directory}/visuals/market_share/"
            )
            
            return landscape_viz and market_viz
        except Exception as e:
            self.logger.error(f"Error creating competitive visualizations: {str(e)}")
            return False
