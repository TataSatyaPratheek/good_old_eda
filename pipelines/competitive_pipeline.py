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

    # Helper and analysis methods
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

    def _analyze_competitive_intensity(self, primary_data: pd.DataFrame, competitor_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze competitive intensity"""
        try:
            return {
                'intensity_score': 0.5,
                'key_competitors': list(competitor_data.keys()),
                'competitive_pressure': 'medium'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing competitive intensity: {str(e)}")
            return {}

    def _identify_market_opportunities(self, market_analysis, market_metrics):
        """Identify market opportunities"""
        try:
            opportunities = []

            # Analyze market size potential
            market_size = market_metrics.get('total_market_traffic', 0)
            primary_share = market_metrics.get('primary_market_share', 0)

            if primary_share < 0.3 and market_size > 1000:
                opportunities.append({
                    'type': 'market_expansion',
                    'description': 'Significant market share growth potential',
                    'priority': 'high',
                    'estimated_impact': market_size * 0.1
                })

            # Analyze competitive gaps
            competitive_gaps = market_analysis.get('gap_analysis', {})
            if competitive_gaps:
                opportunities.append({
                    'type': 'competitive_gaps',
                    'description': 'Keyword gaps identified vs competitors',
                    'priority': 'medium'
                })

            return {
                'opportunities': opportunities,
                'segments': ['high-value-keywords', 'content-gaps', 'serp-features'],
                'market_potential_score': min(primary_share * 2, 1.0)
            }
        except Exception as e:
            self.logger.error(f"Error identifying market opportunities: {str(e)}")
            return {'opportunities': [], 'segments': [], 'market_potential_score': 0}

    async def _analyze_market_evolution(self, primary_data, competitor_data):
        """Analyze market evolution trends"""
        try:
            # Use time series analyzer for market evolution
            evolution_data = self.time_series_analyzer.analyze_trend_patterns(
                primary_data, time_column='date' if 'date' in primary_data.columns else None
            )
            
            return {
                'evolution_trends': evolution_data,
                'market_maturity': 'developing',
                'growth_stage': 'expansion'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing market evolution: {str(e)}")
            return {}

    def _extract_landscape_insights(self, market_analysis, market_metrics, competitive_intensity):
        """Extract key landscape insights"""
        try:
            insights = []
            
            # Market size insights
            market_size = market_metrics.get('total_market_traffic', 0)
            if market_size > 10000:
                insights.append("Large market opportunity with significant traffic potential")
            
            # Competitive intensity insights
            intensity = competitive_intensity.get('intensity_score', 0)
            if intensity > 0.7:
                insights.append("Highly competitive market environment")
            elif intensity < 0.3:
                insights.append("Low competition presents opportunity for market share gains")
            
            return insights
        except Exception:
            return []

    def _analyze_competitive_strengths_weaknesses(self, position_comparison, traffic_comparison):
        """Analyze competitive strengths and weaknesses"""
        try:
            strengths = []
            weaknesses = []

            # Analyze position strengths
            if position_comparison.get('keywords_ahead', 0) > position_comparison.get('keywords_behind', 0):
                strengths.append("Superior keyword positioning vs competitors")
            else:
                weaknesses.append("Lagging in keyword positions vs competitors")

            # Analyze traffic performance
            traffic_performance = traffic_comparison.get('relative_position', 'unknown')
            if traffic_performance in ['leading', 'strong']:
                strengths.append("Strong traffic performance")
            elif traffic_performance in ['lagging', 'weak']:
                weaknesses.append("Below-average traffic performance")

            return {
                'strengths': strengths,
                'weaknesses': weaknesses,
                'competitive_advantages': [s for s in strengths if 'superior' in s.lower()],
                'improvement_areas': weaknesses,
                'strength_score': len(strengths) / (len(strengths) + len(weaknesses)) if (strengths or weaknesses) else 0.5
            }
        except Exception as e:
            self.logger.error(f"Error analyzing strengths/weaknesses: {str(e)}")
            return {'strengths': [], 'weaknesses': [], 'competitive_advantages': [], 'improvement_areas': [], 'strength_score': 0}

    def _create_competitive_position_matrix(self, primary_data, competitor_data):
        """Create competitive position matrix"""
        try:
            # Create position matrix based on traffic and keyword coverage
            matrix_data = {}
            
            # Primary data position
            primary_traffic = primary_data.get('Traffic (%)', pd.Series()).sum()
            primary_keywords = len(primary_data)
            matrix_data['primary'] = {'traffic': primary_traffic, 'keywords': primary_keywords}
            
            # Competitor positions
            for competitor, data in competitor_data.items():
                comp_traffic = data.get('Traffic (%)', pd.Series()).sum()
                comp_keywords = len(data)
                matrix_data[competitor] = {'traffic': comp_traffic, 'keywords': comp_keywords}
            
            return matrix_data
        except Exception as e:
            self.logger.error(f"Error creating position matrix: {str(e)}")
            return {}

    def _analyze_competitive_differentiation(self, primary_data, competitor_data):
        """Analyze competitive differentiation"""
        try:
            differentiation_factors = []
            
            # Analyze unique keywords
            primary_keywords = set(primary_data['Keyword'].str.lower())
            all_competitor_keywords = set()
            for data in competitor_data.values():
                all_competitor_keywords.update(data['Keyword'].str.lower())
            
            unique_keywords = primary_keywords - all_competitor_keywords
            if len(unique_keywords) > 100:
                differentiation_factors.append("Strong unique keyword portfolio")
            
            return {
                'differentiation_factors': differentiation_factors,
                'unique_keyword_count': len(unique_keywords),
                'differentiation_score': len(unique_keywords) / len(primary_keywords) if primary_keywords else 0
            }
        except Exception as e:
            self.logger.error(f"Error analyzing differentiation: {str(e)}")
            return {}

    def _extract_positioning_insights(self, position_comparison, traffic_comparison, strengths_weaknesses):
        """Extract positioning insights"""
        try:
            insights = []
            
            # Position insights
            if position_comparison.get('keywords_ahead', 0) > 100:
                insights.append("Strong competitive positioning with significant keyword advantages")
            
            # Traffic insights
            traffic_efficiency = traffic_comparison.get('traffic_efficiency', 0)
            if traffic_efficiency > 1.2:
                insights.append("Above-average traffic efficiency per keyword")
            
            return insights
        except Exception:
            return []

    def _calculate_competitive_rankings(self, position_comparison, traffic_comparison):
        """Calculate competitive rankings"""
        try:
            # Simple ranking calculation based on available metrics
            position_score = position_comparison.get('keywords_ahead', 0) - position_comparison.get('keywords_behind', 0)
            traffic_score = traffic_comparison.get('total_traffic', 0)
            
            overall_score = position_score * 0.6 + traffic_score * 0.4
            
            return {
                'position_rank': 1 if position_score > 0 else 3,
                'traffic_rank': 1 if traffic_score > 1000 else 3,
                'overall_rank': 1 if overall_score > 500 else 3,
                'ranking_methodology': 'position_and_traffic_weighted'
            }
        except Exception:
            return {'position_rank': 5, 'traffic_rank': 5, 'overall_rank': 5}

    def _analyze_content_gaps(self, primary_data: pd.DataFrame, competitor_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Analyze content gaps"""
        try:
            gaps = []
            
            # Analyze topic coverage gaps
            primary_topics = set()
            if 'Keyword' in primary_data.columns:
                primary_topics = set([keyword.split()[0].lower() for keyword in primary_data['Keyword'] if keyword])
            
            for competitor, comp_data in competitor_data.items():
                if 'Keyword' in comp_data.columns:
                    comp_topics = set([keyword.split()[0].lower() for keyword in comp_data['Keyword'] if keyword])
                    topic_gaps = comp_topics - primary_topics
                    
                    for topic in list(topic_gaps)[:10]:  # Limit to top 10 gaps
                        gaps.append({
                            'gap_type': 'topic_coverage',
                            'topic': topic,
                            'competitor': competitor,
                            'priority': 'medium'
                        })
            
            return gaps
        except Exception as e:
            self.logger.error(f"Error analyzing content gaps: {str(e)}")
            return []

    def _analyze_serp_feature_gaps(self, primary_data, competitor_data):
        """Analyze SERP feature gaps"""
        try:
            gaps = []

            # Check if SERP features column exists
            serp_col = 'SERP Features by Keyword'
            if serp_col not in primary_data.columns:
                return gaps

            # Analyze primary SERP feature presence
            primary_features = set()
            for features_str in primary_data[serp_col].dropna():
                if features_str:
                    features = [f.strip() for f in str(features_str).split(',')]
                    primary_features.update(features)

            # Compare with competitors
            for competitor, comp_data in competitor_data.items():
                if serp_col in comp_data.columns:
                    comp_features = set()
                    for features_str in comp_data[serp_col].dropna():
                        if features_str:
                            features = [f.strip() for f in str(features_str).split(',')]
                            comp_features.update(features)

                    # Identify gaps
                    missing_features = comp_features - primary_features
                    for feature in missing_features:
                        gaps.append({
                            'feature': feature,
                            'competitor': competitor,
                            'gap_type': 'missing_feature',
                            'priority': 'medium'
                        })

            return gaps
        except Exception as e:
            self.logger.error(f"Error analyzing SERP feature gaps: {str(e)}")
            return []

    def _analyze_performance_gaps(self, primary_data, competitor_data):
        """Analyze performance gaps"""
        try:
            gaps = []
            
            # Analyze traffic performance gaps
            primary_avg_traffic = primary_data.get('Traffic (%)', pd.Series()).mean()
            
            for competitor, comp_data in competitor_data.items():
                comp_avg_traffic = comp_data.get('Traffic (%)', pd.Series()).mean()
                
                if comp_avg_traffic > primary_avg_traffic * 1.2:  # 20% threshold
                    gaps.append({
                        'gap_type': 'traffic_performance',
                        'competitor': competitor,
                        'gap_magnitude': comp_avg_traffic - primary_avg_traffic,
                        'priority': 'high'
                    })
            
            return gaps
        except Exception as e:
            self.logger.error(f"Error analyzing performance gaps: {str(e)}")
            return []

    def _generate_additional_gap_insights(self, content_gaps, serp_gaps, performance_gaps):
        """Generate additional insights from gap analysis"""
        try:
            insights = []
            
            if len(content_gaps) > 10:
                insights.append("Significant content coverage gaps identified across multiple competitors")
            
            if len(serp_gaps) > 5:
                insights.append("Missing key SERP features compared to competitors")
            
            if len(performance_gaps) > 2:
                insights.append("Performance optimization opportunities exist vs top competitors")
            
            return insights
        except Exception:
            return []

    def _analyze_growth_trajectories(self, primary_data: pd.DataFrame, competitor_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze growth trajectories"""
        try:
            # Simplified growth analysis
            growth_patterns = {}
            
            # Analyze primary growth if date column exists
            if 'date' in primary_data.columns:
                # Use time series analyzer for growth trends
                primary_growth = self.time_series_analyzer.calculate_growth_trends(primary_data)
                growth_patterns['primary'] = primary_growth
            
            # Analyze competitor growth
            for competitor, data in competitor_data.items():
                if 'date' in data.columns:
                    comp_growth = self.time_series_analyzer.calculate_growth_trends(data)
                    growth_patterns[competitor] = comp_growth
            
            return {
                'growth_patterns': growth_patterns,
                'trajectory_analysis': self._analyze_trajectory_patterns(growth_patterns),
                'future_projections': self._project_future_growth(growth_patterns)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing growth trajectories: {str(e)}")
            return {'growth_patterns': {}, 'trajectory_analysis': {}, 'future_projections': {}}

    def _analyze_trajectory_patterns(self, growth_patterns):
        """Analyze trajectory patterns"""
        try:
            patterns = {}
            for entity, growth_data in growth_patterns.items():
                if growth_data and 'trend_direction' in growth_data:
                    patterns[entity] = {
                        'trend': growth_data['trend_direction'],
                        'stability': growth_data.get('trend_stability', 'unknown'),
                        'growth_rate': growth_data.get('growth_rate', 0)
                    }
            return patterns
        except Exception:
            return {}

    def _project_future_growth(self, growth_patterns):
        """Project future growth based on patterns"""
        try:
            projections = {}
            for entity, growth_data in growth_patterns.items():
                if growth_data and 'growth_rate' in growth_data:
                    current_rate = growth_data['growth_rate']
                    projections[entity] = {
                        '3_month_projection': current_rate * 3,
                        '6_month_projection': current_rate * 6,
                        'confidence': 'medium'
                    }
            return projections
        except Exception:
            return {}

    def _analyze_market_momentum(self, trend_analysis, growth_trajectories):
        """Analyze market momentum"""
        try:
            momentum_indicators = []
            overall_momentum = 'stable'

            # Analyze trend patterns
            if trend_analysis and 'trend_patterns' in trend_analysis:
                trends = trend_analysis['trend_patterns']
                if trends:
                    positive_trends = sum(1 for t in trends.values() if t.get('direction') == 'positive')
                    total_trends = len(trends)
                    
                    if total_trends > 0:
                        if positive_trends / total_trends > 0.6:
                            overall_momentum = 'accelerating'
                            momentum_indicators.append("Majority of trends showing positive momentum")
                        elif positive_trends / total_trends < 0.4:
                            overall_momentum = 'decelerating'
                            momentum_indicators.append("Declining trend momentum detected")

            return {
                'overall_momentum': overall_momentum,
                'momentum_indicators': momentum_indicators,
                'momentum_score': 0.5  # Default value
            }
        except Exception as e:
            self.logger.error(f"Error analyzing market momentum: {str(e)}")
            return {'overall_momentum': 'unknown', 'momentum_indicators': []}

    def _analyze_seasonal_patterns(self, primary_data, competitor_data):
        """Analyze seasonal patterns"""
        try:
            patterns = {}
            
            # Check if date column exists for seasonal analysis
            if 'date' in primary_data.columns:
                primary_seasonal = self.time_series_analyzer.detect_seasonal_patterns(primary_data)
                patterns['primary'] = primary_seasonal
            
            for competitor, data in competitor_data.items():
                if 'date' in data.columns:
                    comp_seasonal = self.time_series_analyzer.detect_seasonal_patterns(data)
                    patterns[competitor] = comp_seasonal
            
            return {
                'seasonal_patterns': patterns,
                'seasonal_insights': self._extract_seasonal_insights(patterns)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing seasonal patterns: {str(e)}")
            return {}

    def _extract_seasonal_insights(self, patterns):
        """Extract insights from seasonal patterns"""
        try:
            insights = []
            
            for entity, pattern_data in patterns.items():
                if pattern_data and pattern_data.get('has_seasonality'):
                    peak_season = pattern_data.get('peak_season', 'unknown')
                    insights.append(f"{entity} shows seasonal peaks in {peak_season}")
            
            return insights
        except Exception:
            return []

    def _analyze_competitive_velocity(self, trend_analysis):
        """Analyze competitive velocity"""
        try:
            velocity_metrics = {}
            
            if trend_analysis and 'competitor_trends' in trend_analysis:
                for competitor, trend_data in trend_analysis['competitor_trends'].items():
                    velocity_score = trend_data.get('trend_strength', 0) * trend_data.get('trend_consistency', 1)
                    velocity_metrics[competitor] = {
                        'velocity_score': velocity_score,
                        'trend_direction': trend_data.get('direction', 'stable'),
                        'acceleration': trend_data.get('acceleration', 0)
                    }
            
            return {
                'competitor_velocities': velocity_metrics,
                'overall_market_velocity': 'moderate',  # Simplified
                'velocity_insights': self._extract_velocity_insights(velocity_metrics)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing competitive velocity: {str(e)}")
            return {}

    def _extract_velocity_insights(self, velocity_metrics):
        """Extract insights from velocity analysis"""
        try:
            insights = []
            
            high_velocity_competitors = [comp for comp, metrics in velocity_metrics.items() 
                                       if metrics.get('velocity_score', 0) > 0.7]
            
            if high_velocity_competitors:
                insights.append(f"High velocity competitors identified: {', '.join(high_velocity_competitors)}")
            
            return insights
        except Exception:
            return []

    def _extract_trend_insights(self, trend_analysis, growth_trajectories, market_momentum):
        """Extract trend insights"""
        try:
            insights = []
            
            # Market momentum insights
            momentum = market_momentum.get('overall_momentum', 'stable')
            if momentum == 'accelerating':
                insights.append("Market showing strong positive momentum")
            elif momentum == 'decelerating':
                insights.append("Market momentum declining - potential risk")
            
            # Growth trajectory insights
            if growth_trajectories and 'growth_patterns' in growth_trajectories:
                patterns = growth_trajectories['growth_patterns']
                if patterns:
                    insights.append(f"Growth patterns identified for {len(patterns)} entities")
            
            return insights
        except Exception:
            return []

    def _generate_trend_forecasts(self, trend_analysis, growth_trajectories):
        """Generate trend forecasts"""
        try:
            forecasts = {}
            
            # Generate forecasts based on growth trajectories
            if growth_trajectories and 'future_projections' in growth_trajectories:
                projections = growth_trajectories['future_projections']
                for entity, projection in projections.items():
                    forecasts[entity] = {
                        'forecast_direction': 'positive' if projection.get('6_month_projection', 0) > 0 else 'negative',
                        'confidence_level': projection.get('confidence', 'medium'),
                        'projection_period': '6_months'
                    }
            
            return {
                'entity_forecasts': forecasts,
                'market_forecast': self._generate_market_forecast(forecasts),
                'forecast_confidence': 'medium'
            }
        except Exception as e:
            self.logger.error(f"Error generating trend forecasts: {str(e)}")
            return {}

    def _generate_market_forecast(self, entity_forecasts):
        """Generate overall market forecast"""
        try:
            if not entity_forecasts:
                return {'direction': 'stable', 'confidence': 'low'}
            
            positive_forecasts = sum(1 for forecast in entity_forecasts.values() 
                                   if forecast.get('forecast_direction') == 'positive')
            total_forecasts = len(entity_forecasts)
            
            if positive_forecasts / total_forecasts > 0.6:
                return {'direction': 'growth', 'confidence': 'medium'}
            elif positive_forecasts / total_forecasts < 0.4:
                return {'direction': 'decline', 'confidence': 'medium'}
            else:
                return {'direction': 'stable', 'confidence': 'medium'}
        except Exception:
            return {'direction': 'unknown', 'confidence': 'low'}

    def _calculate_threat_scores(self, positioning_analysis, prepared_data):
        """Calculate threat scores for competitors"""
        try:
            threat_scores = {}
            
            competitor_data = prepared_data.get('competitor_data', {})
            for competitor, data in competitor_data.items():
                # Calculate threat score based on various factors
                traffic_threat = data.get('Traffic (%)', pd.Series()).sum() / 1000  # Normalize
                keyword_threat = len(data) / 1000  # Normalize
                
                # Position threat from positioning analysis
                position_threat = 0.5  # Default value
                if positioning_analysis and 'competitive_rankings' in positioning_analysis:
                    rankings = positioning_analysis['competitive_rankings']
                    competitor_rank = rankings.get(f'{competitor}_rank', 5)
                    position_threat = max(0, (6 - competitor_rank) / 5)  # Higher threat for better ranks
                
                overall_threat = (traffic_threat * 0.4 + keyword_threat * 0.3 + position_threat * 0.3)
                threat_scores[competitor] = min(1.0, overall_threat)  # Cap at 1.0
            
            return threat_scores
        except Exception as e:
            self.logger.error(f"Error calculating threat scores: {str(e)}")
            return {}

    def _analyze_market_disruption_potential(self, competitor_data, positioning_analysis):
        """Analyze market disruption potential"""
        try:
            disruption_factors = []
            disruption_risk = 'low'

            # Analyze competitor growth rates
            for competitor, data in competitor_data.items():
                if len(data) > 0:
                    # Check for rapid competitor growth (simplified)
                    if 'Traffic (%)' in data.columns:
                        avg_traffic = data['Traffic (%)'].mean()
                        if avg_traffic > 5:  # Threshold for significant traffic
                            disruption_factors.append(f"{competitor} showing strong traffic performance")

            # Assess overall disruption risk
            if len(disruption_factors) > 2:
                disruption_risk = 'high'
            elif len(disruption_factors) > 0:
                disruption_risk = 'medium'

            return {
                'disruption_risk_level': disruption_risk,
                'disruption_factors': disruption_factors,
                'potential_disruptors': [factor.split()[0] for factor in disruption_factors],
                'disruption_timeline': 'medium-term',
                'mitigation_strategies': [
                    'Monitor competitor movements closely',
                    'Strengthen competitive advantages',
                    'Diversify keyword portfolio'
                ]
            }
        except Exception as e:
            self.logger.error(f"Error analyzing market disruption: {str(e)}")
            return {'disruption_risk_level': 'unknown', 'disruption_factors': [], 'potential_disruptors': []}

    def _analyze_competitive_responses(self, primary_data, competitor_data):
        """Analyze competitive responses"""
        try:
            responses = {}
            
            for competitor, comp_data in competitor_data.items():
                # Analyze keyword overlap and potential competitive responses
                primary_keywords = set(primary_data['Keyword'].str.lower())
                comp_keywords = set(comp_data['Keyword'].str.lower())
                
                overlap = len(primary_keywords.intersection(comp_keywords))
                total_comp_keywords = len(comp_keywords)
                
                overlap_ratio = overlap / total_comp_keywords if total_comp_keywords > 0 else 0
                
                responses[competitor] = {
                    'keyword_overlap_ratio': overlap_ratio,
                    'response_intensity': 'high' if overlap_ratio > 0.7 else 'medium' if overlap_ratio > 0.4 else 'low',
                    'competitive_focus': 'direct' if overlap_ratio > 0.5 else 'adjacent'
                }
            
            return {
                'competitor_responses': responses,
                'overall_response_level': 'medium',  # Simplified
                'response_insights': self._extract_response_insights(responses)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing competitive responses: {str(e)}")
            return {}

    def _extract_response_insights(self, responses):
        """Extract insights from competitive response analysis"""
        try:
            insights = []
            
            high_overlap_competitors = [comp for comp, data in responses.items() 
                                      if data.get('keyword_overlap_ratio', 0) > 0.7]
            
            if high_overlap_competitors:
                insights.append(f"High competitive overlap with: {', '.join(high_overlap_competitors)}")
            
            return insights
        except Exception:
            return []

    def _identify_emerging_threats(self, competitor_data, positioning_analysis):
        """Identify emerging competitive threats"""
        try:
            emerging_threats = []
            
            for competitor, data in competitor_data.items():
                # Identify emerging threats based on growth and performance
                if len(data) > 100:  # Significant keyword portfolio
                    avg_traffic = data.get('Traffic (%)', pd.Series()).mean()
                    if avg_traffic > 3:  # Above threshold
                        emerging_threats.append({
                            'competitor': competitor,
                            'threat_type': 'growth',
                            'threat_level': 'medium',
                            'indicators': ['significant_keyword_portfolio', 'above_average_traffic']
                        })
            
            return {
                'emerging_threats': emerging_threats,
                'threat_count': len(emerging_threats),
                'monitoring_priority': 'high' if len(emerging_threats) > 2 else 'medium'
            }
        except Exception as e:
            self.logger.error(f"Error identifying emerging threats: {str(e)}")
            return {'emerging_threats': [], 'threat_count': 0, 'monitoring_priority': 'low'}

    def _create_competitive_risk_matrix(self, threat_scores, disruption_analysis, emerging_threats):
        """Create competitive risk matrix"""
        try:
            risk_matrix = {}
            
            # Combine threat scores with disruption and emerging threat data
            for competitor, threat_score in threat_scores.items():
                disruption_level = 0.5  # Default
                if competitor in disruption_analysis.get('potential_disruptors', []):
                    disruption_level = 0.8
                
                emerging_threat_level = 0.3  # Default
                emerging_competitors = [t['competitor'] for t in emerging_threats.get('emerging_threats', [])]
                if competitor in emerging_competitors:
                    emerging_threat_level = 0.7
                
                overall_risk = (threat_score * 0.4 + disruption_level * 0.3 + emerging_threat_level * 0.3)
                
                risk_matrix[competitor] = {
                    'threat_score': threat_score,
                    'disruption_risk': disruption_level,
                    'emerging_threat_risk': emerging_threat_level,
                    'overall_risk_score': overall_risk,
                    'risk_category': 'high' if overall_risk > 0.7 else 'medium' if overall_risk > 0.4 else 'low'
                }
            
            return risk_matrix
        except Exception as e:
            self.logger.error(f"Error creating risk matrix: {str(e)}")
            return {}

    def _establish_early_warning_indicators(self, prepared_data, positioning_analysis):
        """Establish early warning indicators"""
        try:
            indicators = []
            
            # Traffic-based indicators
            indicators.append({
                'indicator': 'competitor_traffic_surge',
                'threshold': '20% increase in competitor traffic',
                'monitoring_frequency': 'weekly',
                'priority': 'high'
            })
            
            # Position-based indicators
            indicators.append({
                'indicator': 'ranking_position_loss',
                'threshold': 'Loss of top 3 positions for key keywords',
                'monitoring_frequency': 'daily',
                'priority': 'critical'
            })
            
            # Market share indicators
            indicators.append({
                'indicator': 'market_share_decline',
                'threshold': '5% market share decline',
                'monitoring_frequency': 'monthly',
                'priority': 'high'
            })
            
            return {
                'warning_indicators': indicators,
                'monitoring_plan': {
                    'daily_checks': ['ranking_position_loss'],
                    'weekly_checks': ['competitor_traffic_surge'],
                    'monthly_checks': ['market_share_decline']
                }
            }
        except Exception as e:
            self.logger.error(f"Error establishing warning indicators: {str(e)}")
            return {}

    def _develop_threat_mitigation_strategies(self, threat_scores, disruption_analysis, emerging_threats):
        """Develop threat mitigation strategies"""
        try:
            strategies = []
            
            # High threat competitors
            high_threat_competitors = [comp for comp, score in threat_scores.items() if score > 0.7]
            if high_threat_competitors:
                strategies.append({
                    'strategy': 'competitive_monitoring_enhancement',
                    'description': f'Enhanced monitoring of high-threat competitors: {", ".join(high_threat_competitors)}',
                    'timeline': 'immediate',
                    'resources_required': ['analytics', 'competitive_intelligence']
                })
            
            # Disruption risk mitigation
            if disruption_analysis.get('disruption_risk_level') == 'high':
                strategies.append({
                    'strategy': 'market_diversification',
                    'description': 'Diversify market presence to reduce disruption impact',
                    'timeline': 'medium-term',
                    'resources_required': ['product', 'marketing', 'strategy']
                })
            
            # Emerging threat response
            if emerging_threats.get('threat_count', 0) > 2:
                strategies.append({
                    'strategy': 'proactive_competitive_response',
                    'description': 'Develop proactive responses to emerging competitive threats',
                    'timeline': 'short-term',
                    'resources_required': ['strategy', 'product', 'marketing']
                })
            
            return strategies
        except Exception as e:
            self.logger.error(f"Error developing mitigation strategies: {str(e)}")
            return []

    def _create_threat_monitoring_plan(self, warning_indicators, emerging_threats):
        """Create threat monitoring plan"""
        try:
            monitoring_plan = {
                'monitoring_frequency': {
                    'daily': ['SERP position changes', 'competitor traffic spikes'],
                    'weekly': ['competitor content updates', 'new keyword targeting'],
                    'monthly': ['market share analysis', 'competitive landscape review']
                },
                'alert_thresholds': {
                    'critical': 'Immediate action required',
                    'high': 'Action required within 24 hours',
                    'medium': 'Action required within week'
                },
                'escalation_procedures': {
                    'level_1': 'Team notification',
                    'level_2': 'Management notification',
                    'level_3': 'Executive notification'
                },
                'monitoring_tools': ['competitive_intelligence_platform', 'SERP_monitoring', 'traffic_analysis'],
                'review_schedule': 'monthly'
            }
            
            return monitoring_plan
        except Exception as e:
            self.logger.error(f"Error creating monitoring plan: {str(e)}")
            return {}

    def _create_competitive_executive_summary(self, all_analysis):
        """Create executive summary for competitive analysis"""
        try:
            return {
                'analysis_scope': 'comprehensive_competitive_intelligence',
                'competitors_analyzed': len(all_analysis.get('prepared_data', {}).get('competitor_data', {})),
                'market_position': self._assess_market_position(all_analysis),
                'competitive_threats': self._count_competitive_threats(all_analysis),
                'opportunities_identified': self._count_opportunities(all_analysis),
                'strategic_priority': 'high',
                'analysis_timestamp': datetime.now(),
                'key_findings': self._extract_key_findings(all_analysis),
                'recommended_actions': self._extract_recommended_actions(all_analysis)
            }
        except Exception as e:
            self.logger.error(f"Error creating executive summary: {str(e)}")
            return {}

    def _assess_market_position(self, all_analysis):
        """Assess overall market position"""
        try:
            positioning = all_analysis.get('positioning_analysis', {})
            if positioning and 'competitive_rankings' in positioning:
                rankings = positioning['competitive_rankings']
                overall_rank = rankings.get('overall_rank', 5)
                return 'leading' if overall_rank <= 2 else 'competitive' if overall_rank <= 3 else 'challenging'
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
            
            # Alternative counting method
            market_analysis = all_analysis.get('market_analysis', {})
            opportunities = market_analysis.get('market_opportunities', {}).get('opportunities', [])
            return len(opportunities)
        except Exception:
            return 0

    def _extract_key_findings(self, all_analysis):
        """Extract key findings from analysis"""
        try:
            findings = []
            
            # Market position findings
            position = self._assess_market_position(all_analysis)
            findings.append(f"Current market position: {position}")
            
            # Threat findings
            threat_count = self._count_competitive_threats(all_analysis)
            if threat_count > 0:
                findings.append(f"{threat_count} high-priority competitive threats identified")
            
            # Opportunity findings
            opportunity_count = self._count_opportunities(all_analysis)
            if opportunity_count > 0:
                findings.append(f"{opportunity_count} growth opportunities identified")
            
            return findings
        except Exception:
            return []

    def _extract_recommended_actions(self, all_analysis):
        """Extract recommended actions from analysis"""
        try:
            actions = []
            
            # Based on threat level
            threat_count = self._count_competitive_threats(all_analysis)
            if threat_count > 2:
                actions.append("Implement enhanced competitive monitoring")
            
            # Based on opportunities
            opportunity_count = self._count_opportunities(all_analysis)
            if opportunity_count > 10:
                actions.append("Prioritize high-value opportunity capture")
            
            # Based on market position
            position = self._assess_market_position(all_analysis)
            if position == 'challenging':
                actions.append("Develop competitive differentiation strategy")
            
            return actions
        except Exception:
            return []

    def _extract_key_competitive_insights(self, all_analysis):
        """Extract key competitive insights"""
        try:
            insights = []
            
            # Market insights
            market_analysis = all_analysis.get('market_analysis', {})
            if market_analysis:
                market_metrics = market_analysis.get('market_metrics', {})
                market_share = market_metrics.get('primary_market_share', 0)
                insights.append({
                    'category': 'market_share',
                    'insight': f"Current market share: {market_share:.1%}",
                    'impact': 'high'
                })
            
            # Competitive positioning insights
            positioning = all_analysis.get('positioning_analysis', {})
            if positioning:
                strengths_weaknesses = positioning.get('strengths_weaknesses', {})
                strength_score = strengths_weaknesses.get('strength_score', 0)
                insights.append({
                    'category': 'competitive_strength',
                    'insight': f"Competitive strength score: {strength_score:.2f}",
                    'impact': 'medium'
                })
            
            return insights
        except Exception as e:
            self.logger.error(f"Error extracting key insights: {str(e)}")
            return []

    def _analyze_strategic_implications(self, all_analysis):
        """Analyze strategic implications"""
        try:
            implications = []

            # Market analysis implications
            market_analysis = all_analysis.get('market_analysis', {})
            if market_analysis:
                market_opportunities = market_analysis.get('market_opportunities', {})
                if market_opportunities.get('market_potential_score', 0) > 0.7:
                    implications.append({
                        'category': 'market_expansion',
                        'implication': 'High market growth potential identified',
                        'strategic_action': 'Increase market share capture initiatives',
                        'priority': 'high'
                    })

            # Competitive positioning implications
            positioning = all_analysis.get('positioning_analysis', {})
            if positioning:
                strengths_weaknesses = positioning.get('strengths_weaknesses', {})
                if len(strengths_weaknesses.get('weaknesses', [])) > len(strengths_weaknesses.get('strengths', [])):
                    implications.append({
                        'category': 'competitive_positioning',
                        'implication': 'Competitive disadvantages outweigh advantages',
                        'strategic_action': 'Focus on strengthening competitive position',
                        'priority': 'high'
                    })

            # Threat analysis implications
            threat_analysis = all_analysis.get('threat_analysis', {})
            if threat_analysis:
                disruption_analysis = threat_analysis.get('disruption_analysis', {})
                if disruption_analysis.get('disruption_risk_level') == 'high':
                    implications.append({
                        'category': 'risk_management',
                        'implication': 'High market disruption risk detected',
                        'strategic_action': 'Implement risk mitigation strategies',
                        'priority': 'critical'
                    })

            return {
                'strategic_implications': implications,
                'key_themes': [imp['category'] for imp in implications],
                'priority_actions': [imp for imp in implications if imp['priority'] in ['high', 'critical']],
                'strategic_focus_areas': list(set([imp['category'] for imp in implications]))
            }
        except Exception as e:
            self.logger.error(f"Error analyzing strategic implications: {str(e)}")
            return {'strategic_implications': [], 'key_themes': [], 'priority_actions': [], 'strategic_focus_areas': []}

    def _create_intelligence_scorecard(self, all_analysis):
        """Create intelligence scorecard"""
        try:
            scorecard = {}
            
            # Market position score
            position = self._assess_market_position(all_analysis)
            position_score = {'leading': 0.9, 'competitive': 0.7, 'challenging': 0.4}.get(position, 0.5)
            scorecard['market_position_score'] = position_score
            
            # Competitive strength score
            positioning = all_analysis.get('positioning_analysis', {})
            if positioning:
                strength_score = positioning.get('strengths_weaknesses', {}).get('strength_score', 0.5)
                scorecard['competitive_strength_score'] = strength_score
            
            # Opportunity utilization score
            opportunity_count = self._count_opportunities(all_analysis)
            scorecard['opportunity_score'] = min(opportunity_count / 20, 1.0)  # Normalize to max 20 opportunities
            
            # Threat management score
            threat_count = self._count_competitive_threats(all_analysis)
            scorecard['threat_management_score'] = max(0, 1.0 - (threat_count / 5))  # Inverse relationship
            
            # Overall intelligence score
            scores = [scorecard[key] for key in scorecard.keys()]
            scorecard['overall_intelligence_score'] = sum(scores) / len(scores) if scores else 0.5
            
            return scorecard
        except Exception as e:
            self.logger.error(f"Error creating intelligence scorecard: {str(e)}")
            return {}

    def _create_action_priority_matrix(self, all_analysis):
        """Create action priority matrix"""
        try:
            matrix = {
                'high_impact_high_effort': [],
                'high_impact_low_effort': [],
                'low_impact_high_effort': [],
                'low_impact_low_effort': []
            }
            
            # Gap analysis actions
            gap_analysis = all_analysis.get('gap_analysis')
            if gap_analysis and hasattr(gap_analysis, 'priority_keywords') and gap_analysis.priority_keywords:
                matrix['high_impact_low_effort'].append({
                    'action': 'Target high-priority keyword gaps',
                    'description': f'Focus on {len(gap_analysis.priority_keywords)} priority keywords',
                    'estimated_impact': 'high',
                    'estimated_effort': 'low'
                })
            
            # Competitive positioning actions
            positioning = all_analysis.get('positioning_analysis', {})
            if positioning:
                weaknesses = positioning.get('strengths_weaknesses', {}).get('weaknesses', [])
                if len(weaknesses) > 3:
                    matrix['high_impact_high_effort'].append({
                        'action': 'Address competitive weaknesses',
                        'description': f'Address {len(weaknesses)} identified weaknesses',
                        'estimated_impact': 'high',
                        'estimated_effort': 'high'
                    })
            
            return matrix
        except Exception as e:
            self.logger.error(f"Error creating action priority matrix: {str(e)}")
            return {}

    def _create_intelligence_timeline(self, all_analysis):
        """Create intelligence timeline"""
        try:
            timeline = {
                'immediate_actions': [],
                'short_term_actions': [],
                'medium_term_actions': [],
                'long_term_actions': []
            }
            
            # Immediate actions based on threats
            threat_count = self._count_competitive_threats(all_analysis)
            if threat_count > 2:
                timeline['immediate_actions'].append('Implement threat monitoring system')
            
            # Short-term actions based on opportunities
            opportunity_count = self._count_opportunities(all_analysis)
            if opportunity_count > 5:
                timeline['short_term_actions'].append('Capitalize on identified opportunities')
            
            # Medium-term strategic actions
            position = self._assess_market_position(all_analysis)
            if position == 'challenging':
                timeline['medium_term_actions'].append('Develop competitive differentiation strategy')
            
            return timeline
        except Exception as e:
            self.logger.error(f"Error creating intelligence timeline: {str(e)}")
            return {}

    def _assess_integration_quality(self, all_analysis):
        """Assess integration quality of analysis"""
        try:
            quality_factors = []
            
            # Check completeness of analysis components
            expected_components = ['market_analysis', 'positioning_analysis', 'gap_analysis', 'trend_analysis', 'threat_analysis']
            completed_components = [comp for comp in expected_components if comp in all_analysis and all_analysis[comp]]
            
            completeness_score = len(completed_components) / len(expected_components)
            quality_factors.append(completeness_score)
            
            # Check data quality
            prepared_data = all_analysis.get('prepared_data', {})
            if prepared_data:
                data_quality_scores = prepared_data.get('competitive_summary', {}).get('data_quality_scores', {})
                if data_quality_scores:
                    avg_quality = sum(data_quality_scores.values()) / len(data_quality_scores)
                    quality_factors.append(avg_quality)
            
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
        except Exception:
            return 0.5

    def _calculate_intelligence_confidence(self, all_analysis):
        """Calculate intelligence confidence level"""
        try:
            confidence_factors = []
            
            # Data coverage confidence
            prepared_data = all_analysis.get('prepared_data', {})
            if prepared_data:
                coverage_overlap = prepared_data.get('analysis_metadata', {}).get('data_coverage_overlap', 0)
                confidence_factors.append(coverage_overlap)
            
            # Analysis depth confidence
            analysis_components = len([comp for comp in all_analysis.keys() if all_analysis[comp]])
            depth_confidence = min(analysis_components / 6, 1.0)  # Expect 6 main components
            confidence_factors.append(depth_confidence)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        except Exception:
            return 0.5

    def _generate_short_term_recommendations(self, intelligence_synthesis):
        """Generate short-term recommendations"""
        try:
            recommendations = []
            
            # Quick wins from gap analysis
            gap_analysis = intelligence_synthesis.get('detailed_analysis', {}).get('gap_analysis')
            if gap_analysis and hasattr(gap_analysis, 'priority_keywords') and gap_analysis.priority_keywords:
                recommendations.append({
                    'recommendation': 'Target high-priority keyword gaps',
                    'category': 'content',
                    'timeframe': '0-3 months',
                    'priority': 'high',
                    'resources_required': ['content', 'seo'],
                    'expected_outcome': 'Improved keyword rankings and traffic'
                })
            
            # Immediate threat responses
            threat_analysis = intelligence_synthesis.get('detailed_analysis', {}).get('threat_analysis', {})
            if threat_analysis:
                high_threats = [comp for comp, score in threat_analysis.get('threat_scores', {}).items() if score > 0.7]
                if high_threats:
                    recommendations.append({
                        'recommendation': 'Implement competitive monitoring for high-threat competitors',
                        'category': 'intelligence',
                        'timeframe': '0-1 month',
                        'priority': 'critical',
                        'resources_required': ['analytics', 'tools'],
                        'expected_outcome': 'Enhanced competitive awareness'
                    })
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating short-term recommendations: {str(e)}")
            return []

    def _generate_medium_term_recommendations(self, intelligence_synthesis):
        """Generate medium-term recommendations"""
        try:
            recommendations = []
            
            # Extract key insights for medium-term planning
            strategic_implications = intelligence_synthesis.get('strategic_implications', {})
            priority_actions = strategic_implications.get('priority_actions', [])

            for action in priority_actions:
                if action.get('category') == 'market_expansion':
                    recommendations.append({
                        'recommendation': 'Develop comprehensive market expansion strategy',
                        'category': 'growth',
                        'timeframe': '3-6 months',
                        'priority': 'high',
                        'resources_required': ['marketing', 'content', 'seo'],
                        'expected_outcome': 'Increased market share and traffic growth'
                    })

                elif action.get('category') == 'competitive_positioning':
                    recommendations.append({
                        'recommendation': 'Strengthen competitive positioning through differentiation',
                        'category': 'competitive',
                        'timeframe': '6-9 months',
                        'priority': 'medium',
                        'resources_required': ['product', 'marketing', 'content'],
                        'expected_outcome': 'Improved competitive advantages'
                    })

            # Add default medium-term recommendations if none from strategic implications
            if not recommendations:
                recommendations = [
                    {
                        'recommendation': 'Develop competitive intelligence monitoring system',
                        'category': 'intelligence',
                        'timeframe': '3-6 months',
                        'priority': 'medium',
                        'resources_required': ['analytics', 'tools'],
                        'expected_outcome': 'Better competitive awareness'
                    },
                    {
                        'recommendation': 'Implement content gap closure strategy',
                        'category': 'content',
                        'timeframe': '6-12 months',
                        'priority': 'medium',
                        'resources_required': ['content', 'seo'],
                        'expected_outcome': 'Reduced competitive content gaps'
                    }
                ]

            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating medium-term recommendations: {str(e)}")
            return []

    def _generate_long_term_recommendations(self, intelligence_synthesis):
        """Generate long-term recommendations"""
        try:
            recommendations = []
            
            # Strategic positioning recommendations
            market_position = intelligence_synthesis.get('executive_summary', {}).get('market_position', 'unknown')
            if market_position == 'challenging':
                recommendations.append({
                    'recommendation': 'Develop comprehensive market leadership strategy',
                    'category': 'strategic',
                    'timeframe': '12-24 months',
                    'priority': 'high',
                    'resources_required': ['strategy', 'product', 'marketing', 'technology'],
                    'expected_outcome': 'Market leadership position'
                })
            
            # Innovation and differentiation
            recommendations.append({
                'recommendation': 'Invest in innovative competitive advantages',
                'category': 'innovation',
                'timeframe': '18-36 months',
                'priority': 'medium',
                'resources_required': ['r&d', 'product', 'technology'],
                'expected_outcome': 'Sustainable competitive differentiation'
            })
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating long-term recommendations: {str(e)}")
            return []

    def _identify_quick_wins(self, intelligence_synthesis):
        """Identify quick wins"""
        try:
            quick_wins = []
            
            # Gap analysis quick wins
            gap_analysis = intelligence_synthesis.get('detailed_analysis', {}).get('gap_analysis')
            if gap_analysis and hasattr(gap_analysis, 'keyword_gaps') and not gap_analysis.keyword_gaps.empty:
                low_difficulty_gaps = gap_analysis.keyword_gaps.head(5)  # Top 5 easiest gaps
                quick_wins.append({
                    'opportunity': 'Low-competition keyword targeting',
                    'description': f'Target {len(low_difficulty_gaps)} low-competition keywords',
                    'effort': 'low',
                    'impact': 'medium',
                    'timeline': '2-4 weeks'
                })
            
            # Content optimization quick wins
            positioning = intelligence_synthesis.get('detailed_analysis', {}).get('positioning_analysis', {})
            if positioning:
                weaknesses = positioning.get('strengths_weaknesses', {}).get('improvement_areas', [])
                content_weaknesses = [w for w in weaknesses if 'content' in w.lower()]
                if content_weaknesses:
                    quick_wins.append({
                        'opportunity': 'Content optimization',
                        'description': 'Optimize existing content for better performance',
                        'effort': 'low',
                        'impact': 'medium',
                        'timeline': '1-2 weeks'
                    })
            
            return quick_wins
        except Exception as e:
            self.logger.error(f"Error identifying quick wins: {str(e)}")
            return []

    def _recommend_resource_allocation(self, intelligence_synthesis):
        """Recommend resource allocation"""
        try:
            allocation = {}
            
            # Based on strategic priorities
            strategic_implications = intelligence_synthesis.get('strategic_implications', {})
            priority_actions = strategic_implications.get('priority_actions', [])
            
            if any(action.get('priority') == 'critical' for action in priority_actions):
                allocation['competitive_intelligence'] = {
                    'percentage': 30,
                    'focus_areas': ['threat_monitoring', 'competitive_analysis'],
                    'justification': 'Critical competitive threats identified'
                }
            
            # Based on opportunities
            opportunity_count = self._count_opportunities(intelligence_synthesis.get('detailed_analysis', {}))
            if opportunity_count > 10:
                allocation['opportunity_capture'] = {
                    'percentage': 40,
                    'focus_areas': ['content_development', 'keyword_targeting'],
                    'justification': f'{opportunity_count} significant opportunities identified'
                }
            
            # Default allocation
            if not allocation:
                allocation = {
                    'competitive_monitoring': {'percentage': 25, 'focus_areas': ['monitoring', 'analysis']},
                    'content_optimization': {'percentage': 35, 'focus_areas': ['content', 'seo']},
                    'strategic_development': {'percentage': 25, 'focus_areas': ['strategy', 'planning']},
                    'innovation': {'percentage': 15, 'focus_areas': ['r&d', 'innovation']}
                }
            
            return allocation
        except Exception as e:
            self.logger.error(f"Error recommending resource allocation: {str(e)}")
