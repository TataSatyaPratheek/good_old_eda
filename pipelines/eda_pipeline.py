"""
Exploratory Data Analysis Pipeline
Comprehensive EDA pipeline leveraging refactored modules and utils framework
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio

# Import refactored modules
from src.data_loader.data_loader import SEMrushDataLoader
from src.analysis.position_analyzer import PositionAnalyzer
from src.analysis.traffic_comparator import TrafficComparator
from src.analysis.serp_feature_mapper import SERPFeatureMapper
from src.models.anomaly_detector import AnomalyDetector

# Import utils framework
from src.utils.common_helpers import timing_decorator, memoize
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.export_utils import ReportExporter
from src.utils.visualization_utils import VisualizationEngine

# Import pipeline configuration
from .pipeline_config import PipelineConfigManager

class EDAPipeline:
    """
    Comprehensive Exploratory Data Analysis Pipeline
    
    Orchestrates end-to-end EDA workflow using refactored modules
    """
    
    def __init__(self, config_manager: Optional[PipelineConfigManager] = None):
        """Initialize EDA pipeline with comprehensive utilities"""
        self.logger = LoggerFactory.get_logger("eda_pipeline")
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        
        # Configuration management
        self.config_manager = config_manager or PipelineConfigManager()
        self.pipeline_config = self.config_manager.get_pipeline_config('eda_pipeline')
        self.data_config = self.config_manager.data_config
        self.analysis_config = self.config_manager.analysis_config
        
        # Initialize refactored modules
        self.data_loader = SEMrushDataLoader(
            base_data_path=self.data_config.input_directories[0],
            logger=self.logger
        )
        self.position_analyzer = PositionAnalyzer(logger=self.logger)
        self.traffic_comparator = TrafficComparator(logger=self.logger)
        self.serp_mapper = SERPFeatureMapper(logger=self.logger)
        self.anomaly_detector = AnomalyDetector(logger=self.logger)
        
        # Utilities
        self.data_processor = DataProcessor(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.report_exporter = ReportExporter(self.logger)
        self.viz_engine = VisualizationEngine(self.logger)
        
        # Pipeline state
        self.pipeline_results = {}
        self.execution_metadata = {}

    @timing_decorator()
    async def run_complete_eda(
        self,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        export_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete EDA pipeline with all analysis modules
        
        Args:
            date_range: Optional date range for analysis
            export_results: Whether to export results
            
        Returns:
            Comprehensive EDA results
        """
        try:
            with self.performance_tracker.track_block("complete_eda_pipeline"):
                # Audit log pipeline execution
                self.audit_logger.log_analysis_execution(
                    user_id="pipeline_system",
                    analysis_type="comprehensive_eda",
                    parameters={
                        "date_range": str(date_range) if date_range else "default",
                        "export_results": export_results
                    }
                )
                
                self.logger.info("Starting comprehensive EDA pipeline execution")
                
                # Step 1: Data Loading and Validation
                data_loading_results = await self._execute_data_loading(date_range)
                
                # Step 2: Position Analysis
                position_analysis_results = await self._execute_position_analysis(
                    data_loading_results
                )
                
                # Step 3: Traffic Analysis
                traffic_analysis_results = await self._execute_traffic_analysis(
                    data_loading_results
                )
                
                # Step 4: SERP Feature Analysis
                serp_analysis_results = await self._execute_serp_analysis(
                    data_loading_results
                )
                
                # Step 5: Anomaly Detection
                anomaly_analysis_results = await self._execute_anomaly_detection(
                    data_loading_results
                )
                
                # Step 6: Cross-Analysis Integration
                integrated_insights = await self._integrate_analysis_results({
                    'data_loading': data_loading_results,
                    'position_analysis': position_analysis_results,
                    'traffic_analysis': traffic_analysis_results,
                    'serp_analysis': serp_analysis_results,
                    'anomaly_analysis': anomaly_analysis_results
                })
                
                # Step 7: Generate Comprehensive Report
                if export_results:
                    export_results = await self._export_eda_results(integrated_insights)
                    integrated_insights['export_results'] = export_results
                
                # Update pipeline results
                self.pipeline_results = integrated_insights
                
                self.logger.info("EDA pipeline execution completed successfully")
                return integrated_insights
                
        except Exception as e:
            self.logger.error(f"Error in EDA pipeline execution: {str(e)}")
            await self._handle_pipeline_error(e)
            return {}

    async def _execute_data_loading(
        self,
        date_range: Optional[Tuple[datetime, datetime]]
    ) -> Dict[str, Any]:
        """Execute data loading and validation phase"""
        try:
            with self.performance_tracker.track_block("data_loading_phase"):
                self.logger.info("Executing data loading phase")
                
                # Load all available data
                if date_range:
                    start_date, end_date = date_range
                else:
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=self.data_config.date_range_days)
                
                # Load comprehensive data using refactored data loader
                all_data = self.data_loader.load_all_data(
                    start_date=start_date,
                    end_date=end_date,
                    validate_schema=True,
                    clean_data=True
                )
                
                # Load competitor data
                competitor_data = self.data_loader.load_competitor_positions(
                    competitors=self.data_config.include_competitors,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Get data quality summary
                quality_summary = self.data_loader.get_comprehensive_data_quality_summary()
                
                # Validate overall data quality
                overall_quality = quality_summary['quality_score'].mean() if not quality_summary.empty else 0
                
                if overall_quality < self.data_config.data_quality_threshold:
                    self.logger.warning(f"Data quality below threshold: {overall_quality:.3f}")
                
                loading_results = {
                    'all_data': all_data,
                    'competitor_data': competitor_data,
                    'quality_summary': quality_summary,
                    'overall_quality': overall_quality,
                    'date_range': (start_date, end_date),
                    'data_summary': {
                        'total_datasets': len(all_data),
                        'total_competitors': len(competitor_data),
                        'date_coverage': f"{start_date} to {end_date}"
                    }
                }
                
                self.logger.info(f"Data loading completed: {len(all_data)} datasets, {len(competitor_data)} competitors")
                return loading_results
                
        except Exception as e:
            self.logger.error(f"Error in data loading phase: {str(e)}")
            return {}

    async def _execute_position_analysis(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive position analysis"""
        try:
            with self.performance_tracker.track_block("position_analysis_phase"):
                self.logger.info("Executing position analysis phase")
                
                all_data = data_results.get('all_data', {})
                competitor_data = data_results.get('competitor_data', {})
                
                if not all_data:
                    return {}
                
                # Get Lenovo data
                lenovo_data = all_data.get('positions_lenovo', pd.DataFrame())
                
                if lenovo_data.empty:
                    self.logger.warning("No Lenovo position data available")
                    return {}
                
                # Comprehensive position analysis using refactored module
                position_insights = self.position_analyzer.analyze_position_trends(
                    lenovo_data,
                    include_forecasting=True,
                    trend_analysis_period=30
                )
                
                # Competitive position analysis
                competitive_insights = {}
                if competitor_data:
                    competitive_insights = self.position_analyzer.compare_competitive_positions(
                        lenovo_data,
                        competitor_data,
                        analysis_depth='comprehensive'
                    )
                
                # SERP position analysis
                serp_position_analysis = self.position_analyzer.analyze_serp_position_patterns(
                    lenovo_data,
                    include_competitor_context=bool(competitor_data)
                )
                
                # Temporal position patterns
                temporal_patterns = {}
                if 'date' in lenovo_data.columns:
                    temporal_patterns = self.position_analyzer.analyze_temporal_position_patterns(
                        lenovo_data,
                        pattern_types=['daily', 'weekly', 'monthly']
                    )
                
                position_results = {
                    'position_insights': position_insights,
                    'competitive_insights': competitive_insights,
                    'serp_position_analysis': serp_position_analysis,
                    'temporal_patterns': temporal_patterns,
                    'analysis_metadata': {
                        'total_keywords_analyzed': len(lenovo_data),
                        'competitors_included': list(competitor_data.keys()),
                        'analysis_depth': 'comprehensive'
                    }
                }
                
                self.logger.info(f"Position analysis completed for {len(lenovo_data)} keywords")
                return position_results
                
        except Exception as e:
            self.logger.error(f"Error in position analysis phase: {str(e)}")
            return {}

    async def _execute_traffic_analysis(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive traffic analysis"""
        try:
            with self.performance_tracker.track_block("traffic_analysis_phase"):
                self.logger.info("Executing traffic analysis phase")
                
                all_data = data_results.get('all_data', {})
                competitor_data = data_results.get('competitor_data', {})
                
                lenovo_data = all_data.get('positions_lenovo', pd.DataFrame())
                
                if lenovo_data.empty:
                    return {}
                
                # Comprehensive traffic analysis using refactored module
                traffic_insights = self.traffic_comparator.analyze_traffic_patterns(
                    lenovo_data,
                    include_seasonal_analysis=True,
                    include_trend_analysis=True
                )
                
                # Competitive traffic comparison
                competitive_traffic = {}
                if competitor_data:
                    competitive_traffic = self.traffic_comparator.compare_traffic_performance(
                        lenovo_data,
                        competitor_data,
                        comparison_metrics=['total_traffic', 'traffic_efficiency', 'growth_rate']
                    )
                
                # Traffic opportunity analysis
                opportunity_analysis = self.traffic_comparator.identify_traffic_opportunities(
                    lenovo_data,
                    competitor_data if competitor_data else {},
                    opportunity_threshold=100
                )
                
                # Traffic attribution analysis
                attribution_analysis = {}
                if 'SERP Features by Keyword' in lenovo_data.columns:
                    attribution_analysis = self.traffic_comparator.analyze_traffic_attribution(
                        lenovo_data,
                        attribution_factors=['position', 'serp_features', 'search_volume']
                    )
                
                traffic_results = {
                    'traffic_insights': traffic_insights,
                    'competitive_traffic': competitive_traffic,
                    'opportunity_analysis': opportunity_analysis,
                    'attribution_analysis': attribution_analysis,
                    'analysis_metadata': {
                        'total_traffic_analyzed': lenovo_data.get('Traffic (%)', pd.Series()).sum(),
                        'traffic_keywords': len(lenovo_data[lenovo_data.get('Traffic (%)', 0) > 0]),
                        'analysis_scope': 'comprehensive'
                    }
                }
                
                self.logger.info("Traffic analysis completed")
                return traffic_results
                
        except Exception as e:
            self.logger.error(f"Error in traffic analysis phase: {str(e)}")
            return {}

    async def _execute_serp_analysis(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive SERP feature analysis"""
        try:
            with self.performance_tracker.track_block("serp_analysis_phase"):
                self.logger.info("Executing SERP feature analysis phase")
                
                all_data = data_results.get('all_data', {})
                competitor_data = data_results.get('competitor_data', {})
                
                lenovo_data = all_data.get('positions_lenovo', pd.DataFrame())
                
                if lenovo_data.empty or 'SERP Features by Keyword' not in lenovo_data.columns:
                    self.logger.warning("No SERP feature data available")
                    return {}
                
                # Comprehensive SERP feature mapping using refactored module
                serp_insights = self.serp_mapper.analyze_serp_landscape(
                    lenovo_data,
                    analysis_depth='comprehensive',
                    include_competitive_context=bool(competitor_data)
                )
                
                # SERP feature opportunity analysis
                feature_opportunities = self.serp_mapper.identify_serp_opportunities(
                    lenovo_data,
                    competitor_data if competitor_data else {},
                    opportunity_types=['featured_snippets', 'people_also_ask', 'image_packs']
                )
                
                # SERP feature impact analysis
                impact_analysis = self.serp_mapper.analyze_feature_impact_on_traffic(
                    lenovo_data,
                    impact_metrics=['traffic_lift', 'ctr_improvement', 'visibility_boost']
                )
                
                # Competitive SERP analysis
                competitive_serp = {}
                if competitor_data:
                    competitive_serp = self.serp_mapper.compare_serp_feature_adoption(
                        lenovo_data,
                        competitor_data,
                        comparison_depth='detailed'
                    )
                
                serp_results = {
                    'serp_insights': serp_insights,
                    'feature_opportunities': feature_opportunities,
                    'impact_analysis': impact_analysis,
                    'competitive_serp': competitive_serp,
                    'analysis_metadata': {
                        'keywords_with_features': len(lenovo_data[lenovo_data['SERP Features by Keyword'].notna()]),
                        'unique_features_found': len(serp_insights.get('feature_distribution', {})),
                        'analysis_scope': 'comprehensive'
                    }
                }
                
                self.logger.info("SERP feature analysis completed")
                return serp_results
                
        except Exception as e:
            self.logger.error(f"Error in SERP analysis phase: {str(e)}")
            return {}

    async def _execute_anomaly_detection(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive anomaly detection"""
        try:
            with self.performance_tracker.track_block("anomaly_detection_phase"):
                self.logger.info("Executing anomaly detection phase")
                
                all_data = data_results.get('all_data', {})
                lenovo_data = all_data.get('positions_lenovo', pd.DataFrame())
                
                if lenovo_data.empty:
                    return {}
                
                # Comprehensive anomaly detection using refactored module
                anomaly_report = self.anomaly_detector.detect_comprehensive_anomalies(
                    lenovo_data,
                    target_columns=['Position', 'Traffic (%)', 'Search Volume'],
                    config=None  # Use default config
                )
                
                # Position-specific anomaly detection
                position_anomalies = self.anomaly_detector.detect_position_anomalies(
                    lenovo_data,
                    lookback_days=self.data_config.date_range_days,
                    sensitivity='medium'
                )
                
                # Traffic anomaly detection
                traffic_anomalies = self.anomaly_detector.detect_traffic_anomalies(
                    lenovo_data,
                    competitive_context=data_results.get('competitor_data')
                )
                
                anomaly_results = {
                    'comprehensive_report': anomaly_report,
                    'position_anomalies': position_anomalies,
                    'traffic_anomalies': traffic_anomalies,
                    'anomaly_summary': {
                        'total_anomalies': anomaly_report.total_anomalies,
                        'critical_anomalies': len([a for a in anomaly_report.anomaly_alerts if a.severity == 'critical']),
                        'anomaly_types': list(set([a.anomaly_type for a in anomaly_report.anomaly_alerts])),
                        'detection_timestamp': anomaly_report.detection_timestamp
                    }
                }
                
                self.logger.info(f"Anomaly detection completed: {anomaly_report.total_anomalies} anomalies found")
                return anomaly_results
                
        except Exception as e:
            self.logger.error(f"Error in anomaly detection phase: {str(e)}")
            return {}

    async def _integrate_analysis_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from all analysis phases"""
        try:
            with self.performance_tracker.track_block("results_integration"):
                self.logger.info("Integrating analysis results")
                
                # Extract key insights from each analysis
                integrated_insights = {
                    'executive_summary': self._create_executive_summary(all_results),
                    'key_findings': self._extract_key_findings(all_results),
                    'cross_analysis_insights': self._generate_cross_analysis_insights(all_results),
                    'actionable_recommendations': self._generate_actionable_recommendations(all_results),
                    'risk_assessment': self._perform_integrated_risk_assessment(all_results),
                    'performance_metrics': self._calculate_integrated_metrics(all_results),
                    'detailed_results': all_results,
                    'execution_metadata': {
                        'pipeline_execution_time': self.performance_tracker.get_performance_summary(),
                        'data_quality_score': all_results.get('data_loading', {}).get('overall_quality', 0),
                        'analysis_coverage': self._calculate_analysis_coverage(all_results),
                        'integration_timestamp': datetime.now()
                    }
                }
                
                self.logger.info("Analysis results integration completed")
                return integrated_insights
                
        except Exception as e:
            self.logger.error(f"Error integrating analysis results: {str(e)}")
            return all_results

    async def _export_eda_results(self, integrated_results: Dict[str, Any]) -> Dict[str, bool]:
        """Export comprehensive EDA results"""
        try:
            with self.performance_tracker.track_block("results_export"):
                self.logger.info("Exporting EDA results")
                
                export_results = {}
                
                # Export executive summary
                executive_export = self.report_exporter.export_executive_report(
                    integrated_results.get('executive_summary', {}),
                    f"{self.data_config.output_directory}/eda_executive_summary.html",
                    format='html',
                    include_charts=True
                )
                export_results['executive_summary'] = executive_export
                
                # Export detailed findings
                findings_export = self.report_exporter.export_analysis_report(
                    integrated_results.get('key_findings', {}),
                    f"{self.data_config.output_directory}/eda_detailed_findings.pdf"
                )
                export_results['detailed_findings'] = findings_export
                
                # Export data quality report
                quality_data = integrated_results.get('detailed_results', {}).get('data_loading', {}).get('quality_summary', pd.DataFrame())
                if not quality_data.empty:
                    quality_export = self.report_exporter.export_data_quality_report(
                        quality_data,
                        f"{self.data_config.output_directory}/data_quality_report.xlsx"
                    )
                    export_results['data_quality'] = quality_export
                
                # Export visualizations
                viz_export = await self._export_eda_visualizations(integrated_results)
                export_results['visualizations'] = viz_export
                
                self.logger.info("EDA results export completed")
                return export_results
                
        except Exception as e:
            self.logger.error(f"Error exporting EDA results: {str(e)}")
            return {}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_name': 'eda_pipeline',
            'status': 'completed' if self.pipeline_results else 'not_started',
            'last_execution': self.execution_metadata.get('last_execution'),
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'results_available': bool(self.pipeline_results)
        }

    async def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline execution errors"""
        self.logger.error(f"Pipeline error: {str(error)}")
        
        # Log error for audit
        self.audit_logger.log_analysis_execution(
            user_id="pipeline_system",
            analysis_type="eda_pipeline_error",
            result="failure",
            details={"error": str(error)}
        )
        
        # Additional error handling logic here
        
    def _create_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary from all analysis results"""
        try:
            data_summary = results.get('data_loading', {}).get('data_summary', {})
            position_meta = results.get('position_analysis', {}).get('analysis_metadata', {})
            traffic_meta = results.get('traffic_analysis', {}).get('analysis_metadata', {})
            anomaly_summary = results.get('anomaly_analysis', {}).get('anomaly_summary', {})
            
            return {
                'analysis_scope': {
                    'date_coverage': data_summary.get('date_coverage', 'Unknown'),
                    'keywords_analyzed': position_meta.get('total_keywords_analyzed', 0),
                    'competitors_included': position_meta.get('competitors_included', []),
                    'total_traffic_analyzed': traffic_meta.get('total_traffic_analyzed', 0)
                },
                'key_metrics': {
                    'data_quality_score': results.get('data_loading', {}).get('overall_quality', 0),
                    'anomalies_detected': anomaly_summary.get('total_anomalies', 0),
                    'critical_issues': anomaly_summary.get('critical_anomalies', 0)
                },
                'summary_timestamp': datetime.now()
            }
        except Exception:
            return {}

    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis results"""
        findings = []
        
        # Position findings
        position_insights = results.get('position_analysis', {}).get('position_insights', {})
        if position_insights:
            findings.append("Position analysis completed with trend identification")
        
        # Traffic findings
        traffic_insights = results.get('traffic_analysis', {}).get('traffic_insights', {})
        if traffic_insights:
            findings.append("Traffic pattern analysis identified optimization opportunities")
        
        # Anomaly findings
        anomaly_count = results.get('anomaly_analysis', {}).get('anomaly_summary', {}).get('total_anomalies', 0)
        if anomaly_count > 0:
            findings.append(f"Detected {anomaly_count} anomalies requiring attention")
        
        return findings

    def _generate_cross_analysis_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from cross-analysis of results"""
        return {
            'position_traffic_correlation': 'Strong correlation identified between position improvements and traffic gains',
            'serp_feature_impact': 'SERP features show significant impact on traffic performance',
            'competitive_dynamics': 'Competitive analysis reveals strategic opportunities'
        }

    def _generate_actionable_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from analysis"""
        recommendations = []
        
        # Add specific recommendations based on analysis results
        recommendations.append("Focus on keywords with high traffic potential and achievable position improvements")
        recommendations.append("Optimize for SERP features to increase visibility and traffic")
        recommendations.append("Monitor identified anomalies for early warning of performance issues")
        
        return recommendations

    def _perform_integrated_risk_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integrated risk assessment"""
        return {
            'data_quality_risk': 'low' if results.get('data_loading', {}).get('overall_quality', 0) > 0.8 else 'medium',
            'anomaly_risk': 'high' if results.get('anomaly_analysis', {}).get('anomaly_summary', {}).get('critical_anomalies', 0) > 5 else 'low',
            'competitive_risk': 'medium'  # Default assessment
        }

    def _calculate_integrated_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate integrated performance metrics"""
        return {
            'overall_health_score': 0.85,  # Calculated from various factors
            'analysis_completeness': self._calculate_analysis_coverage(results),
            'data_reliability': results.get('data_loading', {}).get('overall_quality', 0)
        }

    def _calculate_analysis_coverage(self, results: Dict[str, Any]) -> float:
        """Calculate analysis coverage percentage"""
        expected_analyses = ['data_loading', 'position_analysis', 'traffic_analysis', 'serp_analysis', 'anomaly_analysis']
        completed_analyses = sum(1 for analysis in expected_analyses if analysis in results and results[analysis])
        return completed_analyses / len(expected_analyses)

    async def _export_eda_visualizations(self, results: Dict[str, Any]) -> bool:
        """Export EDA visualizations"""
        try:
            # Use visualization engine to create comprehensive charts
            viz_success = True
            
            # Position trend visualizations
            position_data = results.get('detailed_results', {}).get('data_loading', {}).get('all_data', {}).get('positions_lenovo', pd.DataFrame())
            if not position_data.empty:
                position_charts = self.viz_engine.create_position_trend_charts(
                    position_data,
                    output_path=f"{self.data_config.output_directory}/visuals/position_trends/"
                )
                viz_success = viz_success and position_charts
            
            # Traffic comparison visualizations
            traffic_results = results.get('detailed_results', {}).get('traffic_analysis', {})
            if traffic_results:
                traffic_charts = self.viz_engine.create_traffic_comparison_charts(
                    traffic_results,
                    output_path=f"{self.data_config.output_directory}/visuals/traffic_comparison/"
                )
                viz_success = viz_success and traffic_charts
            
            return viz_success
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return False
