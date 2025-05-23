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
                        'primary_keywords_analyzed': len(lenovo_data),
                        'competitors_analyzed': len(competitor_data),
                        'analysis_timeframe': data_results.get('date_range'),
                        'analysis_timestamp': datetime.now()
                    }
                }
                
                self.logger.info("Position analysis phase completed")
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
                
                if not all_data:
                    return {}
                
                # Get Lenovo data
                lenovo_data = all_data.get('positions_lenovo', pd.DataFrame())
                
                if lenovo_data.empty:
                    self.logger.warning("No Lenovo traffic data available")
                    return {}
                
                # Traffic performance analysis
                traffic_analysis = self.traffic_comparator.analyze_traffic_performance(
                    lenovo_data,
                    analysis_period=30,
                    include_forecasting=True
                )
                
                # Competitive traffic comparison
                competitive_traffic = {}
                if competitor_data:
                    competitive_traffic = self.traffic_comparator.compare_traffic_performance(
                        lenovo_data,
                        competitor_data,
                    )
                
                traffic_results = {
                    'traffic_analysis': traffic_analysis,
                    'competitive_traffic': competitive_traffic,
                    'traffic_summary': self._create_traffic_summary(traffic_analysis, competitive_traffic)
                }
                
                self.logger.info("Traffic analysis phase completed")
                return traffic_results
                
        except Exception as e:
            self.logger.error(f"Error in traffic analysis phase: {str(e)}")
            return {}

    async def _execute_serp_analysis(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SERP feature analysis"""
        try:
            with self.performance_tracker.track_block("serp_analysis_phase"):
                self.logger.info("Executing SERP analysis phase")
                
                all_data = data_results.get('all_data', {})
                
                if not all_data:
                    return {}
                
                # Get Lenovo data
                lenovo_data = all_data.get('positions_lenovo', pd.DataFrame())
                
                if lenovo_data.empty or 'SERP Features by Keyword' not in lenovo_data.columns:
                    self.logger.warning("No SERP features data available")
                    return {}
                
                # SERP feature mapping and analysis
                serp_analysis = self.serp_mapper.map_serp_features(
                    lenovo_data,
                    feature_analysis=True,
                    competitive_context=True
                )
                
                # SERP opportunity analysis
                serp_opportunities = self.serp_mapper.identify_serp_opportunities(
                    lenovo_data,
                    opportunity_threshold=0.6
                )
                
                serp_results = {
                    'serp_analysis': serp_analysis,
                    'serp_opportunities': serp_opportunities,
                    'serp_summary': self._create_serp_summary(serp_analysis, serp_opportunities)
                }
                
                self.logger.info("SERP analysis phase completed")
                return serp_results
                
        except Exception as e:
            self.logger.error(f"Error in SERP analysis phase: {str(e)}")
            return {}

    async def _execute_anomaly_detection(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute anomaly detection analysis"""
        try:
            with self.performance_tracker.track_block("anomaly_detection_phase"):
                self.logger.info("Executing anomaly detection phase")
                
                all_data = data_results.get('all_data', {})
                
                if not all_data:
                    return {}
                
                # Get Lenovo data
                lenovo_data = all_data.get('positions_lenovo', pd.DataFrame())
                
                if lenovo_data.empty:
                    self.logger.warning("No data available for anomaly detection")
                    return {}
                
                # Detect anomalies in position data
                position_anomalies = self.anomaly_detector.detect_position_anomalies(
                    lenovo_data,
                    detection_methods=['statistical', 'isolation_forest']
                )
                
                # Detect traffic anomalies
                traffic_anomalies = self.anomaly_detector.detect_traffic_anomalies(
                    lenovo_data,
                    sensitivity_threshold=0.3
                )
                
                anomaly_results = {
                    'position_anomalies': position_anomalies,
                    'traffic_anomalies': traffic_anomalies,
                    'anomaly_summary': self._create_anomaly_summary(position_anomalies, traffic_anomalies)
                }
                
                self.logger.info("Anomaly detection phase completed")
                return anomaly_results
                
        except Exception as e:
            self.logger.error(f"Error in anomaly detection phase: {str(e)}")
            return {}

    async def _integrate_analysis_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate all analysis results"""
        try:
            with self.performance_tracker.track_block("analysis_integration"):
                self.logger.info("Integrating analysis results")
                
                # Create integrated insights
                integrated_insights = {
                    'executive_summary': self._create_executive_summary(all_results),
                    'key_findings': self._extract_key_findings(all_results),
                    'data_quality_assessment': self._assess_overall_data_quality(all_results),
                    'performance_metrics': self.performance_tracker.get_performance_summary(),
                    'detailed_results': all_results,
                    'integration_metadata': {
                        'integration_timestamp': datetime.now(),
                        'analysis_components': list(all_results.keys()),
                        'integration_quality': self._assess_integration_quality(all_results)
                    }
                }
                
                self.logger.info("Analysis integration completed")
                return integrated_insights
                
        except Exception as e:
            self.logger.error(f"Error in analysis integration: {str(e)}")
            return all_results

    async def _export_eda_results(self, integrated_results: Dict[str, Any]) -> Dict[str, bool]:
        """Export EDA results"""
        try:
            with self.performance_tracker.track_block("eda_results_export"):
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
                    f"{self.data_config.output_directory}/eda_detailed_findings.json"
                )
                export_results['detailed_findings'] = findings_export
                
                # Create visualizations
                viz_export = await self._create_eda_visualizations(integrated_results)
                export_results['visualizations'] = viz_export
                
                self.logger.info("EDA results export completed")
                return export_results
                
        except Exception as e:
            self.logger.error(f"Error exporting EDA results: {str(e)}")
            return {}

    async def _create_eda_visualizations(self, integrated_insights: Dict[str, Any]) -> bool:
        """Create EDA visualizations"""
        try:
            # Create position trend charts
            position_data = integrated_insights.get('detailed_results', {}).get('data_loading', {}).get('all_data', {}).get('positions_lenovo', pd.DataFrame())
            
            if not position_data.empty:
                position_viz = self.viz_engine.create_position_trend_charts(
                    position_data,
                    f"{self.data_config.output_directory}/visuals/positional_trends/"
                )
                
                # Create traffic comparison charts
                traffic_viz = self.viz_engine.create_traffic_comparison_charts(
                    integrated_insights.get('detailed_results', {}).get('traffic_analysis', {}),
                    f"{self.data_config.output_directory}/visuals/traffic_comparison/"
                )
                
                return position_viz and traffic_viz
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error creating EDA visualizations: {str(e)}")
            return False

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_name': 'eda_pipeline',
            'status': 'completed' if self.pipeline_results else 'not_started',
            'results_available': bool(self.pipeline_results),
            'performance_summary': self.performance_tracker.get_performance_summary()
        }

    async def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline execution errors"""
        self.logger.error(f"EDA pipeline error: {str(error)}")
        self.audit_logger.log_analysis_execution(
            user_id="pipeline_system",
            analysis_type="eda_pipeline_error",
            result="failure",
            details={"error": str(error)}
        )

    # Helper methods
    def _create_traffic_summary(self, traffic_analysis, competitive_traffic):
        """Create traffic analysis summary"""
        return {
            'total_traffic': traffic_analysis.get('total_traffic', 0),
            'traffic_growth_rate': traffic_analysis.get('growth_rate', 0),
            'competitive_position': competitive_traffic.get('relative_position', 'unknown')
        }

    def _create_serp_summary(self, serp_analysis, serp_opportunities):
        """Create SERP analysis summary"""
        return {
            'total_serp_features': len(serp_analysis.get('feature_distribution', {})),
            'opportunities_identified': len(serp_opportunities.get('opportunities', [])),
            'serp_coverage': serp_analysis.get('coverage_percentage', 0)
        }

    def _create_anomaly_summary(self, position_anomalies, traffic_anomalies):
        """Create anomaly detection summary"""
        return {
            'position_anomalies_count': len(position_anomalies.get('anomalies', [])),
            'traffic_anomalies_count': len(traffic_anomalies.get('anomalies', [])),
            'anomaly_severity': 'low'  # Would be calculated based on actual anomalies
        }

    def _create_executive_summary(self, all_results):
        """Create executive summary"""
        return {
            'analysis_scope': 'comprehensive_eda',
            'data_quality_score': all_results.get('data_loading', {}).get('overall_quality', 0),
            'key_insights_count': len(self._extract_key_findings(all_results)),
            'anomalies_detected': self._count_total_anomalies(all_results),
            'analysis_timestamp': datetime.now()
        }

    def _extract_key_findings(self, all_results):
        """Extract key findings from all analysis results"""
        findings = []
        
        # Position findings
        position_results = all_results.get('position_analysis', {})
        if position_results:
            findings.append("Position analysis completed with trend identification")
        
        # Traffic findings
        traffic_results = all_results.get('traffic_analysis', {})
        if traffic_results:
            findings.append("Traffic performance analysis completed")
        
        # SERP findings
        serp_results = all_results.get('serp_analysis', {})
        if serp_results:
            findings.append("SERP feature analysis completed")
        
        # Anomaly findings
        anomaly_results = all_results.get('anomaly_analysis', {})
        if anomaly_results:
            findings.append("Anomaly detection analysis completed")
        
        return findings

    def _assess_overall_data_quality(self, all_results):
        """Assess overall data quality"""
        data_loading = all_results.get('data_loading', {})
        return {
            'overall_quality_score': data_loading.get('overall_quality', 0),
            'quality_threshold_met': data_loading.get('overall_quality', 0) >= self.data_config.data_quality_threshold,
            'datasets_analyzed': len(data_loading.get('all_data', {}))
        }

    def _assess_integration_quality(self, all_results):
        """Assess integration quality"""
        completed_phases = sum(1 for result in all_results.values() if result)
        total_phases = len(all_results)
        return completed_phases / total_phases if total_phases > 0 else 0

    def _count_total_anomalies(self, all_results):
        """Count total anomalies detected"""
        anomaly_results = all_results.get('anomaly_analysis', {})
        position_anomalies = len(anomaly_results.get('position_anomalies', {}).get('anomalies', []))
        traffic_anomalies = len(anomaly_results.get('traffic_anomalies', {}).get('anomalies', []))
        return position_anomalies + traffic_anomalies
