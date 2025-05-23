"""
Competitive Analyzer Module for SEO Competitive Intelligence
Advanced competitive analysis leveraging the comprehensive utility framework
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
class CompetitorMetrics:
    """Comprehensive competitor metrics"""
    domain: str
    organic_keywords: int
    organic_traffic: int
    avg_position: float
    traffic_share: float
    keyword_overlap: float
    competitive_strength: float
    growth_rate: float
    market_share: float

@dataclass
class CompetitiveAnalysisResult:
    """Result of competitive analysis"""
    competitor_metrics: Dict[str, CompetitorMetrics]
    market_analysis: Dict[str, Any]
    competitive_insights: Dict[str, Any]
    strategic_recommendations: List[str]
    opportunity_analysis: Dict[str, Any]
    threat_assessment: Dict[str, Any]

@dataclass
class GapAnalysisResult:
    """Result of gap analysis"""
    keyword_gaps: pd.DataFrame
    content_gaps: List[Dict[str, Any]]
    opportunity_score: float
    priority_keywords: List[str]
    competitive_disadvantages: List[str]
    actionable_insights: List[str]

class CompetitiveAnalyzer:
    """
    Advanced competitive analysis for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    competitive analysis capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("competitive_analyzer")
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
        
        # Load competitive analysis configurations
        analysis_config = self.config.get_analysis_config()
        self.competitive_threshold = analysis_config.competitive_threat_threshold
        self.min_keyword_overlap = 10

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def perform_comprehensive_competitive_analysis(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> CompetitiveAnalysisResult:
        """
        Perform comprehensive competitive analysis using utility framework.
        
        Args:
            lenovo_data: Lenovo's SEO data
            competitor_data: Dictionary of competitor data
            analysis_config: Optional analysis configuration
            
        Returns:
            CompetitiveAnalysisResult with comprehensive analysis
        """
        try:
            with self.performance_tracker.track_block("comprehensive_competitive_analysis"):
                # Audit log the competitive analysis operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="competitive_analysis",
                    parameters={
                        "lenovo_records": len(lenovo_data),
                        "competitors": list(competitor_data.keys()),
                        "total_competitor_records": sum(len(df) for df in competitor_data.values())
                    }
                )
                
                if analysis_config is None:
                    analysis_config = {'include_trends': True, 'include_forecasting': True}
                
                # Clean and validate all data using DataProcessor
                cleaned_lenovo = self.data_processor.clean_seo_data(lenovo_data)
                cleaned_competitors = {}
                
                for competitor, df in competitor_data.items():
                    cleaned_df = self.data_processor.clean_seo_data(df)
                    # Validate competitor data quality
                    validation_report = self.data_validator.validate_seo_dataset(cleaned_df, 'positions')
                    if validation_report.quality_score < 0.7:
                        self.logger.warning(f"Low data quality for {competitor}: {validation_report.quality_score:.3f}")
                    cleaned_competitors[competitor] = cleaned_df
                
                # Calculate comprehensive competitor metrics
                competitor_metrics = self._calculate_competitor_metrics(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Perform market analysis using StatisticalCalculator
                market_analysis = self._perform_market_analysis(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Generate competitive insights using advanced analytics
                competitive_insights = self._generate_competitive_insights(
                    cleaned_lenovo, cleaned_competitors, competitor_metrics
                )
                
                # Generate strategic recommendations using OptimizationHelper
                strategic_recommendations = self._generate_strategic_recommendations(
                    competitor_metrics, market_analysis, competitive_insights
                )
                
                # Perform opportunity analysis
                opportunity_analysis = self._analyze_opportunities(
                    cleaned_lenovo, cleaned_competitors, analysis_config
                )
                
                # Assess competitive threats using business rules
                threat_assessment = self._assess_competitive_threats(
                    competitor_metrics, market_analysis
                )
                
                result = CompetitiveAnalysisResult(
                    competitor_metrics=competitor_metrics,
                    market_analysis=market_analysis,
                    competitive_insights=competitive_insights,
                    strategic_recommendations=strategic_recommendations,
                    opportunity_analysis=opportunity_analysis,
                    threat_assessment=threat_assessment
                )
                
                self.logger.info(f"Comprehensive competitive analysis completed for {len(competitor_data)} competitors")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive competitive analysis: {str(e)}")
            return CompetitiveAnalysisResult({}, {}, {}, [], {}, {})

    @timing_decorator()
    def perform_keyword_gap_analysis(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        gap_threshold: int = 10,
        volume_threshold: int = 100
    ) -> GapAnalysisResult:
        """
        Perform keyword gap analysis using advanced data processing.
        
        Args:
            lenovo_data: Lenovo's keyword data
            competitor_data: Competitor keyword data
            gap_threshold: Position gap threshold
            volume_threshold: Minimum search volume
            
        Returns:
            GapAnalysisResult with gap analysis findings
        """
        try:
            with self.performance_tracker.track_block("keyword_gap_analysis"):
                # Clean data using DataProcessor
                cleaned_lenovo = self.data_processor.clean_seo_data(lenovo_data)
                cleaned_competitors = {
                    comp: self.data_processor.clean_seo_data(df)
                    for comp, df in competitor_data.items()
                }
                
                # Identify keyword gaps using set operations and analytics
                all_competitor_keywords = set()
                lenovo_keywords = set(cleaned_lenovo['Keyword'].str.lower().tolist())
                
                # Collect all competitor keywords
                for comp_df in cleaned_competitors.values():
                    comp_keywords = set(comp_df['Keyword'].str.lower().tolist())
                    all_competitor_keywords.update(comp_keywords)
                
                # Find gap keywords (competitors have, Lenovo doesn't)
                gap_keywords = all_competitor_keywords - lenovo_keywords
                
                # Analyze gap keywords using competitor data
                gap_analysis_data = []
                
                for gap_keyword in gap_keywords:
                    keyword_analysis = self._analyze_gap_keyword(
                        gap_keyword, cleaned_competitors, volume_threshold
                    )
                    if keyword_analysis:
                        gap_analysis_data.append(keyword_analysis)
                
                # Create gap analysis DataFrame
                keyword_gaps = pd.DataFrame(gap_analysis_data)
                
                if not keyword_gaps.empty:
                    # Sort by opportunity score using statistical ranking
                    keyword_gaps = keyword_gaps.sort_values('opportunity_score', ascending=False)
                    
                    # Calculate overall opportunity score using StatisticalCalculator
                    opportunity_scores = keyword_gaps['opportunity_score'].values
                    stats_dict = self.stats_calculator.calculate_descriptive_statistics(opportunity_scores)
                    overall_opportunity_score = stats_dict.get('mean', 0)
                    
                    # Identify priority keywords (top 20% by opportunity score)
                    priority_threshold = np.percentile(opportunity_scores, 80)
                    priority_keywords = keyword_gaps[
                        keyword_gaps['opportunity_score'] >= priority_threshold
                    ]['keyword'].tolist()
                    
                else:
                    overall_opportunity_score = 0.0
                    priority_keywords = []
                
                # Identify content gaps using StringHelper
                content_gaps = self._identify_content_gaps(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Identify competitive disadvantages
                competitive_disadvantages = self._identify_competitive_disadvantages(
                    cleaned_lenovo, cleaned_competitors, keyword_gaps
                )
                
                # Generate actionable insights using optimization
                actionable_insights = self._generate_gap_insights(
                    keyword_gaps, content_gaps, competitive_disadvantages
                )
                
                result = GapAnalysisResult(
                    keyword_gaps=keyword_gaps,
                    content_gaps=content_gaps,
                    opportunity_score=overall_opportunity_score,
                    priority_keywords=priority_keywords,
                    competitive_disadvantages=competitive_disadvantages,
                    actionable_insights=actionable_insights
                )
                
                self.logger.info(f"Gap analysis completed: {len(keyword_gaps)} gaps identified")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in keyword gap analysis: {str(e)}")
            return GapAnalysisResult(pd.DataFrame(), [], 0.0, [], [], [])

    @timing_decorator()
    def analyze_competitive_trends(
        self,
        competitor_data: Dict[str, pd.DataFrame],
        time_period_days: int = 90,
        trend_analysis_method: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        Analyze competitive trends using TimeSeriesAnalyzer.
        
        Args:
            competitor_data: Time-series competitor data
            time_period_days: Analysis time period
            trend_analysis_method: Method for trend analysis
            
        Returns:
            Dictionary of trend analysis results
        """
        try:
            with self.performance_tracker.track_block("analyze_competitive_trends"):
                trend_analysis = {}
                
                for competitor, df in competitor_data.items():
                    # Clean data using DataProcessor
                    cleaned_df = self.data_processor.clean_seo_data(df)
                    
                    if 'date' not in cleaned_df.columns or len(cleaned_df) < 10:
                        continue
                    
                    # Prepare time series data
                    ts_data = cleaned_df.set_index('date').sort_index()
                    
                    competitor_trends = {}
                    
                    # Analyze position trends using TimeSeriesAnalyzer
                    if 'Position' in ts_data.columns:
                        position_series = ts_data['Position'].dropna()
                        if len(position_series) >= 10:
                            # Fit trend model
                            trend_model = self.time_series_analyzer.fit_trend_model(
                                position_series, 'linear'
                            )
                            competitor_trends['position_trend'] = trend_model
                            
                            # Detect change points
                            changepoints = self.time_series_analyzer.detect_changepoints(
                                position_series, method='variance'
                            )
                            competitor_trends['position_changepoints'] = changepoints
                    
                    # Analyze traffic trends
                    if 'Traffic (%)' in ts_data.columns:
                        traffic_series = ts_data['Traffic (%)'].dropna()
                        if len(traffic_series) >= 10:
                            traffic_trend = self.time_series_analyzer.fit_trend_model(
                                traffic_series, 'linear'
                            )
                            competitor_trends['traffic_trend'] = traffic_trend
                    
                    # Calculate trend statistics using StatisticalCalculator
                    if 'Position' in ts_data.columns and len(ts_data['Position'].dropna()) > 5:
                        position_stats = self.stats_calculator.calculate_descriptive_statistics(
                            ts_data['Position'].dropna(), include_advanced=True
                        )
                        competitor_trends['position_statistics'] = position_stats
                    
                    # Analyze volatility using time series methods
                    if 'Position' in ts_data.columns:
                        position_volatility = self._calculate_position_volatility(
                            ts_data['Position'].dropna()
                        )
                        competitor_trends['position_volatility'] = position_volatility
                    
                    trend_analysis[competitor] = competitor_trends
                
                # Calculate cross-competitor trends
                cross_competitor_analysis = self._analyze_cross_competitor_trends(
                    competitor_data
                )
                trend_analysis['cross_competitor_analysis'] = cross_competitor_analysis
                
                self.logger.info(f"Competitive trend analysis completed for {len(competitor_data)} competitors")
                return trend_analysis
                
        except Exception as e:
            self.logger.error(f"Error in competitive trend analysis: {str(e)}")
            return {}

    def _calculate_competitor_metrics(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, CompetitorMetrics]:
        """Calculate comprehensive competitor metrics using statistical analysis."""
        try:
            competitor_metrics = {}
            
            # Calculate total market metrics using StatisticalCalculator
            total_traffic = lenovo_data.get('Traffic (%)', pd.Series()).sum()
            total_keywords = len(lenovo_data)
            lenovo_keywords = set(lenovo_data['Keyword'].str.lower().tolist())
            
            for competitor_name, comp_df in competitor_data.items():
                # Basic metrics
                comp_keywords = len(comp_df)
                comp_traffic = comp_df.get('Traffic (%)', pd.Series()).sum()
                
                # Calculate average position using StatisticalCalculator
                position_stats = self.stats_calculator.calculate_descriptive_statistics(
                    comp_df.get('Position', pd.Series()).dropna()
                )
                avg_position = position_stats.get('mean', 50)
                
                # Calculate traffic share
                total_market_traffic = total_traffic + comp_traffic
                traffic_share = safe_divide(comp_traffic, total_market_traffic, 0.0)
                
                # Calculate keyword overlap using set operations
                comp_keyword_set = set(comp_df['Keyword'].str.lower().tolist())
                keyword_overlap = len(lenovo_keywords.intersection(comp_keyword_set))
                overlap_ratio = safe_divide(keyword_overlap, len(lenovo_keywords.union(comp_keyword_set)), 0.0)
                
                # Calculate competitive strength using multiple factors
                competitive_strength = self._calculate_competitive_strength(
                    comp_df, lenovo_data, traffic_share, overlap_ratio
                )
                
                # Calculate growth rate using time series analysis
                growth_rate = 0.0
                if 'date' in comp_df.columns and len(comp_df) > 10:
                    growth_rate = self._calculate_growth_rate(comp_df)
                
                # Calculate market share
                market_share = safe_divide(comp_traffic, total_market_traffic, 0.0)
                
                metrics = CompetitorMetrics(
                    domain=competitor_name,
                    organic_keywords=comp_keywords,
                    organic_traffic=int(comp_traffic),
                    avg_position=avg_position,
                    traffic_share=traffic_share,
                    keyword_overlap=overlap_ratio,
                    competitive_strength=competitive_strength,
                    growth_rate=growth_rate,
                    market_share=market_share
                )
                
                competitor_metrics[competitor_name] = metrics
            
            return competitor_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating competitor metrics: {str(e)}")
            return {}

    def _perform_market_analysis(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Perform market analysis using statistical methods."""
        try:
            market_analysis = {}
            
            # Calculate market concentration using statistical measures
            traffic_values = [lenovo_data.get('Traffic (%)', pd.Series()).sum()]
            traffic_values.extend([
                df.get('Traffic (%)', pd.Series()).sum() 
                for df in competitor_data.values()
            ])
            
            # Use StatisticalCalculator for market concentration analysis
            market_stats = self.stats_calculator.calculate_descriptive_statistics(
                traffic_values, include_advanced=True
            )
            
            market_analysis['market_concentration'] = {
                'gini_coefficient': self._calculate_gini_coefficient(traffic_values),
                'herfindahl_index': self._calculate_herfindahl_index(traffic_values),
                'market_leader_share': max(traffic_values) / sum(traffic_values) if sum(traffic_values) > 0 else 0,
                'traffic_statistics': market_stats
            }
            
            # Analyze keyword universe
            all_keywords = set(lenovo_data['Keyword'].str.lower().tolist())
            for df in competitor_data.values():
                all_keywords.update(df['Keyword'].str.lower().tolist())
            
            market_analysis['keyword_universe'] = {
                'total_keywords': len(all_keywords),
                'lenovo_coverage': len(set(lenovo_data['Keyword'].str.lower().tolist())) / len(all_keywords),
                'average_competitor_coverage': np.mean([
                    len(set(df['Keyword'].str.lower().tolist())) / len(all_keywords)
                    for df in competitor_data.values()
                ])
            }
            
            # Analyze competitive intensity
            market_analysis['competitive_intensity'] = self._calculate_competitive_intensity(
                lenovo_data, competitor_data
            )
            
            return market_analysis
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return {}

    def _generate_competitive_insights(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        competitor_metrics: Dict[str, CompetitorMetrics]
    ) -> Dict[str, Any]:
        """Generate competitive insights using advanced analytics."""
        try:
            insights = {}
            
            # Identify competitive strengths and weaknesses
            insights['strengths_weaknesses'] = self._analyze_competitive_positioning(
                lenovo_data, competitor_data, competitor_metrics
            )
            
            # Identify keyword opportunities using StringHelper and statistical analysis
            insights['keyword_opportunities'] = self._identify_keyword_opportunities(
                lenovo_data, competitor_data
            )
            
            # Analyze competitive gaps
            insights['competitive_gaps'] = self._analyze_competitive_gaps(
                lenovo_data, competitor_data
            )
            
            # Performance benchmarking using StatisticalCalculator
            insights['performance_benchmarks'] = self._calculate_performance_benchmarks(
                lenovo_data, competitor_data, competitor_metrics
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating competitive insights: {str(e)}")
            return {}

    def _generate_strategic_recommendations(
        self,
        competitor_metrics: Dict[str, CompetitorMetrics],
        market_analysis: Dict[str, Any],
        competitive_insights: Dict[str, Any]
    ) -> List[str]:
        """Generate strategic recommendations using optimization principles."""
        try:
            recommendations = []
            
            # Analyze competitor threats
            high_threat_competitors = [
                name for name, metrics in competitor_metrics.items()
                if metrics.competitive_strength > self.competitive_threshold
            ]
            
            if high_threat_competitors:
                recommendations.append(
                    f"Monitor high-threat competitors closely: {', '.join(high_threat_competitors)}"
                )
            
            # Market concentration recommendations
            market_concentration = market_analysis.get('market_concentration', {})
            if market_concentration.get('market_leader_share', 0) > 0.4:
                recommendations.append("Market is dominated by single player - consider niche targeting")
            
            # Keyword opportunity recommendations
            keyword_opportunities = competitive_insights.get('keyword_opportunities', {})
            if keyword_opportunities.get('high_opportunity_count', 0) > 50:
                recommendations.append("Significant keyword opportunities identified - prioritize content expansion")
            
            # Growth-based recommendations
            fast_growing_competitors = [
                name for name, metrics in competitor_metrics.items()
                if metrics.growth_rate > 0.1  # 10% growth
            ]
            
            if fast_growing_competitors:
                recommendations.append(
                    f"Analyze growth strategies of fast-growing competitors: {', '.join(fast_growing_competitors)}"
                )
            
            # Traffic share recommendations
            low_share_competitors = [
                name for name, metrics in competitor_metrics.items()
                if metrics.traffic_share < 0.05  # Less than 5% share
            ]
            
            if low_share_competitors:
                recommendations.append("Consider strategies to outpace low market share competitors")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating strategic recommendations: {str(e)}")
            return []

    def export_competitive_analysis_results(
        self,
        competitive_result: CompetitiveAnalysisResult,
        gap_result: GapAnalysisResult,
        export_directory: str,
        include_detailed_analysis: bool = True
    ) -> Dict[str, bool]:
        """Export comprehensive competitive analysis results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'competitive_summary': {
                    'total_competitors_analyzed': len(competitive_result.competitor_metrics),
                    'market_analysis': competitive_result.market_analysis,
                    'competitive_insights': competitive_result.competitive_insights,
                    'strategic_recommendations': competitive_result.strategic_recommendations
                },
                'competitor_metrics': {
                    name: {
                        'domain': metrics.domain,
                        'organic_keywords': metrics.organic_keywords,
                        'organic_traffic': metrics.organic_traffic,
                        'avg_position': metrics.avg_position,
                        'traffic_share': metrics.traffic_share,
                        'keyword_overlap': metrics.keyword_overlap,
                        'competitive_strength': metrics.competitive_strength,
                        'growth_rate': metrics.growth_rate,
                        'market_share': metrics.market_share
                    }
                    for name, metrics in competitive_result.competitor_metrics.items()
                },
                'gap_analysis': {
                    'total_gaps_identified': len(gap_result.keyword_gaps),
                    'opportunity_score': gap_result.opportunity_score,
                    'priority_keywords_count': len(gap_result.priority_keywords),
                    'content_gaps_count': len(gap_result.content_gaps),
                    'actionable_insights': gap_result.actionable_insights
                }
            }
            
            # Export summary data using DataExporter
            summary_export_success = self.data_exporter.export_analysis_dataset(
                {'competitive_analysis_summary': pd.DataFrame([export_data])},
                export_path / "competitive_analysis_summary.xlsx"
            )
            
            # Export detailed gap analysis
            gap_export_success = True
            if not gap_result.keyword_gaps.empty:
                gap_export_success = self.data_exporter.export_with_metadata(
                    gap_result.keyword_gaps,
                    metadata={'analysis_type': 'keyword_gaps', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "keyword_gap_analysis.xlsx"
                )
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_competitive_analysis_report(
                {name: pd.DataFrame() for name in competitive_result.competitor_metrics.keys()},  # Placeholder
                export_data,
                export_path / "competitive_analysis_executive_report.html"
            )
            
            return {
                'summary_export': summary_export_success,
                'gap_analysis': gap_export_success,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting competitive analysis results: {str(e)}")
            return {}

    # Helper methods using utility framework
    def _calculate_competitive_strength(
        self,
        comp_df: pd.DataFrame,
        lenovo_df: pd.DataFrame,
        traffic_share: float,
        overlap_ratio: float
    ) -> float:
        """Calculate competitive strength using multiple factors."""
        try:
            # Position advantage
            comp_avg_position = comp_df.get('Position', pd.Series()).mean()
            lenovo_avg_position = lenovo_df.get('Position', pd.Series()).mean()
            position_advantage = safe_divide(lenovo_avg_position, comp_avg_position, 1.0) - 1
            
            # Traffic efficiency
            comp_traffic_per_keyword = safe_divide(
                comp_df.get('Traffic (%)', pd.Series()).sum(), len(comp_df), 0.0
            )
            lenovo_traffic_per_keyword = safe_divide(
                lenovo_df.get('Traffic (%)', pd.Series()).sum(), len(lenovo_df), 0.0
            )
            traffic_efficiency = safe_divide(comp_traffic_per_keyword, lenovo_traffic_per_keyword, 0.0)
            
            # Combine factors
            strength_score = (
                traffic_share * 0.4 +
                overlap_ratio * 0.3 +
                min(position_advantage, 1.0) * 0.2 +
                min(traffic_efficiency, 2.0) * 0.1
            )
            
            return min(strength_score, 1.0)
            
        except Exception:
            return 0.0

    def _calculate_growth_rate(self, df: pd.DataFrame) -> float:
        """Calculate growth rate using time series analysis."""
        try:
            if 'date' not in df.columns or 'Traffic (%)' not in df.columns:
                return 0.0
            
            # Prepare time series data
            ts_data = df.groupby('date')['Traffic (%)'].sum().sort_index()
            
            if len(ts_data) < 5:
                return 0.0
            
            # Use TimeSeriesAnalyzer to fit trend
            trend_model = self.time_series_analyzer.fit_trend_model(ts_data, 'linear')
            
            return trend_model.get('slope', 0.0) if trend_model else 0.0
            
        except Exception:
            return 0.0

    def _analyze_gap_keyword(
        self,
        keyword: str,
        competitor_data: Dict[str, pd.DataFrame],
        volume_threshold: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze individual gap keyword using competitor data."""
        try:
            keyword_analysis = {
                'keyword': keyword,
                'competitors_ranking': [],
                'avg_position': 0,
                'total_volume': 0,
                'opportunity_score': 0
            }
            
            positions = []
            volumes = []
            
            for comp_name, comp_df in competitor_data.items():
                # Find keyword in competitor data
                keyword_rows = comp_df[comp_df['Keyword'].str.lower() == keyword.lower()]
                
                if not keyword_rows.empty:
                    position = keyword_rows.iloc[0].get('Position', 100)
                    volume = keyword_rows.iloc[0].get('Search Volume', 0)
                    
                    if volume >= volume_threshold:
                        keyword_analysis['competitors_ranking'].append({
                            'competitor': comp_name,
                            'position': position,
                            'volume': volume
                        })
                        positions.append(position)
                        volumes.append(volume)
            
            if positions:
                keyword_analysis['avg_position'] = np.mean(positions)
                keyword_analysis['total_volume'] = max(volumes) if volumes else 0
                
                # Calculate opportunity score using statistical methods
                avg_position = keyword_analysis['avg_position']
                max_volume = keyword_analysis['total_volume']
                
                # Higher score for keywords with good competitor positions and high volume
                position_score = max(0, (50 - avg_position) / 50)  # Better positions = higher score
                volume_score = min(np.log10(max_volume + 1) / 6, 1.0)  # Log scale for volume
                
                keyword_analysis['opportunity_score'] = (position_score + volume_score) / 2
                
                return keyword_analysis
            
            return None
            
        except Exception:
            return None

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for market concentration."""
        try:
            values = sorted([v for v in values if v > 0])
            n = len(values)
            if n == 0:
                return 0.0
            
            cumsum = np.cumsum(values)
            return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(values))
        except Exception:
            return 0.0

    def _calculate_herfindahl_index(self, values: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index."""
        try:
            total = sum(values)
            if total == 0:
                return 0.0
            
            shares = [v / total for v in values]
            return sum(share ** 2 for share in shares)
        except Exception:
            return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for competitive analysis operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods would be implemented here...
    def _identify_keyword_opportunities(self, lenovo_data, competitor_data):
        """Identify keyword opportunities using competitor analysis."""
        try:
            opportunities = {'high_opportunity_count': 0, 'opportunities': []}
            
            # Simplified implementation - would be more sophisticated in practice
            lenovo_keywords = set(lenovo_data['Keyword'].str.lower().tolist())
            
            for comp_name, comp_df in competitor_data.items():
                comp_keywords = set(comp_df['Keyword'].str.lower().tolist())
                unique_keywords = comp_keywords - lenovo_keywords
                
                for keyword in list(unique_keywords)[:10]:  # Top 10 for performance
                    keyword_data = comp_df[comp_df['Keyword'].str.lower() == keyword]
                    if not keyword_data.empty:
                        volume = keyword_data.iloc[0].get('Search Volume', 0)
                        position = keyword_data.iloc[0].get('Position', 100)
                        
                        if volume >= 100 and position <= 20:
                            opportunities['opportunities'].append({
                                'keyword': keyword,
                                'competitor': comp_name,
                                'volume': volume,
                                'position': position
                            })
            
            opportunities['high_opportunity_count'] = len(opportunities['opportunities'])
            return opportunities
        except Exception:
            return {'high_opportunity_count': 0, 'opportunities': []}
