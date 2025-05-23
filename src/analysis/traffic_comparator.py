"""
Traffic Comparison Module for SEO Competitive Intelligence
Advanced traffic analysis, comparison metrics, and competitive traffic assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Import our utilities to eliminate redundancy
from src.utils.common_helpers import StringHelper, memoize, timing_decorator, safe_divide
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.math_utils import StatisticalCalculator, OptimizationHelper
from src.utils.logging_utils import LoggerFactory, PerformanceTracker
from src.utils.config_utils import ConfigManager
from src.utils.validation_utils import SchemaValidator

@dataclass
class TrafficMetrics:
    """Data class for traffic comparison results"""
    total_traffic: float
    traffic_share: float
    traffic_growth: float
    traffic_volatility: float
    avg_traffic_per_keyword: float
    traffic_concentration: float
    top_traffic_keywords: List[Dict[str, Any]]

@dataclass
class CompetitiveTrafficAnalysis:
    """Data class for competitive traffic analysis results"""
    competitor_rankings: Dict[str, int]
    traffic_gaps: Dict[str, float]
    opportunity_matrix: pd.DataFrame
    competitive_advantage_score: float
    market_share_analysis: Dict[str, float]
    growth_opportunities: List[Dict[str, Any]]

class TrafficComparator:
    """
    Advanced traffic comparison for SEO competitive intelligence.
    
    Provides comprehensive traffic analysis including competitive comparison,
    market share analysis, and traffic opportunity identification.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config_manager: Optional[ConfigManager] = None):
        """Initialize with utilities instead of custom implementations."""
        self.logger = logger or LoggerFactory.get_logger("traffic_comparator")
        self.config = config_manager or ConfigManager()
        
        # Initialize utility classes
        self.data_processor = DataProcessor(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.performance_tracker = PerformanceTracker(self.logger)
        self.schema_validator = SchemaValidator(self.logger)
        self.optimization_helper = OptimizationHelper(self.logger)
        
        # Get analysis config instead of hardcoding
        analysis_config = self.config.get_analysis_config()
        self.min_traffic_threshold = analysis_config.min_search_volume  # Reuse config value
        self.traffic_volatility_threshold = analysis_config.traffic_anomaly_threshold

    @timing_decorator()
    @memoize(ttl=1800)  # Cache for 30 minutes
    def calculate_traffic_metrics(
        self,
        df: pd.DataFrame,
        traffic_column: str = 'Traffic (%)',
        keyword_column: str = 'Keyword'
    ) -> TrafficMetrics:
        """
        Calculate comprehensive traffic metrics using statistical utilities.
        
        Args:
            df: DataFrame with traffic data
            traffic_column: Column containing traffic data
            keyword_column: Column containing keywords
            
        Returns:
            TrafficMetrics with comprehensive traffic analysis
        """
        try:
            with self.performance_tracker.track_block("calculate_traffic_metrics"):
                # Validate data first
                validation_report = self.data_validator.validate_seo_dataset(df, 'positions')
                if validation_report.quality_score < 0.7:
                    self.logger.warning(f"Low data quality score: {validation_report.quality_score:.3f}")
                
                # Clean data using our data processor
                cleaned_df = self.data_processor.clean_seo_data(df)
                
                if traffic_column not in cleaned_df.columns:
                    self.logger.error(f"Traffic column '{traffic_column}' not found")
                    return TrafficMetrics(0, 0, 0, 0, 0, 0, [])
                
                traffic_data = cleaned_df[traffic_column].dropna()
                
                if len(traffic_data) == 0:
                    return TrafficMetrics(0, 0, 0, 0, 0, 0, [])
                
                # Use StatisticalCalculator for robust statistics
                traffic_stats = self.stats_calculator.calculate_descriptive_statistics(
                    traffic_data, include_advanced=True
                )
                
                total_traffic = traffic_stats.get('sum', traffic_data.sum())
                avg_traffic = traffic_stats.get('mean', 0)
                traffic_volatility = traffic_stats.get('coefficient_of_variation', 0)
                
                # Calculate traffic concentration using Gini coefficient (from math utils)
                traffic_concentration = self._calculate_traffic_gini_coefficient(traffic_data)
                
                # Get top traffic keywords
                top_keywords = self._get_top_traffic_keywords(
                    cleaned_df, traffic_column, keyword_column
                )
                
                # Calculate traffic growth if previous data available
                traffic_growth = self._calculate_traffic_growth(cleaned_df, traffic_column)
                
                metrics = TrafficMetrics(
                    total_traffic=total_traffic,
                    traffic_share=1.0,  # Will be calculated in competitive context
                    traffic_growth=traffic_growth,
                    traffic_volatility=traffic_volatility,
                    avg_traffic_per_keyword=avg_traffic,
                    traffic_concentration=traffic_concentration,
                    top_traffic_keywords=top_keywords
                )
                
                self.logger.info(f"Traffic metrics calculated: {total_traffic:.2f} total traffic")
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error calculating traffic metrics: {str(e)}")
            return TrafficMetrics(0, 0, 0, 0, 0, 0, [])

    @timing_decorator()
    def compare_competitive_traffic(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        traffic_column: str = 'Traffic (%)'
    ) -> CompetitiveTrafficAnalysis:
        """
        Perform comprehensive competitive traffic analysis.
        
        Args:
            lenovo_data: Lenovo traffic data
            competitor_data: Dictionary of competitor traffic data
            traffic_column: Column containing traffic data
            
        Returns:
            CompetitiveTrafficAnalysis with comprehensive comparison
        """
        try:
            with self.performance_tracker.track_block("compare_competitive_traffic"):
                # Calculate metrics for all competitors
                all_metrics = {}
                
                # Process Lenovo data
                lenovo_metrics = self.calculate_traffic_metrics(lenovo_data, traffic_column)
                all_metrics['lenovo'] = lenovo_metrics
                
                # Process competitor data
                for competitor_name, comp_df in competitor_data.items():
                    comp_metrics = self.calculate_traffic_metrics(comp_df, traffic_column)
                    all_metrics[competitor_name] = comp_metrics
                
                # Calculate market share using statistical utilities
                total_market_traffic = sum(
                    metrics.total_traffic for metrics in all_metrics.values()
                )
                
                market_share_analysis = {}
                for competitor, metrics in all_metrics.items():
                    share = safe_divide(metrics.total_traffic, total_market_traffic, 0.0)
                    market_share_analysis[competitor] = share
                
                # Rank competitors by traffic
                competitor_rankings = self._rank_competitors_by_traffic(all_metrics)
                
                # Calculate traffic gaps
                traffic_gaps = self._calculate_traffic_gaps(all_metrics, lenovo_metrics)
                
                # Create opportunity matrix using optimization helper
                opportunity_matrix = self._create_traffic_opportunity_matrix(
                    lenovo_data, competitor_data, traffic_column
                )
                
                # Calculate competitive advantage score
                competitive_advantage_score = self._calculate_competitive_advantage_score(
                    lenovo_metrics, all_metrics
                )
                
                # Identify growth opportunities
                growth_opportunities = self._identify_traffic_growth_opportunities(
                    lenovo_data, competitor_data, all_metrics
                )
                
                analysis = CompetitiveTrafficAnalysis(
                    competitor_rankings=competitor_rankings,
                    traffic_gaps=traffic_gaps,
                    opportunity_matrix=opportunity_matrix,
                    competitive_advantage_score=competitive_advantage_score,
                    market_share_analysis=market_share_analysis,
                    growth_opportunities=growth_opportunities
                )
                
                self.logger.info(f"Competitive traffic analysis completed for {len(competitor_data)} competitors")
                return analysis
                
        except Exception as e:
            self.logger.error(f"Error in competitive traffic analysis: {str(e)}")
            return CompetitiveTrafficAnalysis({}, {}, pd.DataFrame(), 0.0, {}, [])

    @timing_decorator()
    def identify_traffic_opportunities(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        min_opportunity_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Identify traffic opportunities using competitive analysis and optimization.
        
        Args:
            lenovo_data: Lenovo traffic data
            competitor_data: Competitor traffic data
            min_opportunity_score: Minimum opportunity score threshold
            
        Returns:
            List of traffic opportunities
        """
        try:
            opportunities = []
            
            # Combine all datasets for analysis
            all_datasets = {'lenovo': lenovo_data, **competitor_data}
            processed_datasets = {}
            
            for competitor, df in all_datasets.items():
                # Clean and process data
                cleaned_df = self.data_processor.clean_seo_data(df)
                processed_datasets[competitor] = cleaned_df
            
            # Find keywords where competitors have traffic but Lenovo doesn't
            lenovo_keywords = set(processed_datasets['lenovo']['Keyword'].tolist())
            
            for competitor_name, comp_df in processed_datasets.items():
                if competitor_name == 'lenovo':
                    continue
                
                comp_keywords = set(comp_df['Keyword'].tolist())
                
                # Find gap keywords (competitor has, Lenovo doesn't)
                gap_keywords = comp_keywords - lenovo_keywords
                
                for keyword in gap_keywords:
                    keyword_data = comp_df[comp_df['Keyword'] == keyword]
                    
                    if keyword_data.empty:
                        continue
                    
                    keyword_row = keyword_data.iloc[0]
                    
                    # Calculate opportunity score using multiple factors
                    opportunity_score = self._calculate_keyword_opportunity_score(
                        keyword_row, competitor_name
                    )
                    
                    if opportunity_score >= min_opportunity_score:
                        opportunity = {
                            'keyword': keyword,
                            'opportunity_score': opportunity_score,
                            'competitor': competitor_name,
                            'competitor_traffic': keyword_row.get('Traffic (%)', 0),
                            'competitor_position': keyword_row.get('Position', 100),
                            'search_volume': keyword_row.get('Search Volume', 0),
                            'difficulty': keyword_row.get('Keyword Difficulty', 50),
                            'estimated_traffic_potential': self._estimate_traffic_potential(keyword_row),
                            'implementation_priority': self._calculate_implementation_priority(keyword_row),
                            'competitive_advantage': self._assess_competitive_advantage(keyword, processed_datasets)
                        }
                        opportunities.append(opportunity)
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
            self.logger.info(f"Identified {len(opportunities)} traffic opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying traffic opportunities: {str(e)}")
            return []

    @timing_decorator()
    def analyze_traffic_trends(
        self,
        historical_data: pd.DataFrame,
        traffic_column: str = 'Traffic (%)',
        date_column: str = 'date'
    ) -> Dict[str, Any]:
        """
        Analyze traffic trends using time series analysis from math utils.
        
        Args:
            historical_data: Historical traffic data
            traffic_column: Column containing traffic data
            date_column: Column containing dates
            
        Returns:
            Traffic trend analysis results
        """
        try:
            with self.performance_tracker.track_block("analyze_traffic_trends"):
                # Clean and prepare data
                cleaned_df = self.data_processor.clean_seo_data(historical_data)
                
                if date_column not in cleaned_df.columns or traffic_column not in cleaned_df.columns:
                    self.logger.error("Required columns not found for trend analysis")
                    return {}
                
                # Prepare time series data
                ts_data = cleaned_df.groupby(date_column)[traffic_column].sum().sort_index()
                
                if len(ts_data) < 10:
                    self.logger.warning("Insufficient data for trend analysis")
                    return {}
                
                # Use TimeSeriesAnalyzer from math utils
                from src.utils.math_utils import TimeSeriesAnalyzer
                ts_analyzer = TimeSeriesAnalyzer(self.logger)
                
                # Decompose time series
                decomposition = ts_analyzer.decompose_time_series(ts_data)
                
                # Fit trend model
                trend_model = ts_analyzer.fit_trend_model(ts_data)
                
                # Detect anomalies
                anomalies_mask, anomaly_scores = ts_analyzer.detect_anomalies_in_series(ts_data)
                
                # Calculate trend statistics using our statistical calculator
                if len(ts_data) > 1:
                    trend_stats = self.stats_calculator.calculate_descriptive_statistics(
                        ts_data.values, include_advanced=True
                    )
                else:
                    trend_stats = {}
                
                trend_analysis = {
                    'trend_direction': self._determine_trend_direction(trend_model),
                    'trend_strength': trend_model.get('r_squared', 0),
                    'seasonality_detected': len(decomposition.get('seasonal', pd.Series())) > 0,
                    'anomalies_detected': np.sum(anomalies_mask) if len(anomalies_mask) > 0 else 0,
                    'volatility': trend_stats.get('coefficient_of_variation', 0),
                    'growth_rate': trend_model.get('slope', 0) if trend_model.get('model_type') == 'linear' else 0,
                    'trend_model': trend_model,
                    'statistical_summary': trend_stats,
                    'forecast_confidence': self._calculate_forecast_confidence(trend_model)
                }
                
                self.logger.info("Traffic trend analysis completed")
                return trend_analysis
                
        except Exception as e:
            self.logger.error(f"Error in traffic trend analysis: {str(e)}")
            return {}

    def _calculate_traffic_gini_coefficient(self, traffic_data: pd.Series) -> float:
        """Calculate Gini coefficient for traffic concentration using math utils."""
        try:
            # Use the same Gini calculation method as in position_analyzer
            traffic_values = traffic_data.dropna().values
            if len(traffic_values) == 0:
                return 0.0

            # Use statistical calculator for robust calculation
            stats_dict = self.stats_calculator.calculate_descriptive_statistics(traffic_values)
            n = len(traffic_values)
            mean_traffic = stats_dict.get('mean', 0)
            
            if mean_traffic == 0:
                return 0.0

            # Calculate Gini coefficient
            diff_sum = sum(abs(xi - xj) for xi in traffic_values for xj in traffic_values)
            gini = diff_sum / (2 * n * n * mean_traffic)
            
            return gini
            
        except Exception as e:
            self.logger.error(f"Error calculating Gini coefficient: {str(e)}")
            return 0.0

    def _get_top_traffic_keywords(
        self,
        df: pd.DataFrame,
        traffic_column: str,
        keyword_column: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top traffic-driving keywords."""
        try:
            top_keywords_df = df.nlargest(top_n, traffic_column)
            
            top_keywords = []
            for _, row in top_keywords_df.iterrows():
                keyword_info = {
                    'keyword': row[keyword_column],
                    'traffic': row[traffic_column],
                    'position': row.get('Position', 'N/A'),
                    'search_volume': row.get('Search Volume', 'N/A'),
                    'traffic_share': safe_divide(row[traffic_column], df[traffic_column].sum(), 0.0)
                }
                top_keywords.append(keyword_info)
            
            return top_keywords
            
        except Exception as e:
            self.logger.error(f"Error getting top traffic keywords: {str(e)}")
            return []

    def _calculate_traffic_growth(self, df: pd.DataFrame, traffic_column: str) -> float:
        """Calculate traffic growth if previous data is available."""
        try:
            # Simple growth calculation - in practice, would use time series data
            current_traffic = df[traffic_column].sum()
            
            # If we have previous position data, calculate growth
            if 'Previous position' in df.columns:
                # Estimate previous traffic based on position changes
                # This is a simplified approach
                return 0.0  # Placeholder for more sophisticated calculation
            
            return 0.0
            
        except Exception:
            return 0.0

    def _rank_competitors_by_traffic(self, all_metrics: Dict[str, TrafficMetrics]) -> Dict[str, int]:
        """Rank competitors by total traffic."""
        try:
            # Sort by total traffic descending
            sorted_competitors = sorted(
                all_metrics.items(),
                key=lambda x: x[1].total_traffic,
                reverse=True
            )
            
            rankings = {}
            for rank, (competitor, _) in enumerate(sorted_competitors, 1):
                rankings[competitor] = rank
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"Error ranking competitors: {str(e)}")
            return {}

    def _calculate_traffic_gaps(
        self,
        all_metrics: Dict[str, TrafficMetrics],
        lenovo_metrics: TrafficMetrics
    ) -> Dict[str, float]:
        """Calculate traffic gaps between Lenovo and competitors."""
        try:
            gaps = {}
            
            for competitor, metrics in all_metrics.items():
                if competitor == 'lenovo':
                    continue
                
                gap = metrics.total_traffic - lenovo_metrics.total_traffic
                gaps[competitor] = gap
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error calculating traffic gaps: {str(e)}")
            return {}

    def _create_traffic_opportunity_matrix(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        traffic_column: str
    ) -> pd.DataFrame:
        """Create opportunity matrix showing traffic potential by keyword."""
        try:
            # This would create a matrix showing traffic opportunities
            # Simplified implementation for now
            opportunity_data = []
            
            # Get all unique keywords
            all_keywords = set(lenovo_data['Keyword'].tolist())
            for comp_df in competitor_data.values():
                all_keywords.update(comp_df['Keyword'].tolist())
            
            # For each keyword, calculate opportunity score
            for keyword in list(all_keywords)[:50]:  # Limit for performance
                row_data = {'keyword': keyword}
                
                # Get Lenovo traffic for this keyword
                lenovo_traffic = lenovo_data[
                    lenovo_data['Keyword'] == keyword
                ][traffic_column].sum()
                row_data['lenovo_traffic'] = lenovo_traffic
                
                # Get competitor traffic
                for comp_name, comp_df in competitor_data.items():
                    comp_traffic = comp_df[
                        comp_df['Keyword'] == keyword
                    ][traffic_column].sum()
                    row_data[f'{comp_name}_traffic'] = comp_traffic
                
                opportunity_data.append(row_data)
            
            return pd.DataFrame(opportunity_data)
            
        except Exception as e:
            self.logger.error(f"Error creating opportunity matrix: {str(e)}")
            return pd.DataFrame()

    def _calculate_competitive_advantage_score(
        self,
        lenovo_metrics: TrafficMetrics,
        all_metrics: Dict[str, TrafficMetrics]
    ) -> float:
        """Calculate competitive advantage score."""
        try:
            # Calculate relative performance vs competitors
            competitor_metrics = [
                metrics for name, metrics in all_metrics.items() 
                if name != 'lenovo'
            ]
            
            if not competitor_metrics:
                return 0.5
            
            avg_competitor_traffic = np.mean([m.total_traffic for m in competitor_metrics])
            
            if avg_competitor_traffic == 0:
                return 1.0 if lenovo_metrics.total_traffic > 0 else 0.5
            
            # Score based on relative performance
            relative_performance = lenovo_metrics.total_traffic / avg_competitor_traffic
            
            # Normalize to 0-1 scale
            advantage_score = min(1.0, relative_performance / 2.0)
            
            return advantage_score
            
        except Exception as e:
            self.logger.error(f"Error calculating competitive advantage: {str(e)}")
            return 0.5

    def _identify_traffic_growth_opportunities(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        all_metrics: Dict[str, TrafficMetrics]
    ) -> List[Dict[str, Any]]:
        """Identify specific traffic growth opportunities."""
        try:
            opportunities = []
            
            # Analyze top competitor keywords
            for competitor_name, comp_df in competitor_data.items():
                comp_metrics = all_metrics.get(competitor_name)
                if not comp_metrics:
                    continue
                
                # Get top traffic keywords from competitor
                top_comp_keywords = comp_df.nlargest(10, 'Traffic (%)')
                
                for _, keyword_row in top_comp_keywords.iterrows():
                    keyword = keyword_row['Keyword']
                    
                    # Check if Lenovo ranks for this keyword
                    lenovo_keyword_data = lenovo_data[lenovo_data['Keyword'] == keyword]
                    
                    if lenovo_keyword_data.empty:
                        # Gap opportunity
                        opportunity = {
                            'type': 'gap_opportunity',
                            'keyword': keyword,
                            'competitor': competitor_name,
                            'competitor_traffic': keyword_row['Traffic (%)'],
                            'estimated_potential': keyword_row['Traffic (%)'] * 0.5,  # Conservative estimate
                            'priority': 'high' if keyword_row['Traffic (%)'] > 5 else 'medium'
                        }
                        opportunities.append(opportunity)
                    else:
                        # Position improvement opportunity
                        lenovo_row = lenovo_keyword_data.iloc[0]
                        position_gap = lenovo_row['Position'] - keyword_row['Position']
                        
                        if position_gap > 5:  # Significant position gap
                            opportunity = {
                                'type': 'position_improvement',
                                'keyword': keyword,
                                'competitor': competitor_name,
                                'position_gap': position_gap,
                                'traffic_upside': keyword_row['Traffic (%)'] - lenovo_row['Traffic (%)'],
                                'priority': 'high' if position_gap > 10 else 'medium'
                            }
                            opportunities.append(opportunity)
            
            return opportunities[:20]  # Return top 20 opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying growth opportunities: {str(e)}")
            return []

    def _calculate_keyword_opportunity_score(
        self,
        keyword_row: pd.Series,
        competitor_name: str
    ) -> float:
        """Calculate opportunity score for a keyword."""
        try:
            # Factors: traffic potential, search volume, difficulty, position
            traffic_factor = min(keyword_row.get('Traffic (%)', 0) / 10.0, 1.0)  # Normalize
            volume_factor = min(keyword_row.get('Search Volume', 0) / 10000.0, 1.0)  # Normalize
            position_factor = max(0, (50 - keyword_row.get('Position', 50)) / 50.0)  # Better position = higher score
            difficulty_factor = max(0, (100 - keyword_row.get('Keyword Difficulty', 50)) / 100.0)  # Lower difficulty = higher score
            
            # Weighted combination
            opportunity_score = (
                traffic_factor * 0.4 +
                volume_factor * 0.3 +
                position_factor * 0.2 +
                difficulty_factor * 0.1
            )
            
            return opportunity_score
            
        except Exception:
            return 0.0

    def _estimate_traffic_potential(self, keyword_row: pd.Series) -> float:
        """Estimate traffic potential for a keyword."""
        try:
            search_volume = keyword_row.get('Search Volume', 0)
            competitor_position = keyword_row.get('Position', 50)
            
            # Simple CTR model
            ctr_rates = {1: 0.284, 2: 0.147, 3: 0.094, 4: 0.067, 5: 0.051}
            estimated_ctr = ctr_rates.get(min(competitor_position, 5), 0.02)
            
            # Estimate potential if we achieve similar position
            potential_traffic = search_volume * estimated_ctr
            
            return potential_traffic
            
        except Exception:
            return 0.0

    def _calculate_implementation_priority(self, keyword_row: pd.Series) -> str:
        """Calculate implementation priority for a keyword."""
        try:
            difficulty = keyword_row.get('Keyword Difficulty', 50)
            traffic_potential = self._estimate_traffic_potential(keyword_row)
            
            if traffic_potential > 1000 and difficulty < 30:
                return 'high'
            elif traffic_potential > 500 and difficulty < 50:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'low'

    def _assess_competitive_advantage(
        self,
        keyword: str,
        processed_datasets: Dict[str, pd.DataFrame]
    ) -> str:
        """Assess competitive advantage for a keyword."""
        try:
            # Count how many competitors rank for this keyword
            competitor_count = 0
            
            for competitor, df in processed_datasets.items():
                if competitor == 'lenovo':
                    continue
                
                if keyword in df['Keyword'].values:
                    competitor_count += 1
            
            if competitor_count <= 1:
                return 'high'
            elif competitor_count <= 3:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'medium'

    def _determine_trend_direction(self, trend_model: Dict[str, Any]) -> str:
        """Determine trend direction from model."""
        try:
            if trend_model.get('model_type') == 'linear':
                slope = trend_model.get('slope', 0)
                if slope > 0.1:
                    return 'increasing'
                elif slope < -0.1:
                    return 'decreasing'
                else:
                    return 'stable'
            
            return 'unknown'
            
        except Exception:
            return 'unknown'

    def _calculate_forecast_confidence(self, trend_model: Dict[str, Any]) -> float:
        """Calculate confidence level for forecasts."""
        try:
            r_squared = trend_model.get('r_squared', 0)
            return min(r_squared, 1.0)
            
        except Exception:
            return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()
