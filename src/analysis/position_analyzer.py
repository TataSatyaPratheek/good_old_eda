"""
Position Analysis Module for SEO Competitive Intelligence
Handles ranking analysis, position volatility, and competitive positioning metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Import our utilities to eliminate redundancy
from src.utils.common_helpers import StringHelper, memoize, timing_decorator
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.math_utils import StatisticalCalculator
from src.utils.logging_utils import LoggerFactory, PerformanceTracker
from src.utils.config_utils import ConfigManager

@dataclass
class PositionMetrics:
    """Data class for position analysis results"""
    avg_position: float
    position_volatility: float
    top_3_ratio: float
    top_10_ratio: float
    position_distribution: Dict[str, int]
    branded_vs_nonbranded: Dict[str, float]

class PositionAnalyzer:
    """
    Advanced position analysis for SEO competitive intelligence.
    
    Implements traffic concentration analysis, position stability metrics,
    and competitive positioning assessment based on SEMrush data.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config_manager: Optional[ConfigManager] = None):
        """Initialize the position analyzer with utilities."""
        self.logger = logger or LoggerFactory.get_logger("position_analyzer")
        self.config = config_manager or ConfigManager()
        
        # Initialize utility classes
        self.data_processor = DataProcessor(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.performance_tracker = PerformanceTracker(self.logger)
        
        # Get brand keywords from config instead of hardcoding
        analysis_config = self.config.get_analysis_config()
        self.brand_keywords = ['lenovo', 'thinkpad', 'legion', 'ideapad']  # Could be from config

    @timing_decorator()
    @memoize(ttl=3600)  # Cache results for 1 hour
    def calculate_position_stability_index(
        self,
        df: pd.DataFrame,
        position_col: str = 'Position',
        prev_position_col: str = 'Previous position'
    ) -> float:
        """
        Calculate Position Stability Index (PSI) - percentage of keywords
        maintaining rank ±2 positions.
        
        Formula: PSI = (Keywords with |current_pos - prev_pos| <= 2) / Total Keywords
        """
        try:
            # Use data validator to check data quality first
            validation_report = self.data_validator.validate_seo_dataset(df, 'positions')
            if validation_report.quality_score < 0.7:
                self.logger.warning(f"Low data quality score: {validation_report.quality_score:.3f}")
            
            # Handle missing previous position data using our data processor
            valid_data = df.dropna(subset=[position_col, prev_position_col])
            if valid_data.empty:
                self.logger.warning("No valid position data for stability calculation")
                return 0.0

            position_diff = abs(valid_data[position_col] - valid_data[prev_position_col])
            stable_keywords = (position_diff <= 2).sum()
            psi = stable_keywords / len(valid_data)
            
            self.logger.info(f"Position Stability Index: {psi:.3f}")
            return psi
            
        except Exception as e:
            self.logger.error(f"Error calculating PSI: {str(e)}")
            return 0.0

    @timing_decorator()
    def calculate_traffic_concentration_gini(
        self,
        df: pd.DataFrame,
        traffic_pct_col: str = 'Traffic (%)'
    ) -> float:
        """
        Calculate Gini coefficient for traffic concentration analysis using our math utils.
        Formula: G = Σ|xi - xj| / (2n²μ)
        Higher Gini = more concentrated traffic (risky)
        """
        try:
            traffic_shares = df[traffic_pct_col].dropna().values
            if len(traffic_shares) == 0:
                return 0.0

            # Use our statistical calculator for more robust calculation
            stats_dict = self.stats_calculator.calculate_descriptive_statistics(traffic_shares)
            
            n = len(traffic_shares)
            mean_traffic = stats_dict.get('mean', 0)
            
            if mean_traffic == 0:
                return 0.0

            # Calculate Gini coefficient
            diff_sum = sum(abs(xi - xj) for xi in traffic_shares for xj in traffic_shares)
            gini = diff_sum / (2 * n * n * mean_traffic)
            
            self.logger.info(f"Traffic Gini Coefficient: {gini:.3f}")
            return gini
            
        except Exception as e:
            self.logger.error(f"Error calculating Gini coefficient: {str(e)}")
            return 0.0

    @timing_decorator()
    def analyze_branded_vs_nonbranded(
        self,
        df: pd.DataFrame,
        keyword_col: str = 'Keyword',
        traffic_pct_col: str = 'Traffic (%)'
    ) -> Dict[str, float]:
        """
        Analyze branded vs non-branded keyword performance using string utilities.
        Returns ratio of traffic from branded vs non-branded terms.
        """
        try:
            df_clean = df.dropna(subset=[keyword_col, traffic_pct_col])
            
            # Use StringHelper to extract brand keywords
            keywords_list = df_clean[keyword_col].tolist()
            branded_keywords, nonbranded_keywords = StringHelper.extract_brand_keywords(
                keywords_list, self.brand_keywords
            )
            
            # Create masks for branded vs non-branded
            branded_mask = df_clean[keyword_col].isin(branded_keywords)
            
            branded_traffic = df_clean.loc[branded_mask, traffic_pct_col].sum()
            nonbranded_traffic = df_clean.loc[~branded_mask, traffic_pct_col].sum()
            total_traffic = branded_traffic + nonbranded_traffic

            if total_traffic == 0:
                return {'branded_ratio': 0.0, 'nonbranded_ratio': 0.0}

            result = {
                'branded_ratio': branded_traffic / total_traffic,
                'nonbranded_ratio': nonbranded_traffic / total_traffic,
                'branded_traffic_pct': branded_traffic,
                'nonbranded_traffic_pct': nonbranded_traffic
            }
            
            self.logger.info(f"Branded traffic ratio: {result['branded_ratio']:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in branded analysis: {str(e)}")
            return {'branded_ratio': 0.0, 'nonbranded_ratio': 0.0}

    @timing_decorator()
    def calculate_competitive_pressure_index(
        self,
        df: pd.DataFrame,
        position_col: str = 'Position',
        kd_col: str = 'Keyword Difficulty',
        volume_col: str = 'Search Volume'
    ) -> pd.Series:
        """
        Calculate Competitive Pressure Index for each keyword.
        Formula: CPI = (KD/100) × log(Search Volume) × (11 - Position) / 10
        Higher CPI = more competitive pressure
        """
        try:
            # Clean data using our data processor
            cleaned_df = self.data_processor.clean_seo_data(df)
            df_clean = cleaned_df.dropna(subset=[position_col, kd_col, volume_col])
            
            # Avoid log(0) by adding 1 to search volume
            log_volume = np.log1p(df_clean[volume_col])
            position_factor = (11 - df_clean[position_col].clip(1, 10)) / 10
            difficulty_factor = df_clean[kd_col] / 100
            
            cpi = difficulty_factor * log_volume * position_factor
            
            self.logger.info(f"CPI calculated for {len(cpi)} keywords")
            return cpi
            
        except Exception as e:
            self.logger.error(f"Error calculating CPI: {str(e)}")
            return pd.Series()

    def get_position_distribution(
        self,
        df: pd.DataFrame,
        position_col: str = 'Position'
    ) -> Dict[str, int]:
        """Get distribution of positions across ranking tiers."""
        try:
            positions = df[position_col].dropna()
            
            distribution = {
                'top_3': (positions <= 3).sum(),
                'top_5': (positions <= 5).sum(),
                'top_10': (positions <= 10).sum(),
                'top_20': (positions <= 20).sum(),
                'beyond_20': (positions > 20).sum()
            }
            
            self.logger.info(f"Position distribution calculated: {distribution}")
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error in position distribution: {str(e)}")
            return {}

    @timing_decorator()
    def analyze_keyword_intent_performance(
        self,
        df: pd.DataFrame,
        intent_col: str = 'Keyword Intents',
        position_col: str = 'Position',
        traffic_col: str = 'Traffic (%)'
    ) -> pd.DataFrame:
        """
        Analyze performance across different keyword intents.
        Returns aggregated metrics by intent type.
        """
        try:
            df_clean = df.dropna(subset=[intent_col, position_col, traffic_col])
            
            # Handle multiple intents (comma-separated)
            intent_analysis = []
            intents = ['navigational', 'informational', 'transactional', 'commercial']
            
            for intent in intents:
                intent_mask = df_clean[intent_col].str.contains(intent, na=False, case=False)
                intent_data = df_clean[intent_mask]
                
                if len(intent_data) > 0:
                    # Use our statistical calculator for robust stats
                    position_stats = self.stats_calculator.calculate_descriptive_statistics(
                        intent_data[position_col]
                    )
                    
                    analysis = {
                        'intent': intent,
                        'keyword_count': len(intent_data),
                        'avg_position': position_stats.get('mean', 0),
                        'position_std': position_stats.get('std', 0),
                        'total_traffic_pct': intent_data[traffic_col].sum(),
                        'top_3_ratio': (intent_data[position_col] <= 3).mean(),
                        'avg_search_volume': intent_data.get('Search Volume', pd.Series()).mean()
                    }
                    intent_analysis.append(analysis)

            result_df = pd.DataFrame(intent_analysis)
            self.logger.info(f"Intent analysis completed for {len(result_df)} intents")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in intent analysis: {str(e)}")
            return pd.DataFrame()

    @timing_decorator()
    def comprehensive_position_analysis(
        self,
        df: pd.DataFrame
    ) -> PositionMetrics:
        """
        Perform comprehensive position analysis returning all key metrics.
        Returns PositionMetrics dataclass with all calculated metrics.
        """
        try:
            with self.performance_tracker.track_block("comprehensive_position_analysis"):
                # Clean and validate data first
                cleaned_df = self.data_processor.clean_seo_data(df)
                
                # Calculate all metrics using our optimized methods
                positions = cleaned_df['Position'].dropna() if 'Position' in cleaned_df.columns else pd.Series()
                
                # Use statistical calculator for robust calculations
                if len(positions) > 0:
                    position_stats = self.stats_calculator.calculate_descriptive_statistics(positions)
                    avg_position = position_stats.get('mean', 0.0)
                    top_3_ratio = (positions <= 3).mean()
                    top_10_ratio = (positions <= 10).mean()
                else:
                    avg_position = 0.0
                    top_3_ratio = 0.0
                    top_10_ratio = 0.0
                
                position_volatility = self.calculate_position_stability_index(cleaned_df)
                position_distribution = self.get_position_distribution(cleaned_df)
                branded_analysis = self.analyze_branded_vs_nonbranded(cleaned_df)

                metrics = PositionMetrics(
                    avg_position=avg_position,
                    position_volatility=position_volatility,
                    top_3_ratio=top_3_ratio,
                    top_10_ratio=top_10_ratio,
                    position_distribution=position_distribution,
                    branded_vs_nonbranded=branded_analysis
                )
                
                self.logger.info("Comprehensive position analysis completed")
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            return PositionMetrics(0.0, 0.0, 0.0, 0.0, {}, {})

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()
