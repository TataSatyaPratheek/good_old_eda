"""
SERP Feature Mapping Module for SEO Competitive Intelligence
Handles SERP feature detection, mapping, and competitive analysis with advanced categorization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

# Import our utilities to eliminate redundancy
from src.utils.common_helpers import StringHelper, memoize, timing_decorator, ensure_list
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.logging_utils import LoggerFactory, PerformanceTracker
from src.utils.config_utils import ConfigManager
from src.utils.math_utils import StatisticalCalculator
from src.utils.validation_utils import SchemaValidator

@dataclass
class SERPFeatureAnalysis:
    """Data class for SERP feature analysis results"""
    total_features: int
    feature_distribution: Dict[str, int]
    competitive_coverage: Dict[str, Dict[str, float]]
    feature_overlap_matrix: pd.DataFrame
    opportunity_score: float
    recommendations: List[str]

class SERPFeatureMapper:
    """
    Advanced SERP feature mapping and analysis for SEO competitive intelligence.
    
    Handles detection, categorization, and competitive analysis of SERP features
    with sophisticated pattern matching and opportunity identification.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities instead of custom implementations."""
        self.logger = logger or LoggerFactory.get_logger("serp_feature_mapper")
        self.config = config_manager or ConfigManager()
        
        # Initialize utility classes
        self.data_processor = DataProcessor(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.performance_tracker = PerformanceTracker(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.schema_validator = SchemaValidator(self.logger)
        
        # Load SERP feature mappings from config instead of hardcoding
        self.feature_mappings = self._load_feature_mappings()
        self.feature_categories = self._load_feature_categories()
        self.competitive_weights = self._load_competitive_weights()

    @timing_decorator()
    @memoize(ttl=1800)  # Cache for 30 minutes
    def map_serp_features(
        self,
        df: pd.DataFrame,
        features_column: str = 'SERP Features by Keyword'
    ) -> pd.DataFrame:
        """
        Map and standardize SERP features using string utilities.
        
        Args:
            df: DataFrame with SERP features data
            features_column: Column containing SERP features
            
        Returns:
            DataFrame with mapped and standardized features
        """
        try:
            with self.performance_tracker.track_block("map_serp_features"):
                # Validate data first
                validation_report = self.data_validator.validate_seo_dataset(df, 'positions')
                if validation_report.quality_score < 0.7:
                    self.logger.warning(f"Low data quality score: {validation_report.quality_score:.3f}")
                
                # Clean data using our data processor
                cleaned_df = self.data_processor.clean_seo_data(df)
                
                if features_column not in cleaned_df.columns:
                    self.logger.error(f"Column '{features_column}' not found")
                    return cleaned_df
                
                # Process SERP features using StringHelper
                cleaned_df['normalized_serp_features'] = cleaned_df[features_column].apply(
                    lambda x: StringHelper.normalize_serp_features(x) if pd.notna(x) else []
                )
                
                # Map features to standardized names
                cleaned_df['mapped_features'] = cleaned_df['normalized_serp_features'].apply(
                    self._map_features_to_standard
                )
                
                # Add feature categories
                cleaned_df['feature_categories'] = cleaned_df['mapped_features'].apply(
                    self._categorize_features
                )
                
                # Calculate feature counts and metrics
                cleaned_df['feature_count'] = cleaned_df['mapped_features'].apply(len)
                cleaned_df['has_rich_features'] = cleaned_df['mapped_features'].apply(
                    lambda x: any(feature in self.feature_categories.get('rich_results', []) for feature in x)
                )
                
                self.logger.info(f"Mapped SERP features for {len(cleaned_df)} keywords")
                return cleaned_df
                
        except Exception as e:
            self.logger.error(f"Error mapping SERP features: {str(e)}")
            return df

    @timing_decorator()
    def analyze_competitive_serp_landscape(
        self,
        competitor_data: Dict[str, pd.DataFrame],
        lenovo_data: pd.DataFrame,
        features_column: str = 'SERP Features by Keyword'
    ) -> SERPFeatureAnalysis:
        """
        Analyze competitive SERP feature landscape using statistical utilities.
        
        Args:
            competitor_data: Dictionary of competitor DataFrames
            lenovo_data: Lenovo SERP features data
            features_column: Column containing SERP features
            
        Returns:
            SERPFeatureAnalysis with comprehensive competitive analysis
        """
        try:
            with self.performance_tracker.track_block("analyze_competitive_serp_landscape"):
                # Process all datasets
                all_datasets = {'lenovo': lenovo_data, **competitor_data}
                processed_datasets = {}
                
                for competitor, df in all_datasets.items():
                    processed_df = self.map_serp_features(df, features_column)
                    processed_datasets[competitor] = processed_df
                
                # Extract all unique features
                all_features = set()
                for df in processed_datasets.values():
                    for feature_list in df['mapped_features'].dropna():
                        all_features.update(feature_list)
                
                # Calculate feature distribution using statistical utilities
                feature_counts = Counter()
                for df in processed_datasets.values():
                    for feature_list in df['mapped_features'].dropna():
                        feature_counts.update(feature_list)
                
                # Use StatisticalCalculator for distribution analysis
                feature_values = list(feature_counts.values())
                distribution_stats = self.stats_calculator.calculate_descriptive_statistics(
                    feature_values, include_advanced=True
                )
                
                # Calculate competitive coverage
                competitive_coverage = {}
                for competitor, df in processed_datasets.items():
                    coverage = {}
                    total_keywords = len(df)
                    
                    for feature in all_features:
                        feature_presence = df['mapped_features'].apply(
                            lambda x: feature in x if isinstance(x, list) else False
                        )
                        coverage[feature] = feature_presence.sum() / total_keywords if total_keywords > 0 else 0
                    
                    competitive_coverage[competitor] = coverage
                
                # Create feature overlap matrix
                overlap_matrix = self._calculate_feature_overlap_matrix(
                    processed_datasets, all_features
                )
                
                # Calculate opportunity score using our statistical calculator
                opportunity_score = self._calculate_serp_opportunity_score(
                    processed_datasets['lenovo'], competitive_coverage
                )
                
                # Generate recommendations
                recommendations = self._generate_serp_recommendations(
                    competitive_coverage, opportunity_score
                )
                
                analysis = SERPFeatureAnalysis(
                    total_features=len(all_features),
                    feature_distribution=dict(feature_counts),
                    competitive_coverage=competitive_coverage,
                    feature_overlap_matrix=overlap_matrix,
                    opportunity_score=opportunity_score,
                    recommendations=recommendations
                )
                
                self.logger.info(f"SERP competitive analysis completed for {len(competitor_data)} competitors")
                return analysis
                
        except Exception as e:
            self.logger.error(f"Error in competitive SERP analysis: {str(e)}")
            return SERPFeatureAnalysis(0, {}, {}, pd.DataFrame(), 0.0, [])

    @timing_decorator()
    def identify_serp_opportunities(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        min_opportunity_score: float = 0.6
    ) -> List[Dict[str, any]]:
        """
        Identify SERP feature opportunities using competitive analysis.
        
        Args:
            lenovo_data: Lenovo SERP data
            competitor_data: Competitor SERP data
            min_opportunity_score: Minimum score for opportunities
            
        Returns:
            List of SERP feature opportunities
        """
        try:
            # Analyze competitive landscape
            analysis = self.analyze_competitive_serp_landscape(
                competitor_data, lenovo_data
            )
            
            opportunities = []
            lenovo_coverage = analysis.competitive_coverage.get('lenovo', {})
            
            # Find features where competitors have high coverage but Lenovo doesn't
            for feature, competitors_coverage in analysis.competitive_coverage.items():
                if feature == 'lenovo':
                    continue
                
                lenovo_feature_coverage = lenovo_coverage.get(feature, 0)
                
                # Calculate average competitor coverage
                competitor_coverages = [
                    coverage.get(feature, 0) 
                    for comp_name, coverage in competitors_coverage.items() 
                    if comp_name != 'lenovo'
                ]
                
                if competitor_coverages:
                    # Use statistical calculator for robust average
                    avg_competitor_coverage = self.stats_calculator.calculate_descriptive_statistics(
                        competitor_coverages
                    ).get('mean', 0)
                    
                    # Calculate opportunity score
                    coverage_gap = max(0, avg_competitor_coverage - lenovo_feature_coverage)
                    opportunity_score = coverage_gap * self.competitive_weights.get(feature, 1.0)
                    
                    if opportunity_score >= min_opportunity_score:
                        opportunity = {
                            'feature': feature,
                            'opportunity_score': opportunity_score,
                            'lenovo_coverage': lenovo_feature_coverage,
                            'avg_competitor_coverage': avg_competitor_coverage,
                            'coverage_gap': coverage_gap,
                            'category': self._get_feature_category(feature),
                            'priority': self._calculate_feature_priority(feature, opportunity_score),
                            'implementation_difficulty': self._estimate_implementation_difficulty(feature),
                            'potential_impact': self._estimate_potential_impact(feature, coverage_gap)
                        }
                        opportunities.append(opportunity)
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
            self.logger.info(f"Identified {len(opportunities)} SERP feature opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying SERP opportunities: {str(e)}")
            return []

    def _load_feature_mappings(self) -> Dict[str, str]:
        """Load feature mappings from configuration."""
        try:
            # Try to get from config, fallback to default
            analysis_config = self.config.get_analysis_config()
            return getattr(analysis_config, 'serp_feature_mappings', self._get_default_feature_mappings())
        except Exception:
            return self._get_default_feature_mappings()

    def _get_default_feature_mappings(self) -> Dict[str, str]:
        """Default SERP feature mappings."""
        return {
            'featured snippet': 'featured_snippet',
            'featured snippets': 'featured_snippet',
            'people also ask': 'people_also_ask',
            'paa': 'people_also_ask',
            'knowledge panel': 'knowledge_panel',
            'knowledge graph': 'knowledge_panel',
            'image pack': 'image_pack',
            'images': 'image_pack',
            'video carousel': 'video_carousel',
            'video': 'video_carousel',
            'videos': 'video_carousel',
            'shopping results': 'shopping_results',
            'shopping': 'shopping_results',
            'ads': 'ads_top',
            'google ads': 'ads_top',
            'sitelinks': 'sitelinks',
            'site links': 'sitelinks',
            'reviews': 'reviews',
            'review stars': 'reviews',
            'local pack': 'local_pack',
            'map pack': 'local_pack',
            'related searches': 'related_searches',
            'related questions': 'people_also_ask'
        }

    def _load_feature_categories(self) -> Dict[str, List[str]]:
        """Load feature categories from configuration."""
        try:
            analysis_config = self.config.get_analysis_config()
            return getattr(analysis_config, 'serp_feature_categories', self._get_default_feature_categories())
        except Exception:
            return self._get_default_feature_categories()

    def _get_default_feature_categories(self) -> Dict[str, List[str]]:
        """Default SERP feature categories."""
        return {
            'rich_results': [
                'featured_snippet', 'knowledge_panel', 'image_pack', 
                'video_carousel', 'shopping_results', 'reviews'
            ],
            'navigational': ['sitelinks', 'knowledge_panel'],
            'commercial': ['shopping_results', 'ads_top', 'reviews'],
            'informational': [
                'featured_snippet', 'people_also_ask', 'video_carousel', 
                'image_pack', 'related_searches'
            ],
            'local': ['local_pack', 'reviews']
        }

    def _load_competitive_weights(self) -> Dict[str, float]:
        """Load competitive weights from configuration."""
        try:
            analysis_config = self.config.get_analysis_config()
            return getattr(analysis_config, 'serp_competitive_weights', self._get_default_competitive_weights())
        except Exception:
            return self._get_default_competitive_weights()

    def _get_default_competitive_weights(self) -> Dict[str, float]:
        """Default competitive weights for SERP features."""
        return {
            'featured_snippet': 2.0,
            'knowledge_panel': 1.8,
            'shopping_results': 1.6,
            'image_pack': 1.4,
            'video_carousel': 1.3,
            'people_also_ask': 1.2,
            'sitelinks': 1.1,
            'reviews': 1.5,
            'ads_top': 1.0,
            'local_pack': 1.7,
            'related_searches': 0.8
        }

    def _map_features_to_standard(self, features: List[str]) -> List[str]:
        """Map feature list to standardized names."""
        if not features:
            return []
        
        mapped = []
        for feature in features:
            # Use the feature mappings to standardize
            standardized = self.feature_mappings.get(feature.lower(), feature)
            if standardized not in mapped:
                mapped.append(standardized)
        
        return mapped

    def _categorize_features(self, features: List[str]) -> List[str]:
        """Categorize features into high-level categories."""
        if not features:
            return []
        
        categories = set()
        for feature in features:
            for category, category_features in self.feature_categories.items():
                if feature in category_features:
                    categories.add(category)
        
        return list(categories)

    def _calculate_feature_overlap_matrix(
        self,
        processed_datasets: Dict[str, pd.DataFrame],
        all_features: Set[str]
    ) -> pd.DataFrame:
        """Calculate feature overlap matrix between competitors."""
        try:
            competitors = list(processed_datasets.keys())
            overlap_matrix = pd.DataFrame(
                index=competitors,
                columns=competitors,
                dtype=float
            )
            
            for i, comp1 in enumerate(competitors):
                for j, comp2 in enumerate(competitors):
                    if i <= j:
                        if i == j:
                            overlap_matrix.loc[comp1, comp2] = 1.0
                        else:
                            # Calculate Jaccard similarity
                            features1 = set()
                            features2 = set()
                            
                            for feature_list in processed_datasets[comp1]['mapped_features'].dropna():
                                features1.update(feature_list)
                            
                            for feature_list in processed_datasets[comp2]['mapped_features'].dropna():
                                features2.update(feature_list)
                            
                            if features1 or features2:
                                intersection = len(features1.intersection(features2))
                                union = len(features1.union(features2))
                                jaccard = intersection / union if union > 0 else 0
                            else:
                                jaccard = 0
                            
                            overlap_matrix.loc[comp1, comp2] = jaccard
                            overlap_matrix.loc[comp2, comp1] = jaccard
            
            return overlap_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating overlap matrix: {str(e)}")
            return pd.DataFrame()

    def _calculate_serp_opportunity_score(
        self,
        lenovo_data: pd.DataFrame,
        competitive_coverage: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate overall SERP opportunity score."""
        try:
            if not competitive_coverage or 'lenovo' not in competitive_coverage:
                return 0.0
            
            lenovo_coverage = competitive_coverage['lenovo']
            competitor_coverages = {
                comp: coverage for comp, coverage in competitive_coverage.items() 
                if comp != 'lenovo'
            }
            
            if not competitor_coverages:
                return 0.0
            
            opportunity_scores = []
            
            for feature in lenovo_coverage.keys():
                lenovo_feature_coverage = lenovo_coverage[feature]
                
                # Calculate max competitor coverage for this feature
                max_competitor_coverage = max(
                    coverage.get(feature, 0) for coverage in competitor_coverages.values()
                )
                
                # Calculate opportunity (gap * weight)
                gap = max(0, max_competitor_coverage - lenovo_feature_coverage)
                weight = self.competitive_weights.get(feature, 1.0)
                opportunity = gap * weight
                
                opportunity_scores.append(opportunity)
            
            # Use statistical calculator for robust average
            if opportunity_scores:
                stats = self.stats_calculator.calculate_descriptive_statistics(opportunity_scores)
                return stats.get('mean', 0.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.0

    def _generate_serp_recommendations(
        self,
        competitive_coverage: Dict[str, Dict[str, float]],
        opportunity_score: float
    ) -> List[str]:
        """Generate actionable SERP recommendations."""
        recommendations = []
        
        try:
            if opportunity_score > 0.7:
                recommendations.append("High SERP feature opportunity detected - prioritize feature optimization")
            elif opportunity_score > 0.4:
                recommendations.append("Moderate SERP opportunities available - consider targeted improvements")
            
            # Feature-specific recommendations
            if 'lenovo' in competitive_coverage:
                lenovo_coverage = competitive_coverage['lenovo']
                
                # Check for specific low-coverage features
                if lenovo_coverage.get('featured_snippet', 0) < 0.2:
                    recommendations.append("Focus on featured snippet optimization for key informational queries")
                
                if lenovo_coverage.get('shopping_results', 0) < 0.3:
                    recommendations.append("Enhance product data and shopping feed optimization")
                
                if lenovo_coverage.get('image_pack', 0) < 0.25:
                    recommendations.append("Improve image SEO and visual content strategy")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Review SERP feature performance and competitive positioning"]

    def _get_feature_category(self, feature: str) -> str:
        """Get category for a specific feature."""
        for category, features in self.feature_categories.items():
            if feature in features:
                return category
        return 'other'

    def _calculate_feature_priority(self, feature: str, opportunity_score: float) -> str:
        """Calculate priority level for feature optimization."""
        weight = self.competitive_weights.get(feature, 1.0)
        weighted_score = opportunity_score * weight
        
        if weighted_score > 1.5:
            return 'high'
        elif weighted_score > 0.8:
            return 'medium'
        else:
            return 'low'

    def _estimate_implementation_difficulty(self, feature: str) -> str:
        """Estimate implementation difficulty for feature."""
        difficulty_map = {
            'featured_snippet': 'medium',
            'sitelinks': 'easy',
            'reviews': 'medium', 
            'image_pack': 'easy',
            'video_carousel': 'medium',
            'shopping_results': 'hard',
            'knowledge_panel': 'hard',
            'people_also_ask': 'medium',
            'local_pack': 'medium'
        }
        return difficulty_map.get(feature, 'medium')

    def _estimate_potential_impact(self, feature: str, coverage_gap: float) -> str:
        """Estimate potential impact of feature optimization."""
        weight = self.competitive_weights.get(feature, 1.0)
        impact_score = coverage_gap * weight
        
        if impact_score > 1.0:
            return 'high'
        elif impact_score > 0.5:
            return 'medium'
        else:
            return 'low'

    def get_performance_summary(self) -> Dict[str, any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()
