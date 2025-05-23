"""
Competitive Features Module for SEO Competitive Intelligence
Advanced competitive analysis features leveraging the comprehensive utility framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Import our utilities to eliminate ALL redundancy
from src.utils.common_helpers import StringHelper, DateHelper, memoize, timing_decorator, safe_divide, ensure_list, deep_merge_dicts
from src.utils.data_utils import DataProcessor, DataValidator, DataTransformer
from src.utils.math_utils import StatisticalCalculator, OptimizationHelper, TimeSeriesAnalyzer
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager, PathManager
from src.utils.validation_utils import SchemaValidator, BusinessRuleValidator
from src.utils.export_utils import ReportExporter, DataExporter
from src.utils.file_utils import FileManager, ExportManager

@dataclass
class CompetitorProfile:
    """Comprehensive competitor profile"""
    domain: str
    market_share: float
    organic_keywords: int
    organic_traffic: int
    competitive_strength: float
    growth_rate: float
    top_keywords: List[Dict[str, Any]]
    serp_feature_coverage: Dict[str, float]
    content_gaps: List[str]
    technical_advantages: List[str]

@dataclass
class MarketAnalysis:
    """Market analysis results"""
    total_market_size: int
    market_concentration: float
    growth_opportunities: List[Dict[str, Any]]
    competitive_threats: List[Dict[str, Any]]
    market_leaders: List[str]
    emerging_competitors: List[str]
    keyword_distribution: Dict[str, int]

@dataclass
class CompetitiveIntelligence:
    """Comprehensive competitive intelligence"""
    competitor_profiles: Dict[str, CompetitorProfile]
    market_analysis: MarketAnalysis
    competitive_positioning: Dict[str, Any]
    strategic_recommendations: List[str]
    opportunity_matrix: pd.DataFrame
    threat_assessment: Dict[str, Any]

class CompetitiveFeatures:
    """
    Advanced competitive features for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide competitive analysis,
    market research, and strategic intelligence without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("competitive_features")
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
        self.market_share_threshold = 0.05

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def comprehensive_competitor_analysis(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        include_market_analysis: bool = True,
        analysis_depth: str = 'comprehensive'
    ) -> CompetitiveIntelligence:
        """
        Perform comprehensive competitor analysis using utility framework.
        
        Args:
            lenovo_data: Lenovo's SEO data
            competitor_data: Dictionary of competitor data
            include_market_analysis: Whether to include market analysis
            analysis_depth: Analysis depth level
            
        Returns:
            CompetitiveIntelligence with comprehensive insights
        """
        try:
            with self.performance_tracker.track_block("comprehensive_competitor_analysis"):
                # Audit log the analysis
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="competitive_analysis",
                    parameters={
                        "include_market_analysis": include_market_analysis,
                        "analysis_depth": analysis_depth,
                        "competitors_count": len(competitor_data)
                    }
                )
                
                # Clean and validate all datasets using DataProcessor
                cleaned_lenovo = self.data_processor.clean_seo_data(lenovo_data)
                cleaned_competitors = {}
                
                for competitor, df in competitor_data.items():
                    cleaned_df = self.data_processor.clean_seo_data(df)
                    cleaned_competitors[competitor] = cleaned_df
                
                # Build comprehensive competitor profiles
                competitor_profiles = self._build_competitor_profiles(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Perform market analysis if requested
                market_analysis = None
                if include_market_analysis:
                    market_analysis = self._perform_market_analysis(
                        cleaned_lenovo, cleaned_competitors
                    )
                
                # Calculate competitive positioning using statistical analysis
                competitive_positioning = self._calculate_competitive_positioning(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Generate strategic recommendations using optimization
                strategic_recommendations = self._generate_strategic_recommendations(
                    competitor_profiles, market_analysis, competitive_positioning
                )
                
                # Create opportunity matrix using mathematical optimization
                opportunity_matrix = self._create_opportunity_matrix(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Assess competitive threats using business rules
                threat_assessment = self._assess_competitive_threats(
                    competitor_profiles, market_analysis
                )
                
                intelligence = CompetitiveIntelligence(
                    competitor_profiles=competitor_profiles,
                    market_analysis=market_analysis,
                    competitive_positioning=competitive_positioning,
                    strategic_recommendations=strategic_recommendations,
                    opportunity_matrix=opportunity_matrix,
                    threat_assessment=threat_assessment
                )
                
                self.logger.info(f"Comprehensive competitor analysis completed for {len(competitor_data)} competitors")
                return intelligence
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive competitor analysis: {str(e)}")
            return CompetitiveIntelligence({}, None, {}, [], pd.DataFrame(), {})

    @timing_decorator()
    def analyze_competitor_content_gaps(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        gap_analysis_method: str = 'keyword_overlap'
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze content gaps using advanced text analysis and statistical methods.
        
        Args:
            lenovo_data: Lenovo's content/keyword data
            competitor_data: Competitor content data
            gap_analysis_method: Method for gap analysis
            
        Returns:
            Dictionary of content gaps by competitor
        """
        try:
            with self.performance_tracker.track_block("analyze_competitor_content_gaps"):
                content_gaps = {}
                
                # Extract Lenovo keywords using StringHelper
                lenovo_keywords = set()
                if 'Keyword' in lenovo_data.columns:
                    lenovo_keywords = set(
                        StringHelper.clean_keyword(kw) 
                        for kw in lenovo_data['Keyword'].dropna().tolist()
                    )
                
                for competitor, comp_df in competitor_data.items():
                    self.logger.info(f"Analyzing content gaps for {competitor}")
                    
                    # Extract competitor keywords
                    comp_keywords = set()
                    if 'Keyword' in comp_df.columns:
                        comp_keywords = set(
                            StringHelper.clean_keyword(kw) 
                            for kw in comp_df['Keyword'].dropna().tolist()
                        )
                    
                    # Find gap keywords (competitor has, Lenovo doesn't)
                    gap_keywords = comp_keywords - lenovo_keywords
                    
                    # Analyze gaps using different methods
                    if gap_analysis_method == 'keyword_overlap':
                        gaps = self._analyze_keyword_overlap_gaps(
                            gap_keywords, comp_df, lenovo_data
                        )
                    elif gap_analysis_method == 'semantic_similarity':
                        gaps = self._analyze_semantic_content_gaps(
                            gap_keywords, comp_df, lenovo_keywords
                        )
                    else:
                        gaps = self._analyze_statistical_content_gaps(
                            gap_keywords, comp_df, lenovo_data
                        )
                    
                    content_gaps[competitor] = gaps
                
                self.logger.info(f"Content gap analysis completed for {len(competitor_data)} competitors")
                return content_gaps
                
        except Exception as e:
            self.logger.error(f"Error in content gap analysis: {str(e)}")
            return {}

    @timing_decorator()
    def track_competitor_serp_feature_adoption(
        self,
        competitor_data: Dict[str, pd.DataFrame],
        features_column: str = 'SERP Features by Keyword',
        time_series_analysis: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Track SERP feature adoption across competitors using time series analysis.
        
        Args:
            competitor_data: Competitor data with SERP features
            features_column: Column containing SERP features
            time_series_analysis: Whether to perform time series analysis
            
        Returns:
            SERP feature adoption analysis by competitor
        """
        try:
            with self.performance_tracker.track_block("track_competitor_serp_feature_adoption"):
                adoption_analysis = {}
                
                for competitor, df in competitor_data.items():
                    if features_column not in df.columns:
                        continue
                    
                    # Process SERP features using StringHelper
                    serp_features_data = []
                    for _, row in df.iterrows():
                        features_str = row[features_column]
                        if pd.notna(features_str):
                            normalized_features = StringHelper.normalize_serp_features(features_str)
                            serp_features_data.extend(normalized_features)
                    
                    # Count feature frequency
                    from collections import Counter
                    feature_counts = Counter(serp_features_data)
                    
                    # Calculate adoption metrics using StatisticalCalculator
                    total_keywords = len(df)
                    feature_adoption = {}
                    
                    for feature, count in feature_counts.items():
                        adoption_rate = safe_divide(count, total_keywords, 0.0)
                        feature_adoption[feature] = {
                            'adoption_rate': adoption_rate,
                            'keyword_count': count,
                            'competitive_advantage': self._calculate_feature_competitive_advantage(
                                feature, adoption_rate, competitor_data
                            )
                        }
                    
                    # Time series analysis if requested and date data available
                    trend_analysis = {}
                    if time_series_analysis and 'date' in df.columns:
                        trend_analysis = self._analyze_serp_feature_trends(
                            df, features_column
                        )
                    
                    adoption_analysis[competitor] = {
                        'feature_adoption': feature_adoption,
                        'total_features_used': len(feature_counts),
                        'feature_diversity_score': self._calculate_feature_diversity_score(feature_counts),
                        'trend_analysis': trend_analysis,
                        'competitive_positioning': self._assess_serp_competitive_positioning(
                            feature_adoption, competitor_data
                        )
                    }
                
                self.logger.info(f"SERP feature adoption analysis completed for {len(competitor_data)} competitors")
                return adoption_analysis
                
        except Exception as e:
            self.logger.error(f"Error in SERP feature adoption analysis: {str(e)}")
            return {}

    @timing_decorator()
    def competitive_keyword_clustering(
        self,
        competitor_data: Dict[str, pd.DataFrame],
        clustering_method: str = 'semantic',
        n_clusters: int = 10
    ) -> Dict[str, Any]:
        """
        Perform competitive keyword clustering using advanced ML techniques.
        
        Args:
            competitor_data: Dictionary of competitor keyword data
            clustering_method: Clustering method ('semantic', 'statistical', 'hybrid')
            n_clusters: Number of clusters
            
        Returns:
            Keyword clustering results with competitive insights
        """
        try:
            with self.performance_tracker.track_block("competitive_keyword_clustering"):
                # Combine all competitor keywords
                all_keywords = []
                keyword_metadata = []
                
                for competitor, df in competitor_data.items():
                    if 'Keyword' in df.columns:
                        for _, row in df.iterrows():
                            keyword = StringHelper.clean_keyword(row['Keyword'])
                            if keyword:
                                all_keywords.append(keyword)
                                keyword_metadata.append({
                                    'keyword': keyword,
                                    'competitor': competitor,
                                    'position': row.get('Position', 100),
                                    'search_volume': row.get('Search Volume', 0),
                                    'traffic': row.get('Traffic (%)', 0)
                                })
                
                if len(all_keywords) < n_clusters:
                    self.logger.warning("Insufficient keywords for clustering")
                    return {}
                
                # Perform clustering based on method
                if clustering_method == 'semantic':
                    cluster_results = self._semantic_keyword_clustering(
                        all_keywords, keyword_metadata, n_clusters
                    )
                elif clustering_method == 'statistical':
                    cluster_results = self._statistical_keyword_clustering(
                        all_keywords, keyword_metadata, n_clusters
                    )
                else:
                    cluster_results = self._hybrid_keyword_clustering(
                        all_keywords, keyword_metadata, n_clusters
                    )
                
                # Analyze competitive landscape within clusters
                competitive_cluster_analysis = self._analyze_competitive_clusters(
                    cluster_results, competitor_data
                )
                
                clustering_results = {
                    'clusters': cluster_results,
                    'competitive_analysis': competitive_cluster_analysis,
                    'cluster_opportunities': self._identify_cluster_opportunities(
                        cluster_results, competitive_cluster_analysis
                    ),
                    'market_coverage': self._calculate_market_coverage_by_cluster(
                        cluster_results, competitor_data
                    )
                }
                
                self.logger.info(f"Competitive keyword clustering completed: {len(cluster_results)} clusters")
                return clustering_results
                
        except Exception as e:
            self.logger.error(f"Error in competitive keyword clustering: {str(e)}")
            return {}

    def _build_competitor_profiles(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, CompetitorProfile]:
        """Build comprehensive competitor profiles using statistical analysis."""
        try:
            profiles = {}
            
            # Calculate total market metrics using StatisticalCalculator
            all_traffic = []
            all_keywords = []
            
            for df in [lenovo_data] + list(competitor_data.values()):
                if 'Traffic (%)' in df.columns:
                    all_traffic.extend(df['Traffic (%)'].dropna().tolist())
                all_keywords.extend(df.get('Keyword', pd.Series()).dropna().tolist())
            
            total_market_traffic = sum(all_traffic) if all_traffic else 1
            total_market_keywords = len(set(all_keywords))
            
            for competitor, df in competitor_data.items():
                # Basic metrics
                organic_keywords = len(df)
                organic_traffic = df.get('Traffic (%)', pd.Series()).sum()
                market_share = safe_divide(organic_traffic, total_market_traffic, 0.0)
                
                # Calculate competitive strength using multiple factors
                competitive_strength = self._calculate_competitive_strength(
                    df, lenovo_data, market_share
                )
                
                # Calculate growth rate using time series if available
                growth_rate = 0.0
                if 'date' in df.columns and len(df) > 10:
                    growth_rate = self._calculate_growth_rate(df)
                
                # Get top keywords
                top_keywords = self._get_top_competitive_keywords(df, lenovo_data)
                
                # SERP feature coverage
                serp_coverage = self._calculate_serp_feature_coverage(df)
                
                # Content gaps analysis
                content_gaps = self._identify_content_gaps(df, lenovo_data)
                
                # Technical advantages
                technical_advantages = self._assess_technical_advantages(df, lenovo_data)
                
                profile = CompetitorProfile(
                    domain=competitor,
                    market_share=market_share,
                    organic_keywords=organic_keywords,
                    organic_traffic=int(organic_traffic),
                    competitive_strength=competitive_strength,
                    growth_rate=growth_rate,
                    top_keywords=top_keywords,
                    serp_feature_coverage=serp_coverage,
                    content_gaps=content_gaps,
                    technical_advantages=technical_advantages
                )
                
                profiles[competitor] = profile
            
            return profiles
            
        except Exception as e:
            self.logger.error(f"Error building competitor profiles: {str(e)}")
            return {}

    def _perform_market_analysis(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> MarketAnalysis:
        """Perform comprehensive market analysis using statistical methods."""
        try:
            # Calculate total market size
            all_keywords = set()
            all_traffic = 0
            
            for df in [lenovo_data] + list(competitor_data.values()):
                all_keywords.update(df.get('Keyword', pd.Series()).dropna().tolist())
                all_traffic += df.get('Traffic (%)', pd.Series()).sum()
            
            total_market_size = len(all_keywords)
            
            # Calculate market concentration using Gini coefficient
            traffic_shares = [df.get('Traffic (%)', pd.Series()).sum() for df in competitor_data.values()]
            traffic_shares.append(lenovo_data.get('Traffic (%)', pd.Series()).sum())
            
            market_concentration = self._calculate_market_concentration(traffic_shares)
            
            # Identify growth opportunities using optimization
            growth_opportunities = self._identify_market_growth_opportunities(
                lenovo_data, competitor_data
            )
            
            # Assess competitive threats
            competitive_threats = self._assess_market_threats(
                lenovo_data, competitor_data
            )
            
            # Identify market leaders and emerging competitors
            market_leaders, emerging_competitors = self._identify_market_players(
                competitor_data
            )
            
            # Analyze keyword distribution
            keyword_distribution = self._analyze_keyword_distribution(
                lenovo_data, competitor_data
            )
            
            return MarketAnalysis(
                total_market_size=total_market_size,
                market_concentration=market_concentration,
                growth_opportunities=growth_opportunities,
                competitive_threats=competitive_threats,
                market_leaders=market_leaders,
                emerging_competitors=emerging_competitors,
                keyword_distribution=keyword_distribution
            )
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return MarketAnalysis(0, 0.0, [], [], [], [], {})

    def _calculate_competitive_strength(
        self,
        competitor_df: pd.DataFrame,
        lenovo_df: pd.DataFrame,
        market_share: float
    ) -> float:
        """Calculate competitive strength using multiple factors."""
        try:
            # Position advantage
            avg_position = competitor_df.get('Position', pd.Series()).mean()
            position_score = max(0, (50 - avg_position) / 50) if avg_position > 0 else 0
            
            # Traffic efficiency
            total_traffic = competitor_df.get('Traffic (%)', pd.Series()).sum()
            keyword_count = len(competitor_df)
            traffic_efficiency = safe_divide(total_traffic, keyword_count, 0.0)
            efficiency_score = min(traffic_efficiency / 10, 1.0)  # Normalize
            
            # Market share factor
            market_share_score = min(market_share * 10, 1.0)  # Normalize
            
            # Keyword overlap advantage
            competitor_keywords = set(competitor_df.get('Keyword', pd.Series()).dropna().tolist())
            lenovo_keywords = set(lenovo_df.get('Keyword', pd.Series()).dropna().tolist())
            
            overlap = len(competitor_keywords.intersection(lenovo_keywords))
            unique_keywords = len(competitor_keywords - lenovo_keywords)
            
            uniqueness_score = safe_divide(unique_keywords, len(competitor_keywords), 0.0)
            
            # Weighted combination
            competitive_strength = (
                position_score * 0.3 +
                efficiency_score * 0.25 +
                market_share_score * 0.25 +
                uniqueness_score * 0.2
            )
            
            return competitive_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating competitive strength: {str(e)}")
            return 0.0

    def _semantic_keyword_clustering(
        self,
        keywords: List[str],
        metadata: List[Dict[str, Any]],
        n_clusters: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic keyword clustering using NLP techniques."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Vectorize keywords
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Create features from keywords
            keyword_features = vectorizer.fit_transform(keywords)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(keyword_features)
            
            # Calculate clustering quality
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(keyword_features, cluster_labels)
            else:
                silhouette_avg = 0.0
            
            # Organize results by cluster
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_keywords = []
                cluster_metadata = []
                
                for i, label in enumerate(cluster_labels):
                    if label == cluster_id:
                        cluster_keywords.append(keywords[i])
                        cluster_metadata.append(metadata[i])
                
                if cluster_keywords:
                    # Analyze cluster characteristics
                    cluster_stats = self.stats_calculator.calculate_descriptive_statistics(
                        [meta['search_volume'] for meta in cluster_metadata]
                    )
                    
                    clusters.append({
                        'cluster_id': cluster_id,
                        'keywords': cluster_keywords,
                        'metadata': cluster_metadata,
                        'size': len(cluster_keywords),
                        'avg_search_volume': cluster_stats.get('mean', 0),
                        'avg_position': np.mean([meta['position'] for meta in cluster_metadata]),
                        'total_traffic': sum(meta['traffic'] for meta in cluster_metadata),
                        'dominant_competitors': self._get_dominant_competitors_in_cluster(cluster_metadata)
                    })
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in semantic clustering: {str(e)}")
            return []

    def export_competitive_analysis_results(
        self,
        intelligence: CompetitiveIntelligence,
        export_directory: str,
        include_charts: bool = True
    ) -> Dict[str, bool]:
        """Export comprehensive competitive analysis results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'competitor_profiles': {
                    name: {
                        'domain': profile.domain,
                        'market_share': profile.market_share,
                        'organic_keywords': profile.organic_keywords,
                        'organic_traffic': profile.organic_traffic,
                        'competitive_strength': profile.competitive_strength,
                        'growth_rate': profile.growth_rate,
                        'top_keywords': profile.top_keywords[:10],  # Top 10
                        'serp_feature_coverage': profile.serp_feature_coverage,
                        'content_gaps': profile.content_gaps[:10],  # Top 10
                        'technical_advantages': profile.technical_advantages
                    }
                    for name, profile in intelligence.competitor_profiles.items()
                },
                'market_analysis': {
                    'total_market_size': intelligence.market_analysis.total_market_size if intelligence.market_analysis else 0,
                    'market_concentration': intelligence.market_analysis.market_concentration if intelligence.market_analysis else 0,
                    'growth_opportunities': intelligence.market_analysis.growth_opportunities[:10] if intelligence.market_analysis else [],
                    'competitive_threats': intelligence.market_analysis.competitive_threats[:10] if intelligence.market_analysis else [],
                    'market_leaders': intelligence.market_analysis.market_leaders if intelligence.market_analysis else [],
                    'emerging_competitors': intelligence.market_analysis.emerging_competitors if intelligence.market_analysis else []
                },
                'strategic_recommendations': intelligence.strategic_recommendations,
                'threat_assessment': intelligence.threat_assessment
            }
            
            # Export detailed data using DataExporter
            data_export_success = self.data_exporter.export_analysis_dataset(
                {'competitive_analysis': pd.DataFrame([export_data])},
                export_path / "competitive_analysis_detailed.xlsx"
            )
            
            # Export opportunity matrix
            matrix_export_success = True
            if not intelligence.opportunity_matrix.empty:
                matrix_export_success = self.data_exporter.export_with_metadata(
                    intelligence.opportunity_matrix,
                    metadata={'analysis_type': 'opportunity_matrix', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "opportunity_matrix.xlsx"
                )
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_competitive_analysis_report(
                {name: pd.DataFrame() for name in intelligence.competitor_profiles.keys()},  # Placeholder
                export_data,
                export_path / "competitive_analysis_executive_report.html"
            )
            
            return {
                'detailed_data': data_export_success,
                'opportunity_matrix': matrix_export_success,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting competitive analysis results: {str(e)}")
            return {}

    # Helper methods using utility framework
    def _calculate_market_concentration(self, traffic_shares: List[float]) -> float:
        """Calculate market concentration using Gini coefficient."""
        try:
            if not traffic_shares or all(share == 0 for share in traffic_shares):
                return 0.0
            
            # Use statistical calculator for Gini coefficient
            traffic_array = np.array(traffic_shares)
            stats_dict = self.stats_calculator.calculate_descriptive_statistics(traffic_array)
            
            n = len(traffic_array)
            mean_traffic = stats_dict.get('mean', 0)
            
            if mean_traffic == 0:
                return 0.0
            
            # Calculate Gini coefficient
            diff_sum = sum(abs(xi - xj) for xi in traffic_shares for xj in traffic_shares)
            gini = diff_sum / (2 * n * n * mean_traffic)
            
            return gini
            
        except Exception:
            return 0.0

    def _get_dominant_competitors_in_cluster(self, cluster_metadata: List[Dict[str, Any]]) -> List[str]:
        """Get dominant competitors in a keyword cluster."""
        try:
            competitor_counts = {}
            for meta in cluster_metadata:
                competitor = meta['competitor']
                competitor_counts[competitor] = competitor_counts.get(competitor, 0) + 1
            
            # Sort by count and return top 3
            sorted_competitors = sorted(competitor_counts.items(), key=lambda x: x[1], reverse=True)
            return [comp for comp, count in sorted_competitors[:3]]
            
        except Exception:
            return []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for competitive analysis operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional utility methods would go here...
    def _calculate_growth_rate(self, df: pd.DataFrame) -> float:
        """Calculate growth rate using time series analysis."""
        try:
            if 'date' not in df.columns or 'Traffic (%)' not in df.columns:
                return 0.0
            
            # Prepare time series data
            ts_data = df.groupby('date')['Traffic (%)'].sum().sort_index()
            
            if len(ts_data) < 5:
                return 0.0
            
            # Use time series analyzer to fit trend
            trend_model = self.time_series_analyzer.fit_trend_model(ts_data, 'linear')
            
            return trend_model.get('slope', 0.0) if trend_model else 0.0
            
        except Exception:
            return 0.0

    def _get_top_competitive_keywords(self, comp_df: pd.DataFrame, lenovo_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get top competitive keywords for a competitor."""
        try:
            # Find keywords where competitor ranks better than Lenovo
            top_keywords = []
            
            lenovo_keywords = {}
            for _, row in lenovo_df.iterrows():
                keyword = row.get('Keyword', '')
                position = row.get('Position', 100)
                lenovo_keywords[keyword.lower()] = position
            
            for _, row in comp_df.iterrows():
                keyword = row.get('Keyword', '')
                comp_position = row.get('Position', 100)
                traffic = row.get('Traffic (%)', 0)
                search_volume = row.get('Search Volume', 0)
                
                lenovo_position = lenovo_keywords.get(keyword.lower(), 100)
                
                # Competitor ranks better and has significant traffic
                if comp_position < lenovo_position and traffic > 1:
                    top_keywords.append({
                        'keyword': keyword,
                        'competitor_position': comp_position,
                        'lenovo_position': lenovo_position,
                        'position_advantage': lenovo_position - comp_position,
                        'traffic': traffic,
                        'search_volume': search_volume,
                        'opportunity_score': traffic * (lenovo_position - comp_position) / 10
                    })
            
            # Sort by opportunity score and return top 20
            top_keywords.sort(key=lambda x: x['opportunity_score'], reverse=True)
            return top_keywords[:20]
            
        except Exception:
            return []

    def _calculate_serp_feature_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate SERP feature coverage for a competitor."""
        try:
            if 'SERP Features by Keyword' not in df.columns:
                return {}
            
            feature_counts = {}
            total_keywords = len(df)
            
            for _, row in df.iterrows():
                features_str = row['SERP Features by Keyword']
                if pd.notna(features_str):
                    features = StringHelper.normalize_serp_features(features_str)
                    for feature in features:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            # Calculate coverage percentages
            coverage = {
                feature: safe_divide(count, total_keywords, 0.0)
                for feature, count in feature_counts.items()
            }
            
            return coverage
            
        except Exception:
            return {}
