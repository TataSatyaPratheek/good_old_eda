"""
Merge Strategy Module for SEO Competitive Intelligence
Advanced data merging, gap analysis, and strategic recommendation generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Import our utilities to eliminate redundancy
from src.utils.common_helpers import StringHelper, DateHelper, memoize, timing_decorator, deep_merge_dicts, safe_divide
from src.utils.data_utils import DataProcessor, DataValidator, DataTransformer
from src.utils.math_utils import StatisticalCalculator, OptimizationHelper
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager, PathManager
from src.utils.validation_utils import SchemaValidator, BusinessRuleValidator
from src.utils.export_utils import ReportExporter, DataExporter

@dataclass
class MergeResult:
    """Data class for merge operation results"""
    merged_data: pd.DataFrame
    merge_quality_score: float
    conflicts_resolved: int
    data_gaps_identified: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class GapAnalysis:
    """Data class for gap analysis results"""
    keyword_gaps: pd.DataFrame
    traffic_gaps: Dict[str, float]
    position_opportunities: pd.DataFrame
    competitive_threats: List[Dict[str, Any]]
    priority_actions: List[Dict[str, Any]]
    market_share_analysis: Dict[str, float]

@dataclass
class StrategicRecommendations:
    """Data class for strategic recommendations"""
    immediate_actions: List[Dict[str, Any]]
    short_term_goals: List[Dict[str, Any]]
    long_term_strategy: List[Dict[str, Any]]
    resource_allocation: Dict[str, float]
    risk_assessment: Dict[str, Any]
    expected_outcomes: Dict[str, Any]

class MergeStrategy:
    """
    Advanced merge strategy for SEO competitive intelligence.
    
    Handles intelligent data merging, gap analysis, and strategic
    recommendation generation using the comprehensive utility framework.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities instead of custom implementations."""
        self.logger = logger or LoggerFactory.get_logger("merge_strategy")
        self.config = config_manager or ConfigManager()
        
        # Initialize utility classes - no more redundant implementations
        self.data_processor = DataProcessor(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.data_transformer = DataTransformer(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.optimization_helper = OptimizationHelper(self.logger)
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        self.schema_validator = SchemaValidator(self.logger)
        self.business_rule_validator = BusinessRuleValidator(self.logger)
        self.report_exporter = ReportExporter(self.logger)
        self.path_manager = PathManager(config_manager=self.config)
        
        # Load merge strategy configurations
        analysis_config = self.config.get_analysis_config()
        self.merge_threshold = analysis_config.competitive_threat_threshold
        self.min_data_quality = 0.7
        self.similarity_threshold = 0.8

    @timing_decorator()
    @memoize(ttl=1800)  # Cache for 30 minutes
    def intelligent_data_merge(
        self,
        primary_dataset: pd.DataFrame,
        secondary_datasets: Dict[str, pd.DataFrame],
        merge_strategy: str = 'intelligent',
        conflict_resolution: str = 'prioritize_quality'
    ) -> MergeResult:
        """
        Perform intelligent data merging with conflict resolution.
        
        Args:
            primary_dataset: Primary dataset (e.g., Lenovo data)
            secondary_datasets: Dictionary of secondary datasets (competitors)
            merge_strategy: Merge strategy ('intelligent', 'union', 'intersection')
            conflict_resolution: How to resolve conflicts
            
        Returns:
            MergeResult with merged data and metadata
        """
        try:
            with self.performance_tracker.track_block("intelligent_data_merge"):
                # Audit log the merge operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="data_merge",
                    parameters={
                        "merge_strategy": merge_strategy,
                        "conflict_resolution": conflict_resolution,
                        "datasets_count": len(secondary_datasets) + 1
                    }
                )
                
                # Validate all datasets first using DataValidator
                primary_validation = self.data_validator.validate_seo_dataset(primary_dataset, 'positions')
                if primary_validation.quality_score < self.min_data_quality:
                    self.logger.warning(f"Primary dataset quality low: {primary_validation.quality_score:.3f}")
                
                # Clean and standardize all datasets using DataProcessor
                cleaned_primary = self.data_processor.clean_seo_data(primary_dataset)
                cleaned_secondary = {}
                
                for competitor, dataset in secondary_datasets.items():
                    cleaned_data = self.data_processor.clean_seo_data(dataset)
                    cleaned_secondary[competitor] = cleaned_data
                
                # Standardize data formats across all datasets
                all_datasets = {'primary': cleaned_primary, **cleaned_secondary}
                standardized_datasets = self.data_processor.standardize_competitor_data(all_datasets)
                
                # Perform intelligent merge based on strategy
                if merge_strategy == 'intelligent':
                    merged_data = self._perform_intelligent_merge(
                        standardized_datasets, conflict_resolution
                    )
                elif merge_strategy == 'union':
                    merged_data = self._perform_union_merge(standardized_datasets)
                elif merge_strategy == 'intersection':
                    merged_data = self._perform_intersection_merge(standardized_datasets)
                else:
                    raise ValueError(f"Unknown merge strategy: {merge_strategy}")
                
                # Calculate merge quality metrics
                merge_quality = self._calculate_merge_quality(merged_data, standardized_datasets)
                
                # Identify conflicts and gaps
                conflicts_resolved = self._count_conflicts_resolved(standardized_datasets)
                data_gaps = self._identify_data_gaps(merged_data, standardized_datasets)
                
                # Generate merge recommendations
                recommendations = self._generate_merge_recommendations(
                    merged_data, merge_quality, data_gaps
                )
                
                # Create metadata
                metadata = {
                    'merge_timestamp': datetime.now(),
                    'source_datasets': list(secondary_datasets.keys()) + ['primary'],
                    'merge_strategy': merge_strategy,
                    'conflict_resolution': conflict_resolution,
                    'record_counts': {name: len(df) for name, df in standardized_datasets.items()},
                    'final_record_count': len(merged_data)
                }
                
                result = MergeResult(
                    merged_data=merged_data,
                    merge_quality_score=merge_quality,
                    conflicts_resolved=conflicts_resolved,
                    data_gaps_identified=data_gaps,
                    recommendations=recommendations,
                    metadata=metadata
                )
                
                self.logger.info(f"Intelligent merge completed: {len(merged_data)} records, quality: {merge_quality:.3f}")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in intelligent data merge: {str(e)}")
            return MergeResult(pd.DataFrame(), 0.0, 0, [], [], {})

    @timing_decorator()
    def comprehensive_gap_analysis(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        analysis_depth: str = 'comprehensive'
    ) -> GapAnalysis:
        """
        Perform comprehensive gap analysis using multiple analytical approaches.
        
        Args:
            lenovo_data: Lenovo's current data
            competitor_data: Competitor datasets
            analysis_depth: Analysis depth ('basic', 'comprehensive', 'advanced')
            
        Returns:
            GapAnalysis with comprehensive findings
        """
        try:
            with self.performance_tracker.track_block("comprehensive_gap_analysis"):
                # Clean and validate all datasets
                cleaned_lenovo = self.data_processor.clean_seo_data(lenovo_data)
                cleaned_competitors = {}
                
                for competitor, df in competitor_data.items():
                    cleaned_competitors[competitor] = self.data_processor.clean_seo_data(df)
                
                # Identify keyword gaps using StringHelper for matching
                keyword_gaps = self._identify_keyword_gaps(cleaned_lenovo, cleaned_competitors)
                
                # Calculate traffic gaps using statistical analysis
                traffic_gaps = self._calculate_traffic_gaps(cleaned_lenovo, cleaned_competitors)
                
                # Find position opportunities using optimization
                position_opportunities = self._find_position_opportunities(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Assess competitive threats using business rules
                competitive_threats = self._assess_competitive_threats(
                    cleaned_lenovo, cleaned_competitors
                )
                
                # Generate priority actions using optimization helper
                priority_actions = self._generate_priority_actions(
                    keyword_gaps, traffic_gaps, position_opportunities
                )
                
                # Analyze market share using statistical calculator
                market_share_analysis = self._analyze_market_share(
                    cleaned_lenovo, cleaned_competitors
                )
                
                gap_analysis = GapAnalysis(
                    keyword_gaps=keyword_gaps,
                    traffic_gaps=traffic_gaps,
                    position_opportunities=position_opportunities,
                    competitive_threats=competitive_threats,
                    priority_actions=priority_actions,
                    market_share_analysis=market_share_analysis
                )
                
                self.logger.info(f"Gap analysis completed: {len(keyword_gaps)} keyword gaps identified")
                return gap_analysis
                
        except Exception as e:
            self.logger.error(f"Error in gap analysis: {str(e)}")
            return GapAnalysis(pd.DataFrame(), {}, pd.DataFrame(), [], [], {})

    @timing_decorator()
    def generate_strategic_recommendations(
        self,
        gap_analysis: GapAnalysis,
        current_performance: Dict[str, Any],
        business_objectives: Dict[str, Any] = None
    ) -> StrategicRecommendations:
        """
        Generate strategic recommendations based on gap analysis and business objectives.
        
        Args:
            gap_analysis: Results from gap analysis
            current_performance: Current performance metrics
            business_objectives: Business objectives and constraints
            
        Returns:
            StrategicRecommendations with actionable insights
        """
        try:
            with self.performance_tracker.track_block("generate_strategic_recommendations"):
                if business_objectives is None:
                    business_objectives = self._get_default_business_objectives()
                
                # Generate immediate actions (0-30 days)
                immediate_actions = self._generate_immediate_actions(
                    gap_analysis, current_performance
                )
                
                # Generate short-term goals (1-6 months)
                short_term_goals = self._generate_short_term_goals(
                    gap_analysis, business_objectives
                )
                
                # Generate long-term strategy (6+ months)
                long_term_strategy = self._generate_long_term_strategy(
                    gap_analysis, business_objectives
                )
                
                # Optimize resource allocation using optimization helper
                resource_allocation = self._optimize_resource_allocation(
                    gap_analysis, business_objectives
                )
                
                # Assess risks using business rule validator
                risk_assessment = self._assess_strategic_risks(
                    gap_analysis, current_performance
                )
                
                # Project expected outcomes using statistical modeling
                expected_outcomes = self._project_expected_outcomes(
                    gap_analysis, resource_allocation
                )
                
                recommendations = StrategicRecommendations(
                    immediate_actions=immediate_actions,
                    short_term_goals=short_term_goals,
                    long_term_strategy=long_term_strategy,
                    resource_allocation=resource_allocation,
                    risk_assessment=risk_assessment,
                    expected_outcomes=expected_outcomes
                )
                
                self.logger.info(f"Strategic recommendations generated: {len(immediate_actions)} immediate actions")
                return recommendations
                
        except Exception as e:
            self.logger.error(f"Error generating strategic recommendations: {str(e)}")
            return StrategicRecommendations([], [], [], {}, {}, {})

    def _perform_intelligent_merge(
        self,
        datasets: Dict[str, pd.DataFrame],
        conflict_resolution: str
    ) -> pd.DataFrame:
        """Perform intelligent merge with conflict resolution."""
        try:
            # Find common keywords across all datasets
            all_keywords = set()
            for df in datasets.values():
                if 'Keyword' in df.columns:
                    all_keywords.update(df['Keyword'].str.lower().tolist())
            
            merged_records = []
            
            for keyword in all_keywords:
                # Get records for this keyword from all datasets
                keyword_records = {}
                for dataset_name, df in datasets.items():
                    if 'Keyword' in df.columns:
                        keyword_mask = df['Keyword'].str.lower() == keyword.lower()
                        keyword_data = df[keyword_mask]
                        if not keyword_data.empty:
                            keyword_records[dataset_name] = keyword_data.iloc[0]
                
                if keyword_records:
                    # Resolve conflicts and merge
                    merged_record = self._resolve_record_conflicts(
                        keyword_records, conflict_resolution
                    )
                    merged_records.append(merged_record)
            
            merged_df = pd.DataFrame(merged_records)
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error in intelligent merge: {str(e)}")
            return pd.DataFrame()

    def _resolve_record_conflicts(
        self,
        keyword_records: Dict[str, pd.Series],
        resolution_strategy: str
    ) -> Dict[str, Any]:
        """Resolve conflicts between records for the same keyword."""
        try:
            if resolution_strategy == 'prioritize_quality':
                # Use primary dataset as base, fill gaps from others
                if 'primary' in keyword_records:
                    base_record = keyword_records['primary'].to_dict()
                else:
                    base_record = list(keyword_records.values())[0].to_dict()
                
                # Fill missing values from other sources
                for dataset_name, record in keyword_records.items():
                    for field, value in record.to_dict().items():
                        if pd.isna(base_record.get(field)) and pd.notna(value):
                            base_record[field] = value
                
                return base_record
                
            elif resolution_strategy == 'average_numeric':
                # Average numeric fields, use most common for categorical
                merged_record = {}
                
                # Get all unique fields
                all_fields = set()
                for record in keyword_records.values():
                    all_fields.update(record.index)
                
                for field in all_fields:
                    values = [record.get(field) for record in keyword_records.values() 
                             if pd.notna(record.get(field))]
                    
                    if values:
                        if field in ['Position', 'Search Volume', 'Traffic', 'CPC']:
                            # Average numeric fields
                            merged_record[field] = np.mean([v for v in values if isinstance(v, (int, float))])
                        else:
                            # Use most common value for categorical
                            merged_record[field] = max(set(values), key=values.count)
                
                return merged_record
                
            else:
                # Default: use first available record
                return list(keyword_records.values())[0].to_dict()
                
        except Exception as e:
            self.logger.error(f"Error resolving record conflicts: {str(e)}")
            return {}

    def _identify_keyword_gaps(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Identify keyword gaps using string similarity analysis."""
        try:
            lenovo_keywords = set(lenovo_data['Keyword'].str.lower().tolist())
            gap_keywords = []
            
            for competitor, df in competitor_data.items():
                competitor_keywords = set(df['Keyword'].str.lower().tolist())
                
                # Find keywords competitor has but Lenovo doesn't
                missing_keywords = competitor_keywords - lenovo_keywords
                
                for keyword in missing_keywords:
                    keyword_data = df[df['Keyword'].str.lower() == keyword]
                    if not keyword_data.empty:
                        row = keyword_data.iloc[0]
                        
                        # Calculate opportunity score
                        opportunity_score = self._calculate_keyword_opportunity_score(row, competitor)
                        
                        gap_keywords.append({
                            'keyword': keyword,
                            'competitor': competitor,
                            'position': row.get('Position', 100),
                            'search_volume': row.get('Search Volume', 0),
                            'traffic': row.get('Traffic (%)', 0),
                            'opportunity_score': opportunity_score,
                            'difficulty': row.get('Keyword Difficulty', 50),
                            'gap_type': 'missing_keyword'
                        })
            
            # Also find keywords where Lenovo ranks poorly vs competitors
            for keyword in lenovo_keywords:
                lenovo_row = lenovo_data[lenovo_data['Keyword'].str.lower() == keyword]
                if lenovo_row.empty:
                    continue
                
                lenovo_position = lenovo_row.iloc[0].get('Position', 100)
                
                for competitor, df in competitor_data.items():
                    comp_row = df[df['Keyword'].str.lower() == keyword]
                    if not comp_row.empty:
                        comp_position = comp_row.iloc[0].get('Position', 100)
                        
                        # If competitor ranks significantly better
                        if comp_position < lenovo_position - 10:
                            opportunity_score = self._calculate_position_improvement_score(
                                lenovo_position, comp_position, lenovo_row.iloc[0]
                            )
                            
                            gap_keywords.append({
                                'keyword': keyword,
                                'competitor': competitor,
                                'lenovo_position': lenovo_position,
                                'competitor_position': comp_position,
                                'position_gap': lenovo_position - comp_position,
                                'search_volume': lenovo_row.iloc[0].get('Search Volume', 0),
                                'opportunity_score': opportunity_score,
                                'gap_type': 'position_improvement'
                            })
            
            return pd.DataFrame(gap_keywords)
            
        except Exception as e:
            self.logger.error(f"Error identifying keyword gaps: {str(e)}")
            return pd.DataFrame()

    def _calculate_traffic_gaps(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate traffic gaps using statistical analysis."""
        try:
            lenovo_traffic = lenovo_data.get('Traffic (%)', pd.Series()).sum()
            traffic_gaps = {}
            
            for competitor, df in competitor_data.items():
                competitor_traffic = df.get('Traffic (%)', pd.Series()).sum()
                gap = competitor_traffic - lenovo_traffic
                traffic_gaps[competitor] = gap
            
            # Calculate relative gaps
            total_competitor_traffic = sum(
                df.get('Traffic (%)', pd.Series()).sum() 
                for df in competitor_data.values()
            )
            
            if total_competitor_traffic > 0:
                traffic_gaps['total_gap'] = total_competitor_traffic - lenovo_traffic
                traffic_gaps['relative_gap'] = safe_divide(
                    traffic_gaps['total_gap'], total_competitor_traffic, 0.0
                )
            
            return traffic_gaps
            
        except Exception as e:
            self.logger.error(f"Error calculating traffic gaps: {str(e)}")
            return {}

    def _find_position_opportunities(
        self,
        lenovo_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Find position improvement opportunities using optimization."""
        try:
            opportunities = []
            
            # Use optimization helper to find best opportunities
            if not lenovo_data.empty and competitor_data:
                # Prepare data for optimization
                keyword_data = lenovo_data.copy()
                
                # Add competitor position data
                for competitor, df in competitor_data.items():
                    competitor_positions = {}
                    for _, row in df.iterrows():
                        keyword = row.get('Keyword', '').lower()
                        position = row.get('Position', 100)
                        competitor_positions[keyword] = position
                    
                    keyword_data[f'{competitor}_position'] = keyword_data['Keyword'].str.lower().map(
                        competitor_positions
                    ).fillna(100)
                
                # Calculate opportunity scores
                for _, row in keyword_data.iterrows():
                    keyword = row['Keyword']
                    lenovo_position = row.get('Position', 100)
                    
                    # Find best competitor position for this keyword
                    competitor_positions = [
                        row.get(f'{comp}_position', 100) 
                        for comp in competitor_data.keys()
                    ]
                    best_competitor_position = min(competitor_positions) if competitor_positions else 100
                    
                    if best_competitor_position < lenovo_position - 5:  # Significant opportunity
                        opportunity = {
                            'keyword': keyword,
                            'current_position': lenovo_position,
                            'best_competitor_position': best_competitor_position,
                            'position_gap': lenovo_position - best_competitor_position,
                            'search_volume': row.get('Search Volume', 0),
                            'traffic_potential': self._estimate_traffic_potential(row, best_competitor_position),
                            'effort_required': self._estimate_effort_required(row, best_competitor_position),
                            'roi_score': self._calculate_roi_score(row, best_competitor_position)
                        }
                        opportunities.append(opportunity)
            
            opportunities_df = pd.DataFrame(opportunities)
            
            # Sort by ROI score
            if not opportunities_df.empty:
                opportunities_df = opportunities_df.sort_values('roi_score', ascending=False)
            
            return opportunities_df
            
        except Exception as e:
            self.logger.error(f"Error finding position opportunities: {str(e)}")
            return pd.DataFrame()

    def _calculate_keyword_opportunity_score(self, keyword_row: pd.Series, competitor: str) -> float:
        """Calculate opportunity score for a keyword gap."""
        try:
            # Factors: search volume, competitor position, traffic potential
            search_volume = keyword_row.get('Search Volume', 0)
            position = keyword_row.get('Position', 50)
            traffic = keyword_row.get('Traffic (%)', 0)
            difficulty = keyword_row.get('Keyword Difficulty', 50)
            
            # Normalize factors
            volume_score = min(search_volume / 10000, 1.0)
            position_score = max(0, (50 - position) / 50)
            traffic_score = min(traffic / 10, 1.0)
            difficulty_score = max(0, (100 - difficulty) / 100)
            
            # Weighted combination
            opportunity_score = (
                volume_score * 0.3 +
                position_score * 0.25 +
                traffic_score * 0.25 +
                difficulty_score * 0.2
            )
            
            return opportunity_score
            
        except Exception:
            return 0.0

    def _get_default_business_objectives(self) -> Dict[str, Any]:
        """Get default business objectives from configuration."""
        return {
            'target_traffic_increase': 0.25,  # 25% increase
            'budget_constraint': 100000,  # $100k
            'timeline_months': 12,
            'priority_keywords': [],
            'target_positions': {'top_3': 0.15, 'top_10': 0.40},
            'risk_tolerance': 'medium'
        }

    def export_merge_results(
        self,
        merge_result: MergeResult,
        gap_analysis: GapAnalysis,
        recommendations: StrategicRecommendations,
        export_directory: str
    ) -> Dict[str, bool]:
        """Export comprehensive merge results using export utilities."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Export datasets
            datasets_to_export = {
                'merged_data': merge_result.merged_data,
                'keyword_gaps': gap_analysis.keyword_gaps,
                'position_opportunities': gap_analysis.position_opportunities
            }
            
            # Export using DataExporter
            data_exporter = DataExporter(self.logger)
            export_results = data_exporter.export_analysis_dataset(
                datasets_to_export, export_path / "merge_analysis_results.xlsx"
            )
            
            # Export strategic report using ReportExporter
            analysis_results = {
                'gap_analysis': gap_analysis,
                'recommendations': recommendations,
                'merge_metadata': merge_result.metadata
            }
            
            report_success = self.report_exporter.export_executive_report(
                analysis_results, export_path / "strategic_analysis_report.html"
            )
            
            export_results['strategic_report'] = report_success
            
            self.logger.info(f"Merge results exported to {export_path}")
            return export_results
            
        except Exception as e:
            self.logger.error(f"Error exporting merge results: {str(e)}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for merge operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )
