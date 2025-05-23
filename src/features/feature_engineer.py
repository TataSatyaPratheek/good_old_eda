"""
Feature Engineering Module for SEO Competitive Intelligence
Advanced feature engineering leveraging the comprehensive utility framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
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
from src.utils.file_utils import FileManager, ExportManager

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering operations"""
    create_temporal_features: bool = True
    create_text_features: bool = True
    create_statistical_features: bool = True
    create_competitive_features: bool = True
    apply_scaling: bool = True
    feature_selection: bool = True
    max_features: Optional[int] = None

@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering process"""
    engineered_features: pd.DataFrame
    feature_metadata: Dict[str, Any]
    feature_importance: Dict[str, float]
    transformation_pipeline: List[str]
    quality_metrics: Dict[str, float]
    processing_time_seconds: float

class FeatureEngineer:
    """
    Advanced feature engineering for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    feature engineering capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("feature_engineer")
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
        
        # Load feature engineering configurations
        analysis_config = self.config.get_analysis_config()
        self.feature_engineering_config = FeatureEngineeringConfig()

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def comprehensive_feature_engineering(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        competitor_data: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[FeatureEngineeringConfig] = None
    ) -> FeatureEngineeringResult:
        """
        Perform comprehensive feature engineering using utility framework.
        
        Args:
            data: Input DataFrame
            target_column: Target column for supervised learning
            competitor_data: Competitor data for competitive features
            config: Feature engineering configuration
            
        Returns:
            FeatureEngineeringResult with engineered features and metadata
        """
        try:
            with self.performance_tracker.track_block("comprehensive_feature_engineering"):
                # Audit log the feature engineering operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="feature_engineering",
                    parameters={
                        "rows": len(data),
                        "columns": len(data.columns),
                        "target_column": target_column,
                        "has_competitor_data": competitor_data is not None
                    }
                )
                
                if config is None:
                    config = self.feature_engineering_config
                
                # Clean and validate data using DataProcessor
                cleaned_data = self.data_processor.clean_seo_data(data)
                
                # Validate data quality
                validation_report = self.data_validator.validate_seo_dataset(cleaned_data, 'positions')
                if validation_report.quality_score < 0.7:
                    self.logger.warning(f"Low data quality score: {validation_report.quality_score:.3f}")
                
                transformation_pipeline = []
                engineered_data = cleaned_data.copy()
                
                # 1. Temporal Feature Engineering using DataTransformer
                if config.create_temporal_features and 'date' in engineered_data.columns:
                    temporal_features = self.data_transformer.create_temporal_features(
                        engineered_data,
                        date_column='date',
                        value_columns=['Position', 'Traffic (%)', 'Search Volume']
                    )
                    engineered_data = temporal_features
                    transformation_pipeline.append("temporal_features")
                
                # 2. Text/Keyword Feature Engineering using StringHelper
                if config.create_text_features and 'Keyword' in engineered_data.columns:
                    text_features = self._create_text_features(engineered_data)
                    engineered_data = pd.concat([engineered_data, text_features], axis=1)
                    transformation_pipeline.append("text_features")
                
                # 3. Statistical Feature Engineering using StatisticalCalculator
                if config.create_statistical_features:
                    statistical_features = self._create_statistical_features(engineered_data)
                    engineered_data = pd.concat([engineered_data, statistical_features], axis=1)
                    transformation_pipeline.append("statistical_features")
                
                # 4. Competitive Feature Engineering
                if config.create_competitive_features and competitor_data:
                    competitive_features = self._create_competitive_features(
                        engineered_data, competitor_data
                    )
                    engineered_data = pd.concat([engineered_data, competitive_features], axis=1)
                    transformation_pipeline.append("competitive_features")
                
                # 5. SERP Feature Engineering using StringHelper
                if 'SERP Features by Keyword' in engineered_data.columns:
                    serp_features = self._create_serp_features(engineered_data)
                    engineered_data = pd.concat([engineered_data, serp_features], axis=1)
                    transformation_pipeline.append("serp_features")
                
                # 6. Interaction Features using mathematical operations
                interaction_features = self._create_interaction_features(engineered_data)
                engineered_data = pd.concat([engineered_data, interaction_features], axis=1)
                transformation_pipeline.append("interaction_features")
                
                # 7. Scaling using DataTransformer
                if config.apply_scaling:
                    engineered_data = self.data_transformer.apply_scaling(
                        engineered_data,
                        scaling_method='standard',
                        columns_to_scale=None,  # Auto-detect
                        fit_scaler=True
                    )
                    transformation_pipeline.append("scaling")
                
                # 8. Feature Selection using optimization
                if config.feature_selection and target_column and target_column in engineered_data.columns:
                    selected_features = self._perform_feature_selection(
                        engineered_data, target_column, config.max_features
                    )
                    transformation_pipeline.append("feature_selection")
                else:
                    selected_features = engineered_data
                
                # Calculate feature importance and metadata
                feature_metadata = self._generate_feature_metadata(
                    selected_features, data, transformation_pipeline
                )
                
                feature_importance = self._calculate_feature_importance(
                    selected_features, target_column
                )
                
                # Calculate quality metrics
                quality_metrics = self._calculate_feature_quality_metrics(
                    selected_features, data
                )
                
                result = FeatureEngineeringResult(
                    engineered_features=selected_features,
                    feature_metadata=feature_metadata,
                    feature_importance=feature_importance,
                    transformation_pipeline=transformation_pipeline,
                    quality_metrics=quality_metrics,
                    processing_time_seconds=0.0  # Will be filled by performance tracker
                )
                
                self.logger.info(f"Feature engineering completed: {len(selected_features.columns)} features created")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive feature engineering: {str(e)}")
            return FeatureEngineeringResult(
                pd.DataFrame(), {}, {}, [], {}, 0.0
            )

    @timing_decorator()
    def create_advanced_seo_features(
        self,
        data: pd.DataFrame,
        feature_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Create advanced SEO-specific features using utility framework.
        
        Args:
            data: Input SEO data
            feature_types: Types of features to create
            
        Returns:
            DataFrame with advanced SEO features
        """
        try:
            with self.performance_tracker.track_block("create_advanced_seo_features"):
                if feature_types is None:
                    feature_types = ['keyword_metrics', 'position_metrics', 'traffic_metrics', 'serp_metrics']
                
                # Clean data first
                cleaned_data = self.data_processor.clean_seo_data(data)
                advanced_features = pd.DataFrame(index=cleaned_data.index)
                
                # Keyword-based features using StringHelper
                if 'keyword_metrics' in feature_types and 'Keyword' in cleaned_data.columns:
                    keyword_features = self._create_keyword_metrics(cleaned_data)
                    advanced_features = pd.concat([advanced_features, keyword_features], axis=1)
                
                # Position-based features using statistical analysis
                if 'position_metrics' in feature_types and 'Position' in cleaned_data.columns:
                    position_features = self._create_position_metrics(cleaned_data)
                    advanced_features = pd.concat([advanced_features, position_features], axis=1)
                
                # Traffic-based features using mathematical analysis
                if 'traffic_metrics' in feature_types and 'Traffic (%)' in cleaned_data.columns:
                    traffic_features = self._create_traffic_metrics(cleaned_data)
                    advanced_features = pd.concat([advanced_features, traffic_features], axis=1)
                
                # SERP feature analysis
                if 'serp_metrics' in feature_types and 'SERP Features by Keyword' in cleaned_data.columns:
                    serp_features = self._create_advanced_serp_metrics(cleaned_data)
                    advanced_features = pd.concat([advanced_features, serp_features], axis=1)
                
                # Volatility features using time series analysis
                if 'volatility_metrics' in feature_types:
                    volatility_features = self._create_volatility_features(cleaned_data)
                    advanced_features = pd.concat([advanced_features, volatility_features], axis=1)
                
                self.logger.info(f"Created {len(advanced_features.columns)} advanced SEO features")
                return advanced_features
                
        except Exception as e:
            self.logger.error(f"Error creating advanced SEO features: {str(e)}")
            return pd.DataFrame()

    @timing_decorator()
    def create_competitive_features(
        self,
        primary_data: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        feature_prefix: str = "comp_"
    ) -> pd.DataFrame:
        """
        Create competitive analysis features using statistical utilities.
        
        Args:
            primary_data: Primary dataset (e.g., Lenovo)
            competitor_data: Competitor datasets
            feature_prefix: Prefix for competitive features
            
        Returns:
            DataFrame with competitive features
        """
        try:
            with self.performance_tracker.track_block("create_competitive_features"):
                competitive_features = pd.DataFrame(index=primary_data.index)
                
                # Get primary keywords for comparison
                primary_keywords = set(primary_data['Keyword'].str.lower().tolist())
                
                # For each competitor, create comparative features
                for competitor_name, comp_df in competitor_data.items():
                    comp_clean = self.data_processor.clean_seo_data(comp_df)
                    
                    # Keyword overlap features
                    comp_keywords = set(comp_clean['Keyword'].str.lower().tolist())
                    overlap_ratio = len(primary_keywords.intersection(comp_keywords)) / len(primary_keywords.union(comp_keywords))
                    
                    competitive_features[f'{feature_prefix}{competitor_name}_overlap_ratio'] = overlap_ratio
                    
                    # Position comparison features
                    if 'Position' in comp_clean.columns:
                        position_comparison = self._create_position_comparison_features(
                            primary_data, comp_clean, competitor_name, feature_prefix
                        )
                        competitive_features = pd.concat([competitive_features, position_comparison], axis=1)
                    
                    # Traffic comparison features
                    if 'Traffic (%)' in comp_clean.columns:
                        traffic_comparison = self._create_traffic_comparison_features(
                            primary_data, comp_clean, competitor_name, feature_prefix
                        )
                        competitive_features = pd.concat([competitive_features, traffic_comparison], axis=1)
                
                # Aggregate competitive metrics using StatisticalCalculator
                competitive_features = self._add_aggregate_competitive_metrics(
                    competitive_features, feature_prefix
                )
                
                self.logger.info(f"Created {len(competitive_features.columns)} competitive features")
                return competitive_features
                
        except Exception as e:
            self.logger.error(f"Error creating competitive features: {str(e)}")
            return pd.DataFrame()

    def _create_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features using StringHelper utilities."""
        try:
            text_features = pd.DataFrame(index=data.index)
            
            if 'Keyword' in data.columns:
                # Basic text metrics using StringHelper
                text_features['keyword_length'] = data['Keyword'].apply(
                    lambda x: len(StringHelper.clean_keyword(str(x))) if pd.notna(x) else 0
                )
                
                text_features['keyword_word_count'] = data['Keyword'].apply(
                    lambda x: len(StringHelper.clean_keyword(str(x)).split()) if pd.notna(x) else 0
                )
                
                # Brand keyword identification using StringHelper
                brand_terms = ['lenovo', 'thinkpad', 'legion', 'ideapad']
                
                def is_branded_keyword(keyword):
                    if pd.isna(keyword):
                        return False
                    branded, _ = StringHelper.extract_brand_keywords([str(keyword)], brand_terms)
                    return len(branded) > 0
                
                text_features['is_branded'] = data['Keyword'].apply(is_branded_keyword)
                
                # Keyword intent classification (simplified)
                text_features['is_commercial'] = data['Keyword'].str.contains(
                    r'\b(buy|purchase|price|cost|cheap|deal|discount|shop)\b', 
                    case=False, na=False
                )
                
                text_features['is_informational'] = data['Keyword'].str.contains(
                    r'\b(how|what|why|when|where|guide|tutorial|tips|review)\b', 
                    case=False, na=False
                )
                
                text_features['is_navigational'] = data['Keyword'].str.contains(
                    r'\b(login|sign in|website|official|homepage)\b', 
                    case=False, na=False
                )
                
                # Keyword complexity features
                text_features['has_numbers'] = data['Keyword'].str.contains(r'\d', na=False)
                text_features['has_special_chars'] = data['Keyword'].str.contains(r'[^\w\s]', na=False)
                
                # Long-tail keyword identification
                text_features['is_long_tail'] = (text_features['keyword_word_count'] >= 4) & (text_features['keyword_length'] >= 20)
            
            return text_features
            
        except Exception as e:
            self.logger.error(f"Error creating text features: {str(e)}")
            return pd.DataFrame()

    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features using StatisticalCalculator."""
        try:
            stat_features = pd.DataFrame(index=data.index)
            
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['Position', 'Search Volume', 'Traffic (%)', 'CPC']:
                    col_data = data[col].dropna()
                    
                    if len(col_data) > 10:
                        # Calculate statistical measures using StatisticalCalculator
                        stats_dict = self.stats_calculator.calculate_descriptive_statistics(col_data)
                        
                        # Z-score (standardized value)
                        mean_val = stats_dict.get('mean', 0)
                        std_val = stats_dict.get('std', 1)
                        
                        if std_val > 0:
                            stat_features[f'{col}_zscore'] = (data[col] - mean_val) / std_val
                        
                        # Percentile rank
                        stat_features[f'{col}_percentile'] = data[col].rank(pct=True)
                        
                        # Outlier indicators
                        q25 = stats_dict.get('q25', 0)
                        q75 = stats_dict.get('q75', 0)
                        iqr = stats_dict.get('iqr', 0)
                        
                        if iqr > 0:
                            lower_bound = q25 - 1.5 * iqr
                            upper_bound = q75 + 1.5 * iqr
                            stat_features[f'{col}_is_outlier'] = (
                                (data[col] < lower_bound) | (data[col] > upper_bound)
                            ).astype(int)
            
            # Cross-column statistical features
            if all(col in data.columns for col in ['Position', 'Search Volume']):
                # Opportunity score (high volume, poor position)
                position_inverted = 101 - data['Position']  # Invert position (higher is better)
                volume_normalized = np.log1p(data['Search Volume'])  # Log normalize volume
                stat_features['opportunity_score'] = position_inverted * volume_normalized
            
            if all(col in data.columns for col in ['Traffic (%)', 'Search Volume']):
                # Traffic efficiency (traffic per unit of search volume)
                stat_features['traffic_efficiency'] = safe_divide(
                    data['Traffic (%)'], data['Search Volume'], 0.0
                )
            
            return stat_features
            
        except Exception as e:
            self.logger.error(f"Error creating statistical features: {str(e)}")
            return pd.DataFrame()

    def _create_serp_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create SERP feature-based features using StringHelper."""
        try:
            serp_features = pd.DataFrame(index=data.index)
            
            if 'SERP Features by Keyword' not in data.columns:
                return serp_features
            
            # Process SERP features using StringHelper
            def extract_serp_features(features_str):
                if pd.isna(features_str):
                    return []
                return StringHelper.normalize_serp_features(str(features_str))
            
            data['serp_features_list'] = data['SERP Features by Keyword'].apply(extract_serp_features)
            
            # Count of SERP features
            serp_features['serp_feature_count'] = data['serp_features_list'].apply(len)
            
            # Specific SERP feature indicators
            important_features = [
                'featured_snippet', 'people_also_ask', 'knowledge_panel', 
                'image_pack', 'video_carousel', 'shopping_results', 'ads_top'
            ]
            
            for feature in important_features:
                serp_features[f'has_{feature}'] = data['serp_features_list'].apply(
                    lambda x: int(feature in x)
                )
            
            # SERP feature diversity (using set operations)
            serp_features['serp_diversity_score'] = data['serp_features_list'].apply(
                lambda x: len(set(x)) / max(len(x), 1) if x else 0
            )
            
            # Rich results indicator
            rich_features = ['featured_snippet', 'knowledge_panel', 'image_pack', 'video_carousel']
            serp_features['has_rich_results'] = data['serp_features_list'].apply(
                lambda x: int(any(feature in x for feature in rich_features))
            )
            
            return serp_features
            
        except Exception as e:
            self.logger.error(f"Error creating SERP features: {str(e)}")
            return pd.DataFrame()

    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features using mathematical operations."""
        try:
            interaction_features = pd.DataFrame(index=data.index)
            
            # Position × Search Volume interaction
            if all(col in data.columns for col in ['Position', 'Search Volume']):
                interaction_features['position_volume_interaction'] = (
                    (101 - data['Position']) * np.log1p(data['Search Volume'])
                )
            
            # Position × Difficulty interaction
            if all(col in data.columns for col in ['Position', 'Keyword Difficulty']):
                interaction_features['position_difficulty_interaction'] = (
                    data['Position'] * data['Keyword Difficulty']
                )
            
            # Traffic × CPC interaction (revenue potential)
            if all(col in data.columns for col in ['Traffic (%)', 'CPC']):
                interaction_features['revenue_potential'] = (
                    data['Traffic (%)'] * data['CPC']
                )
            
            # Volume × CPC interaction (market value)
            if all(col in data.columns for col in ['Search Volume', 'CPC']):
                interaction_features['market_value'] = (
                    np.log1p(data['Search Volume']) * data['CPC']
                )
            
            return interaction_features
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {str(e)}")
            return pd.DataFrame()

    def _perform_feature_selection(
        self,
        data: pd.DataFrame,
        target_column: str,
        max_features: Optional[int] = None
    ) -> pd.DataFrame:
        """Perform feature selection using optimization utilities."""
        try:
            if target_column not in data.columns:
                return data
            
            # Separate features and target
            feature_columns = [col for col in data.columns if col != target_column]
            X = data[feature_columns].select_dtypes(include=[np.number])
            y = data[target_column]
            
            if X.empty or len(X.columns) == 0:
                return data
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Correlation-based feature selection
            correlation_with_target = X.corrwith(y).abs().sort_values(ascending=False)
            
            # Remove highly correlated features (multicollinearity)
            correlation_matrix = X.corr().abs()
            
            # Find pairs of features with high correlation
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.9:
                        col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                        # Keep the one with higher correlation with target
                        if correlation_with_target[col1] > correlation_with_target[col2]:
                            high_corr_pairs.append(col2)
                        else:
                            high_corr_pairs.append(col1)
            
            # Remove highly correlated features
            features_to_keep = [col for col in X.columns if col not in high_corr_pairs]
            
            # If max_features specified, select top features by correlation with target
            if max_features and len(features_to_keep) > max_features:
                top_features = correlation_with_target[features_to_keep].head(max_features).index.tolist()
                features_to_keep = top_features
            
            # Return selected features plus target
            selected_columns = features_to_keep + [target_column]
            selected_data = data[selected_columns].copy()
            
            self.logger.info(f"Feature selection: {len(selected_columns)-1} features selected from {len(feature_columns)}")
            return selected_data
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            return data

    def _calculate_feature_importance(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate feature importance using statistical methods."""
        try:
            if not target_column or target_column not in data.columns:
                return {}
            
            # Use correlation with target as importance measure
            numeric_data = data.select_dtypes(include=[np.number])
            if target_column not in numeric_data.columns:
                return {}
            
            target = numeric_data[target_column]
            features = numeric_data.drop(columns=[target_column])
            
            importance_scores = {}
            for feature in features.columns:
                correlation = features[feature].corr(target)
                importance_scores[feature] = abs(correlation) if not pd.isna(correlation) else 0.0
            
            # Normalize importance scores
            max_importance = max(importance_scores.values()) if importance_scores else 1.0
            if max_importance > 0:
                importance_scores = {
                    feature: score / max_importance 
                    for feature, score in importance_scores.items()
                }
            
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return {}

    def _generate_feature_metadata(
        self,
        engineered_data: pd.DataFrame,
        original_data: pd.DataFrame,
        pipeline: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive feature metadata."""
        try:
            metadata = {
                'original_features': len(original_data.columns),
                'engineered_features': len(engineered_data.columns),
                'features_added': len(engineered_data.columns) - len(original_data.columns),
                'transformation_pipeline': pipeline,
                'feature_types': {},
                'data_types': engineered_data.dtypes.astype(str).to_dict(),
                'missing_value_counts': engineered_data.isnull().sum().to_dict(),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            # Categorize features by type
            for column in engineered_data.columns:
                if 'temporal' in column or any(temp in column for temp in ['lag', 'rolling', 'year', 'month']):
                    metadata['feature_types'][column] = 'temporal'
                elif 'text' in column or 'keyword' in column:
                    metadata['feature_types'][column] = 'text'
                elif 'stat' in column or any(stat in column for stat in ['zscore', 'percentile', 'outlier']):
                    metadata['feature_types'][column] = 'statistical'
                elif 'comp_' in column or 'competitive' in column:
                    metadata['feature_types'][column] = 'competitive'
                elif 'serp' in column:
                    metadata['feature_types'][column] = 'serp'
                elif 'interaction' in column:
                    metadata['feature_types'][column] = 'interaction'
                else:
                    metadata['feature_types'][column] = 'original'
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error generating feature metadata: {str(e)}")
            return {}

    def _calculate_feature_quality_metrics(
        self,
        engineered_data: pd.DataFrame,
        original_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate feature quality metrics."""
        try:
            quality_metrics = {}
            
            # Data completeness
            total_cells = engineered_data.size
            missing_cells = engineered_data.isnull().sum().sum()
            quality_metrics['completeness'] = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
            
            # Feature variance (diversity)
            numeric_data = engineered_data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                variances = numeric_data.var()
                quality_metrics['avg_feature_variance'] = variances.mean()
                quality_metrics['feature_variance_std'] = variances.std()
            
            # Correlation diversity (how correlated features are with each other)
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                # Get upper triangle of correlation matrix (excluding diagonal)
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                correlations = upper_triangle.stack().abs()
                quality_metrics['avg_feature_correlation'] = correlations.mean()
                quality_metrics['max_feature_correlation'] = correlations.max()
            
            # Dimensionality increase
            quality_metrics['dimensionality_increase'] = len(engineered_data.columns) / len(original_data.columns)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {str(e)}")
            return {}

    def export_feature_engineering_results(
        self,
        result: FeatureEngineeringResult,
        export_directory: str,
        include_feature_analysis: bool = True
    ) -> Dict[str, bool]:
        """Export comprehensive feature engineering results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Export engineered features
            features_export_success = self.data_exporter.export_with_metadata(
                result.engineered_features,
                metadata=result.feature_metadata,
                export_path=export_path / "engineered_features.xlsx"
            )
            
            # Export feature analysis report if requested
            analysis_export_success = True
            if include_feature_analysis:
                analysis_data = {
                    'feature_importance': result.feature_importance,
                    'transformation_pipeline': result.transformation_pipeline,
                    'quality_metrics': result.quality_metrics,
                    'metadata': result.feature_metadata
                }
                
                analysis_export_success = self.report_exporter.export_executive_report(
                    analysis_data,
                    export_path / "feature_engineering_analysis.html",
                    format='html',
                    include_charts=True
                )
            
            return {
                'engineered_features': features_export_success,
                'feature_analysis': analysis_export_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting feature engineering results: {str(e)}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for feature engineering operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods for specific feature types...
    def _create_keyword_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create keyword-specific metrics using StringHelper."""
        try:
            keyword_features = pd.DataFrame(index=data.index)
            
            if 'Keyword' in data.columns:
                # Advanced keyword analysis
                keyword_features['keyword_complexity'] = data['Keyword'].apply(
                    lambda x: len(set(StringHelper.clean_keyword(str(x)).split())) if pd.notna(x) else 0
                )
                
                keyword_features['keyword_uniqueness'] = data['Keyword'].apply(
                    lambda x: len(set(StringHelper.clean_keyword(str(x)))) / len(StringHelper.clean_keyword(str(x))) if pd.notna(x) and len(StringHelper.clean_keyword(str(x))) > 0 else 0
                )
            
            return keyword_features
            
        except Exception:
            return pd.DataFrame()

    def _create_position_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific metrics."""
        try:
            position_features = pd.DataFrame(index=data.index)
            
            if 'Position' in data.columns:
                # Position tier classification
                position_features['position_tier'] = pd.cut(
                    data['Position'], 
                    bins=[0, 3, 10, 20, 50, 100], 
                    labels=['top_3', 'top_10', 'top_20', 'top_50', 'beyond_50']
                )
                
                # CTR estimation based on position
                ctr_map = {1: 0.284, 2: 0.147, 3: 0.094, 4: 0.067, 5: 0.051}
                position_features['estimated_ctr'] = data['Position'].map(
                    lambda pos: ctr_map.get(pos, 0.01)
                )
            
            return position_features
            
        except Exception:
            return pd.DataFrame()

    def _create_traffic_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create traffic-specific metrics."""
        try:
            traffic_features = pd.DataFrame(index=data.index)
            
            if 'Traffic (%)' in data.columns:
                # Traffic tiers
                traffic_features['traffic_tier'] = pd.cut(
                    data['Traffic (%)'], 
                    bins=[0, 1, 5, 10, 25, float('inf')], 
                    labels=['very_low', 'low', 'medium', 'high', 'very_high']
                )
                
                # Traffic concentration
                total_traffic = data['Traffic (%)'].sum()
                traffic_features['traffic_share'] = data['Traffic (%)'] / total_traffic if total_traffic > 0 else 0
            
            return traffic_features
            
        except Exception:
            return pd.DataFrame()
