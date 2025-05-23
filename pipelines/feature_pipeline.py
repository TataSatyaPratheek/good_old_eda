"""
Feature Engineering Pipeline
Comprehensive feature engineering pipeline leveraging refactored modules and src/utils
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio

# Import refactored modules
from src.features.feature_engineer import FeatureEngineer, FeatureEngineeringResult
from src.features.feature_selector import FeatureSelector, FeatureSelectionResult
from src.features.feature_validator import FeatureValidator, FeatureValidationResult
from src.features.temporal_features import TemporalFeatureEngineer, TemporalFeatureResult
from src.features.competitive_features import CompetitiveFeatures, CompetitiveIntelligence

# Import utils framework
from src.utils.common_helpers import timing_decorator, memoize
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager
from src.utils.data_utils import DataProcessor, DataValidator, DataTransformer
from src.utils.export_utils import ReportExporter, DataExporter
from src.utils.math_utils import StatisticalCalculator

# Import pipeline configuration
from .pipeline_config import PipelineConfigManager

class FeaturePipeline:
    """
    Advanced Feature Engineering Pipeline
    
    Orchestrates comprehensive feature engineering using all refactored modules
    """
    
    def __init__(self, config_manager: Optional[PipelineConfigManager] = None):
        """Initialize feature pipeline with comprehensive utilities"""
        self.logger = LoggerFactory.get_logger("feature_pipeline")
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        
        # Configuration management
        self.config_manager = config_manager or PipelineConfigManager()
        self.pipeline_config = self.config_manager.get_pipeline_config('feature_pipeline')
        self.data_config = self.config_manager.data_config
        self.analysis_config = self.config_manager.analysis_config
        
        # Initialize refactored feature modules
        self.feature_engineer = FeatureEngineer(logger=self.logger)
        self.feature_selector = FeatureSelector(logger=self.logger)
        self.feature_validator = FeatureValidator(logger=self.logger)
        self.temporal_engineer = TemporalFeatureEngineer(logger=self.logger)
        self.competitive_features = CompetitiveFeatures(logger=self.logger)
        
        # Utilities
        self.data_processor = DataProcessor(self.logger)
        self.data_transformer = DataTransformer(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.report_exporter = ReportExporter(self.logger)
        self.data_exporter = DataExporter(self.logger)
        
        # Pipeline state
        self.pipeline_results = {}
        self.feature_catalog = {}

    @timing_decorator()
    async def run_comprehensive_feature_engineering(
        self,
        primary_data: pd.DataFrame,
        competitive_data: Optional[Dict[str, pd.DataFrame]] = None,
        target_column: Optional[str] = None,
        feature_objectives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive feature engineering pipeline
        
        Args:
            primary_data: Primary dataset (e.g., Lenovo data)
            competitive_data: Competitive datasets
            target_column: Target variable for supervised feature engineering
            feature_objectives: Specific feature engineering objectives
            
        Returns:
            Comprehensive feature engineering results
        """
        try:
            with self.performance_tracker.track_block("comprehensive_feature_engineering"):
                # Audit log pipeline execution
                self.audit_logger.log_analysis_execution(
                    user_id="pipeline_system",
                    analysis_type="comprehensive_feature_engineering",
                    parameters={
                        "primary_data_rows": len(primary_data),
                        "competitive_datasets": len(competitive_data) if competitive_data else 0,
                        "target_column": target_column,
                        "feature_objectives": feature_objectives
                    }
                )
                
                self.logger.info("Starting comprehensive feature engineering pipeline")
                
                # Phase 1: Data Preparation and Validation
                prepared_data = await self._prepare_feature_data(
                    primary_data, competitive_data
                )
                
                # Phase 2: Basic Feature Engineering
                basic_features = await self._execute_basic_feature_engineering(
                    prepared_data, target_column
                )
                
                # Phase 3: Temporal Feature Engineering
                temporal_features = await self._execute_temporal_feature_engineering(
                    prepared_data, basic_features
                )
                
                # Phase 4: Competitive Feature Engineering
                competitive_features = await self._execute_competitive_feature_engineering(
                    prepared_data, competitive_data, temporal_features
                )
                
                # Phase 5: Advanced Feature Engineering
                advanced_features = await self._execute_advanced_feature_engineering(
                    competitive_features, target_column
                )
                
                # Phase 6: Feature Selection and Optimization
                optimized_features = await self._execute_feature_selection(
                    advanced_features, target_column
                )
                
                # Phase 7: Feature Validation and Quality Assessment
                validation_results = await self._execute_feature_validation(
                    optimized_features, target_column
                )
                
                # Phase 8: Feature Catalog Creation
                feature_catalog = await self._create_feature_catalog(
                    optimized_features, validation_results
                )
                
                # Phase 9: Results Integration and Export
                integrated_results = await self._integrate_feature_results({
                    'prepared_data': prepared_data,
                    'basic_features': basic_features,
                    'temporal_features': temporal_features,
                    'competitive_features': competitive_features,
                    'advanced_features': advanced_features,
                    'optimized_features': optimized_features,
                    'validation_results': validation_results,
                    'feature_catalog': feature_catalog
                })
                
                # Export comprehensive results
                export_results = await self._export_feature_results(integrated_results)
                integrated_results['export_results'] = export_results
                
                self.pipeline_results = integrated_results
                self.logger.info("Comprehensive feature engineering pipeline completed")
                return integrated_results
                
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            await self._handle_pipeline_error(e)
            return {}

    async def _prepare_feature_data(
        self,
        primary_data: pd.DataFrame,
        competitive_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Dict[str, Any]:
        """Prepare data for feature engineering"""
        try:
            with self.performance_tracker.track_block("data_preparation"):
                self.logger.info("Preparing data for feature engineering")
                
                # Clean primary data using DataProcessor
                cleaned_primary = self.data_processor.clean_seo_data(primary_data)
                
                # Validate data quality
                validation_report = self.data_processor.validate_data_quality(cleaned_primary)
                
                # Prepare competitive data if available
                cleaned_competitive = {}
                if competitive_data:
                    for competitor, data in competitive_data.items():
                        cleaned_competitive[competitor] = self.data_processor.clean_seo_data(data)
                
                # Create data summary
                data_summary = {
                    'primary_data_shape': cleaned_primary.shape,
                    'primary_data_columns': list(cleaned_primary.columns),
                    'competitive_data_summary': {
                        comp: data.shape for comp, data in cleaned_competitive.items()
                    },
                    'data_quality_score': validation_report.quality_score,
                    'preparation_timestamp': datetime.now()
                }
                
                prepared_data = {
                    'primary_data': cleaned_primary,
                    'competitive_data': cleaned_competitive,
                    'validation_report': validation_report,
                    'data_summary': data_summary
                }
                
                self.logger.info(f"Data preparation completed: {cleaned_primary.shape} primary, {len(cleaned_competitive)} competitive")
                return prepared_data
                
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            return {}

    async def _execute_basic_feature_engineering(
        self,
        prepared_data: Dict[str, Any],
        target_column: Optional[str]
    ) -> FeatureEngineeringResult:
        """Execute basic feature engineering using FeatureEngineer"""
        try:
            with self.performance_tracker.track_block("basic_feature_engineering"):
                self.logger.info("Executing basic feature engineering")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                competitive_data = prepared_data.get('competitive_data', {})
                
                # Use comprehensive feature engineering from refactored module
                basic_result = self.feature_engineer.comprehensive_feature_engineering(
                    data=primary_data,
                    target_column=target_column,
                    competitor_data=competitive_data if competitive_data else None,
                    config=None  # Use default config
                )
                
                self.logger.info(f"Basic feature engineering completed: {len(basic_result.engineered_features.columns)} features")
                return basic_result
                
        except Exception as e:
            self.logger.error(f"Error in basic feature engineering: {str(e)}")
            return FeatureEngineeringResult(pd.DataFrame(), {}, {}, [], {}, 0.0)

    async def _execute_temporal_feature_engineering(
        self,
        prepared_data: Dict[str, Any],
        basic_features: FeatureEngineeringResult
    ) -> TemporalFeatureResult:
        """Execute temporal feature engineering using TemporalFeatureEngineer"""
        try:
            with self.performance_tracker.track_block("temporal_feature_engineering"):
                self.logger.info("Executing temporal feature engineering")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                
                # Check if date column exists
                if 'date' not in primary_data.columns:
                    self.logger.warning("No date column found for temporal features")
                    return TemporalFeatureResult(
                        pd.DataFrame(), {}, 
                        None, {}, {}
                    )
                
                # Use temporal feature engineering from refactored module
                temporal_result = self.temporal_engineer.comprehensive_temporal_feature_engineering(
                    df=primary_data,
                    date_column='date',
                    value_columns=['Position', 'Traffic (%)', 'Search Volume'],
                    config=None,  # Use default config
                    entity_column='Keyword'
                )
                
                self.logger.info(f"Temporal feature engineering completed: {len(temporal_result.engineered_features.columns)} features")
                return temporal_result
                
        except Exception as e:
            self.logger.error(f"Error in temporal feature engineering: {str(e)}")
            return TemporalFeatureResult(pd.DataFrame(), {}, None, {}, {})

    async def _execute_competitive_feature_engineering(
        self,
        prepared_data: Dict[str, Any],
        competitive_data: Optional[Dict[str, pd.DataFrame]],
        temporal_features: TemporalFeatureResult
    ) -> CompetitiveIntelligence:
        """Execute competitive feature engineering using CompetitiveFeatures"""
        try:
            with self.performance_tracker.track_block("competitive_feature_engineering"):
                self.logger.info("Executing competitive feature engineering")
                
                primary_data = prepared_data.get('primary_data', pd.DataFrame())
                
                if not competitive_data:
                    self.logger.warning("No competitive data available for competitive features")
                    return CompetitiveIntelligence({}, None, {}, [], pd.DataFrame(), {})
                
                # Use competitive feature engineering from refactored module
                competitive_result = self.competitive_features.comprehensive_competitor_analysis(
                    lenovo_data=primary_data,
                    competitor_data=competitive_data,
                    include_market_analysis=True,
                    analysis_depth='comprehensive'
                )
                
                self.logger.info("Competitive feature engineering completed")
                return competitive_result
                
        except Exception as e:
            self.logger.error(f"Error in competitive feature engineering: {str(e)}")
            return CompetitiveIntelligence({}, None, {}, [], pd.DataFrame(), {})

    async def _execute_advanced_feature_engineering(
        self,
        competitive_features: CompetitiveIntelligence,
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Execute advanced feature engineering techniques"""
        try:
            with self.performance_tracker.track_block("advanced_feature_engineering"):
                self.logger.info("Executing advanced feature engineering")
                
                # Extract base features from competitive analysis
                base_features = competitive_features.opportunity_matrix
                
                if base_features.empty:
                    return {'advanced_features': pd.DataFrame(), 'metadata': {}}
                
                # Apply advanced transformations using DataTransformer
                scaled_features = self.data_transformer.apply_scaling(
                    base_features,
                    scaling_method='standard',
                    fit_scaler=True
                )
                
                # Create polynomial features for important interactions
                interaction_features = self._create_interaction_features(scaled_features)
                
                # Combine all advanced features
                advanced_features = pd.concat([scaled_features, interaction_features], axis=1)
                
                # Calculate feature statistics using StatisticalCalculator
                feature_stats = {}
                for column in advanced_features.select_dtypes(include=[np.number]).columns:
                    stats_dict = self.stats_calculator.calculate_descriptive_statistics(
                        advanced_features[column], include_advanced=True
                    )
                    feature_stats[column] = stats_dict
                
                advanced_result = {
                    'advanced_features': advanced_features,
                    'feature_statistics': feature_stats,
                    'transformation_methods': ['scaling', 'interaction_creation'],
                    'advanced_metadata': {
                        'total_advanced_features': len(advanced_features.columns),
                        'interaction_features_count': len(interaction_features.columns),
                        'processing_timestamp': datetime.now()
                    }
                }
                
                self.logger.info(f"Advanced feature engineering completed: {len(advanced_features.columns)} features")
                return advanced_result
                
        except Exception as e:
            self.logger.error(f"Error in advanced feature engineering: {str(e)}")
            return {'advanced_features': pd.DataFrame(), 'metadata': {}}

    async def _execute_feature_selection(
        self,
        advanced_features: Dict[str, Any],
        target_column: Optional[str]
    ) -> FeatureSelectionResult:
        """Execute feature selection using FeatureSelector"""
        try:
            with self.performance_tracker.track_block("feature_selection"):
                self.logger.info("Executing feature selection")
                
                features_df = advanced_features.get('advanced_features', pd.DataFrame())
                
                if features_df.empty or not target_column or target_column not in features_df.columns:
                    self.logger.warning("Cannot perform feature selection: missing data or target")
                    return FeatureSelectionResult([], {}, {}, {}, [], [], {}, pd.DataFrame())
                
                # Separate features and target
                X = features_df.drop(columns=[target_column])
                y = features_df[target_column]
                
                # Use comprehensive feature selection from refactored module
                selection_result = self.feature_selector.comprehensive_feature_selection(
                    X=X,
                    y=y,
                    config=None,  # Use default config
                    feature_metadata=advanced_features.get('feature_statistics', {})
                )
                
                self.logger.info(f"Feature selection completed: {len(selection_result.selected_features)} features selected")
                return selection_result
                
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            return FeatureSelectionResult([], {}, {}, {}, [], [], {}, pd.DataFrame())

    async def _execute_feature_validation(
        self,
        optimized_features: FeatureSelectionResult,
        target_column: Optional[str]
    ) -> FeatureValidationResult:
        """Execute feature validation using FeatureValidator"""
        try:
            with self.performance_tracker.track_block("feature_validation"):
                self.logger.info("Executing feature validation")
                
                if not optimized_features.selected_features:
                    return FeatureValidationResult({}, {}, {}, {}, {}, {})
                
                # Get selected features data
                features_data = optimized_features.feature_importance_matrix
                
                if features_data.empty:
                    return FeatureValidationResult({}, {}, {}, {}, {}, {})
                
                # Use comprehensive feature validation from refactored module
                validation_result = self.feature_validator.comprehensive_feature_validation(
                    features_df=features_data,
                    target_column=target_column,
                    feature_metadata=optimized_features.selection_metadata,
                    config=None  # Use default config
                )
                
                self.logger.info("Feature validation completed")
                return validation_result
                
        except Exception as e:
            self.logger.error(f"Error in feature validation: {str(e)}")
            return FeatureValidationResult({}, {}, {}, {}, {}, {})

    async def _create_feature_catalog(
        self,
        optimized_features: FeatureSelectionResult,
        validation_results: FeatureValidationResult
    ) -> Dict[str, Any]:
        """Create comprehensive feature catalog"""
        try:
            with self.performance_tracker.track_block("feature_catalog_creation"):
                self.logger.info("Creating feature catalog")
                
                # Build feature catalog with comprehensive metadata
                feature_catalog = {}
                
                for feature in optimized_features.selected_features:
                    feature_info = {
                        'feature_name': feature,
                        'feature_type': self._determine_feature_type(feature),
                        'importance_score': optimized_features.feature_scores.get(feature, 0),
                        'ranking': optimized_features.feature_rankings.get(feature, 999),
                        'quality_metrics': validation_results.quality_metrics.feature_statistics.get(feature, {}),
                        'validation_status': 'validated',
                        'creation_timestamp': datetime.now(),
                        'selection_metadata': optimized_features.selection_metadata,
                        'recommended_usage': self._get_feature_usage_recommendations(feature)
                    }
                    feature_catalog[feature] = feature_info
                
                # Add catalog summary
                catalog_summary = {
                    'total_features': len(feature_catalog),
                    'feature_types': self._summarize_feature_types(feature_catalog),
                    'quality_distribution': self._summarize_quality_distribution(feature_catalog),
                    'creation_timestamp': datetime.now(),
                    'catalog_version': '1.0'
                }
                
                complete_catalog = {
                    'feature_catalog': feature_catalog,
                    'catalog_summary': catalog_summary,
                    'catalog_metadata': {
                        'total_features_processed': len(optimized_features.selected_features) + len(optimized_features.removed_features),
                        'features_selected': len(optimized_features.selected_features),
                        'features_removed': len(optimized_features.removed_features),
                        'selection_ratio': len(optimized_features.selected_features) / max(len(optimized_features.selected_features) + len(optimized_features.removed_features), 1)
                    }
                }
                
                self.feature_catalog = complete_catalog
                self.logger.info(f"Feature catalog created with {len(feature_catalog)} features")
                return complete_catalog
                
        except Exception as e:
            self.logger.error(f"Error creating feature catalog: {str(e)}")
            return {}

    async def _integrate_feature_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from all feature engineering phases"""
        try:
            with self.performance_tracker.track_block("feature_results_integration"):
                self.logger.info("Integrating feature engineering results")
                
                # Create comprehensive summary
                integrated_results = {
                    'executive_summary': self._create_feature_executive_summary(all_results),
                    'feature_engineering_summary': {
                        'basic_features_count': len(all_results.get('basic_features', FeatureEngineeringResult(pd.DataFrame(), {}, {}, [], {}, 0.0)).engineered_features.columns),
                        'temporal_features_count': len(all_results.get('temporal_features', TemporalFeatureResult(pd.DataFrame(), {}, None, {}, {})).engineered_features.columns),
                        'competitive_features_available': bool(all_results.get('competitive_features')),
                        'advanced_features_count': len(all_results.get('advanced_features', {}).get('advanced_features', pd.DataFrame()).columns),
                        'final_selected_features': len(all_results.get('optimized_features', FeatureSelectionResult([], {}, {}, {}, [], [], {}, pd.DataFrame())).selected_features)
                    },
                    'quality_assessment': {
                        'overall_quality_score': all_results.get('validation_results', FeatureValidationResult({}, {}, {}, {}, {}, {})).quality_metrics.overall_quality_score if hasattr(all_results.get('validation_results', FeatureValidationResult({}, {}, {}, {}, {}, {})).quality_metrics, 'overall_quality_score') else 0,
                        'data_preparation_quality': all_results.get('prepared_data', {}).get('validation_report', type('obj', (object,), {'quality_score': 0})).quality_score,
                        'feature_catalog_completeness': 1.0 if all_results.get('feature_catalog') else 0.0
                    },
                    'recommendations': self._generate_feature_recommendations(all_results),
                    'detailed_results': all_results,
                    'processing_metadata': {
                        'pipeline_execution_time': self.performance_tracker.get_performance_summary(),
                        'integration_timestamp': datetime.now(),
                        'processing_stages_completed': len([k for k, v in all_results.items() if v])
                    }
                }
                
                self.logger.info("Feature engineering results integration completed")
                return integrated_results
                
        except Exception as e:
            self.logger.error(f"Error integrating feature results: {str(e)}")
            return all_results

    async def _export_feature_results(self, integrated_results: Dict[str, Any]) -> Dict[str, bool]:
        """Export comprehensive feature engineering results"""
        try:
            with self.performance_tracker.track_block("feature_results_export"):
                self.logger.info("Exporting feature engineering results")
                
                export_results = {}
                
                # Export feature catalog
                feature_catalog = integrated_results.get('detailed_results', {}).get('feature_catalog', {})
                if feature_catalog:
                    catalog_export = self.data_exporter.export_analysis_dataset(
                        {'feature_catalog': pd.DataFrame.from_dict(feature_catalog.get('feature_catalog', {}), orient='index')},
                        f"{self.data_config.output_directory}/feature_catalog.xlsx"
                    )
                    export_results['feature_catalog'] = catalog_export
                
                # Export selected features
                optimized_features = integrated_results.get('detailed_results', {}).get('optimized_features')
                if optimized_features and hasattr(optimized_features, 'feature_importance_matrix') and not optimized_features.feature_importance_matrix.empty:
                    features_export = self.data_exporter.export_with_metadata(
                        optimized_features.feature_importance_matrix,
                        metadata={'analysis_type': 'selected_features', 'generation_timestamp': datetime.now()},
                        export_path=f"{self.data_config.output_directory}/selected_features.xlsx"
                    )
                    export_results['selected_features'] = features_export
                
                # Export comprehensive report
                executive_report = self.report_exporter.export_executive_report(
                    integrated_results.get('executive_summary', {}),
                    f"{self.data_config.output_directory}/feature_engineering_executive_report.html",
                    format='html',
                    include_charts=True
                )
                export_results['executive_report'] = executive_report
                
                self.logger.info("Feature engineering results export completed")
                return export_results
                
        except Exception as e:
            self.logger.error(f"Error exporting feature results: {str(e)}")
            return {}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_name': 'feature_pipeline',
            'status': 'completed' if self.pipeline_results else 'not_started',
            'feature_catalog_available': bool(self.feature_catalog),
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'results_available': bool(self.pipeline_results)
        }

    async def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline execution errors"""
        self.logger.error(f"Feature pipeline error: {str(error)}")
        self.audit_logger.log_analysis_execution(
            user_id="pipeline_system",
            analysis_type="feature_pipeline_error",
            result="failure",
            details={"error": str(error)}
        )

    # Helper methods
    def _create_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        try:
            interaction_features = pd.DataFrame(index=features_df.index)
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            # Create limited interactions to avoid explosion
            important_cols = numeric_cols[:5]  # Top 5 numeric columns
            
            for i, col1 in enumerate(important_cols):
                for col2 in important_cols[i+1:]:
                    interaction_name = f"{col1}_x_{col2}"
                    interaction_features[interaction_name] = features_df[col1] * features_df[col2]
            
            return interaction_features
        except Exception:
            return pd.DataFrame()

    def _determine_feature_type(self, feature_name: str) -> str:
        """Determine feature type from name"""
        if 'temporal' in feature_name or any(temp in feature_name for temp in ['lag', 'rolling', 'trend']):
            return 'temporal'
        elif 'competitive' in feature_name or 'comp_' in feature_name:
            return 'competitive'
        elif 'interaction' in feature_name or '_x_' in feature_name:
            return 'interaction'
        elif any(basic in feature_name for basic in ['position', 'traffic', 'volume']):
            return 'seo_metric'
        else:
            return 'engineered'

    def _get_feature_usage_recommendations(self, feature_name: str) -> List[str]:
        """Get usage recommendations for feature"""
        feature_type = self._determine_feature_type(feature_name)
        
        recommendations = {
            'temporal': ['Use for time series modeling', 'Good for trend prediction'],
            'competitive': ['Use for competitive analysis', 'Include in market intelligence models'],
            'interaction': ['Use for complex relationships', 'Good for ensemble models'],
            'seo_metric': ['Core SEO feature', 'Use in all SEO models'],
            'engineered': ['Derived feature', 'Test importance before using']
        }
        
        return recommendations.get(feature_type, ['General purpose feature'])

    def _summarize_feature_types(self, catalog: Dict[str, Any]) -> Dict[str, int]:
        """Summarize feature types in catalog"""
        type_counts = {}
        for feature_info in catalog.values():
            feature_type = feature_info.get('feature_type', 'unknown')
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1
        return type_counts

    def _summarize_quality_distribution(self, catalog: Dict[str, Any]) -> Dict[str, int]:
        """Summarize quality distribution"""
        quality_distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for feature_info in catalog.values():
            importance = feature_info.get('importance_score', 0)
            if importance > 0.7:
                quality_distribution['high'] += 1
            elif importance > 0.3:
                quality_distribution['medium'] += 1
            else:
                quality_distribution['low'] += 1
        
        return quality_distribution

    def _create_feature_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary for feature engineering"""
        return {
            'feature_engineering_scope': 'comprehensive',
            'total_features_created': sum([
                len(results.get('basic_features', FeatureEngineeringResult(pd.DataFrame(), {}, {}, [], {}, 0.0)).engineered_features.columns),
                len(results.get('temporal_features', TemporalFeatureResult(pd.DataFrame(), {}, None, {}, {})).engineered_features.columns),
                len(results.get('advanced_features', {}).get('advanced_features', pd.DataFrame()).columns)
            ]),
            'final_feature_count': len(results.get('optimized_features', FeatureSelectionResult([], {}, {}, {}, [], [], {}, pd.DataFrame())).selected_features),
            'processing_timestamp': datetime.now(),
            'pipeline_success': bool(results.get('feature_catalog'))
        }

    def _generate_feature_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate feature engineering recommendations"""
        recommendations = []
        
        selected_count = len(results.get('optimized_features', FeatureSelectionResult([], {}, {}, {}, [], [], {}, pd.DataFrame())).selected_features)
        
        if selected_count > 50:
            recommendations.append("Consider additional feature selection to reduce dimensionality")
        elif selected_count < 10:
            recommendations.append("Consider creating additional features to improve model performance")
        
        if results.get('competitive_features'):
            recommendations.append("Competitive features available - leverage for market intelligence")
        
        temporal_features = results.get('temporal_features')
        if temporal_features and hasattr(temporal_features, 'engineered_features') and not temporal_features.engineered_features.empty:
            recommendations.append("Temporal features created - use for time series modeling")
        
        return recommendations
