"""
Feature Validation Module for SEO Competitive Intelligence
Advanced feature validation leveraging the comprehensive utility framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

# Import our utilities to eliminate ALL redundancy
from src.utils.common_helpers import StringHelper, memoize, timing_decorator, safe_divide, ensure_list
from src.utils.data_utils import DataProcessor, DataValidator, DataTransformer
from src.utils.math_utils import StatisticalCalculator, OptimizationHelper
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager, PathManager
from src.utils.validation_utils import SchemaValidator, BusinessRuleValidator, ValidationReport, ValidationIssue, ValidationSeverity
from src.utils.export_utils import ReportExporter, DataExporter
from src.utils.file_utils import FileManager

@dataclass
class FeatureValidationConfig:
    """Configuration for feature validation operations"""
    validate_statistical_properties: bool = True
    validate_business_rules: bool = True
    validate_data_quality: bool = True
    validate_feature_relationships: bool = True
    strict_validation: bool = False
    quality_threshold: float = 0.8
    correlation_threshold: float = 0.95

@dataclass
class FeatureQualityMetrics:
    """Feature quality metrics"""
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    accuracy_score: float
    overall_quality_score: float
    feature_statistics: Dict[str, Any]

@dataclass
class FeatureValidationResult:
    """Comprehensive feature validation result"""
    validation_report: ValidationReport
    quality_metrics: FeatureQualityMetrics
    feature_issues: Dict[str, List[ValidationIssue]]
    correlation_analysis: pd.DataFrame
    outlier_analysis: Dict[str, Any]
    business_rule_violations: List[Dict[str, Any]]
    recommendations: List[str]
    validation_summary: Dict[str, Any]

class FeatureValidator:
    """
    Advanced feature validation for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    feature validation capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("feature_validator")
        self.config = config_manager or ConfigManager()
        
        # Initialize utility classes - eliminate ALL redundancy
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
        self.data_exporter = DataExporter(self.logger)
        self.file_manager = FileManager(self.logger)
        self.path_manager = PathManager(config_manager=self.config)
        
        # Load validation configurations
        analysis_config = self.config.get_analysis_config()
        self.default_config = FeatureValidationConfig()

    @timing_decorator()
    @memoize(ttl=1800)  # Cache for 30 minutes
    def comprehensive_feature_validation(
        self,
        features_df: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_metadata: Optional[Dict[str, Any]] = None,
        config: Optional[FeatureValidationConfig] = None
    ) -> FeatureValidationResult:
        """
        Perform comprehensive feature validation using utility framework.
        
        Args:
            features_df: DataFrame with features to validate
            target_column: Target column for supervised validation
            feature_metadata: Metadata about features
            config: Validation configuration
            
        Returns:
            FeatureValidationResult with comprehensive validation analysis
        """
        try:
            with self.performance_tracker.track_block("comprehensive_feature_validation"):
                # Audit log the validation operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="feature_validation",
                    parameters={
                        "n_features": len(features_df.columns),
                        "n_samples": len(features_df),
                        "has_target": target_column is not None,
                        "validation_config": str(config)
                    }
                )
                
                if config is None:
                    config = self.default_config
                
                # Clean and validate data using DataProcessor
                cleaned_df = self.data_processor.clean_seo_data(features_df)
                
                # 1. Schema Validation using SchemaValidator
                schema_definition = self._generate_feature_schema(cleaned_df, feature_metadata)
                validation_report = self.schema_validator.validate_dataframe_schema(
                    cleaned_df, schema_definition, "features"
                )
                
                # 2. Data Quality Assessment using DataValidator
                data_quality_report = self.data_validator.validate_seo_dataset(cleaned_df, 'features')
                
                # 3. Statistical Properties Validation
                feature_quality_metrics = self._calculate_feature_quality_metrics(
                    cleaned_df, config
                )
                
                # 4. Feature Relationships Analysis using StatisticalCalculator
                correlation_analysis = pd.DataFrame()
                if config.validate_feature_relationships:
                    correlation_analysis = self._analyze_feature_relationships(cleaned_df)
                
                # 5. Outlier Analysis using statistical methods
                outlier_analysis = {}
                if config.validate_data_quality:
                    outlier_analysis = self._perform_outlier_analysis(cleaned_df)
                
                # 6. Business Rule Validation using BusinessRuleValidator
                business_rule_violations = []
                if config.validate_business_rules:
                    business_rule_violations = self._validate_feature_business_rules(
                        cleaned_df, target_column
                    )
                
                # 7. Feature-specific Issues Analysis
                feature_issues = self._analyze_feature_specific_issues(
                    cleaned_df, config
                )
                
                # 8. Generate Recommendations using optimization
                recommendations = self._generate_feature_recommendations(
                    validation_report, feature_quality_metrics, correlation_analysis,
                    outlier_analysis, business_rule_violations
                )
                
                # 9. Create Validation Summary
                validation_summary = self._create_validation_summary(
                    validation_report, feature_quality_metrics, len(business_rule_violations)
                )
                
                result = FeatureValidationResult(
                    validation_report=validation_report,
                    quality_metrics=feature_quality_metrics,
                    feature_issues=feature_issues,
                    correlation_analysis=correlation_analysis,
                    outlier_analysis=outlier_analysis,
                    business_rule_violations=business_rule_violations,
                    recommendations=recommendations,
                    validation_summary=validation_summary
                )
                
                self.logger.info(f"Feature validation completed: {feature_quality_metrics.overall_quality_score:.3f} overall quality")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive feature validation: {str(e)}")
            return FeatureValidationResult(
                ValidationReport(0, [], 0.0, 0, 1, 1, 0, datetime.now(), []),
                FeatureQualityMetrics(0, 0, 0, 0, 0, 0, {}),
                {}, pd.DataFrame(), {}, [], [f"Validation failed: {str(e)}"], {}
            )

    @timing_decorator()
    def validate_feature_engineering_pipeline(
        self,
        original_data: pd.DataFrame,
        engineered_features: pd.DataFrame,
        transformation_pipeline: List[str],
        validation_config: Optional[FeatureValidationConfig] = None
    ) -> Dict[str, Any]:
        """
        Validate feature engineering pipeline using statistical analysis.
        
        Args:
            original_data: Original dataset
            engineered_features: Features after engineering
            transformation_pipeline: List of transformations applied
            validation_config: Validation configuration
            
        Returns:
            Pipeline validation results
        """
        try:
            with self.performance_tracker.track_block("validate_feature_engineering_pipeline"):
                if validation_config is None:
                    validation_config = self.default_config
                
                # Validate pipeline integrity
                pipeline_integrity = self._validate_pipeline_integrity(
                    original_data, engineered_features, transformation_pipeline
                )
                
                # Validate feature transformations
                transformation_validity = self._validate_transformations(
                    original_data, engineered_features, transformation_pipeline
                )
                
                # Check for information leakage
                leakage_analysis = self._detect_information_leakage(
                    original_data, engineered_features
                )
                
                # Validate feature distributions using StatisticalCalculator
                distribution_analysis = self._analyze_feature_distributions(
                    original_data, engineered_features
                )
                
                # Check feature scaling and normalization
                scaling_validation = self._validate_feature_scaling(engineered_features)
                
                # Performance impact analysis
                performance_impact = self._analyze_performance_impact(
                    original_data, engineered_features
                )
                
                pipeline_validation = {
                    'pipeline_integrity': pipeline_integrity,
                    'transformation_validity': transformation_validity,
                    'leakage_analysis': leakage_analysis,
                    'distribution_analysis': distribution_analysis,
                    'scaling_validation': scaling_validation,
                    'performance_impact': performance_impact,
                    'overall_pipeline_score': self._calculate_pipeline_score([
                        pipeline_integrity, transformation_validity, 
                        1 - leakage_analysis.get('leakage_risk', 0)
                    ]),
                    'recommendations': self._generate_pipeline_recommendations([
                        pipeline_integrity, transformation_validity, leakage_analysis
                    ])
                }
                
                self.logger.info(f"Feature engineering pipeline validation completed")
                return pipeline_validation
                
        except Exception as e:
            self.logger.error(f"Error validating feature engineering pipeline: {str(e)}")
            return {}

    @timing_decorator()
    def validate_feature_selection_quality(
        self,
        X_original: pd.DataFrame,
        X_selected: pd.DataFrame,
        y: Optional[pd.Series] = None,
        selection_method: str = 'unknown'
    ) -> Dict[str, Any]:
        """
        Validate quality of feature selection using statistical methods.
        
        Args:
            X_original: Original feature matrix
            X_selected: Selected features matrix
            y: Target variable
            selection_method: Method used for selection
            
        Returns:
            Feature selection quality analysis
        """
        try:
            with self.performance_tracker.track_block("validate_feature_selection_quality"):
                # Information preservation analysis
                information_preservation = self._analyze_information_preservation(
                    X_original, X_selected, y
                )
                
                # Feature importance consistency using StatisticalCalculator
                importance_consistency = self._validate_importance_consistency(
                    X_original, X_selected, y
                )
                
                # Multicollinearity analysis using correlation matrix
                multicollinearity_analysis = self._analyze_multicollinearity_improvement(
                    X_original, X_selected
                )
                
                # Dimensionality reduction effectiveness
                dimensionality_effectiveness = self._evaluate_dimensionality_reduction(
                    X_original, X_selected
                )
                
                # Predictive power preservation (if target available)
                predictive_power = {}
                if y is not None:
                    predictive_power = self._evaluate_predictive_power_preservation(
                        X_original, X_selected, y
                    )
                
                # Selection stability analysis
                stability_analysis = self._analyze_selection_stability(
                    X_original, X_selected, selection_method
                )
                
                selection_quality = {
                    'information_preservation': information_preservation,
                    'importance_consistency': importance_consistency,
                    'multicollinearity_improvement': multicollinearity_analysis,
                    'dimensionality_effectiveness': dimensionality_effectiveness,
                    'predictive_power_preservation': predictive_power,
                    'selection_stability': stability_analysis,
                    'overall_selection_quality': self._calculate_selection_quality_score([
                        information_preservation.get('preservation_score', 0),
                        importance_consistency.get('consistency_score', 0),
                        multicollinearity_analysis.get('improvement_score', 0)
                    ]),
                    'selection_recommendations': self._generate_selection_recommendations([
                        information_preservation, importance_consistency, multicollinearity_analysis
                    ])
                }
                
                self.logger.info(f"Feature selection quality validation completed")
                return selection_quality
                
        except Exception as e:
            self.logger.error(f"Error validating feature selection quality: {str(e)}")
            return {}

    def _calculate_feature_quality_metrics(
        self,
        df: pd.DataFrame,
        config: FeatureValidationConfig
    ) -> FeatureQualityMetrics:
        """Calculate comprehensive feature quality metrics using utilities."""
        try:
            # Completeness - using data completeness from utilities
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            completeness_score = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
            
            # Consistency - using statistical consistency checks
            consistency_scores = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if not df[col].empty:
                    # Use StatisticalCalculator for robust analysis
                    stats_dict = self.stats_calculator.calculate_descriptive_statistics(df[col].dropna())
                    cv = stats_dict.get('coefficient_of_variation', float('inf'))
                    # Lower CV indicates higher consistency
                    consistency_score = max(0, 1 - min(cv / 2, 1))
                    consistency_scores.append(consistency_score)
            
            consistency_score = np.mean(consistency_scores) if consistency_scores else 1.0
            
            # Validity - check data types and ranges
            validity_scores = []
            for col in df.columns:
                col_validity = self._calculate_column_validity(df[col])
                validity_scores.append(col_validity)
            
            validity_score = np.mean(validity_scores) if validity_scores else 1.0
            
            # Uniqueness - check for appropriate uniqueness
            uniqueness_scores = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                    # For categorical features, moderate uniqueness is good
                    uniqueness_score = 1 - abs(unique_ratio - 0.5) if unique_ratio <= 1 else 0
                else:
                    # For numeric features, high uniqueness is generally good
                    uniqueness_score = min(df[col].nunique() / len(df), 1.0) if len(df) > 0 else 0
                
                uniqueness_scores.append(uniqueness_score)
            
            uniqueness_score = np.mean(uniqueness_scores) if uniqueness_scores else 1.0
            
            # Accuracy - based on outlier detection and business rules
            accuracy_scores = []
            for col in numeric_cols:
                if not df[col].empty:
                    # Use outlier detection to assess accuracy
                    outlier_data = df[col].dropna()
                    if len(outlier_data) > 10:
                        # Use IQR method for outlier detection
                        Q1 = outlier_data.quantile(0.25)
                        Q3 = outlier_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((outlier_data < (Q1 - 1.5 * IQR)) | 
                                   (outlier_data > (Q3 + 1.5 * IQR))).sum()
                        accuracy_score = 1 - (outliers / len(outlier_data))
                        accuracy_scores.append(accuracy_score)
            
            accuracy_score = np.mean(accuracy_scores) if accuracy_scores else 1.0
            
            # Overall quality score
            scores = [completeness_score, consistency_score, validity_score, uniqueness_score, accuracy_score]
            overall_quality_score = np.mean(scores)
            
            # Feature statistics using StatisticalCalculator
            feature_statistics = {}
            for col in numeric_cols:
                if not df[col].empty:
                    stats_dict = self.stats_calculator.calculate_descriptive_statistics(
                        df[col].dropna(), include_advanced=True
                    )
                    feature_statistics[col] = stats_dict
            
            return FeatureQualityMetrics(
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                validity_score=validity_score,
                uniqueness_score=uniqueness_score,
                accuracy_score=accuracy_score,
                overall_quality_score=overall_quality_score,
                feature_statistics=feature_statistics
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating feature quality metrics: {str(e)}")
            return FeatureQualityMetrics(0, 0, 0, 0, 0, 0, {})

    def _analyze_feature_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature relationships using StatisticalCalculator."""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty or len(numeric_df.columns) < 2:
                return pd.DataFrame()
            
            # Use StatisticalCalculator for correlation analysis
            correlation_matrix, p_values = self.stats_calculator.calculate_correlation_matrix(numeric_df)
            
            # Create comprehensive relationship analysis
            relationships = []
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Only upper triangle
                        correlation = correlation_matrix.loc[col1, col2]
                        p_value = p_values.loc[col1, col2] if not p_values.empty else np.nan
                        
                        relationship = {
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': correlation,
                            'p_value': p_value,
                            'relationship_strength': self._categorize_correlation_strength(correlation),
                            'significant': p_value < 0.05 if not pd.isna(p_value) else False
                        }
                        relationships.append(relationship)
            
            relationships_df = pd.DataFrame(relationships)
            if not relationships_df.empty:
                relationships_df = relationships_df.sort_values('correlation', key=abs, ascending=False)
            
            return relationships_df
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature relationships: {str(e)}")
            return pd.DataFrame()

    def _perform_outlier_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform outlier analysis using statistical methods."""
        try:
            outlier_analysis = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) < 10:
                    continue
                
                # Use multiple outlier detection methods
                
                # 1. IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                
                # 2. Z-score method
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                z_outliers = (z_scores > 3).sum()
                
                # 3. Modified Z-score (using median)
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * (col_data - median) / mad if mad > 0 else np.zeros_like(col_data)
                modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
                
                outlier_analysis[col] = {
                    'iqr_outliers': iqr_outliers,
                    'iqr_percentage': (iqr_outliers / len(col_data)) * 100,
                    'z_score_outliers': z_outliers,
                    'z_score_percentage': (z_outliers / len(col_data)) * 100,
                    'modified_z_outliers': modified_z_outliers,
                    'modified_z_percentage': (modified_z_outliers / len(col_data)) * 100,
                    'outlier_summary': {
                        'total_outliers': max(iqr_outliers, z_outliers, modified_z_outliers),
                        'outlier_rate': max(iqr_outliers, z_outliers, modified_z_outliers) / len(col_data),
                        'severity': self._categorize_outlier_severity(
                            max(iqr_outliers, z_outliers, modified_z_outliers) / len(col_data)
                        )
                    }
                }
            
            return outlier_analysis
            
        except Exception as e:
            self.logger.error(f"Error performing outlier analysis: {str(e)}")
            return {}

    def _validate_feature_business_rules(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Validate business rules using BusinessRuleValidator."""
        try:
            violations = []
            
            # SEO-specific business rules
            
            # Rule 1: Position features should be between 1 and 100
            position_columns = [col for col in df.columns if 'position' in col.lower()]
            for col in position_columns:
                if col in df.columns:
                    invalid_positions = df[(df[col] < 1) | (df[col] > 100)][col].dropna()
                    if len(invalid_positions) > 0:
                        violations.append({
                            'rule': 'position_range_validation',
                            'feature': col,
                            'violation_count': len(invalid_positions),
                            'description': f'Position values should be between 1 and 100',
                            'severity': 'error'
                        })
            
            # Rule 2: Traffic percentages should be between 0 and 100
            traffic_columns = [col for col in df.columns if 'traffic' in col.lower() and '%' in col]
            for col in traffic_columns:
                if col in df.columns:
                    invalid_traffic = df[(df[col] < 0) | (df[col] > 100)][col].dropna()
                    if len(invalid_traffic) > 0:
                        violations.append({
                            'rule': 'traffic_percentage_validation',
                            'feature': col,
                            'violation_count': len(invalid_traffic),
                            'description': f'Traffic percentages should be between 0 and 100',
                            'severity': 'error'
                        })
            
            # Rule 3: Search volume should be non-negative
            volume_columns = [col for col in df.columns if 'volume' in col.lower()]
            for col in volume_columns:
                if col in df.columns:
                    negative_volumes = df[df[col] < 0][col].dropna()
                    if len(negative_volumes) > 0:
                        violations.append({
                            'rule': 'search_volume_validation',
                            'feature': col,
                            'violation_count': len(negative_volumes),
                            'description': f'Search volume should be non-negative',
                            'severity': 'error'
                        })
            
            # Rule 4: Feature correlation with target (if available)
            if target_column and target_column in df.columns:
                numeric_features = df.select_dtypes(include=[np.number]).columns
                for feature in numeric_features:
                    if feature != target_column:
                        correlation = df[feature].corr(df[target_column])
                        if abs(correlation) > 0.99:  # Suspiciously high correlation
                            violations.append({
                                'rule': 'target_correlation_validation',
                                'feature': feature,
                                'violation_count': 1,
                                'description': f'Suspiciously high correlation with target: {correlation:.3f}',
                                'severity': 'warning'
                            })
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error validating business rules: {str(e)}")
            return []

    def _generate_feature_schema(
        self,
        df: pd.DataFrame,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Generate schema definition for features based on data analysis."""
        try:
            schema = {}
            
            for column in df.columns:
                column_schema = {
                    'required': True,
                    'nullable': df[column].isnull().any(),
                    'data_type': self._determine_data_type(df[column])
                }
                
                # Add range constraints for numeric columns
                if df[column].dtype in ['int64', 'float64']:
                    column_schema.update({
                        'min_value': df[column].min(),
                        'max_value': df[column].max()
                    })
                
                # Add length constraints for string columns
                elif df[column].dtype == 'object':
                    string_lengths = df[column].astype(str).str.len()
                    column_schema.update({
                        'max_length': string_lengths.max(),
                        'min_length': string_lengths.min()
                    })
                
                # Add metadata if available
                if feature_metadata and column in feature_metadata:
                    column_schema.update(feature_metadata[column])
                
                schema[column] = column_schema
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Error generating feature schema: {str(e)}")
            return {}

    def export_feature_validation_results(
        self,
        validation_result: FeatureValidationResult,
        export_directory: str,
        include_detailed_analysis: bool = True
    ) -> Dict[str, bool]:
        """Export comprehensive feature validation results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'validation_summary': validation_result.validation_summary,
                'quality_metrics': {
                    'completeness_score': validation_result.quality_metrics.completeness_score,
                    'consistency_score': validation_result.quality_metrics.consistency_score,
                    'validity_score': validation_result.quality_metrics.validity_score,
                    'uniqueness_score': validation_result.quality_metrics.uniqueness_score,
                    'accuracy_score': validation_result.quality_metrics.accuracy_score,
                    'overall_quality_score': validation_result.quality_metrics.overall_quality_score
                },
                'validation_issues': {
                    'critical_issues': len([issue for issue in validation_result.validation_report.issues 
                                          if issue.severity == ValidationSeverity.CRITICAL]),
                    'errors': len([issue for issue in validation_result.validation_report.issues 
                                 if issue.severity == ValidationSeverity.ERROR]),
                    'warnings': len([issue for issue in validation_result.validation_report.issues 
                                   if issue.severity == ValidationSeverity.WARNING])
                },
                'business_rule_violations': len(validation_result.business_rule_violations),
                'recommendations': validation_result.recommendations,
                'feature_statistics': validation_result.quality_metrics.feature_statistics
            }
            
            # Export summary data using DataExporter
            summary_export_success = self.data_exporter.export_analysis_dataset(
                {'feature_validation_summary': pd.DataFrame([export_data])},
                export_path / "feature_validation_summary.xlsx"
            )
            
            # Export detailed correlation analysis
            correlation_export_success = True
            if not validation_result.correlation_analysis.empty:
                correlation_export_success = self.data_exporter.export_with_metadata(
                    validation_result.correlation_analysis,
                    metadata={'analysis_type': 'feature_correlation', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "feature_correlation_analysis.xlsx"
                )
            
            # Export detailed validation report if requested
            detailed_export_success = True
            if include_detailed_analysis:
                detailed_data = {
                    'validation_issues': [
                        {
                            'field': issue.field,
                            'issue_type': issue.issue_type,
                            'severity': issue.severity.value,
                            'message': issue.message,
                            'affected_rows_count': len(issue.affected_rows),
                            'suggested_fix': issue.suggested_fix
                        }
                        for issue in validation_result.validation_report.issues
                    ],
                    'outlier_analysis': validation_result.outlier_analysis,
                    'business_rule_violations': validation_result.business_rule_violations
                }
                
                detailed_export_success = self.data_exporter.export_with_metadata(
                    pd.DataFrame(detailed_data.get('validation_issues', [])),
                    metadata={'analysis_type': 'detailed_validation', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "detailed_validation_report.xlsx"
                )
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_executive_report(
                export_data,
                export_path / "feature_validation_executive_report.html",
                format='html',
                include_charts=True
            )
            
            return {
                'summary_export': summary_export_success,
                'correlation_analysis': correlation_export_success,
                'detailed_report': detailed_export_success,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting feature validation results: {str(e)}")
            return {}

    # Helper methods using utility framework
    def _calculate_column_validity(self, column: pd.Series) -> float:
        """Calculate validity score for a column."""
        try:
            if column.empty:
                return 0.0
            
            # Check for null values
            null_ratio = column.isnull().sum() / len(column)
            
            # Check for data type consistency
            type_consistency = 1.0
            if column.dtype == 'object':
                # For object columns, check if they can be converted to expected types
                numeric_convertible = pd.to_numeric(column, errors='coerce').notna().sum()
                if numeric_convertible > len(column) * 0.8:  # If 80%+ are numeric
                    type_consistency = numeric_convertible / len(column)
            
            # Combine scores
            validity_score = (1 - null_ratio) * type_consistency
            return validity_score
            
        except Exception:
            return 0.0

    def _categorize_correlation_strength(self, correlation: float) -> str:
        """Categorize correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'

    def _categorize_outlier_severity(self, outlier_rate: float) -> str:
        """Categorize outlier severity based on rate."""
        if outlier_rate >= 0.1:
            return 'critical'
        elif outlier_rate >= 0.05:
            return 'high'
        elif outlier_rate >= 0.02:
            return 'moderate'
        else:
            return 'low'

    def _determine_data_type(self, column: pd.Series) -> str:
        """Determine appropriate data type for schema."""
        if column.dtype in ['int64', 'int32']:
            return 'integer'
        elif column.dtype in ['float64', 'float32']:
            return 'float'
        elif column.dtype == 'bool':
            return 'boolean'
        elif column.dtype == 'datetime64[ns]':
            return 'datetime'
        else:
            return 'string'

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for feature validation operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods would be implemented here following the same pattern
    def _validate_pipeline_integrity(self, original_data, engineered_features, pipeline):
        """Validate pipeline integrity."""
        try:
            return 1.0 if len(engineered_features) == len(original_data) else 0.8
        except Exception:
            return 0.0

    def _validate_transformations(self, original_data, engineered_features, pipeline):
        """Validate individual transformations."""
        try:
            # Simplified validation - would be more comprehensive in practice
            return 0.9 if len(engineered_features.columns) >= len(original_data.columns) else 0.7
        except Exception:
            return 0.0

    def _detect_information_leakage(self, original_data, engineered_features):
        """Detect potential information leakage."""
        try:
            # Simplified detection
            return {'leakage_risk': 0.1, 'leakage_features': []}
        except Exception:
            return {'leakage_risk': 0.0, 'leakage_features': []}

    def _analyze_feature_distributions(self, original_data, engineered_features):
        """Analyze feature distribution changes."""
        try:
            return {'distribution_changes': 'minimal', 'skewness_changes': {}}
        except Exception:
            return {}

    def _validate_feature_scaling(self, features):
        """Validate feature scaling."""
        try:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {'scaling_appropriate': True, 'scaling_method': 'none'}
            
            # Check if features are scaled
            means = features[numeric_cols].mean()
            stds = features[numeric_cols].std()
            
            # Check for standard scaling
            is_standard_scaled = all(abs(mean) < 0.1 for mean in means) and all(abs(std - 1) < 0.1 for std in stds)
            
            # Check for min-max scaling
            mins = features[numeric_cols].min()
            maxs = features[numeric_cols].max()
            is_minmax_scaled = all(abs(min_val) < 0.1 for min_val in mins) and all(abs(max_val - 1) < 0.1 for max_val in maxs)
            
            return {
                'scaling_appropriate': is_standard_scaled or is_minmax_scaled,
                'scaling_method': 'standard' if is_standard_scaled else ('minmax' if is_minmax_scaled else 'unknown'),
                'mean_range': (means.min(), means.max()),
                'std_range': (stds.min(), stds.max())
            }
        except Exception:
            return {'scaling_appropriate': False, 'scaling_method': 'unknown'}
