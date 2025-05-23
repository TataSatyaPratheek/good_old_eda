"""
Schema Validation Module for SEO Competitive Intelligence
Advanced schema validation leveraging the comprehensive utility framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Import our utilities to eliminate ALL redundancy
from src.utils.validation_utils import SchemaValidator as BaseSchemaValidator, ValidationReport, ValidationIssue, ValidationSeverity
from src.utils.common_helpers import StringHelper, memoize, timing_decorator, ensure_list
from src.utils.data_utils import DataValidator, DataProcessor
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager
from src.utils.math_utils import StatisticalCalculator

@dataclass
class SEOSchemaValidationResult:
    """Extended validation result for SEO-specific validation"""
    base_validation_report: ValidationReport
    seo_specific_issues: List[ValidationIssue]
    data_quality_metrics: Dict[str, float]
    business_rule_violations: List[Dict[str, Any]]
    recommendations: List[str]
    performance_metrics: Dict[str, Any]

class SchemaValidator:
    """
    SEO-specific schema validator that leverages the comprehensive utility framework.
    
    This class extends the base validation capabilities with SEO domain-specific
    validation rules while eliminating all redundant code by using our utilities.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("schema_validator")
        self.config = config_manager or ConfigManager()
        
        # Use existing utility classes instead of reimplementing
        self.base_validator = BaseSchemaValidator(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.data_processor = DataProcessor(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        
        # Load SEO-specific schemas from config instead of hardcoding
        self.seo_schemas = self._load_seo_schemas_from_config()
        
        # Get validation thresholds from config
        analysis_config = self.config.get_analysis_config()
        self.quality_threshold = 0.8
        self.critical_issue_threshold = 0

    @timing_decorator()
    @memoize(ttl=1800)  # Cache schema validation for 30 minutes
    def validate_seo_dataset(
        self,
        df: pd.DataFrame,
        dataset_type: str = 'positions',
        include_business_rules: bool = True,
        include_quality_metrics: bool = True
    ) -> SEOSchemaValidationResult:
        """
        Comprehensive SEO dataset validation using utility framework.
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of SEO dataset ('positions', 'competitors', 'gap_keywords')
            include_business_rules: Whether to validate business rules
            include_quality_metrics: Whether to calculate quality metrics
            
        Returns:
            SEOSchemaValidationResult with comprehensive validation
        """
        try:
            with self.performance_tracker.track_block("validate_seo_dataset"):
                # Audit log the validation request
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="schema_validation",
                    parameters={
                        "dataset_type": dataset_type,
                        "record_count": len(df),
                        "include_business_rules": include_business_rules
                    }
                )
                
                # Get appropriate schema for dataset type
                schema_definition = self.seo_schemas.get(dataset_type, {})
                if not schema_definition:
                    raise ValueError(f"No schema definition found for dataset type: {dataset_type}")
                
                # Perform base validation using existing SchemaValidator
                base_report = self.base_validator.validate_dataframe_schema(
                    df, schema_definition, dataset_type
                )
                
                # SEO-specific validation using existing DataValidator
                seo_validation_report = self.data_validator.validate_seo_dataset(df, dataset_type)
                
                # Extract SEO-specific issues
                seo_specific_issues = self._extract_seo_specific_issues(
                    df, dataset_type, seo_validation_report
                )
                
                # Calculate data quality metrics using StatisticalCalculator
                quality_metrics = {}
                if include_quality_metrics:
                    quality_metrics = self._calculate_quality_metrics(df, dataset_type)
                
                # Validate business rules if requested
                business_rule_violations = []
                if include_business_rules:
                    business_rule_violations = self._validate_seo_business_rules(df, dataset_type)
                
                # Generate enhanced recommendations
                recommendations = self._generate_enhanced_recommendations(
                    base_report, seo_specific_issues, quality_metrics, business_rule_violations
                )
                
                # Get performance metrics
                performance_metrics = self.performance_tracker.get_performance_summary(
                    operation_filter="validate_seo_dataset",
                    time_window_minutes=5
                )
                
                result = SEOSchemaValidationResult(
                    base_validation_report=base_report,
                    seo_specific_issues=seo_specific_issues,
                    data_quality_metrics=quality_metrics,
                    business_rule_violations=business_rule_violations,
                    recommendations=recommendations,
                    performance_metrics=performance_metrics
                )
                
                self.logger.info(f"SEO schema validation completed for {dataset_type}: {len(seo_specific_issues)} SEO issues found")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in SEO schema validation: {str(e)}")
            self.audit_logger.log_analysis_execution(
                user_id="system",
                analysis_type="schema_validation",
                parameters={"dataset_type": dataset_type},
                result="failure",
                details={"error": str(e)}
            )
            return SEOSchemaValidationResult(
                ValidationReport(0, [], 0.0, 0, 1, 1, 0, datetime.now(), []),
                [], {}, [], [f"Validation failed: {str(e)}"], {}
            )

    @timing_decorator()
    def validate_keyword_quality(
        self,
        keywords: Union[pd.Series, List[str]],
        strict_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Validate keyword quality using StringHelper utilities.
        
        Args:
            keywords: Keywords to validate
            strict_mode: Whether to use strict validation rules
            
        Returns:
            Keyword quality validation results
        """
        try:
            with self.performance_tracker.track_block("validate_keyword_quality"):
                keyword_list = ensure_list(keywords)
                if isinstance(keywords, pd.Series):
                    keyword_list = keywords.tolist()
                
                quality_results = {
                    'total_keywords': len(keyword_list),
                    'valid_keywords': 0,
                    'invalid_keywords': [],
                    'quality_issues': [],
                    'cleaned_keywords': [],
                    'duplicate_keywords': [],
                    'quality_score': 0.0
                }
                
                # Process each keyword using StringHelper
                seen_keywords = set()
                for i, keyword in enumerate(keyword_list):
                    if pd.isna(keyword):
                        quality_results['invalid_keywords'].append({
                            'index': i,
                            'keyword': keyword,
                            'issue': 'null_value'
                        })
                        continue
                    
                    # Clean keyword using StringHelper
                    cleaned_keyword = StringHelper.clean_keyword(str(keyword))
                    quality_results['cleaned_keywords'].append(cleaned_keyword)
                    
                    # Check for duplicates
                    if cleaned_keyword.lower() in seen_keywords:
                        quality_results['duplicate_keywords'].append({
                            'index': i,
                            'keyword': keyword,
                            'cleaned': cleaned_keyword
                        })
                    else:
                        seen_keywords.add(cleaned_keyword.lower())
                    
                    # Validate keyword quality
                    keyword_issues = self._validate_single_keyword(keyword, cleaned_keyword, strict_mode)
                    if keyword_issues:
                        quality_results['quality_issues'].extend(keyword_issues)
                    else:
                        quality_results['valid_keywords'] += 1
                
                # Calculate quality score
                if quality_results['total_keywords'] > 0:
                    quality_results['quality_score'] = (
                        quality_results['valid_keywords'] / quality_results['total_keywords']
                    )
                
                self.logger.info(f"Keyword quality validation: {quality_results['valid_keywords']}/{quality_results['total_keywords']} valid")
                return quality_results
                
        except Exception as e:
            self.logger.error(f"Error in keyword quality validation: {str(e)}")
            return {}

    @timing_decorator()
    def validate_competitor_data_consistency(
        self,
        competitor_datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Validate consistency across competitor datasets using DataProcessor.
        
        Args:
            competitor_datasets: Dictionary of competitor DataFrames
            
        Returns:
            Consistency validation results
        """
        try:
            with self.performance_tracker.track_block("validate_competitor_data_consistency"):
                # Use DataProcessor to check consistency
                consistency_results = {
                    'total_datasets': len(competitor_datasets),
                    'schema_consistency': {},
                    'data_consistency': {},
                    'standardization_needed': [],
                    'overall_consistency_score': 0.0
                }
                
                if not competitor_datasets:
                    return consistency_results
                
                # Get reference schema from first dataset
                reference_name, reference_df = next(iter(competitor_datasets.items()))
                reference_columns = set(reference_df.columns)
                reference_dtypes = reference_df.dtypes.to_dict()
                
                schema_scores = []
                
                for competitor_name, df in competitor_datasets.items():
                    competitor_columns = set(df.columns)
                    competitor_dtypes = df.dtypes.to_dict()
                    
                    # Check schema consistency
                    missing_columns = reference_columns - competitor_columns
                    extra_columns = competitor_columns - reference_columns
                    
                    # Check data type consistency
                    dtype_mismatches = []
                    for col in reference_columns.intersection(competitor_columns):
                        if reference_dtypes[col] != competitor_dtypes[col]:
                            dtype_mismatches.append({
                                'column': col,
                                'reference_type': str(reference_dtypes[col]),
                                'competitor_type': str(competitor_dtypes[col])
                            })
                    
                    # Calculate consistency score for this competitor
                    schema_score = 1.0
                    if missing_columns:
                        schema_score -= len(missing_columns) / len(reference_columns) * 0.5
                    if extra_columns:
                        schema_score -= len(extra_columns) / len(reference_columns) * 0.3
                    if dtype_mismatches:
                        schema_score -= len(dtype_mismatches) / len(reference_columns) * 0.2
                    
                    schema_scores.append(max(0.0, schema_score))
                    
                    consistency_results['schema_consistency'][competitor_name] = {
                        'missing_columns': list(missing_columns),
                        'extra_columns': list(extra_columns),
                        'dtype_mismatches': dtype_mismatches,
                        'schema_score': schema_score
                    }
                    
                    # Check if standardization is needed
                    if missing_columns or extra_columns or dtype_mismatches:
                        consistency_results['standardization_needed'].append(competitor_name)
                
                # Calculate overall consistency score
                consistency_results['overall_consistency_score'] = np.mean(schema_scores) if schema_scores else 0.0
                
                self.logger.info(f"Competitor data consistency validation completed: {consistency_results['overall_consistency_score']:.3f} score")
                return consistency_results
                
        except Exception as e:
            self.logger.error(f"Error in competitor data consistency validation: {str(e)}")
            return {}

    def _load_seo_schemas_from_config(self) -> Dict[str, Dict[str, Any]]:
        """Load SEO schema definitions from configuration."""
        try:
            # Try to get from config, fallback to defaults
            config_schemas = getattr(self.config, 'seo_schemas', {})
            
            # Merge with defaults from base validator
            default_schemas = {
                'positions': {
                    'Keyword': {'required': True, 'data_type': 'string', 'nullable': False},
                    'Position': {'required': True, 'data_type': 'integer', 'min_value': 1, 'max_value': 100},
                    'Search Volume': {'required': True, 'data_type': 'integer', 'min_value': 0},
                    'Traffic (%)': {'required': False, 'data_type': 'float', 'min_value': 0},
                    'URL': {'required': False, 'data_type': 'string'},
                    'CPC': {'required': False, 'data_type': 'float', 'min_value': 0},
                    'Keyword Difficulty': {'required': False, 'data_type': 'integer', 'min_value': 0, 'max_value': 100},
                    'SERP Features by Keyword': {'required': False, 'data_type': 'string'}
                },
                'competitors': {
                    'Domain': {'required': True, 'data_type': 'string', 'nullable': False},
                    'Organic Keywords': {'required': True, 'data_type': 'integer', 'min_value': 0},
                    'Organic Traffic': {'required': True, 'data_type': 'integer', 'min_value': 0},
                    'Competitor Relevance': {'required': False, 'data_type': 'float', 'min_value': 0, 'max_value': 1}
                },
                'gap_keywords': {
                    'Keyword': {'required': True, 'data_type': 'string', 'nullable': False},
                    'Volume': {'required': True, 'data_type': 'integer', 'min_value': 0},
                    'lenovo.com': {'required': False, 'data_type': 'integer', 'min_value': 0, 'max_value': 100},
                    'dell.com': {'required': False, 'data_type': 'integer', 'min_value': 0, 'max_value': 100},
                    'hp.com': {'required': False, 'data_type': 'integer', 'min_value': 0, 'max_value': 100}
                }
            }
            
            # Merge config with defaults
            for schema_type, schema_def in default_schemas.items():
                if schema_type in config_schemas:
                    schema_def.update(config_schemas[schema_type])
            
            return default_schemas
            
        except Exception as e:
            self.logger.error(f"Error loading SEO schemas: {str(e)}")
            return {}

    def _extract_seo_specific_issues(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        validation_report
    ) -> List[ValidationIssue]:
        """Extract SEO-specific validation issues."""
        try:
            seo_issues = []
            
            # URL validation using StringHelper
            if 'URL' in df.columns:
                for idx, url in enumerate(df['URL']):
                    if pd.notna(url):
                        domain = StringHelper.extract_domain_from_url(str(url))
                        if not domain:
                            seo_issues.append(ValidationIssue(
                                field='URL',
                                issue_type='invalid_url',
                                severity=ValidationSeverity.ERROR,
                                message=f"Invalid URL format: {url}",
                                affected_rows=[idx],
                                suggested_fix="Correct URL format"
                            ))
            
            # SERP features validation using StringHelper
            if 'SERP Features by Keyword' in df.columns:
                for idx, features in enumerate(df['SERP Features by Keyword']):
                    if pd.notna(features):
                        normalized_features = StringHelper.normalize_serp_features(str(features))
                        if not normalized_features:
                            seo_issues.append(ValidationIssue(
                                field='SERP Features by Keyword',
                                issue_type='invalid_serp_features',
                                severity=ValidationSeverity.WARNING,
                                message=f"Could not parse SERP features: {features}",
                                affected_rows=[idx],
                                suggested_fix="Check SERP features format"
                            ))
            
            # Position-traffic correlation check using StatisticalCalculator
            if all(col in df.columns for col in ['Position', 'Traffic (%)']):
                correlation_data = df[['Position', 'Traffic (%)']].dropna()
                if len(correlation_data) > 5:
                    correlation = correlation_data['Position'].corr(correlation_data['Traffic (%)'])
                    if correlation > 0.1:  # Expecting negative correlation
                        seo_issues.append(ValidationIssue(
                            field='Position-Traffic',
                            issue_type='unexpected_correlation',
                            severity=ValidationSeverity.WARNING,
                            message=f"Unexpected positive correlation between position and traffic: {correlation:.3f}",
                            affected_rows=[],
                            suggested_fix="Review data quality - better positions should typically have more traffic"
                        ))
            
            return seo_issues
            
        except Exception as e:
            self.logger.error(f"Error extracting SEO-specific issues: {str(e)}")
            return []

    def _calculate_quality_metrics(
        self,
        df: pd.DataFrame,
        dataset_type: str
    ) -> Dict[str, float]:
        """Calculate data quality metrics using StatisticalCalculator."""
        try:
            quality_metrics = {}
            
            # Completeness metrics
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            quality_metrics['completeness'] = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
            
            # Keyword quality metrics
            if 'Keyword' in df.columns:
                keyword_quality = self.validate_keyword_quality(df['Keyword'])
                quality_metrics['keyword_quality'] = keyword_quality.get('quality_score', 0)
            
            # Numerical data quality using StatisticalCalculator
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                outlier_ratios = []
                for col in numeric_columns:
                    if col in ['Position', 'Search Volume', 'Traffic (%)']:
                        col_data = df[col].dropna()
                        if len(col_data) > 10:
                            stats_dict = self.stats_calculator.calculate_descriptive_statistics(col_data)
                            q1, q3 = stats_dict.get('q25', 0), stats_dict.get('q75', 0)
                            iqr = q3 - q1
                            if iqr > 0:
                                outliers = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()
                                outlier_ratio = outliers / len(col_data)
                                outlier_ratios.append(outlier_ratio)
                
                quality_metrics['numerical_quality'] = 1 - np.mean(outlier_ratios) if outlier_ratios else 1.0
            
            # Overall quality score
            quality_scores = [v for v in quality_metrics.values() if isinstance(v, (int, float))]
            quality_metrics['overall_quality'] = np.mean(quality_scores) if quality_scores else 0.0
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {str(e)}")
            return {}

    def _validate_single_keyword(
        self,
        original_keyword: str,
        cleaned_keyword: str,
        strict_mode: bool
    ) -> List[Dict[str, Any]]:
        """Validate a single keyword using validation rules."""
        try:
            issues = []
            
            # Length validation
            if len(cleaned_keyword) == 0:
                issues.append({
                    'issue': 'empty_keyword',
                    'message': 'Keyword is empty after cleaning',
                    'original': original_keyword
                })
            elif len(cleaned_keyword) > 200:
                issues.append({
                    'issue': 'keyword_too_long',
                    'message': f'Keyword exceeds maximum length: {len(cleaned_keyword)} characters',
                    'original': original_keyword
                })
            
            # Character validation in strict mode
            if strict_mode and cleaned_keyword:
                # Check for suspicious patterns
                if re.search(r'[^\w\s\-\'\"]', cleaned_keyword):
                    issues.append({
                        'issue': 'special_characters',
                        'message': 'Keyword contains special characters',
                        'original': original_keyword
                    })
                
                # Check for excessive repetition
                words = cleaned_keyword.split()
                if len(words) > 1:
                    word_counts = {word: words.count(word) for word in set(words)}
                    if any(count > 3 for count in word_counts.values()):
                        issues.append({
                            'issue': 'excessive_repetition',
                            'message': 'Keyword has excessive word repetition',
                            'original': original_keyword
                        })
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error validating keyword: {str(e)}")
            return []

    def _validate_seo_business_rules(
        self,
        df: pd.DataFrame,
        dataset_type: str
    ) -> List[Dict[str, Any]]:
        """Validate SEO business rules."""
        try:
            violations = []
            
            # Position-based business rules
            if 'Position' in df.columns:
                # Check for positions that seem too good for competitive keywords
                if 'Keyword Difficulty' in df.columns:
                    suspicious = df[(df['Position'] <= 3) & (df['Keyword Difficulty'] > 80)]
                    if len(suspicious) > 0:
                        violations.append({
                            'rule': 'suspicious_high_rankings',
                            'description': f'Found {len(suspicious)} keywords ranking in top 3 despite high difficulty',
                            'severity': 'warning',
                            'affected_keywords': suspicious['Keyword'].tolist()[:5] if 'Keyword' in df.columns else []
                        })
            
            # Traffic consistency rules
            if all(col in df.columns for col in ['Search Volume', 'Traffic (%)', 'Position']):
                # Check for high traffic with low search volume
                inconsistent_traffic = df[
                    (df['Traffic (%)'] > 5) & 
                    (df['Search Volume'] < 100) & 
                    (df['Position'] > 10)
                ]
                if len(inconsistent_traffic) > 0:
                    violations.append({
                        'rule': 'traffic_volume_inconsistency',
                        'description': f'Found {len(inconsistent_traffic)} keywords with high traffic but low search volume',
                        'severity': 'warning',
                        'affected_keywords': inconsistent_traffic['Keyword'].tolist()[:5] if 'Keyword' in df.columns else []
                    })
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error validating business rules: {str(e)}")
            return []

    def _generate_enhanced_recommendations(
        self,
        base_report: ValidationReport,
        seo_issues: List[ValidationIssue],
        quality_metrics: Dict[str, float],
        business_violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate enhanced recommendations based on all validation results."""
        try:
            recommendations = []
            
            # Base recommendations from validation report
            recommendations.extend(base_report.recommendations)
            
            # Quality-based recommendations
            overall_quality = quality_metrics.get('overall_quality', 0)
            if overall_quality < 0.7:
                recommendations.append("Data quality is below acceptable threshold - implement data cleaning procedures")
            
            completeness = quality_metrics.get('completeness', 0)
            if completeness < 0.8:
                recommendations.append("Address missing data issues to improve analysis accuracy")
            
            keyword_quality = quality_metrics.get('keyword_quality', 0)
            if keyword_quality < 0.9:
                recommendations.append("Improve keyword data quality through standardized collection processes")
            
            # SEO-specific recommendations
            if len(seo_issues) > 5:
                recommendations.append("Multiple SEO data format issues detected - review data import processes")
            
            # Business rule recommendations
            if business_violations:
                recommendations.append("Business rule violations detected - verify data accuracy and collection methods")
            
            # Performance recommendations
            if len(base_report.issues) > 20:
                recommendations.append("Consider implementing automated data validation in your data pipeline")
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Review data quality and rerun validation"]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_validation_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for validation operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )
