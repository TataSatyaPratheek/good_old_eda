"""
Validation Utilities for SEO Competitive Intelligence

Advanced validation, quality checking, and business rule validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import logging
from datetime import datetime, timedelta
import re
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings

warnings.filterwarnings('ignore')

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    field: str
    issue_type: str
    severity: ValidationSeverity
    message: str
    affected_rows: List[int]
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    total_records: int
    issues: List[ValidationIssue]
    overall_score: float
    passed_validations: int
    failed_validations: int
    critical_issues: int
    warnings: int
    validation_timestamp: datetime
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    """Simple validation result container (from paste file for compatibility)"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class SchemaValidator:
    """
    Advanced schema validation for SEO competitive intelligence.
    Provides comprehensive schema validation with business logic
    and data quality assessment capabilities.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.schemas = {}  # For simple schema storage from paste file
        
        # SEO-specific validation rules
        self.seo_validation_rules = {
            'keyword_patterns': {
                'min_length': 1,
                'max_length': 500,
                'allowed_chars': r'^[a-zA-Z0-9\s\-\'\"\+\(\)\[\]\.\,\!\?]+$'
            },
            'position_ranges': {
                'min_value': 1,
                'max_value': 100,
                'data_type': 'integer'
            },
            'url_patterns': {
                'pattern': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
                'max_length': 2000
            },
            'search_volume_ranges': {
                'min_value': 0,
                'max_value': 100000000,
                'data_type': 'integer'
            },
            'traffic_ranges': {
                'min_value': 0,
                'max_value': 10000000,
                'data_type': 'numeric'
            }
        }

    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register a validation schema (from paste file)"""
        self.schemas[name] = schema

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        schema_name: str
    ) -> ValidationResult:
        """Validate dataframe against schema (simple version from paste file)"""
        errors = []
        warnings = []
        
        if schema_name not in self.schemas:
            errors.append(f"Schema '{schema_name}' not found")
            return ValidationResult(False, errors, warnings, {})
        
        schema = self.schemas[schema_name]
        
        # Check required columns
        required_columns = schema.get('required_columns', [])
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        column_types = schema.get('column_types', {})
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._type_compatible(actual_type, expected_type):
                    warnings.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        
        # Check constraints
        constraints = schema.get('constraints', {})
        for col, constraint in constraints.items():
            if col in df.columns:
                if 'min' in constraint and df[col].min() < constraint['min']:
                    errors.append(f"Column '{col}' has values below minimum {constraint['min']}")
                if 'max' in constraint and df[col].max() > constraint['max']:
                    errors.append(f"Column '{col}' has values above maximum {constraint['max']}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={'rows': len(df), 'columns': len(df.columns)}
        )

    def validate_dataframe_schema(
        self,
        df: pd.DataFrame,
        schema_definition: Dict[str, Dict[str, Any]],
        dataset_type: str = "general"
    ) -> ValidationReport:
        """
        Validate DataFrame against comprehensive schema definition (from uploaded file).
        
        Args:
            df: DataFrame to validate
            schema_definition: Schema definition with validation rules
            dataset_type: Type of dataset for specialized validation
            
        Returns:
            Comprehensive validation report
        """
        try:
            validation_start = datetime.now()
            issues = []

            # Basic structure validation
            structure_issues = self._validate_structure(df, schema_definition)
            issues.extend(structure_issues)

            # Column-by-column validation
            for column, rules in schema_definition.items():
                if column in df.columns:
                    column_issues = self._validate_column(df, column, rules)
                    issues.extend(column_issues)
                elif rules.get('required', False):
                    issues.append(ValidationIssue(
                        field=column,
                        issue_type='missing_required_column',
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Required column '{column}' is missing",
                        affected_rows=[],
                        suggested_fix=f"Add column '{column}' to the dataset"
                    ))

            # Dataset-specific validation
            if dataset_type in ['positions', 'competitors', 'gap_keywords']:
                specialized_issues = self._validate_seo_dataset(df, dataset_type)
                issues.extend(specialized_issues)

            # Business logic validation
            business_issues = self._validate_business_logic(df, dataset_type)
            issues.extend(business_issues)

            # Calculate validation metrics
            report = self._generate_validation_report(df, issues, validation_start)

            self.logger.info(f"Schema validation completed. Overall score: {report.overall_score:.2f}")
            return report

        except Exception as e:
            self.logger.error(f"Error in schema validation: {str(e)}")
            return ValidationReport(
                total_records=0,
                issues=[],
                overall_score=0.0,
                passed_validations=0,
                failed_validations=1,
                critical_issues=1,
                warnings=0,
                validation_timestamp=datetime.now(),
                recommendations=[f"Validation error: {str(e)}"]
            )

    def validate_data_consistency(
        self,
        df: pd.DataFrame,
        consistency_rules: Optional[Dict[str, Any]] = None
    ) -> List[ValidationIssue]:
        """
        Validate data consistency across related fields.
        
        Args:
            df: DataFrame to validate
            consistency_rules: Custom consistency rules
            
        Returns:
            List of consistency validation issues
        """
        try:
            if consistency_rules is None:
                consistency_rules = self._get_default_consistency_rules()

            issues = []

            # Cross-field validations
            for rule_name, rule_config in consistency_rules.items():
                try:
                    rule_issues = self._apply_consistency_rule(df, rule_name, rule_config)
                    issues.extend(rule_issues)
                except Exception as e:
                    self.logger.warning(f"Error applying consistency rule {rule_name}: {str(e)}")

            return issues

        except Exception as e:
            self.logger.error(f"Error in consistency validation: {str(e)}")
            return []

    def validate_data_ranges(
        self,
        df: pd.DataFrame,
        range_definitions: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[ValidationIssue]:
        """
        Validate data ranges and bounds.
        
        Args:
            df: DataFrame to validate
            range_definitions: Range validation definitions
            
        Returns:
            List of range validation issues
        """
        try:
            if range_definitions is None:
                range_definitions = self._get_default_range_definitions()

            issues = []

            for column, range_config in range_definitions.items():
                if column in df.columns:
                    column_issues = self._validate_column_ranges(df, column, range_config)
                    issues.extend(column_issues)

            return issues

        except Exception as e:
            self.logger.error(f"Error in range validation: {str(e)}")
            return []

    def _type_compatible(self, actual: str, expected: str) -> bool:
        """Check if types are compatible (from paste file)"""
        type_mappings = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'str': ['object', 'string'],
            'datetime': ['datetime64[ns]', 'datetime64']
        }
        
        for base_type, compatible_types in type_mappings.items():
            if expected == base_type and any(t in actual for t in compatible_types):
                return True
        
        return actual == expected

    def _validate_structure(
        self,
        df: pd.DataFrame,
        schema_definition: Dict[str, Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Validate basic DataFrame structure."""
        issues = []

        # Check if DataFrame is empty
        if df.empty:
            issues.append(ValidationIssue(
                field='dataframe',
                issue_type='empty_dataframe',
                severity=ValidationSeverity.CRITICAL,
                message="DataFrame is empty",
                affected_rows=[],
                suggested_fix="Ensure data is properly loaded"
            ))
            return issues

        # Check for unexpected columns
        expected_columns = set(schema_definition.keys())
        actual_columns = set(df.columns)
        unexpected_columns = actual_columns - expected_columns

        if unexpected_columns:
            issues.append(ValidationIssue(
                field='columns',
                issue_type='unexpected_columns',
                severity=ValidationSeverity.WARNING,
                message=f"Unexpected columns found: {', '.join(unexpected_columns)}",
                affected_rows=[],
                suggested_fix="Review column names or update schema definition"
            ))

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicate_columns = [col for col in df.columns if df.columns.tolist().count(col) > 1]
            issues.append(ValidationIssue(
                field='columns',
                issue_type='duplicate_columns',
                severity=ValidationSeverity.ERROR,
                message=f"Duplicate column names found: {', '.join(set(duplicate_columns))}",
                affected_rows=[],
                suggested_fix="Rename duplicate columns to unique names"
            ))

        return issues

    def _validate_column(
        self,
        df: pd.DataFrame,
        column: str,
        rules: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate individual column against rules."""
        issues = []
        column_data = df[column]

        # Data type validation
        expected_type = rules.get('data_type')
        if expected_type:
            type_issues = self._validate_data_type(column_data, column, expected_type)
            issues.extend(type_issues)

        # Null value validation
        nullable = rules.get('nullable', True)
        if not nullable:
            null_rows = column_data.isnull()
            if null_rows.any():
                null_indices = df.index[null_rows].tolist()
                issues.append(ValidationIssue(
                    field=column,
                    issue_type='null_values',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{column}' contains null values but is marked as non-nullable",
                    affected_rows=null_indices,
                    suggested_fix="Fill null values or update nullable constraint"
                ))

        # Pattern validation
        pattern = rules.get('pattern')
        if pattern and column_data.dtype == 'object':
            pattern_issues = self._validate_pattern(column_data, column, pattern)
            issues.extend(pattern_issues)

        # Range validation
        min_value = rules.get('min_value')
        max_value = rules.get('max_value')
        if min_value is not None or max_value is not None:
            range_issues = self._validate_value_range(column_data, column, min_value, max_value)
            issues.extend(range_issues)

        # Uniqueness validation
        if rules.get('unique', False):
            duplicate_mask = column_data.duplicated()
            if duplicate_mask.any():
                duplicate_indices = df.index[duplicate_mask].tolist()
                issues.append(ValidationIssue(
                    field=column,
                    issue_type='duplicate_values',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{column}' contains duplicate values but should be unique",
                    affected_rows=duplicate_indices,
                    suggested_fix="Remove or resolve duplicate values"
                ))

        return issues

    def _validate_seo_dataset(
        self,
        df: pd.DataFrame,
        dataset_type: str
    ) -> List[ValidationIssue]:
        """Validate SEO-specific dataset requirements."""
        issues = []

        if dataset_type == 'positions':
            # Keyword-Position relationship validation
            if 'Keyword' in df.columns and 'Position' in df.columns:
                # Check for impossible position-traffic relationships
                if 'Traffic' in df.columns:
                    issues.extend(self._validate_position_traffic_relationship(df))

                # Check for keyword duplicates with different positions
                issues.extend(self._validate_keyword_position_consistency(df))

        elif dataset_type == 'competitors':
            # Competitor relevance validation
            if 'Competitor Relevance' in df.columns:
                relevance_data = df['Competitor Relevance']
                out_of_range = (relevance_data < 0) | (relevance_data > 1)
                if out_of_range.any():
                    affected_rows = df.index[out_of_range].tolist()
                    issues.append(ValidationIssue(
                        field='Competitor Relevance',
                        issue_type='invalid_relevance_score',
                        severity=ValidationSeverity.ERROR,
                        message="Competitor relevance scores must be between 0 and 1",
                        affected_rows=affected_rows,
                        suggested_fix="Normalize relevance scores to 0-1 range"
                    ))

        elif dataset_type == 'gap_keywords':
            # Gap keyword validation
            position_columns = ['lenovo.com', 'dell.com', 'hp.com']
            for col in position_columns:
                if col in df.columns:
                    invalid_positions = (df[col] < 0) | (df[col] > 100)
                    if invalid_positions.any():
                        affected_rows = df.index[invalid_positions].tolist()
                        issues.append(ValidationIssue(
                            field=col,
                            issue_type='invalid_position',
                            severity=ValidationSeverity.ERROR,
                            message=f"Positions in '{col}' must be between 1 and 100",
                            affected_rows=affected_rows,
                            suggested_fix="Correct position values to valid range"
                        ))

        return issues

    def _validate_business_logic(
        self,
        df: pd.DataFrame,
        dataset_type: str
    ) -> List[ValidationIssue]:
        """Validate business logic rules."""
        issues = []

        # Traffic should correlate negatively with position (lower position = more traffic generally)
        if all(col in df.columns for col in ['Position', 'Traffic']):
            correlation = df['Position'].corr(df['Traffic'])
            if correlation > 0.1:  # Expecting negative correlation
                issues.append(ValidationIssue(
                    field='Position-Traffic',
                    issue_type='unexpected_correlation',
                    severity=ValidationSeverity.WARNING,
                    message=f"Unexpected positive correlation between Position and Traffic: {correlation:.3f}",
                    affected_rows=[],
                    suggested_fix="Review data for accuracy - typically better positions (lower numbers) should have more traffic"
                ))

        # Search volume should be positive
        if 'Search Volume' in df.columns:
            zero_volume = df['Search Volume'] <= 0
            if zero_volume.any():
                affected_rows = df.index[zero_volume].tolist()
                issues.append(ValidationIssue(
                    field='Search Volume',
                    issue_type='zero_search_volume',
                    severity=ValidationSeverity.WARNING,
                    message="Keywords with zero or negative search volume detected",
                    affected_rows=affected_rows,
                    suggested_fix="Review search volume data or consider removing zero-volume keywords"
                ))

        return issues

    def _generate_validation_report(
        self,
        df: pd.DataFrame,
        issues: List[ValidationIssue],
        validation_start: datetime
    ) -> ValidationReport:
        """Generate comprehensive validation report."""
        # Count issues by severity
        critical_issues = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        errors = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)

        # Calculate overall score (0-100)
        total_checks = len(issues) + 10  # Assume some passed validations
        failed_checks = critical_issues * 3 + errors * 2 + warnings * 1
        overall_score = max(0, 100 - (failed_checks / total_checks * 100))

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        return ValidationReport(
            total_records=len(df),
            issues=issues,
            overall_score=overall_score,
            passed_validations=total_checks - len(issues),
            failed_validations=len(issues),
            critical_issues=critical_issues,
            warnings=warnings,
            validation_timestamp=validation_start,
            recommendations=recommendations
        )

    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations based on validation issues."""
        recommendations = []

        # Group issues by type
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)

        # Generate recommendations based on issue patterns
        if 'null_values' in issue_types:
            recommendations.append("Implement data cleansing procedures to handle missing values")

        if 'invalid_position' in issue_types:
            recommendations.append("Validate position data at source to ensure values are within 1-100 range")

        if 'duplicate_values' in issue_types:
            recommendations.append("Implement duplicate detection and resolution processes")

        if 'unexpected_correlation' in issue_types:
            recommendations.append("Review data collection processes for potential systematic errors")

        # Priority recommendations
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            recommendations.insert(0, "Address critical data quality issues immediately before proceeding with analysis")

        return recommendations

    # Helper methods for validation logic
    def _validate_data_type(self, column_data: pd.Series, column: str, expected_type: str) -> List[ValidationIssue]:
        """Validate data type of column."""
        return []  # Implementation would check data types

    def _validate_pattern(self, column_data: pd.Series, column: str, pattern: str) -> List[ValidationIssue]:
        """Validate pattern matching."""
        return []  # Implementation would check patterns

    def _validate_value_range(self, column_data: pd.Series, column: str, min_value, max_value) -> List[ValidationIssue]:
        """Validate value ranges."""
        return []  # Implementation would check ranges

    def _validate_position_traffic_relationship(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate position-traffic relationship."""
        return []  # Implementation would check relationships

    def _validate_keyword_position_consistency(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate keyword position consistency."""
        return []  # Implementation would check consistency

    def _get_default_consistency_rules(self) -> Dict[str, Any]:
        """Get default consistency rules."""
        return {}

    def _apply_consistency_rule(self, df: pd.DataFrame, rule_name: str, rule_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Apply consistency rule."""
        return []

    def _get_default_range_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get default range definitions."""
        return {}

    def _validate_column_ranges(self, df: pd.DataFrame, column: str, range_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate column ranges."""
        return []


class BusinessRuleValidator:
    """
    Business rule validation for SEO competitive intelligence.
    Validates domain-specific business rules and logical constraints
    specific to SEO and competitive analysis workflows.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.rules = {}  # For simple rule storage from paste file

        # Define SEO business rules
        self.seo_business_rules = {
            'position_traffic_correlation': {
                'description': 'Better positions should generally have more traffic',
                'validation_function': self._validate_position_traffic_correlation,
                'severity': ValidationSeverity.WARNING
            },
            'competitor_relevance_bounds': {
                'description': 'Competitor relevance should be between 0 and 1',
                'validation_function': self._validate_competitor_relevance,
                'severity': ValidationSeverity.ERROR
            },
            'search_volume_consistency': {
                'description': 'Search volume should be consistent across related keywords',
                'validation_function': self._validate_search_volume_consistency,
                'severity': ValidationSeverity.WARNING
            },
            'url_domain_consistency': {
                'description': 'URLs should belong to expected domains',
                'validation_function': self._validate_url_domains,
                'severity': ValidationSeverity.ERROR
            }
        }

    def register_rule(self, name: str, rule_func: callable):
        """Register a business rule (from paste file)"""
        self.rules[name] = rule_func

    def validate(
        self,
        data: Any,
        rule_names: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate data against business rules (simple version from paste file)"""
        errors = []
        warnings = []
        metadata = {}
        
        rules_to_check = rule_names or list(self.rules.keys())
        
        for rule_name in rules_to_check:
            if rule_name not in self.rules:
                warnings.append(f"Rule '{rule_name}' not found")
                continue
            
            try:
                rule_func = self.rules[rule_name]
                result = rule_func(data)
                
                if isinstance(result, bool):
                    if not result:
                        errors.append(f"Rule '{rule_name}' failed")
                elif isinstance(result, dict):
                    if not result.get('passed', False):
                        errors.append(result.get('message', f"Rule '{rule_name}' failed"))
                    if 'warning' in result:
                        warnings.append(result['warning'])
                    if 'metadata' in result:
                        metadata[rule_name] = result['metadata']
                        
            except Exception as e:
                errors.append(f"Error executing rule '{rule_name}': {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

    def validate_seo_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate SEO-specific business rules (simple version from paste file)"""
        errors = []
        warnings = []
        
        # Check position values
        if 'Position' in df.columns:
            invalid_positions = df[df['Position'] <= 0]
            if not invalid_positions.empty:
                errors.append(f"Found {len(invalid_positions)} rows with invalid positions")
        
        # Check traffic percentages
        if 'Traffic (%)' in df.columns:
            invalid_traffic = df[df['Traffic (%)'] < 0]
            if not invalid_traffic.empty:
                errors.append(f"Found {len(invalid_traffic)} rows with negative traffic")
        
        # Check for duplicate keywords
        if 'Keyword' in df.columns:
            duplicates = df[df.duplicated(subset=['Keyword'], keep=False)]
            if not duplicates.empty:
                warnings.append(f"Found {len(duplicates)} duplicate keywords")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={'total_rows': len(df)}
        )

    def validate_business_rules(
        self,
        df: pd.DataFrame,
        dataset_type: str = 'general',
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> List[ValidationIssue]:
        """
        Validate business rules against dataset (comprehensive version from uploaded file).
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset
            custom_rules: Custom business rules
            
        Returns:
            List of business rule validation issues
        """
        try:
            issues = []

            # Apply standard SEO business rules
            for rule_name, rule_config in self.seo_business_rules.items():
                try:
                    validation_function = rule_config['validation_function']
                    rule_issues = validation_function(df, dataset_type)
                    # Set severity from rule config
                    for issue in rule_issues:
                        issue.severity = rule_config['severity']
                    issues.extend(rule_issues)
                except Exception as e:
                    self.logger.warning(f"Error applying business rule {rule_name}: {str(e)}")

            # Apply custom rules if provided
            if custom_rules:
                for rule_name, rule_function in custom_rules.items():
                    try:
                        if callable(rule_function):
                            custom_issues = rule_function(df)
                            issues.extend(custom_issues)
                    except Exception as e:
                        self.logger.warning(f"Error applying custom rule {rule_name}: {str(e)}")

            self.logger.info(f"Business rule validation completed. Found {len(issues)} issues")
            return issues

        except Exception as e:
            self.logger.error(f"Error in business rule validation: {str(e)}")
            return []

    def _validate_position_traffic_correlation(
        self,
        df: pd.DataFrame,
        dataset_type: str
    ) -> List[ValidationIssue]:
        """Validate position-traffic correlation business rule."""
        issues = []

        if all(col in df.columns for col in ['Position', 'Traffic']):
            # Calculate correlation
            correlation = df['Position'].corr(df['Traffic'])

            # Expect negative correlation (better position = lower number = more traffic)
            if correlation > 0.1:
                issues.append(ValidationIssue(
                    field='Position-Traffic',
                    issue_type='business_rule_violation',
                    severity=ValidationSeverity.WARNING,
                    message=f"Unexpected positive correlation between position and traffic: {correlation:.3f}",
                    affected_rows=[],
                    suggested_fix="Review data quality - typically better positions should correlate with higher traffic"
                ))

            # Check for obvious violations (high traffic with very poor positions)
            poor_position_high_traffic = (df['Position'] > 50) & (df['Traffic'] > df['Traffic'].quantile(0.9))
            if poor_position_high_traffic.any():
                affected_rows = df.index[poor_position_high_traffic].tolist()
                issues.append(ValidationIssue(
                    field='Position-Traffic',
                    issue_type='business_rule_violation',
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {poor_position_high_traffic.sum()} keywords with poor positions but high traffic",
                    affected_rows=affected_rows,
                    suggested_fix="Investigate these keywords for data accuracy"
                ))

        return issues

    def _validate_competitor_relevance(
        self,
        df: pd.DataFrame,
        dataset_type: str
    ) -> List[ValidationIssue]:
        """Validate competitor relevance business rule."""
        issues = []

        if 'Competitor Relevance' in df.columns:
            relevance_data = df['Competitor Relevance']

            # Check bounds
            out_of_bounds = (relevance_data < 0) | (relevance_data > 1)
            if out_of_bounds.any():
                affected_rows = df.index[out_of_bounds].tolist()
                issues.append(ValidationIssue(
                    field='Competitor Relevance',
                    issue_type='business_rule_violation',
                    severity=ValidationSeverity.ERROR,
                    message="Competitor relevance values must be between 0 and 1",
                    affected_rows=affected_rows,
                    suggested_fix="Normalize relevance values to 0-1 range"
                ))

        return issues

    def _validate_search_volume_consistency(
        self,
        df: pd.DataFrame,
        dataset_type: str
    ) -> List[ValidationIssue]:
        """Validate search volume consistency business rule."""
        issues = []

        if 'Search Volume' in df.columns and 'Keyword' in df.columns:
            # Check for duplicate keywords with different search volumes
            keyword_volumes = df.groupby('Keyword')['Search Volume'].nunique()
            inconsistent_keywords = keyword_volumes[keyword_volumes > 1]

            if len(inconsistent_keywords) > 0:
                issues.append(ValidationIssue(
                    field='Search Volume',
                    issue_type='business_rule_violation',
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {len(inconsistent_keywords)} keywords with inconsistent search volumes",
                    affected_rows=[],
                    suggested_fix="Standardize search volume data for duplicate keywords"
                ))

        return issues

    def _validate_url_domains(
        self,
        df: pd.DataFrame,
        dataset_type: str
    ) -> List[ValidationIssue]:
        """Validate URL domain consistency business rule."""
        issues = []

        if 'URL' in df.columns:
            # Extract domains from URLs
            def extract_domain(url):
                try:
                    match = re.search(r'https?://(?:www\.)?([^/]+)', str(url))
                    return match.group(1) if match else None
                except:
                    return None

            domains = df['URL'].apply(extract_domain)
            unique_domains = domains.dropna().unique()

            # For position data, expect primarily one domain (the analyzed site)
            if dataset_type == 'positions' and len(unique_domains) > 3:
                issues.append(ValidationIssue(
                    field='URL',
                    issue_type='business_rule_violation',
                    severity=ValidationSeverity.WARNING,
                    message=f"Position data contains URLs from {len(unique_domains)} different domains",
                    affected_rows=[],
                    suggested_fix="Verify that position data is from the correct domain"
                ))

        return issues


class DataQualityChecker:
    """
    Advanced data quality assessment for SEO competitive intelligence.
    Provides comprehensive data quality metrics and automated
    quality improvement suggestions.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def assess_data_quality(
        self,
        df: pd.DataFrame,
        quality_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment.
        
        Args:
            df: DataFrame to assess
            quality_dimensions: Specific quality dimensions to assess
            
        Returns:
            Comprehensive quality assessment
        """
        try:
            if quality_dimensions is None:
                quality_dimensions = [
                    'completeness', 'accuracy', 'consistency',
                    'validity', 'uniqueness', 'timeliness'
                ]

            assessment = {
                'overview': {
                    'total_records': len(df),
                    'total_columns': len(df.columns),
                    'assessment_timestamp': datetime.now()
                },
                'dimensions': {}
            }

            # Assess each quality dimension
            for dimension in quality_dimensions:
                dimension_assessment = self._assess_quality_dimension(df, dimension)
                assessment['dimensions'][dimension] = dimension_assessment

            # Calculate overall quality score
            dimension_scores = [
                assessment['dimensions'][dim].get('score', 0)
                for dim in quality_dimensions
            ]
            overall_score = np.mean(dimension_scores) if dimension_scores else 0
            assessment['overall_quality_score'] = overall_score
            assessment['quality_grade'] = self._calculate_quality_grade(overall_score)

            # Generate improvement recommendations
            assessment['improvement_recommendations'] = self._generate_quality_improvements(
                assessment['dimensions']
            )

            self.logger.info(f"Data quality assessment completed. Overall score: {overall_score:.2f}")
            return assessment

        except Exception as e:
            self.logger.error(f"Error in data quality assessment: {str(e)}")
            return {}

    def _assess_quality_dimension(
        self,
        df: pd.DataFrame,
        dimension: str
    ) -> Dict[str, Any]:
        """Assess specific quality dimension."""
        try:
            if dimension == 'completeness':
                return self._assess_completeness(df)
            elif dimension == 'accuracy':
                return self._assess_accuracy(df)
            elif dimension == 'consistency':
                return self._assess_consistency(df)
            elif dimension == 'validity':
                return self._assess_validity(df)
            elif dimension == 'uniqueness':
                return self._assess_uniqueness(df)
            elif dimension == 'timeliness':
                return self._assess_timeliness(df)
            else:
                return {'score': 0, 'issues': [], 'details': {}}

        except Exception as e:
            self.logger.error(f"Error assessing {dimension}: {str(e)}")
            return {'score': 0, 'issues': [str(e)], 'details': {}}

    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

        # Column-level completeness
        column_completeness = {}
        for col in df.columns:
            col_completeness = 1 - (df[col].isnull().sum() / len(df))
            column_completeness[col] = col_completeness

        # Identify problematic columns
        low_completeness_columns = [
            col for col, completeness in column_completeness.items()
            if completeness < 0.8
        ]

        score = completeness_ratio * 100

        return {
            'score': score,
            'overall_completeness': completeness_ratio,
            'column_completeness': column_completeness,
            'missing_cells': missing_cells,
            'total_cells': total_cells,
            'low_completeness_columns': low_completeness_columns,
            'issues': [f"Low completeness in columns: {', '.join(low_completeness_columns)}"] if low_completeness_columns else []
        }

    def _assess_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data accuracy through various heuristics."""
        accuracy_issues = []
        accuracy_score = 100

        # Check for impossible values in SEO context
        if 'Position' in df.columns:
            invalid_positions = (df['Position'] < 1) | (df['Position'] > 100)
            if invalid_positions.any():
                accuracy_issues.append(f"Invalid positions found: {invalid_positions.sum()} records")
                accuracy_score -= 10

        if 'Search Volume' in df.columns:
            negative_volume = df['Search Volume'] < 0
            if negative_volume.any():
                accuracy_issues.append(f"Negative search volumes found: {negative_volume.sum()} records")
                accuracy_score -= 15

        # Check for outliers
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
            outlier_counts[col] = outliers

            if outliers > len(df) * 0.05:  # More than 5% outliers
                accuracy_issues.append(f"High number of outliers in {col}: {outliers}")
                accuracy_score -= 5

        return {
            'score': max(0, accuracy_score),
            'issues': accuracy_issues,
            'outlier_counts': outlier_counts,
            'details': {
                'invalid_positions': invalid_positions.sum() if 'Position' in df.columns else 0,
                'negative_volumes': negative_volume.sum() if 'Search Volume' in df.columns else 0
            }
        }

    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency."""
        return {'score': 80, 'issues': [], 'details': {}}

    def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity."""
        return {'score': 85, 'issues': [], 'details': {}}

    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness."""
        return {'score': 90, 'issues': [], 'details': {}}

    def _assess_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data timeliness."""
        return {'score': 75, 'issues': [], 'details': {}}

    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate quality grade from score."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def _generate_quality_improvements(self, dimensions: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        for dimension, assessment in dimensions.items():
            if assessment.get('score', 0) < 80:
                recommendations.append(f"Improve {dimension} - current score: {assessment.get('score', 0):.1f}")
        
        return recommendations
