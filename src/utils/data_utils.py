"""
Data Processing Utilities for SEO Competitive Intelligence

Advanced data processing, validation, and transformation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from dataclasses import dataclass
import re
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')

@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_records: int
    missing_values: Dict[str, int]
    duplicate_records: int
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    quality_score: float
    recommendations: List[str]

@dataclass
class DataValidationResult:
    """Data validation result container"""
    is_valid: bool
    quality_score: float
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class DataProcessor:
    """
    Advanced data processing utilities for SEO competitive intelligence.
    Provides comprehensive data cleaning, transformation, and preprocessing
    capabilities optimized for SEO and competitive analysis datasets.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.scalers = {}
        self.transformation_history = []

        # SEO-specific data patterns
        self.seo_patterns = {
            'url_pattern': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            'keyword_pattern': r'^[a-zA-Z0-9\s\-\_\+\(\)\'\"\.]+$',
            'domain_pattern': r'^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$',
            'serp_features_pattern': r'[a-zA-Z\s,\-\_]+'
        }

    def clean_seo_data(
        self,
        df: pd.DataFrame,
        cleaning_config: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Comprehensive SEO data cleaning with domain-specific rules.
        
        Args:
            df: Input DataFrame
            cleaning_config: Configuration for cleaning operations
            
        Returns:
            Cleaned DataFrame
        """
        try:
            if cleaning_config is None:
                cleaning_config = self._get_default_cleaning_config()

            cleaned_df = df.copy()
            cleaning_steps = []

            # Handle empty dataframe
            if df.empty:
                return df

            # Remove completely empty rows
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(how='all')
            rows_removed = initial_count - len(cleaned_df)
            if rows_removed > 0:
                cleaning_steps.append(f"Removed {rows_removed} completely empty rows")

            # Clean keyword data
            if 'Keyword' in cleaned_df.columns:
                cleaned_df = self._clean_keyword_column(cleaned_df, cleaning_config)
                cleaning_steps.append("Cleaned keyword column")

            # Clean URL data
            if 'URL' in cleaned_df.columns:
                cleaned_df = self._clean_url_column(cleaned_df, cleaning_config)
                cleaning_steps.append("Cleaned URL column")

            # Clean position data
            if 'Position' in cleaned_df.columns:
                cleaned_df = self._clean_position_data(cleaned_df, cleaning_config)
                cleaning_steps.append("Cleaned position data")

            # Clean traffic data
            traffic_columns = ['Traffic', 'Traffic (%)', 'Organic Traffic']
            for col in traffic_columns:
                if col in cleaned_df.columns:
                    cleaned_df = self._clean_traffic_data(cleaned_df, col, cleaning_config)
                    cleaning_steps.append(f"Cleaned {col} column")

            # Clean search volume data
            if 'Search Volume' in cleaned_df.columns:
                cleaned_df = self._clean_search_volume_data(cleaned_df, cleaning_config)
                cleaning_steps.append("Cleaned search volume data")

            # Handle SERP features
            if 'SERP Features by Keyword' in cleaned_df.columns:
                cleaned_df = self._clean_serp_features(cleaned_df, cleaning_config)
                cleaning_steps.append("Cleaned SERP features")

            # Remove duplicates
            if cleaning_config.get('remove_duplicates', True):
                duplicate_count = cleaned_df.duplicated().sum()
                cleaned_df = cleaned_df.drop_duplicates()
                if duplicate_count > 0:
                    cleaning_steps.append(f"Removed {duplicate_count} duplicate records")

            # Additional simple cleaning from paste file
            cleaned_df = self._apply_simple_cleaning(cleaned_df)

            # Log cleaning summary
            self.logger.info(f"Data cleaning completed. Steps: {'; '.join(cleaning_steps)}")

            self.transformation_history.append({
                'operation': 'clean_seo_data',
                'timestamp': datetime.now(),
                'steps': cleaning_steps,
                'input_shape': df.shape,
                'output_shape': cleaned_df.shape
            })

            return cleaned_df

        except Exception as e:
            self.logger.error(f"Error in SEO data cleaning: {str(e)}")
            return df

    def _apply_simple_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply simple cleaning from paste file approach."""
        try:
            df_clean = df.copy()

            # Handle missing values for numeric columns
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in ['Position', 'Traffic (%)', 'Search Volume', 'CPC']:
                    if col == 'Position':
                        df_clean[col] = df_clean[col].fillna(100)  # Assume not ranking
                    else:
                        df_clean[col] = df_clean[col].fillna(0)

            # Clean string columns
            string_columns = df_clean.select_dtypes(include=[object]).columns
            for col in string_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna('').astype(str).str.strip()

            # Standardize column names
            df_clean.columns = [col.strip() for col in df_clean.columns]

            # Handle date columns
            if 'date' in df_clean.columns:
                try:
                    df_clean['date'] = pd.to_datetime(df_clean['date'])
                except:
                    self.logger.warning("Could not convert date column to datetime")

            # Remove invalid position values
            if 'Position' in df_clean.columns:
                df_clean = df_clean[df_clean['Position'] > 0]

            return df_clean

        except Exception as e:
            self.logger.error(f"Error in simple cleaning: {str(e)}")
            return df

    def validate_data_quality(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Validate data quality and completeness (from paste file)
        
        Args:
            df: Dataframe to validate
            
        Returns:
            DataValidationResult object
        """
        issues = []
        warnings = []
        metrics = {}

        if df.empty:
            return DataValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=["Empty dataframe"],
                warnings=[],
                metrics={}
            )

        # Check for required columns
        required_columns = ['Keyword', 'Position']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")

        # Calculate completeness
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        metrics['completeness'] = completeness

        # Check data types
        if 'Position' in df.columns:
            try:
                pd.to_numeric(df['Position'], errors='coerce')
            except:
                issues.append("Position column contains non-numeric values")

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")
        metrics['duplicate_count'] = duplicate_count

        # Calculate quality score
        quality_score = completeness * 0.4
        quality_score += (1 - min(duplicate_count / len(df), 1)) * 0.3
        quality_score += (1 - len(issues) / 10) * 0.3

        return DataValidationResult(
            is_valid=len(issues) == 0,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )

    def merge_datasets(
        self,
        datasets: List[pd.DataFrame],
        merge_on: Union[str, List[str]],
        how: str = 'outer'
    ) -> pd.DataFrame:
        """
        Merge multiple datasets (from paste file)
        
        Args:
            datasets: List of dataframes to merge
            merge_on: Column(s) to merge on
            how: Merge type (outer, inner, left, right)
            
        Returns:
            Merged dataframe
        """
        if not datasets:
            return pd.DataFrame()

        if len(datasets) == 1:
            return datasets[0]

        result = datasets[0]
        for df in datasets[1:]:
            result = pd.merge(result, df, on=merge_on, how=how, suffixes=('', '_y'))
            # Remove duplicate columns
            result = result.loc[:, ~result.columns.str.endswith('_y')]

        return result

    def standardize_competitor_data(
        self,
        competitor_datasets: Dict[str, pd.DataFrame],
        standardization_rules: Dict[str, Any] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Standardize data formats across multiple competitor datasets.
        
        Args:
            competitor_datasets: Dictionary of competitor DataFrames
            standardization_rules: Rules for standardization
            
        Returns:
            Dictionary of standardized DataFrames
        """
        try:
            if standardization_rules is None:
                standardization_rules = self._get_default_standardization_rules()

            standardized_datasets = {}

            # Identify common schema
            common_schema = self._identify_common_schema(competitor_datasets)

            for competitor_name, df in competitor_datasets.items():
                self.logger.info(f"Standardizing data for {competitor_name}")

                standardized_df = df.copy()

                # Standardize column names
                standardized_df = self._standardize_column_names(
                    standardized_df, common_schema
                )

                # Standardize data types
                standardized_df = self._standardize_data_types(
                    standardized_df, standardization_rules
                )

                # Standardize value formats
                standardized_df = self._standardize_value_formats(
                    standardized_df, standardization_rules
                )

                # Add competitor identifier
                standardized_df['competitor'] = competitor_name

                # Ensure common schema compliance
                standardized_df = self._ensure_schema_compliance(
                    standardized_df, common_schema
                )

                standardized_datasets[competitor_name] = standardized_df

            self.logger.info(f"Standardized {len(standardized_datasets)} competitor datasets")
            return standardized_datasets

        except Exception as e:
            self.logger.error(f"Error standardizing competitor data: {str(e)}")
            return competitor_datasets

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'intelligent',
        custom_strategies: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Intelligent missing value handling for SEO data.
        
        Args:
            df: Input DataFrame
            strategy: Overall strategy ('intelligent', 'drop', 'fill')
            custom_strategies: Column-specific strategies
            
        Returns:
            DataFrame with handled missing values
        """
        try:
            processed_df = df.copy()

            if custom_strategies is None:
                custom_strategies = {}

            missing_summary = {}

            for column in processed_df.columns:
                missing_count = processed_df[column].isnull().sum()
                missing_percentage = (missing_count / len(processed_df)) * 100

                if missing_count == 0:
                    continue

                missing_summary[column] = {
                    'count': missing_count,
                    'percentage': missing_percentage
                }

                # Determine strategy for this column
                if column in custom_strategies:
                    col_strategy = custom_strategies[column]
                elif strategy == 'intelligent':
                    col_strategy = self._determine_intelligent_strategy(
                        processed_df, column, missing_percentage
                    )
                else:
                    col_strategy = strategy

                # Apply strategy
                processed_df = self._apply_missing_value_strategy(
                    processed_df, column, col_strategy
                )

            self.logger.info(f"Handled missing values in {len(missing_summary)} columns")
            return processed_df

        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            return df

    def detect_and_handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'isolation_forest',
        threshold: float = 0.1
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and handle outliers in SEO data.
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr', 'z_score', 'isolation_forest')
            threshold: Outlier threshold
            
        Returns:
            Tuple of (processed DataFrame, outlier report)
        """
        try:
            from sklearn.ensemble import IsolationForest

            processed_df = df.copy()
            outlier_report = {}

            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns

            for column in numeric_columns:
                # Skip certain columns that shouldn't have outlier treatment
                if column in ['Position', 'Keyword Difficulty', 'Competition']:
                    continue

                column_data = processed_df[column].dropna()

                if len(column_data) < 10:  # Need minimum data for outlier detection
                    continue

                outliers_mask = self._detect_outliers(column_data, method, threshold)
                outlier_count = outliers_mask.sum()

                if outlier_count > 0:
                    outlier_report[column] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(column_data)) * 100,
                        'method': method
                    }

                    # Handle outliers based on column type
                    if column in ['Traffic', 'Search Volume', 'Organic Traffic']:
                        # For traffic/volume, cap at 95th percentile
                        cap_value = column_data.quantile(0.95)
                        processed_df.loc[processed_df[column] > cap_value, column] = cap_value
                    else:
                        # For other metrics, use IQR capping
                        Q1 = column_data.quantile(0.25)
                        Q3 = column_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        processed_df.loc[processed_df[column] < lower_bound, column] = lower_bound
                        processed_df.loc[processed_df[column] > upper_bound, column] = upper_bound

            self.logger.info(f"Processed outliers in {len(outlier_report)} columns")
            return processed_df, outlier_report

        except Exception as e:
            self.logger.error(f"Error in outlier detection: {str(e)}")
            return df, {}

    def _clean_keyword_column(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Clean keyword data with SEO-specific rules."""
        try:
            cleaned_df = df.copy()

            # Remove non-printable characters
            cleaned_df['Keyword'] = cleaned_df['Keyword'].astype(str).str.strip()

            # Remove keywords that are too short or too long
            min_length = config.get('keyword_min_length', 1)
            max_length = config.get('keyword_max_length', 200)

            length_mask = (
                cleaned_df['Keyword'].str.len().between(min_length, max_length)
            )
            cleaned_df = cleaned_df[length_mask]

            # Remove keywords with invalid characters (if strict mode)
            if config.get('strict_keyword_validation', False):
                pattern_mask = cleaned_df['Keyword'].str.match(
                    self.seo_patterns['keyword_pattern']
                )
                cleaned_df = cleaned_df[pattern_mask.fillna(False)]

            # Convert to lowercase if specified
            if config.get('lowercase_keywords', True):
                cleaned_df['Keyword'] = cleaned_df['Keyword'].str.lower()

            return cleaned_df

        except Exception as e:
            self.logger.error(f"Error cleaning keyword column: {str(e)}")
            return df

    def _clean_url_column(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Clean URL data with validation."""
        try:
            cleaned_df = df.copy()

            # Remove malformed URLs
            url_mask = cleaned_df['URL'].astype(str).str.match(
                self.seo_patterns['url_pattern']
            )
            cleaned_df = cleaned_df[url_mask.fillna(False)]

            # Normalize URLs
            if config.get('normalize_urls', True):
                cleaned_df['URL'] = cleaned_df['URL'].str.lower()
                # Remove trailing slashes
                cleaned_df['URL'] = cleaned_df['URL'].str.rstrip('/')

            # Remove URL parameters if specified
            if config.get('remove_url_params', False):
                cleaned_df['URL'] = cleaned_df['URL'].str.split('?').str[0]

            return cleaned_df

        except Exception as e:
            self.logger.error(f"Error cleaning URL column: {str(e)}")
            return df

    def _clean_position_data(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Clean position data with SEO-specific validation."""
        try:
            cleaned_df = df.copy()

            # Convert to numeric, coercing errors to NaN
            cleaned_df['Position'] = pd.to_numeric(
                cleaned_df['Position'], errors='coerce'
            )

            # Remove impossible positions
            min_position = config.get('min_position', 1)
            max_position = config.get('max_position', 100)

            position_mask = cleaned_df['Position'].between(min_position, max_position)
            cleaned_df = cleaned_df[position_mask.fillna(False)]

            # Round to integers
            cleaned_df['Position'] = cleaned_df['Position'].round().astype('Int64')

            return cleaned_df

        except Exception as e:
            self.logger.error(f"Error cleaning position data: {str(e)}")
            return df

    def _clean_traffic_data(
        self,
        df: pd.DataFrame,
        column: str,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Clean traffic data."""
        try:
            cleaned_df = df.copy()
            
            # Convert to numeric
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
            
            # Remove negative values
            cleaned_df = cleaned_df[cleaned_df[column] >= 0]
            
            return cleaned_df
        except Exception as e:
            self.logger.error(f"Error cleaning traffic data: {str(e)}")
            return df

    def _clean_search_volume_data(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Clean search volume data."""
        try:
            cleaned_df = df.copy()
            
            # Convert to numeric
            cleaned_df['Search Volume'] = pd.to_numeric(cleaned_df['Search Volume'], errors='coerce')
            
            # Remove negative values and unrealistic high values
            max_volume = config.get('max_search_volume', 10000000)
            cleaned_df = cleaned_df[
                (cleaned_df['Search Volume'] >= 0) & 
                (cleaned_df['Search Volume'] <= max_volume)
            ]
            
            return cleaned_df
        except Exception as e:
            self.logger.error(f"Error cleaning search volume data: {str(e)}")
            return df

    def _clean_serp_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Clean SERP features data."""
        try:
            cleaned_df = df.copy()
            
            # Clean SERP features column
            cleaned_df['SERP Features by Keyword'] = (
                cleaned_df['SERP Features by Keyword']
                .astype(str)
                .str.strip()
                .replace('nan', '')
            )
            
            return cleaned_df
        except Exception as e:
            self.logger.error(f"Error cleaning SERP features: {str(e)}")
            return df

    def _get_default_cleaning_config(self) -> Dict[str, Any]:
        """Get default cleaning configuration."""
        return {
            'remove_duplicates': True,
            'keyword_min_length': 1,
            'keyword_max_length': 200,
            'lowercase_keywords': True,
            'strict_keyword_validation': False,
            'normalize_urls': True,
            'remove_url_params': False,
            'min_position': 1,
            'max_position': 100,
            'min_traffic': 0,
            'max_search_volume': 10000000
        }

    def _get_default_standardization_rules(self) -> Dict[str, Any]:
        """Get default standardization rules."""
        return {
            'column_mapping': {},
            'data_types': {},
            'value_formats': {}
        }

    def _identify_common_schema(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Identify common schema across datasets."""
        return {}

    def _standardize_column_names(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Standardize column names."""
        return df

    def _standardize_data_types(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Standardize data types."""
        return df

    def _standardize_value_formats(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Standardize value formats."""
        return df

    def _ensure_schema_compliance(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Ensure schema compliance."""
        return df

    def _determine_intelligent_strategy(self, df: pd.DataFrame, column: str, missing_percentage: float) -> str:
        """Determine intelligent strategy for missing values."""
        if missing_percentage > 50:
            return 'drop'
        elif column in ['Position', 'Traffic', 'Search Volume']:
            return 'fill'
        else:
            return 'drop'

    def _apply_missing_value_strategy(self, df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
        """Apply missing value strategy."""
        if strategy == 'drop':
            return df.dropna(subset=[column])
        elif strategy == 'fill':
            if df[column].dtype in ['int64', 'float64']:
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna('')
        return df

    def _detect_outliers(
        self,
        data: pd.Series,
        method: str,
        threshold: float
    ) -> np.ndarray:
        """Detect outliers using specified method."""
        try:
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return (data < lower_bound) | (data > upper_bound)

            elif method == 'z_score':
                z_scores = np.abs(stats.zscore(data))
                return z_scores > 3

            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=threshold, random_state=42)
                outliers = iso_forest.fit_predict(data.values.reshape(-1, 1))
                return outliers == -1

            else:
                return np.zeros(len(data), dtype=bool)

        except Exception:
            return np.zeros(len(data), dtype=bool)


class DataValidator:
    """
    Advanced data validation for SEO competitive intelligence.
    Provides comprehensive validation rules and quality checks
    specific to SEO and competitive analysis datasets.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules = self._initialize_validation_rules()

    def validate_seo_dataset(
        self,
        df: pd.DataFrame,
        dataset_type: str = 'positions'
    ) -> DataQualityReport:
        """
        Comprehensive validation of SEO dataset.
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset ('positions', 'competitors', 'gap_keywords')
            
        Returns:
            DataQualityReport with validation results
        """
        try:
            self.logger.info(f"Validating {dataset_type} dataset with {len(df)} records")

            # Basic data quality checks
            missing_values = self._check_missing_values(df)
            duplicate_records = self._check_duplicates(df)
            data_types = self._check_data_types(df, dataset_type)
            outliers = self._check_outliers(df)

            # Dataset-specific validation
            specific_validation = self._validate_dataset_specific_rules(df, dataset_type)

            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                df, missing_values, duplicate_records, outliers, specific_validation
            )

            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                missing_values, duplicate_records, outliers, specific_validation
            )

            report = DataQualityReport(
                total_records=len(df),
                missing_values=missing_values,
                duplicate_records=duplicate_records,
                outliers=outliers,
                data_types=data_types,
                quality_score=quality_score,
                recommendations=recommendations
            )

            self.logger.info(f"Validation completed. Quality score: {quality_score:.2f}")
            return report

        except Exception as e:
            self.logger.error(f"Error in dataset validation: {str(e)}")
            return DataQualityReport(0, {}, 0, {}, {}, 0.0, [])

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: Dict[str, type]
    ) -> Tuple[bool, List[str]]:
        """
        Validate dataframe against expected schema (from paste file)
        
        Args:
            df: Dataframe to validate
            schema: Expected schema {column_name: expected_type}
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required columns
        missing_cols = set(schema.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # Check data types
        for col, expected_type in schema.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if expected_type == str and actual_type == object:
                    continue  # String type check passes
                elif expected_type in [int, float] and pd.api.types.is_numeric_dtype(actual_type):
                    continue  # Numeric type check passes
                else:
                    errors.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")

        return len(errors) == 0, errors

    def validate_date_range(
        self,
        df: pd.DataFrame,
        date_column: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Validate date range in dataframe (from paste file)
        
        Args:
            df: Dataframe to validate
            date_column: Name of date column
            start_date: Expected start date
            end_date: Expected end date
            
        Returns:
            Tuple of (is_valid, message)
        """
        if date_column not in df.columns:
            return False, f"Date column '{date_column}' not found"

        try:
            dates = pd.to_datetime(df[date_column])

            if start_date and dates.min() < start_date:
                return False, f"Data contains dates before {start_date}"

            if end_date and dates.max() > end_date:
                return False, f"Data contains dates after {end_date}"

            return True, "Date range is valid"

        except Exception as e:
            return False, f"Error validating dates: {str(e)}"

    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for different dataset types."""
        return {
            'positions': {
                'required_columns': ['Keyword', 'Position', 'Search Volume'],
                'numeric_columns': ['Position', 'Search Volume', 'Traffic', 'CPC'],
                'position_range': (1, 100),
                'search_volume_range': (0, 10000000),
                'traffic_range': (0, float('inf'))
            },
            'competitors': {
                'required_columns': ['Domain', 'Organic Keywords', 'Organic Traffic'],
                'numeric_columns': ['Organic Keywords', 'Organic Traffic', 'Competitor Relevance'],
                'relevance_range': (0, 1),
                'keywords_range': (0, float('inf'))
            },
            'gap_keywords': {
                'required_columns': ['Keyword', 'Volume', 'lenovo.com', 'dell.com', 'hp.com'],
                'numeric_columns': ['Volume', 'lenovo.com', 'dell.com', 'hp.com'],
                'volume_range': (0, 10000000),
                'position_range': (0, 100)
            }
        }

    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for missing values across all columns."""
        missing_values = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_values[column] = missing_count
        return missing_values

    def _check_duplicates(self, df: pd.DataFrame) -> int:
        """Check for duplicate records."""
        return df.duplicated().sum()

    def _check_data_types(self, df: pd.DataFrame, dataset_type: str) -> Dict[str, str]:
        """Check data types for all columns."""
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for outliers in numeric columns."""
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            data = df[column].dropna()
            if len(data) > 10:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
                if outlier_count > 0:
                    outliers[column] = outlier_count
        
        return outliers

    def _validate_dataset_specific_rules(self, df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
        """Validate dataset-specific rules."""
        return {'failures': 0}

    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        missing_values: Dict[str, int],
        duplicate_records: int,
        outliers: Dict[str, int],
        specific_validation: Dict[str, Any]
    ) -> float:
        """Calculate overall data quality score (0-1)."""
        try:
            score = 1.0

            # Deduct for missing values
            if missing_values:
                missing_ratio = sum(missing_values.values()) / (len(df) * len(df.columns))
                score -= missing_ratio * 0.3

            # Deduct for duplicates
            if duplicate_records > 0:
                duplicate_ratio = duplicate_records / len(df)
                score -= duplicate_ratio * 0.2

            # Deduct for outliers
            if outliers:
                outlier_ratio = sum(outliers.values()) / (len(df) * len(outliers))
                score -= outlier_ratio * 0.2

            # Deduct for specific validation failures
            validation_failures = specific_validation.get('failures', 0)
            if validation_failures > 0:
                failure_ratio = validation_failures / len(df)
                score -= failure_ratio * 0.3

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5

    def _generate_quality_recommendations(
        self,
        missing_values: Dict[str, int],
        duplicate_records: int,
        outliers: Dict[str, int],
        specific_validation: Dict[str, Any]
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if missing_values:
            recommendations.append("Handle missing values in affected columns")
        
        if duplicate_records > 0:
            recommendations.append("Remove duplicate records")
        
        if outliers:
            recommendations.append("Review and handle outliers in numeric columns")
        
        return recommendations


class DataTransformer:
    """
    Advanced data transformation utilities for SEO competitive intelligence.
    Provides feature engineering, scaling, and transformation capabilities
    optimized for machine learning on SEO datasets.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fitted_transformers = {}
        self.transformation_pipeline = []
        self.scalers = {}

    def create_temporal_features(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        value_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Create temporal features for time series analysis.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            value_columns: Columns to create temporal features for
            
        Returns:
            DataFrame with temporal features
        """
        try:
            if value_columns is None:
                value_columns = ['Position', 'Traffic', 'Search Volume']

            transformed_df = df.copy()

            # Ensure date column is datetime
            if date_column in transformed_df.columns:
                transformed_df[date_column] = pd.to_datetime(transformed_df[date_column])

                # Add date-based features
                transformed_df['year'] = transformed_df[date_column].dt.year
                transformed_df['month'] = transformed_df[date_column].dt.month
                transformed_df['day_of_week'] = transformed_df[date_column].dt.dayofweek
                transformed_df['day_of_year'] = transformed_df[date_column].dt.dayofyear
                transformed_df['week_of_year'] = transformed_df[date_column].dt.isocalendar().week

                # Add lag features
                for value_col in value_columns:
                    if value_col in transformed_df.columns:
                        for lag in [1, 7, 14, 30]:
                            lag_col = f"{value_col}_lag_{lag}"
                            transformed_df[lag_col] = transformed_df.groupby('Keyword')[value_col].shift(lag)

                # Add rolling statistics
                for value_col in value_columns:
                    if value_col in transformed_df.columns:
                        for window in [7, 14, 30]:
                            rolling_mean_col = f"{value_col}_rolling_mean_{window}"
                            rolling_std_col = f"{value_col}_rolling_std_{window}"

                            transformed_df[rolling_mean_col] = (
                                transformed_df.groupby('Keyword')[value_col]
                                .rolling(window=window, min_periods=1)
                                .mean()
                                .reset_index(0, drop=True)
                            )

                            transformed_df[rolling_std_col] = (
                                transformed_df.groupby('Keyword')[value_col]
                                .rolling(window=window, min_periods=1)
                                .std()
                                .reset_index(0, drop=True)
                            )

            self.logger.info(f"Created temporal features. New shape: {transformed_df.shape}")
            return transformed_df

        except Exception as e:
            self.logger.error(f"Error creating temporal features: {str(e)}")
            return df

    def apply_scaling(
        self,
        df: pd.DataFrame,
        scaling_method: str = 'standard',
        columns_to_scale: List[str] = None,
        fit_scaler: bool = True,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply scaling to numerical columns.
        
        Args:
            df: Input DataFrame
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
            columns_to_scale: Specific columns to scale (legacy parameter)
            fit_scaler: Whether to fit the scaler
            columns: Columns to scale (from paste file, takes precedence)
            
        Returns:
            DataFrame with scaled features
        """
        try:
            scaled_df = df.copy()

            # Use columns parameter if provided (from paste file), otherwise use columns_to_scale
            if columns is not None:
                target_columns = columns
            elif columns_to_scale is not None:
                target_columns = columns_to_scale
            else:
                target_columns = scaled_df.select_dtypes(include=[np.number]).columns.tolist()

            # Remove columns that shouldn't be scaled
            columns_to_exclude = ['Position', 'year', 'month', 'day_of_week']
            target_columns = [col for col in target_columns if col not in columns_to_exclude and col in scaled_df.columns]

            if not target_columns:
                return scaled_df

            # Select scaler
            scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler()
            }

            scaler = scalers.get(scaling_method, StandardScaler())

            # Fit and transform
            if fit_scaler:
                scaled_values = scaler.fit_transform(scaled_df[target_columns])
                self.fitted_transformers[f"{scaling_method}_scaler"] = scaler
                self.scalers[scaling_method] = scaler  # For backward compatibility
            else:
                if f"{scaling_method}_scaler" in self.fitted_transformers:
                    scaler = self.fitted_transformers[f"{scaling_method}_scaler"]
                    scaled_values = scaler.transform(scaled_df[target_columns])
                elif scaling_method in self.scalers:  # Backward compatibility
                    scaler = self.scalers[scaling_method]
                    scaled_values = scaler.transform(scaled_df[target_columns])
                else:
                    self.logger.warning(f"No fitted {scaling_method} scaler found")
                    return scaled_df

            # Update DataFrame
            scaled_df[target_columns] = scaled_values

            self.logger.info(f"Applied {scaling_method} scaling to {len(target_columns)} columns")
            return scaled_df

        except Exception as e:
            self.logger.error(f"Error applying scaling: {str(e)}")
            return df

    def create_time_features(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """
        Create time-based features (from paste file)
        
        Args:
            df: Input dataframe
            date_column: Name of date column
            
        Returns:
            Dataframe with time features
        """
        df_features = df.copy()

        if date_column not in df.columns:
            return df_features

        try:
            df_features[date_column] = pd.to_datetime(df_features[date_column])

            # Extract time components
            df_features[f'{date_column}_year'] = df_features[date_column].dt.year
            df_features[f'{date_column}_month'] = df_features[date_column].dt.month
            df_features[f'{date_column}_day'] = df_features[date_column].dt.day
            df_features[f'{date_column}_dayofweek'] = df_features[date_column].dt.dayofweek
            df_features[f'{date_column}_quarter'] = df_features[date_column].dt.quarter
            df_features[f'{date_column}_weekofyear'] = df_features[date_column].dt.isocalendar().week

        except Exception as e:
            self.logger.error(f"Error creating time features: {str(e)}")

        return df_features
