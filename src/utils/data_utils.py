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
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Apply scaling to numerical columns.
        
        Args:
            df: Input DataFrame
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
            columns_to_scale: Specific columns to scale
            fit_scaler: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        try:
            scaled_df = df.copy()
            
            if columns_to_scale is None:
                columns_to_scale = scaled_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove columns that shouldn't be scaled
            columns_to_exclude = ['Position', 'year', 'month', 'day_of_week']
            columns_to_scale = [col for col in columns_to_scale if col not in columns_to_exclude]
            
            if not columns_to_scale:
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
                scaled_values = scaler.fit_transform(scaled_df[columns_to_scale])
                self.fitted_transformers[f"{scaling_method}_scaler"] = scaler
            else:
                if f"{scaling_method}_scaler" in self.fitted_transformers:
                    scaler = self.fitted_transformers[f"{scaling_method}_scaler"]
                    scaled_values = scaler.transform(scaled_df[columns_to_scale])
                else:
                    raise ValueError(f"No fitted {scaling_method} scaler found")
            
            # Update DataFrame
            scaled_df[columns_to_scale] = scaled_values
            
            self.logger.info(f"Applied {scaling_method} scaling to {len(columns_to_scale)} columns")
            return scaled_df
            
        except Exception as e:
            self.logger.error(f"Error applying scaling: {str(e)}")
            return df
