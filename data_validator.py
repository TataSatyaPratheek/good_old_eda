"""
Data Quality Validator (Brick 4)
Ensures data integrity and quality for reliable analysis
"""

import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

class DataValidator:
    """Validate data quality and completeness"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run comprehensive data validation"""
        
        print("🔍 Validating data quality...")
        
        validation_results = {}
        
        for dataset_name, df in data.items():
            print(f"   📊 Validating {dataset_name}...")
            validation_results[dataset_name] = self._validate_dataset(df, dataset_name)
        
        # Overall summary
        validation_results['summary'] = self._create_validation_summary(validation_results)
        
        print("✅ Data validation complete!")
        return validation_results
    
    def _validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate a single dataset"""
        
        if df.empty:
            return {
                'status': 'empty',
                'row_count': 0,
                'issues': ['Dataset is empty'],
                'quality_score': 0.0,
                'recommendations': ['Check data loading process']
            }
        
        issues = []
        recommendations = []
        quality_factors = []
        
        # Check required columns
        required_columns = self._get_required_columns(dataset_name)
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            recommendations.append(f"Ensure {missing_columns} are included in source data")
            quality_factors.append(0.3)
        else:
            quality_factors.append(1.0)
        
        # Check data completeness
        completeness = self._check_completeness(df)
        if completeness < 0.8:
            issues.append(f"Low data completeness: {completeness:.1%}")
            recommendations.append("Review data source for missing values")
        quality_factors.append(completeness)
        
        # Check for duplicates
        duplicate_info = self._check_duplicates(df)
        if duplicate_info['count'] > 0:
            issues.append(f"{duplicate_info['count']} duplicate rows found")
            recommendations.append("Remove duplicate records")
            quality_factors.append(max(0, 1 - (duplicate_info['count'] / len(df))))
        else:
            quality_factors.append(1.0)
        
        # Check data types and ranges
        data_quality_info = self._check_data_quality(df, dataset_name)
        quality_factors.append(data_quality_info['score'])
        issues.extend(data_quality_info['issues'])
        recommendations.extend(data_quality_info['recommendations'])
        
        # Check for outliers
        outlier_info = self._check_outliers(df, dataset_name)
        if outlier_info['outlier_count'] > 0:
            issues.append(f"{outlier_info['outlier_count']} potential outliers detected")
            recommendations.append("Review outlier values for accuracy")
        
        overall_quality = sum(quality_factors) / len(quality_factors)
        
        # Determine status
        if overall_quality > 0.8:
            status = 'excellent'
        elif overall_quality > 0.7:
            status = 'good'
        elif overall_quality > 0.5:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'status': status,
            'row_count': len(df),
            'column_count': len(df.columns),
            'completeness': completeness,
            'duplicate_info': duplicate_info,
            'outlier_info': outlier_info,
            'issues': issues,
            'recommendations': recommendations,
            'quality_score': overall_quality,
            'quality_breakdown': {
                'column_completeness': quality_factors[0] if quality_factors else 0,
                'data_completeness': quality_factors[1] if len(quality_factors) > 1 else 0,
                'duplicate_check': quality_factors[2] if len(quality_factors) > 2 else 0,
                'data_validity': quality_factors[3] if len(quality_factors) > 3 else 0
            }
        }
    
    def _get_required_columns(self, dataset_name: str) -> List[str]:
        """Get required columns for each dataset type"""
        
        column_requirements = {
            'lenovo': ['Keyword', 'Position', 'Traffic (%)'],
            'dell': ['Keyword', 'Position', 'Traffic (%)'],
            'hp': ['Keyword', 'Position', 'Traffic (%)'],
            'gap_keywords': ['Keyword', 'Volume', 'Keyword Difficulty']
        }
        
        return column_requirements.get(dataset_name, ['Keyword'])
    
    def _check_completeness(self, df: pd.DataFrame) -> float:
        """Check data completeness (non-null values)"""
        
        total_cells = df.size
        non_null_cells = df.count().sum()
        
        return non_null_cells / total_cells if total_cells > 0 else 0
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate records"""
        
        duplicate_count = df.duplicated().sum()
        
        # Check for keyword duplicates specifically
        keyword_duplicates = 0
        if 'Keyword' in df.columns:
            keyword_duplicates = df['Keyword'].duplicated().sum()
        
        return {
            'count': duplicate_count,
            'percentage': (duplicate_count / len(df)) * 100 if len(df) > 0 else 0,
            'keyword_duplicates': keyword_duplicates
        }
    
    def _check_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Check data quality specific to dataset type"""
        
        quality_score = 1.0
        issues = []
        recommendations = []
        
        # Position values validation
        if 'Position' in df.columns:
            invalid_positions = df[(df['Position'] <= 0) | (df['Position'] > 100)]
            if len(invalid_positions) > 0:
                quality_score -= 0.2
                issues.append(f"{len(invalid_positions)} invalid position values (outside 1-100 range)")
                recommendations.append("Review position data for values outside normal range")
        
        # Traffic validation
        if 'Traffic (%)' in df.columns:
            invalid_traffic = df[df['Traffic (%)'] < 0]
            if len(invalid_traffic) > 0:
                quality_score -= 0.2
                issues.append(f"{len(invalid_traffic)} negative traffic values")
                recommendations.append("Check traffic percentage calculations")
        
        # Keywords validation
        if 'Keyword' in df.columns:
            empty_keywords = df[df['Keyword'].isna() | (df['Keyword'] == '')]
            if len(empty_keywords) > 0:
                quality_score -= 0.3
                issues.append(f"{len(empty_keywords)} empty keyword values")
                recommendations.append("Ensure all records have valid keywords")
            
            # Check for suspiciously short keywords
            if not df.empty:
                short_keywords = df[df['Keyword'].str.len() < 3]
                if len(short_keywords) > len(df) * 0.1:  # More than 10%
                    quality_score -= 0.1
                    issues.append(f"{len(short_keywords)} suspiciously short keywords")
                    recommendations.append("Review keywords with fewer than 3 characters")
        
        # Volume validation (for gap keywords)
        if 'Volume' in df.columns:
            invalid_volume = df[df['Volume'] < 0]
            if len(invalid_volume) > 0:
                quality_score -= 0.2
                issues.append(f"{len(invalid_volume)} negative search volume values")
                recommendations.append("Verify search volume data source")
        
        # Keyword difficulty validation
        if 'Keyword Difficulty' in df.columns:
            invalid_difficulty = df[(df['Keyword Difficulty'] < 0) | (df['Keyword Difficulty'] > 100)]
            if len(invalid_difficulty) > 0:
                quality_score -= 0.2
                issues.append(f"{len(invalid_difficulty)} invalid keyword difficulty values")
                recommendations.append("Ensure keyword difficulty is between 0-100")
        
        return {
            'score': max(0, quality_score),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_outliers(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Check for statistical outliers"""
        
        outlier_count = 0
        outlier_details = []
        
        # Check position outliers
        if 'Position' in df.columns:
            positions = df['Position'].dropna()
            if len(positions) > 0:
                q75, q25 = np.percentile(positions, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - (1.5 * iqr)
                upper_bound = q75 + (1.5 * iqr)
                
                position_outliers = df[(df['Position'] < lower_bound) | (df['Position'] > upper_bound)]
                outlier_count += len(position_outliers)
                
                if len(position_outliers) > 0:
                    outlier_details.append(f"{len(position_outliers)} position outliers")
        
        # Check traffic outliers
        if 'Traffic (%)' in df.columns:
            traffic = df['Traffic (%)'].dropna()
            if len(traffic) > 0 and traffic.std() > 0:
                # Use z-score for traffic outliers
                z_scores = np.abs((traffic - traffic.mean()) / traffic.std())
                traffic_outliers = df[df['Traffic (%)'].isin(traffic[z_scores > 3])]
                outlier_count += len(traffic_outliers)
                
                if len(traffic_outliers) > 0:
                    outlier_details.append(f"{len(traffic_outliers)} traffic outliers")
        
        return {
            'outlier_count': outlier_count,
            'details': outlier_details
        }
    
    def _create_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create overall validation summary"""
        
        dataset_results = {k: v for k, v in validation_results.items() if k != 'summary'}
        
        if not dataset_results:
            return {
                'total_datasets': 0,
                'valid_datasets': 0,
                'total_rows': 0,
                'average_quality_score': 0,
                'overall_status': 'no_data'
            }
        
        total_rows = sum(result['row_count'] for result in dataset_results.values())
        quality_scores = [result['quality_score'] for result in dataset_results.values()]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        excellent_datasets = len([r for r in dataset_results.values() if r['status'] == 'excellent'])
        good_datasets = len([r for r in dataset_results.values() if r['status'] == 'good'])
        fair_datasets = len([r for r in dataset_results.values() if r['status'] == 'fair'])
        poor_datasets = len([r for r in dataset_results.values() if r['status'] == 'poor'])
        
        # Determine overall status
        if avg_quality > 0.8:
            overall_status = 'excellent'
        elif avg_quality > 0.7:
            overall_status = 'good'
        elif avg_quality > 0.5:
            overall_status = 'fair'
        else:
            overall_status = 'poor'
        
        return {
            'total_datasets': len(dataset_results),
            'excellent_datasets': excellent_datasets,
            'good_datasets': good_datasets,
            'fair_datasets': fair_datasets,
            'poor_datasets': poor_datasets,
            'total_rows': total_rows,
            'average_quality_score': avg_quality,
            'overall_status': overall_status,
            'quality_distribution': {
                'excellent': excellent_datasets,
                'good': good_datasets,
                'fair': fair_datasets,
                'poor': poor_datasets
            },
            'recommendations': self._generate_overall_recommendations(dataset_results)
        }
    
    def _generate_overall_recommendations(self, dataset_results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations for data quality improvement"""
        
        recommendations = []
        
        # Check for common issues across datasets
        poor_datasets = [name for name, result in dataset_results.items() if result['quality_score'] < 0.5]
        if poor_datasets:
            recommendations.append(f"Priority: Improve data quality for {', '.join(poor_datasets)}")
        
        # Check for completeness issues
        incomplete_datasets = [name for name, result in dataset_results.items() if result['completeness'] < 0.8]
        if incomplete_datasets:
            recommendations.append(f"Address data completeness issues in {', '.join(incomplete_datasets)}")
        
        # Check for duplicate issues
        duplicate_datasets = [name for name, result in dataset_results.items() 
                            if result['duplicate_info']['count'] > 0]
        if duplicate_datasets:
            recommendations.append(f"Remove duplicates from {', '.join(duplicate_datasets)}")
        
        # General recommendations
        avg_quality = sum(result['quality_score'] for result in dataset_results.values()) / len(dataset_results)
        if avg_quality < 0.7:
            recommendations.append("Consider implementing automated data quality checks")
            recommendations.append("Review data collection and processing procedures")
        
        return recommendations
