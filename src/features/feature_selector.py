"""
Feature Selection Module for SEO Competitive Intelligence
Advanced feature selection leveraging the comprehensive utility framework
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
from src.utils.validation_utils import SchemaValidator, BusinessRuleValidator
from src.utils.export_utils import ReportExporter, DataExporter
from src.utils.file_utils import FileManager

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection operations"""
    selection_method: str = 'mutual_info'
    n_features_to_select: Optional[int] = None
    feature_selection_ratio: float = 0.5
    cross_validation_folds: int = 5
    scoring_metric: str = 'accuracy'
    include_statistical_tests: bool = True
    include_correlation_analysis: bool = True
    remove_multicollinear_features: bool = True
    multicollinearity_threshold: float = 0.95

@dataclass
class FeatureSelectionResult:
    """Result of feature selection process"""
    selected_features: List[str]
    feature_scores: Dict[str, float]
    feature_rankings: Dict[str, int]
    selection_metadata: Dict[str, Any]
    removed_features: List[str]
    multicollinear_groups: List[List[str]]
    cross_validation_scores: Dict[str, float]
    feature_importance_matrix: pd.DataFrame

@dataclass
class FeatureImportanceAnalysis:
    """Feature importance analysis results"""
    importance_scores: Dict[str, float]
    importance_rankings: Dict[str, int]
    statistical_significance: Dict[str, float]
    stability_scores: Dict[str, float]
    feature_interactions: pd.DataFrame
    dimensionality_analysis: Dict[str, Any]

class FeatureSelector:
    """
    Advanced feature selection for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    feature selection capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("feature_selector")
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
        
        # Load feature selection configurations
        analysis_config = self.config.get_analysis_config()
        self.default_config = FeatureSelectionConfig()

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def comprehensive_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Optional[FeatureSelectionConfig] = None,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureSelectionResult:
        """
        Perform comprehensive feature selection using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            config: Feature selection configuration
            feature_metadata: Metadata about features
            
        Returns:
            FeatureSelectionResult with comprehensive selection analysis
        """
        try:
            with self.performance_tracker.track_block("comprehensive_feature_selection"):
                # Audit log the feature selection operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="feature_selection",
                    parameters={
                        "n_features": len(X.columns),
                        "n_samples": len(X),
                        "selection_method": config.selection_method if config else "default"
                    }
                )
                
                if config is None:
                    config = self.default_config
                
                # Clean and validate data using DataProcessor
                X_clean = self.data_processor.clean_seo_data(X)
                
                # Handle missing values in target
                if y.isnull().any():
                    self.logger.warning("Target variable contains missing values")
                    valid_indices = ~(X_clean.isnull().any(axis=1) | y.isnull())
                    X_clean = X_clean[valid_indices]
                    y = y[valid_indices]
                
                # Validate data quality
                validation_report = self.data_validator.validate_seo_dataset(X_clean, 'features')
                if validation_report.quality_score < 0.7:
                    self.logger.warning(f"Low feature data quality: {validation_report.quality_score:.3f}")
                
                # 1. Remove multicollinear features using statistical analysis
                if config.remove_multicollinear_features:
                    X_clean, multicollinear_groups = self._remove_multicollinear_features(
                        X_clean, config.multicollinearity_threshold
                    )
                else:
                    multicollinear_groups = []
                
                # 2. Calculate feature importance using multiple methods
                feature_scores = self._calculate_comprehensive_feature_scores(
                    X_clean, y, config
                )
                
                # 3. Perform feature selection based on method
                selected_features = self._select_features_by_method(
                    X_clean, y, feature_scores, config
                )
                
                # 4. Cross-validation evaluation
                cv_scores = self._evaluate_feature_selection_cv(
                    X_clean[selected_features], y, config
                )
                
                # 5. Generate feature rankings
                feature_rankings = self._generate_feature_rankings(feature_scores)
                
                # 6. Create feature importance matrix
                importance_matrix = self._create_feature_importance_matrix(
                    X_clean, y, selected_features
                )
                
                # 7. Generate metadata
                selection_metadata = self._generate_selection_metadata(
                    X_clean, selected_features, config, feature_metadata
                )
                
                removed_features = [f for f in X.columns if f not in selected_features]
                
                result = FeatureSelectionResult(
                    selected_features=selected_features,
                    feature_scores=feature_scores,
                    feature_rankings=feature_rankings,
                    selection_metadata=selection_metadata,
                    removed_features=removed_features,
                    multicollinear_groups=multicollinear_groups,
                    cross_validation_scores=cv_scores,
                    feature_importance_matrix=importance_matrix
                )
                
                self.logger.info(f"Feature selection completed: {len(selected_features)}/{len(X.columns)} features selected")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive feature selection: {str(e)}")
            return FeatureSelectionResult([], {}, {}, {}, [], [], {}, pd.DataFrame())

    @timing_decorator()
    def analyze_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str] = None,
        include_interactions: bool = True
    ) -> FeatureImportanceAnalysis:
        """
        Analyze feature importance using multiple methods and statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            methods: List of importance calculation methods
            include_interactions: Whether to analyze feature interactions
            
        Returns:
            FeatureImportanceAnalysis with comprehensive importance analysis
        """
        try:
            with self.performance_tracker.track_block("analyze_feature_importance"):
                if methods is None:
                    methods = ['mutual_info', 'correlation', 'random_forest', 'chi2']
                
                # Clean data using DataProcessor
                X_clean = self.data_processor.clean_seo_data(X)
                
                # Calculate importance scores using multiple methods
                importance_scores = {}
                for method in methods:
                    method_scores = self._calculate_importance_by_method(X_clean, y, method)
                    for feature, score in method_scores.items():
                        if feature not in importance_scores:
                            importance_scores[feature] = []
                        importance_scores[feature].append(score)
                
                # Average importance scores across methods
                averaged_scores = {
                    feature: np.mean(scores) for feature, scores in importance_scores.items()
                }
                
                # Calculate statistical significance using statistical calculator
                significance_scores = self._calculate_feature_significance(X_clean, y)
                
                # Calculate stability scores using cross-validation
                stability_scores = self._calculate_feature_stability(X_clean, y, methods)
                
                # Generate feature rankings
                rankings = {
                    feature: rank for rank, (feature, score) in enumerate(
                        sorted(averaged_scores.items(), key=lambda x: x[1], reverse=True), 1
                    )
                }
                
                # Analyze feature interactions if requested
                interactions_df = pd.DataFrame()
                if include_interactions:
                    interactions_df = self._analyze_feature_interactions(X_clean, y)
                
                # Dimensionality analysis using PCA and other methods
                dimensionality_analysis = self._analyze_feature_dimensionality(X_clean)
                
                analysis = FeatureImportanceAnalysis(
                    importance_scores=averaged_scores,
                    importance_rankings=rankings,
                    statistical_significance=significance_scores,
                    stability_scores=stability_scores,
                    feature_interactions=interactions_df,
                    dimensionality_analysis=dimensionality_analysis
                )
                
                self.logger.info(f"Feature importance analysis completed for {len(X_clean.columns)} features")
                return analysis
                
        except Exception as e:
            self.logger.error(f"Error in feature importance analysis: {str(e)}")
            return FeatureImportanceAnalysis({}, {}, {}, {}, pd.DataFrame(), {})

    @timing_decorator()
    def optimize_feature_subset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimization_method: str = 'genetic_algorithm',
        objective_function: str = 'accuracy',
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize feature subset using advanced optimization techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            optimization_method: Optimization algorithm to use
            objective_function: Objective function to optimize
            constraints: Optimization constraints
            
        Returns:
            Optimization results with optimal feature subset
        """
        try:
            with self.performance_tracker.track_block("optimize_feature_subset"):
                # Clean data
                X_clean = self.data_processor.clean_seo_data(X)
                
                if constraints is None:
                    constraints = {
                        'max_features': len(X_clean.columns) // 2,
                        'min_features': 5,
                        'max_correlation': 0.95
                    }
                
                if optimization_method == 'genetic_algorithm':
                    return self._genetic_algorithm_feature_selection(
                        X_clean, y, objective_function, constraints
                    )
                elif optimization_method == 'particle_swarm':
                    return self._particle_swarm_feature_selection(
                        X_clean, y, objective_function, constraints
                    )
                elif optimization_method == 'recursive_elimination':
                    return self._recursive_feature_elimination(
                        X_clean, y, objective_function, constraints
                    )
                else:
                    raise ValueError(f"Unsupported optimization method: {optimization_method}")
                    
        except Exception as e:
            self.logger.error(f"Error in feature subset optimization: {str(e)}")
            return {}

    def _remove_multicollinear_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        """Remove multicollinear features using correlation analysis."""
        try:
            # Calculate correlation matrix using StatisticalCalculator
            corr_matrix, p_values = self.stats_calculator.calculate_correlation_matrix(X)
            
            if corr_matrix.empty:
                return X, []
            
            # Find highly correlated pairs
            multicollinear_groups = []
            features_to_remove = set()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    correlation = abs(corr_matrix.iloc[i, j])
                    
                    if correlation > threshold:
                        feature1 = corr_matrix.columns[i]
                        feature2 = corr_matrix.columns[j]
                        
                        # Create or add to multicollinear group
                        group_found = False
                        for group in multicollinear_groups:
                            if feature1 in group or feature2 in group:
                                group.extend([feature1, feature2])
                                group_found = True
                                break
                        
                        if not group_found:
                            multicollinear_groups.append([feature1, feature2])
                        
                        # Remove the feature with lower variance (less informative)
                        if X[feature1].var() < X[feature2].var():
                            features_to_remove.add(feature1)
                        else:
                            features_to_remove.add(feature2)
            
            # Remove features and clean up groups
            X_filtered = X.drop(columns=list(features_to_remove))
            multicollinear_groups = [list(set(group)) for group in multicollinear_groups]
            
            self.logger.info(f"Removed {len(features_to_remove)} multicollinear features")
            return X_filtered, multicollinear_groups
            
        except Exception as e:
            self.logger.error(f"Error removing multicollinear features: {str(e)}")
            return X, []

    def _calculate_comprehensive_feature_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: FeatureSelectionConfig
    ) -> Dict[str, float]:
        """Calculate feature scores using multiple methods."""
        try:
            feature_scores = {}
            
            # Method 1: Mutual Information
            if config.selection_method in ['mutual_info', 'comprehensive']:
                mi_scores = self._calculate_mutual_information(X, y)
                for feature, score in mi_scores.items():
                    feature_scores[f"{feature}_mi"] = score
            
            # Method 2: Statistical tests (correlation, chi2, f-test)
            if config.include_statistical_tests:
                stat_scores = self._calculate_statistical_scores(X, y)
                for feature, score in stat_scores.items():
                    feature_scores[f"{feature}_stat"] = score
            
            # Method 3: Tree-based importance
            if config.selection_method in ['tree_based', 'comprehensive']:
                tree_scores = self._calculate_tree_importance(X, y)
                for feature, score in tree_scores.items():
                    feature_scores[f"{feature}_tree"] = score
            
            # Method 4: Correlation with target
            if config.include_correlation_analysis:
                corr_scores = self._calculate_correlation_scores(X, y)
                for feature, score in corr_scores.items():
                    feature_scores[f"{feature}_corr"] = score
            
            # Aggregate scores by feature
            aggregated_scores = {}
            for feature in X.columns:
                feature_score_list = [
                    score for score_name, score in feature_scores.items()
                    if score_name.startswith(feature)
                ]
                if feature_score_list:
                    aggregated_scores[feature] = np.mean(feature_score_list)
            
            return aggregated_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating feature scores: {str(e)}")
            return {}

    def _calculate_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate mutual information scores."""
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from sklearn.preprocessing import LabelEncoder
            
            # Determine if regression or classification
            if y.dtype == 'object' or len(y.unique()) < 10:
                # Classification
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
            else:
                # Regression
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            return dict(zip(X.columns, mi_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating mutual information: {str(e)}")
            return {}

    def _calculate_statistical_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate statistical test scores using StatisticalCalculator."""
        try:
            scores = {}
            
            for feature in X.columns:
                feature_data = X[feature].dropna()
                
                if len(feature_data) == 0:
                    scores[feature] = 0.0
                    continue
                
                # Calculate correlation with target
                try:
                    correlation = feature_data.corr(y)
                    if pd.isna(correlation):
                        correlation = 0.0
                    scores[feature] = abs(correlation)
                except Exception:
                    scores[feature] = 0.0
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical scores: {str(e)}")
            return {}

    def _calculate_tree_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate tree-based feature importance."""
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Determine if regression or classification
            if y.dtype == 'object' or len(y.unique()) < 10:
                # Classification
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y_encoded)
            else:
                # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
            
            importance_scores = model.feature_importances_
            return dict(zip(X.columns, importance_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating tree importance: {str(e)}")
            return {}

    def _calculate_correlation_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate correlation scores with target variable."""
        try:
            scores = {}
            
            for feature in X.columns:
                try:
                    correlation = X[feature].corr(y)
                    scores[feature] = abs(correlation) if not pd.isna(correlation) else 0.0
                except Exception:
                    scores[feature] = 0.0
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation scores: {str(e)}")
            return {}

    def _select_features_by_method(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_scores: Dict[str, float],
        config: FeatureSelectionConfig
    ) -> List[str]:
        """Select features based on specified method."""
        try:
            if config.n_features_to_select:
                n_select = min(config.n_features_to_select, len(feature_scores))
            else:
                n_select = int(len(feature_scores) * config.feature_selection_ratio)
            
            # Sort features by score
            sorted_features = sorted(
                feature_scores.items(), key=lambda x: x[1], reverse=True
            )
            
            # Select top N features
            selected_features = [feature for feature, score in sorted_features[:n_select]]
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            return list(X.columns)

    def _evaluate_feature_selection_cv(
        self,
        X_selected: pd.DataFrame,
        y: pd.Series,
        config: FeatureSelectionConfig
    ) -> Dict[str, float]:
        """Evaluate feature selection using cross-validation."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            if len(X_selected.columns) == 0:
                return {'cv_score': 0.0}
            
            # Choose appropriate model
            if y.dtype == 'object' or len(y.unique()) < 10:
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                target = y_encoded
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                target = y
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_selected, target, 
                cv=config.cross_validation_folds, 
                scoring=config.scoring_metric
            )
            
            return {
                'cv_score_mean': np.mean(cv_scores),
                'cv_score_std': np.std(cv_scores),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation evaluation: {str(e)}")
            return {'cv_score': 0.0}

    def _generate_feature_rankings(self, feature_scores: Dict[str, float]) -> Dict[str, int]:
        """Generate feature rankings based on scores."""
        try:
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            rankings = {feature: rank for rank, (feature, score) in enumerate(sorted_features, 1)}
            return rankings
        except Exception:
            return {}

    def _create_feature_importance_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str]
    ) -> pd.DataFrame:
        """Create comprehensive feature importance matrix."""
        try:
            importance_data = []
            
            for feature in X.columns:
                importance_row = {
                    'feature': feature,
                    'selected': feature in selected_features,
                    'correlation_with_target': X[feature].corr(y) if not X[feature].isnull().all() else 0,
                    'variance': X[feature].var(),
                    'missing_ratio': X[feature].isnull().mean(),
                    'unique_values': X[feature].nunique()
                }
                importance_data.append(importance_row)
            
            return pd.DataFrame(importance_data)
            
        except Exception as e:
            self.logger.error(f"Error creating importance matrix: {str(e)}")
            return pd.DataFrame()

    def _generate_selection_metadata(
        self,
        X: pd.DataFrame,
        selected_features: List[str],
        config: FeatureSelectionConfig,
        feature_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive selection metadata."""
        try:
            metadata = {
                'original_features_count': len(X.columns),
                'selected_features_count': len(selected_features),
                'selection_ratio': len(selected_features) / len(X.columns),
                'selection_method': config.selection_method,
                'selection_timestamp': datetime.now().isoformat(),
                'data_shape': X.shape,
                'feature_types': X.dtypes.astype(str).to_dict(),
                'missing_values_count': X.isnull().sum().to_dict()
            }
            
            if feature_metadata:
                metadata['feature_metadata'] = feature_metadata
            
            return metadata
            
        except Exception:
            return {}

    def _calculate_feature_significance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate statistical significance of features."""
        try:
            significance_scores = {}
            
            for feature in X.columns:
                try:
                    # Perform statistical test using StatisticalCalculator
                    test_result = self.stats_calculator.perform_hypothesis_test(
                        X[feature].dropna(), alternative='two-sided'
                    )
                    p_value = test_result.get('p_value', 1.0)
                    significance_scores[feature] = 1 - p_value  # Higher is more significant
                except Exception:
                    significance_scores[feature] = 0.0
            
            return significance_scores
            
        except Exception:
            return {}

    def _calculate_feature_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str]
    ) -> Dict[str, float]:
        """Calculate feature selection stability across different samples."""
        try:
            from sklearn.model_selection import StratifiedKFold, KFold
            
            stability_scores = {feature: [] for feature in X.columns}
            
            # Use appropriate cross-validation
            if y.dtype == 'object' or len(y.unique()) < 10:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, _ in cv.split(X, y):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                
                # Calculate importance for this fold
                fold_scores = self._calculate_importance_by_method(X_train, y_train, methods[0])
                
                for feature, score in fold_scores.items():
                    stability_scores[feature].append(score)
            
            # Calculate stability as 1 - coefficient of variation
            final_stability = {}
            for feature, scores in stability_scores.items():
                if scores and np.mean(scores) > 0:
                    cv_score = np.std(scores) / np.mean(scores)
                    final_stability[feature] = max(0, 1 - cv_score)
                else:
                    final_stability[feature] = 0.0
            
            return final_stability
            
        except Exception as e:
            self.logger.error(f"Error calculating feature stability: {str(e)}")
            return {}

    def _calculate_importance_by_method(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str
    ) -> Dict[str, float]:
        """Calculate feature importance using specific method."""
        try:
            if method == 'mutual_info':
                return self._calculate_mutual_information(X, y)
            elif method == 'correlation':
                return self._calculate_correlation_scores(X, y)
            elif method == 'random_forest':
                return self._calculate_tree_importance(X, y)
            elif method == 'chi2':
                return self._calculate_statistical_scores(X, y)
            else:
                return {}
        except Exception:
            return {}

    def _analyze_feature_interactions(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Analyze feature interactions using statistical methods."""
        try:
            interactions = []
            features = X.columns.tolist()
            
            # Analyze pairwise interactions (limited to avoid exponential complexity)
            for i, feature1 in enumerate(features[:20]):  # Limit for performance
                for j, feature2 in enumerate(features[i+1:21]):
                    try:
                        # Create interaction term
                        interaction_term = X[feature1] * X[feature2]
                        
                        # Calculate correlation with target
                        interaction_corr = interaction_term.corr(y)
                        
                        if not pd.isna(interaction_corr):
                            interactions.append({
                                'feature1': feature1,
                                'feature2': feature2,
                                'interaction_strength': abs(interaction_corr),
                                'interaction_direction': 'positive' if interaction_corr > 0 else 'negative'
                            })
                    except Exception:
                        continue
            
            interactions_df = pd.DataFrame(interactions)
            if not interactions_df.empty:
                interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)
            
            return interactions_df
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature interactions: {str(e)}")
            return pd.DataFrame()

    def _analyze_feature_dimensionality(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature dimensionality using PCA and other methods."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA()
            pca.fit(X_scaled)
            
            # Calculate cumulative explained variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            
            # Find number of components for different variance thresholds
            n_components_80 = np.argmax(cumsum_var >= 0.8) + 1
            n_components_90 = np.argmax(cumsum_var >= 0.9) + 1
            n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
            
            analysis = {
                'original_dimensions': X.shape[1],
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': cumsum_var.tolist(),
                'n_components_80_variance': n_components_80,
                'n_components_90_variance': n_components_90,
                'n_components_95_variance': n_components_95,
                'dimensionality_reduction_potential': {
                    '80%': (X.shape[1] - n_components_80) / X.shape[1],
                    '90%': (X.shape[1] - n_components_90) / X.shape[1],
                    '95%': (X.shape[1] - n_components_95) / X.shape[1]
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in dimensionality analysis: {str(e)}")
            return {}

    def _genetic_algorithm_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        objective_function: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Feature selection using genetic algorithm optimization."""
        try:
            # Simplified genetic algorithm implementation
            # In practice, you might use DEAP or similar libraries
            
            n_features = len(X.columns)
            population_size = 50
            n_generations = 100
            mutation_rate = 0.1
            
            # Initialize population (binary chromosomes)
            population = []
            for _ in range(population_size):
                chromosome = np.random.random(n_features) < 0.5
                # Ensure minimum features constraint
                if np.sum(chromosome) < constraints['min_features']:
                    indices = np.random.choice(n_features, constraints['min_features'], replace=False)
                    chromosome[indices] = True
                population.append(chromosome)
            
            best_fitness = -np.inf
            best_chromosome = None
            
            for generation in range(n_generations):
                # Evaluate fitness
                fitness_scores = []
                for chromosome in population:
                    selected_features = X.columns[chromosome].tolist()
                    if len(selected_features) == 0:
                        fitness = 0
                    else:
                        cv_result = self._evaluate_feature_selection_cv(
                            X[selected_features], y, self.default_config
                        )
                        fitness = cv_result.get('cv_score_mean', 0)
                    fitness_scores.append(fitness)
                
                # Track best solution
                max_fitness_idx = np.argmax(fitness_scores)
                if fitness_scores[max_fitness_idx] > best_fitness:
                    best_fitness = fitness_scores[max_fitness_idx]
                    best_chromosome = population[max_fitness_idx].copy()
                
                # Selection, crossover, and mutation would go here
                # Simplified: just keep best solutions and add some random ones
                sorted_indices = np.argsort(fitness_scores)[::-1]
                new_population = []
                
                # Keep top 50%
                for i in range(population_size // 2):
                    new_population.append(population[sorted_indices[i]])
                
                # Add random chromosomes for diversity
                for _ in range(population_size - len(new_population)):
                    chromosome = np.random.random(n_features) < 0.5
                    if np.sum(chromosome) < constraints['min_features']:
                        indices = np.random.choice(n_features, constraints['min_features'], replace=False)
                        chromosome[indices] = True
                    new_population.append(chromosome)
                
                population = new_population
            
            selected_features = X.columns[best_chromosome].tolist()
            
            return {
                'optimization_method': 'genetic_algorithm',
                'optimal_features': selected_features,
                'best_fitness': best_fitness,
                'n_features_selected': len(selected_features),
                'optimization_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in genetic algorithm feature selection: {str(e)}")
            return {}

    def export_feature_selection_results(
        self,
        selection_result: FeatureSelectionResult,
        importance_analysis: FeatureImportanceAnalysis,
        export_directory: str
    ) -> Dict[str, bool]:
        """Export comprehensive feature selection results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'selection_summary': {
                    'selected_features_count': len(selection_result.selected_features),
                    'total_features_count': len(selection_result.selected_features) + len(selection_result.removed_features),
                    'selection_ratio': len(selection_result.selected_features) / (len(selection_result.selected_features) + len(selection_result.removed_features)),
                    'multicollinear_groups_count': len(selection_result.multicollinear_groups),
                    'cross_validation_score': selection_result.cross_validation_scores.get('cv_score_mean', 0)
                },
                'selected_features': selection_result.selected_features,
                'feature_scores': selection_result.feature_scores,
                'feature_rankings': selection_result.feature_rankings,
                'importance_analysis': {
                    'importance_scores': importance_analysis.importance_scores,
                    'importance_rankings': importance_analysis.importance_rankings,
                    'statistical_significance': importance_analysis.statistical_significance,
                    'stability_scores': importance_analysis.stability_scores
                },
                'removed_features': selection_result.removed_features,
                'multicollinear_groups': selection_result.multicollinear_groups,
                'dimensionality_analysis': importance_analysis.dimensionality_analysis
            }
            
            # Export detailed data using DataExporter
            data_export_success = self.data_exporter.export_analysis_dataset(
                {'feature_selection_results': pd.DataFrame([export_data])},
                export_path / "feature_selection_detailed.xlsx"
            )
            
            # Export feature importance matrix
            matrix_export_success = True
            if not selection_result.feature_importance_matrix.empty:
                matrix_export_success = self.data_exporter.export_with_metadata(
                    selection_result.feature_importance_matrix,
                    metadata={'analysis_type': 'feature_importance', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "feature_importance_matrix.xlsx"
                )
            
            # Export feature interactions
            interactions_export_success = True
            if not importance_analysis.feature_interactions.empty:
                interactions_export_success = self.data_exporter.export_with_metadata(
                    importance_analysis.feature_interactions,
                    metadata={'analysis_type': 'feature_interactions'},
                    export_path=export_path / "feature_interactions.xlsx"
                )
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_executive_report(
                export_data,
                export_path / "feature_selection_executive_report.html",
                format='html',
                include_charts=True
            )
            
            return {
                'detailed_data': data_export_success,
                'importance_matrix': matrix_export_success,
                'feature_interactions': interactions_export_success,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting feature selection results: {str(e)}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for feature selection operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )
