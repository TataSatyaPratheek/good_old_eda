"""
Model Evaluator Module for SEO Competitive Intelligence
Advanced model evaluation leveraging the comprehensive utility framework
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
from src.utils.file_utils import FileManager

@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation"""
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    scoring_metrics: List[str] = None
    include_feature_importance: bool = True
    include_residual_analysis: bool = True
    include_learning_curves: bool = True
    include_validation_curves: bool = False

@dataclass
class ModelPerformance:
    """Comprehensive model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    cross_val_scores: List[float]
    cross_val_mean: float
    cross_val_std: float
    training_time: float
    prediction_time: float
    feature_importance: Dict[str, float]

@dataclass
class ModelComparisonResult:
    """Result of model comparison analysis"""
    model_performances: Dict[str, ModelPerformance]
    best_model: str
    performance_ranking: List[str]
    statistical_comparison: Dict[str, Any]
    recommendations: List[str]
    evaluation_summary: Dict[str, Any]

@dataclass
class ModelValidationResult:
    """Model validation analysis result"""
    validation_metrics: Dict[str, float]
    residual_analysis: Dict[str, Any]
    learning_curves: Dict[str, Any]
    validation_curves: Dict[str, Any]
    bias_variance_analysis: Dict[str, Any]
    overfitting_assessment: Dict[str, Any]

class ModelEvaluator:
    """
    Advanced model evaluation for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    model evaluation capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("model_evaluator")
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
        
        # Load model evaluation configurations
        analysis_config = self.config.get_analysis_config()
        self.default_config = ModelEvaluationConfig(
            scoring_metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        )

    @timing_decorator()
    @memoize(ttl=3600)  # Cache for 1 hour
    def comprehensive_model_evaluation(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: Optional[ModelEvaluationConfig] = None
    ) -> ModelComparisonResult:
        """
        Perform comprehensive model evaluation using utility framework.
        
        Args:
            models: Dictionary of trained models
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            config: Evaluation configuration
            
        Returns:
            ModelComparisonResult with comprehensive evaluation
        """
        try:
            with self.performance_tracker.track_block("comprehensive_model_evaluation"):
                # Audit log the evaluation operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="model_evaluation",
                    parameters={
                        "n_models": len(models),
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "n_features": len(X_train.columns)
                    }
                )
                
                if config is None:
                    config = self.default_config
                
                # Clean and validate data using DataProcessor
                X_train_clean = self.data_processor.clean_seo_data(X_train)
                X_test_clean = self.data_processor.clean_seo_data(X_test)
                
                # Validate data quality using DataValidator
                train_validation = self.data_validator.validate_seo_dataset(X_train_clean, 'features')
                test_validation = self.data_validator.validate_seo_dataset(X_test_clean, 'features')
                
                if train_validation.quality_score < 0.7 or test_validation.quality_score < 0.7:
                    self.logger.warning("Low data quality detected in training or test sets")
                
                # Evaluate each model
                model_performances = {}
                for model_name, model in models.items():
                    self.logger.info(f"Evaluating model: {model_name}")
                    
                    performance = self._evaluate_single_model(
                        model, X_train_clean, X_test_clean, y_train, y_test,
                        model_name, config
                    )
                    model_performances[model_name] = performance
                
                # Determine best model using statistical comparison
                best_model, performance_ranking = self._determine_best_model(model_performances)
                
                # Statistical comparison using StatisticalCalculator
                statistical_comparison = self._perform_statistical_model_comparison(
                    model_performances, config
                )
                
                # Generate recommendations using optimization principles
                recommendations = self._generate_model_recommendations(
                    model_performances, statistical_comparison
                )
                
                # Create evaluation summary
                evaluation_summary = self._create_evaluation_summary(
                    model_performances, statistical_comparison
                )
                
                result = ModelComparisonResult(
                    model_performances=model_performances,
                    best_model=best_model,
                    performance_ranking=performance_ranking,
                    statistical_comparison=statistical_comparison,
                    recommendations=recommendations,
                    evaluation_summary=evaluation_summary
                )
                
                self.logger.info(f"Model evaluation completed. Best model: {best_model}")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive model evaluation: {str(e)}")
            return ModelComparisonResult({}, "", [], {}, [], {})

    @timing_decorator()
    def validate_model_robustness(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        validation_methods: List[str] = None,
        config: Optional[ModelEvaluationConfig] = None
    ) -> ModelValidationResult:
        """
        Validate model robustness using multiple validation techniques.
        
        Args:
            model: Trained model to validate
            X: Feature matrix
            y: Target variable
            validation_methods: List of validation methods to apply
            config: Validation configuration
            
        Returns:
            ModelValidationResult with validation analysis
        """
        try:
            with self.performance_tracker.track_block("validate_model_robustness"):
                if config is None:
                    config = self.default_config
                
                if validation_methods is None:
                    validation_methods = ['cross_validation', 'residual_analysis', 'learning_curves']
                
                # Clean data using DataProcessor
                X_clean = self.data_processor.clean_seo_data(X)
                y_clean = y.fillna(y.median())
                
                validation_results = {}
                
                # Cross-validation metrics
                if 'cross_validation' in validation_methods:
                    cv_metrics = self._cross_validation_analysis(
                        model, X_clean, y_clean, config
                    )
                    validation_results['cross_validation'] = cv_metrics
                
                # Residual analysis using statistical methods
                residual_analysis = {}
                if 'residual_analysis' in validation_methods and config.include_residual_analysis:
                    residual_analysis = self._perform_residual_analysis(
                        model, X_clean, y_clean
                    )
                
                # Learning curves using performance tracking
                learning_curves = {}
                if 'learning_curves' in validation_methods and config.include_learning_curves:
                    learning_curves = self._generate_learning_curves(
                        model, X_clean, y_clean, config
                    )
                
                # Validation curves for hyperparameter analysis
                validation_curves = {}
                if 'validation_curves' in validation_methods and config.include_validation_curves:
                    validation_curves = self._generate_validation_curves(
                        model, X_clean, y_clean, config
                    )
                
                # Bias-variance analysis using statistical decomposition
                bias_variance_analysis = self._analyze_bias_variance_tradeoff(
                    model, X_clean, y_clean, config
                )
                
                # Overfitting assessment using validation metrics
                overfitting_assessment = self._assess_overfitting(
                    model, X_clean, y_clean, config
                )
                
                result = ModelValidationResult(
                    validation_metrics=validation_results,
                    residual_analysis=residual_analysis,
                    learning_curves=learning_curves,
                    validation_curves=validation_curves,
                    bias_variance_analysis=bias_variance_analysis,
                    overfitting_assessment=overfitting_assessment
                )
                
                self.logger.info("Model robustness validation completed")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in model robustness validation: {str(e)}")
            return ModelValidationResult({}, {}, {}, {}, {}, {})

    @timing_decorator()
    def evaluate_model_interpretability(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: List[str] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Evaluate model interpretability using various techniques.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Feature names
            sample_size: Sample size for analysis
            
        Returns:
            Dictionary of interpretability analysis results
        """
        try:
            with self.performance_tracker.track_block("evaluate_model_interpretability"):
                # Clean data using DataProcessor
                X_clean = self.data_processor.clean_seo_data(X)
                
                if feature_names is None:
                    feature_names = X_clean.columns.tolist()
                
                interpretability_results = {}
                
                # Global feature importance
                global_importance = self._calculate_global_feature_importance(
                    model, X_clean, feature_names
                )
                interpretability_results['global_importance'] = global_importance
                
                # Permutation importance using statistical analysis
                permutation_importance = self._calculate_permutation_importance(
                    model, X_clean, feature_names, sample_size
                )
                interpretability_results['permutation_importance'] = permutation_importance
                
                # Feature interaction analysis
                interaction_analysis = self._analyze_feature_interactions(
                    model, X_clean, feature_names
                )
                interpretability_results['feature_interactions'] = interaction_analysis
                
                # Model complexity metrics
                complexity_metrics = self._calculate_model_complexity(
                    model, X_clean
                )
                interpretability_results['complexity_metrics'] = complexity_metrics
                
                # Decision boundary analysis (for classification models)
                if hasattr(model, 'predict_proba'):
                    boundary_analysis = self._analyze_decision_boundaries(
                        model, X_clean, sample_size
                    )
                    interpretability_results['decision_boundaries'] = boundary_analysis
                
                self.logger.info("Model interpretability evaluation completed")
                return interpretability_results
                
        except Exception as e:
            self.logger.error(f"Error in model interpretability evaluation: {str(e)}")
            return {}

    def _evaluate_single_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        config: ModelEvaluationConfig
    ) -> ModelPerformance:
        """Evaluate single model performance using comprehensive metrics."""
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, confusion_matrix, classification_report
            )
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            import time
            
            # Training time (if model needs fitting)
            start_time = time.time()
            if hasattr(model, 'fit') and not hasattr(model, 'predict'):
                model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prediction time
            pred_start = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - pred_start
            
            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC (handle both binary and multiclass)
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    if y_pred_proba.shape[1] == 2:  # Binary classification
                        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:  # Multiclass
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    roc_auc = 0.5  # Default for models without probability
            except Exception:
                roc_auc = 0.5
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Cross-validation using proper CV strategy
            cv_strategy = StratifiedKFold(n_splits=config.cross_validation_folds, shuffle=True, random_state=config.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
            
            # Feature importance
            feature_importance = {}
            if config.include_feature_importance:
                feature_importance = self._extract_feature_importance(model, X_train.columns)
            
            performance = ModelPerformance(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                confusion_matrix=conf_matrix,
                classification_report=class_report,
                cross_val_scores=cv_scores.tolist(),
                cross_val_mean=np.mean(cv_scores),
                cross_val_std=np.std(cv_scores),
                training_time=training_time,
                prediction_time=prediction_time,
                feature_importance=feature_importance
            )
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {str(e)}")
            return ModelPerformance(
                model_name, 0.0, 0.0, 0.0, 0.0, 0.5, np.array([]), {}, [], 0.0, 0.0, 0.0, 0.0, {}
            )

    def _determine_best_model(
        self,
        model_performances: Dict[str, ModelPerformance]
    ) -> Tuple[str, List[str]]:
        """Determine best model using multiple criteria."""
        try:
            if not model_performances:
                return "", []
            
            # Score each model using weighted combination of metrics
            model_scores = {}
            
            for model_name, performance in model_performances.items():
                # Weighted scoring (can be configured)
                score = (
                    performance.accuracy * 0.3 +
                    performance.f1_score * 0.3 +
                    performance.roc_auc * 0.2 +
                    performance.cross_val_mean * 0.2
                )
                
                # Penalty for high variance in cross-validation
                cv_penalty = min(performance.cross_val_std * 2, 0.1)
                final_score = score - cv_penalty
                
                model_scores[model_name] = final_score
            
            # Rank models by score
            ranking = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            best_model = ranking[0][0]
            performance_ranking = [model_name for model_name, _ in ranking]
            
            return best_model, performance_ranking
            
        except Exception as e:
            self.logger.error(f"Error determining best model: {str(e)}")
            return list(model_performances.keys())[0] if model_performances else "", []

    def _perform_statistical_model_comparison(
        self,
        model_performances: Dict[str, ModelPerformance],
        config: ModelEvaluationConfig
    ) -> Dict[str, Any]:
        """Perform statistical comparison of models using statistical tests."""
        try:
            comparison_results = {}
            
            # Extract cross-validation scores for comparison
            cv_scores_by_model = {
                name: perf.cross_val_scores 
                for name, perf in model_performances.items()
            }
            
            if len(cv_scores_by_model) < 2:
                return comparison_results
            
            # Pairwise statistical tests using StatisticalCalculator
            model_names = list(cv_scores_by_model.keys())
            pairwise_tests = {}
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    scores1 = cv_scores_by_model[model1]
                    scores2 = cv_scores_by_model[model2]
                    
                    # Paired t-test for cross-validation scores
                    test_result = self.stats_calculator.perform_hypothesis_test(
                        np.array(scores1), np.array(scores2),
                        test_type='ttest', alternative='two-sided'
                    )
                    
                    pair_key = f"{model1}_vs_{model2}"
                    pairwise_tests[pair_key] = test_result
            
            comparison_results['pairwise_tests'] = pairwise_tests
            
            # Overall model statistics using StatisticalCalculator
            model_stats = {}
            for model_name, scores in cv_scores_by_model.items():
                stats_dict = self.stats_calculator.calculate_descriptive_statistics(scores)
                model_stats[model_name] = stats_dict
            
            comparison_results['model_statistics'] = model_stats
            
            # Effect sizes and practical significance
            effect_sizes = self._calculate_effect_sizes(cv_scores_by_model)
            comparison_results['effect_sizes'] = effect_sizes
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error in statistical model comparison: {str(e)}")
            return {}

    def _perform_residual_analysis(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Perform residual analysis using statistical methods."""
        try:
            # Get predictions
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            # Statistical analysis of residuals using StatisticalCalculator
            residual_stats = self.stats_calculator.calculate_descriptive_statistics(
                residuals, include_advanced=True
            )
            
            # Normality tests for residuals
            normality_tests = self.stats_calculator.perform_normality_tests(residuals)
            
            # Autocorrelation analysis using TimeSeriesAnalyzer
            if len(residuals) > 10:
                autocorr, lags = self.time_series_analyzer.calculate_autocorrelation(
                    pd.Series(residuals)
                )
                autocorr_analysis = {
                    'autocorrelations': autocorr.tolist(),
                    'lags': lags.tolist(),
                    'significant_autocorr': np.any(np.abs(autocorr[1:]) > 0.2)
                }
            else:
                autocorr_analysis = {}
            
            # Heteroscedasticity analysis
            heteroscedasticity = self._test_heteroscedasticity(y_pred, residuals)
            
            residual_analysis = {
                'residual_statistics': residual_stats,
                'normality_tests': normality_tests,
                'autocorrelation_analysis': autocorr_analysis,
                'heteroscedasticity': heteroscedasticity,
                'outliers': self._identify_residual_outliers(residuals)
            }
            
            return residual_analysis
            
        except Exception as e:
            self.logger.error(f"Error in residual analysis: {str(e)}")
            return {}

    def _generate_learning_curves(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        config: ModelEvaluationConfig
    ) -> Dict[str, Any]:
        """Generate learning curves using performance tracking."""
        try:
            from sklearn.model_selection import learning_curve
            
            # Define training sizes
            training_sizes = np.linspace(0.1, 1.0, 10)
            
            # Generate learning curves
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=training_sizes,
                cv=config.cross_validation_folds,
                scoring='accuracy',
                random_state=config.random_state,
                n_jobs=-1
            )
            
            # Calculate statistics using StatisticalCalculator
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Learning curve analysis
            learning_efficiency = self._analyze_learning_efficiency(
                train_sizes, train_mean, val_mean
            )
            
            learning_curves = {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_mean.tolist(),
                'train_scores_std': train_std.tolist(),
                'validation_scores_mean': val_mean.tolist(),
                'validation_scores_std': val_std.tolist(),
                'learning_efficiency': learning_efficiency,
                'convergence_analysis': self._analyze_convergence(val_mean)
            }
            
            return learning_curves
            
        except Exception as e:
            self.logger.error(f"Error generating learning curves: {str(e)}")
            return {}

    def export_model_evaluation_results(
        self,
        comparison_result: ModelComparisonResult,
        validation_result: ModelValidationResult,
        export_directory: str,
        include_detailed_analysis: bool = True
    ) -> Dict[str, bool]:
        """Export comprehensive model evaluation results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'evaluation_summary': {
                    'best_model': comparison_result.best_model,
                    'total_models_evaluated': len(comparison_result.model_performances),
                    'performance_ranking': comparison_result.performance_ranking,
                    'evaluation_timestamp': datetime.now().isoformat()
                },
                'model_performances': {
                    name: {
                        'accuracy': perf.accuracy,
                        'precision': perf.precision,
                        'recall': perf.recall,
                        'f1_score': perf.f1_score,
                        'roc_auc': perf.roc_auc,
                        'cross_val_mean': perf.cross_val_mean,
                        'cross_val_std': perf.cross_val_std,
                        'training_time': perf.training_time,
                        'prediction_time': perf.prediction_time
                    }
                    for name, perf in comparison_result.model_performances.items()
                },
                'statistical_comparison': comparison_result.statistical_comparison,
                'validation_analysis': validation_result.validation_metrics,
                'recommendations': comparison_result.recommendations
            }
            
            # Export summary data using DataExporter
            summary_export_success = self.data_exporter.export_analysis_dataset(
                {'model_evaluation_summary': pd.DataFrame([export_data])},
                export_path / "model_evaluation_summary.xlsx"
            )
            
            # Export detailed performance metrics
            performance_data = []
            for model_name, performance in comparison_result.model_performances.items():
                performance_data.append({
                    'model_name': model_name,
                    'accuracy': performance.accuracy,
                    'precision': performance.precision,
                    'recall': performance.recall,
                    'f1_score': performance.f1_score,
                    'roc_auc': performance.roc_auc,
                    'cross_val_mean': performance.cross_val_mean,
                    'cross_val_std': performance.cross_val_std,
                    'training_time': performance.training_time,
                    'prediction_time': performance.prediction_time
                })
            
            if performance_data:
                performance_df = pd.DataFrame(performance_data)
                performance_export_success = self.data_exporter.export_with_metadata(
                    performance_df,
                    metadata={'analysis_type': 'model_performance', 'generation_timestamp': datetime.now()},
                    export_path=export_path / "detailed_model_performance.xlsx"
                )
            else:
                performance_export_success = True
            
            # Export feature importance analysis
            feature_importance_data = []
            for model_name, performance in comparison_result.model_performances.items():
                for feature, importance in performance.feature_importance.items():
                    feature_importance_data.append({
                        'model_name': model_name,
                        'feature': feature,
                        'importance': importance
                    })
            
            if feature_importance_data:
                feature_df = pd.DataFrame(feature_importance_data)
                feature_export_success = self.data_exporter.export_with_metadata(
                    feature_df,
                    metadata={'analysis_type': 'feature_importance'},
                    export_path=export_path / "feature_importance_analysis.xlsx"
                )
            else:
                feature_export_success = True
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_executive_report(
                export_data,
                export_path / "model_evaluation_executive_report.html",
                format='html',
                include_charts=True
            )
            
            return {
                'summary_export': summary_export_success,
                'performance_details': performance_export_success,
                'feature_importance': feature_export_success,
                'executive_report': report_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting model evaluation results: {str(e)}")
            return {}

    # Helper methods using utility framework
    def _extract_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model."""
        try:
            importance = {}
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                importance = dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
                importance = dict(zip(feature_names, coefficients))
            
            return importance
            
        except Exception:
            return {}

    def _calculate_effect_sizes(self, cv_scores_by_model: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate effect sizes between models."""
        try:
            effect_sizes = {}
            model_names = list(cv_scores_by_model.keys())
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    scores1 = np.array(cv_scores_by_model[model1])
                    scores2 = np.array(cv_scores_by_model[model2])
                    
                    # Cohen's d effect size
                    pooled_std = np.sqrt(((scores1.var() + scores2.var()) / 2))
                    effect_size = (scores1.mean() - scores2.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    effect_sizes[f"{model1}_vs_{model2}"] = effect_size
            
            return effect_sizes
            
        except Exception:
            return {}

    def _test_heteroscedasticity(self, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for heteroscedasticity in residuals."""
        try:
            # Simple test: correlation between absolute residuals and predictions
            abs_residuals = np.abs(residuals)
            correlation = np.corrcoef(y_pred, abs_residuals)[0, 1]
            
            # Breusch-Pagan test approximation
            bp_statistic = abs(correlation) * len(residuals) ** 0.5
            
            return {
                'correlation_with_predictions': correlation,
                'bp_statistic': bp_statistic,
                'heteroscedasticity_detected': abs(correlation) > 0.3
            }
            
        except Exception:
            return {'heteroscedasticity_detected': False}

    def _identify_residual_outliers(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Identify outliers in residuals using statistical methods."""
        try:
            # Use statistical calculator for outlier detection
            stats_dict = self.stats_calculator.calculate_descriptive_statistics(residuals)
            
            # IQR-based outliers
            Q1 = stats_dict.get('q25', 0)
            Q3 = stats_dict.get('q75', 0)
            IQR = stats_dict.get('iqr', 0)
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
                outlier_count = np.sum(outlier_mask)
            else:
                outlier_count = 0
            
            return {
                'outlier_count': outlier_count,
                'outlier_percentage': (outlier_count / len(residuals)) * 100,
                'outlier_threshold_lower': Q1 - 1.5 * IQR if IQR > 0 else 0,
                'outlier_threshold_upper': Q3 + 1.5 * IQR if IQR > 0 else 0
            }
            
        except Exception:
            return {'outlier_count': 0, 'outlier_percentage': 0}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for model evaluation operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods would be implemented here...
    def _generate_model_recommendations(
        self,
        model_performances: Dict[str, ModelPerformance],
        statistical_comparison: Dict[str, Any]
    ) -> List[str]:
        """Generate model recommendations based on evaluation results."""
        try:
            recommendations = []
            
            if not model_performances:
                return ["No models to evaluate"]
            
            # Performance-based recommendations
            best_accuracy = max(perf.accuracy for perf in model_performances.values())
            best_f1 = max(perf.f1_score for perf in model_performances.values())
            
            if best_accuracy > 0.9:
                recommendations.append("Excellent model performance achieved - consider deploying to production")
            elif best_accuracy > 0.8:
                recommendations.append("Good model performance - consider hyperparameter tuning for improvement")
            else:
                recommendations.append("Model performance needs improvement - review feature engineering and model selection")
            
            # Cross-validation stability recommendations
            cv_stds = [perf.cross_val_std for perf in model_performances.values()]
            max_cv_std = max(cv_stds) if cv_stds else 0
            
            if max_cv_std > 0.1:
                recommendations.append("High cross-validation variance detected - consider more stable models or better data preprocessing")
            
            # Training time recommendations
            training_times = [perf.training_time for perf in model_performances.values()]
            max_training_time = max(training_times) if training_times else 0
            
            if max_training_time > 300:  # 5 minutes
                recommendations.append("Long training times detected - consider model complexity reduction or distributed training")
            
            # Feature importance recommendations
            feature_counts = [len(perf.feature_importance) for perf in model_performances.values()]
            max_features = max(feature_counts) if feature_counts else 0
            
            if max_features > 100:
                recommendations.append("High feature dimensionality - consider feature selection or dimensionality reduction")
            
            return recommendations
            
        except Exception:
            return ["Review model evaluation results and optimize based on performance metrics"]

    def _analyze_learning_efficiency(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze learning efficiency from learning curves."""
        try:
            # Calculate learning rate (improvement per additional training sample)
            val_improvements = np.diff(val_scores)
            size_increases = np.diff(train_sizes)
            
            learning_rates = val_improvements / size_increases
            avg_learning_rate = np.mean(learning_rates) if len(learning_rates) > 0 else 0
            
            # Convergence analysis
            final_gap = abs(train_scores[-1] - val_scores[-1]) if len(val_scores) > 0 else 0
            
            return {
                'average_learning_rate': avg_learning_rate,
                'final_train_val_gap': final_gap,
                'learning_efficiency_score': max(0, avg_learning_rate - final_gap),
                'data_efficiency': 'high' if avg_learning_rate > 0.01 else 'low'
            }
            
        except Exception:
            return {}

    def _analyze_convergence(self, validation_scores: np.ndarray) -> Dict[str, Any]:
        """Analyze convergence patterns in validation scores."""
        try:
            if len(validation_scores) < 3:
                return {}
            
            # Calculate convergence metrics
            score_diffs = np.diff(validation_scores)
            recent_changes = score_diffs[-3:] if len(score_diffs) >= 3 else score_diffs
            
            convergence_rate = np.mean(np.abs(recent_changes))
            is_converged = convergence_rate < 0.001
            
            # Trend analysis
            if len(validation_scores) >= 5:
                recent_trend = np.polyfit(range(len(validation_scores[-5:])), validation_scores[-5:], 1)[0]
                trend_direction = 'improving' if recent_trend > 0 else 'declining' if recent_trend < -0.001 else 'stable'
            else:
                trend_direction = 'insufficient_data'
            
            return {
                'convergence_rate': convergence_rate,
                'is_converged': is_converged,
                'trend_direction': trend_direction,
                'final_score': validation_scores[-1] if len(validation_scores) > 0 else 0
            }
            
        except Exception:
            return {}
