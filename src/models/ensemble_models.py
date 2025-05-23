"""
Ensemble Models Module for SEO Competitive Intelligence
Advanced ensemble modeling leveraging the comprehensive utility framework
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
class EnsembleConfig:
    """Configuration for ensemble modeling"""
    base_models: List[str] = None
    ensemble_method: str = 'voting'
    cross_validation_folds: int = 5
    hyperparameter_optimization: bool = True
    feature_importance_analysis: bool = True
    model_interpretability: bool = True
    performance_threshold: float = 0.8

@dataclass
class ModelPerformance:
    """Individual model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_time: float
    prediction_time: float
    feature_importance: Dict[str, float]
    cross_val_scores: List[float]

@dataclass
class EnsembleResult:
    """Result of ensemble modeling"""
    ensemble_model: Any
    individual_performances: Dict[str, ModelPerformance]
    ensemble_performance: ModelPerformance
    feature_analysis: Dict[str, Any]
    model_interpretability: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    recommendations: List[str]

class EnsembleModelManager:
    """
    Advanced ensemble modeling for SEO competitive intelligence.
    
    Leverages the comprehensive utility framework to provide sophisticated
    ensemble modeling capabilities without redundancy.
    """
    
    def __init__(self, logger=None, config_manager=None):
        """Initialize with utilities - no more redundant implementations."""
        self.logger = logger or LoggerFactory.get_logger("ensemble_models")
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
        
        # Load ensemble modeling configurations
        analysis_config = self.config.get_analysis_config()
        self.default_config = EnsembleConfig(
            base_models=['random_forest', 'gradient_boosting', 'logistic_regression', 'svm'],
            ensemble_method='voting',
            cross_validation_folds=5
        )

    @timing_decorator()
    @memoize(ttl=7200)  # Cache for 2 hours
    def build_ensemble_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Optional[EnsembleConfig] = None,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> EnsembleResult:
        """
        Build comprehensive ensemble model using utility framework.
        
        Args:
            X: Feature matrix
            y: Target variable
            config: Ensemble configuration
            validation_data: Optional validation data
            
        Returns:
            EnsembleResult with comprehensive modeling results
        """
        try:
            with self.performance_tracker.track_block("build_ensemble_model"):
                # Audit log the ensemble modeling operation
                self.audit_logger.log_analysis_execution(
                    user_id="system",
                    analysis_type="ensemble_modeling",
                    parameters={
                        "n_features": len(X.columns),
                        "n_samples": len(X),
                        "base_models": config.base_models if config else "default",
                        "ensemble_method": config.ensemble_method if config else "voting"
                    }
                )
                
                if config is None:
                    config = self.default_config
                
                # Clean and validate data using DataProcessor
                X_clean = self.data_processor.clean_seo_data(X)
                
                # Validate data quality using DataValidator
                validation_report = self.data_validator.validate_seo_dataset(X_clean, 'features')
                if validation_report.quality_score < 0.7:
                    self.logger.warning(f"Low feature data quality: {validation_report.quality_score:.3f}")
                
                # Prepare data using DataTransformer
                X_processed = self.data_transformer.apply_scaling(
                    X_clean, scaling_method='standard', fit_scaler=True
                )
                
                # Handle missing values in target
                y_clean = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
                
                # Build individual models
                individual_performances = self._build_individual_models(
                    X_processed, y_clean, config
                )
                
                # Create ensemble model
                ensemble_model = self._create_ensemble_model(
                    X_processed, y_clean, individual_performances, config
                )
                
                # Evaluate ensemble performance
                ensemble_performance = self._evaluate_ensemble_performance(
                    ensemble_model, X_processed, y_clean, validation_data, config
                )
                
                # Feature analysis using statistical methods
                feature_analysis = self._analyze_ensemble_features(
                    ensemble_model, individual_performances, X_processed.columns
                )
                
                # Model interpretability analysis
                model_interpretability = {}
                if config.model_interpretability:
                    model_interpretability = self._analyze_model_interpretability(
                        ensemble_model, X_processed, y_clean
                    )
                
                # Generate recommendations using optimization principles
                recommendations = self._generate_ensemble_recommendations(
                    individual_performances, ensemble_performance, feature_analysis
                )
                
                result = EnsembleResult(
                    ensemble_model=ensemble_model,
                    individual_performances=individual_performances,
                    ensemble_performance=ensemble_performance,
                    feature_analysis=feature_analysis,
                    model_interpretability=model_interpretability,
                    optimization_history=[],  # Would be populated during hyperparameter optimization
                    recommendations=recommendations
                )
                
                self.logger.info(f"Ensemble model built successfully. Performance: {ensemble_performance.accuracy:.3f}")
                return result
                
        except Exception as e:
            self.logger.error(f"Error building ensemble model: {str(e)}")
            return EnsembleResult(None, {}, None, {}, {}, [], [f"Modeling failed: {str(e)}"])

    @timing_decorator()
    def optimize_ensemble_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Optional[EnsembleConfig] = None,
        optimization_method: str = 'bayesian'
    ) -> Dict[str, Any]:
        """
        Optimize ensemble hyperparameters using advanced optimization.
        
        Args:
            X: Feature matrix
            y: Target variable
            config: Ensemble configuration
            optimization_method: Optimization method to use
            
        Returns:
            Optimization results with best parameters
        """
        try:
            with self.performance_tracker.track_block("optimize_ensemble_hyperparameters"):
                if config is None:
                    config = self.default_config
                
                # Clean and prepare data
                X_clean = self.data_processor.clean_seo_data(X)
                X_processed = self.data_transformer.apply_scaling(X_clean, scaling_method='standard')
                
                # Define parameter search space
                param_space = self._define_parameter_search_space(config.base_models)
                
                # Perform optimization based on method
                if optimization_method == 'bayesian':
                    optimization_results = self._bayesian_optimization(
                        X_processed, y, param_space, config
                    )
                elif optimization_method == 'grid_search':
                    optimization_results = self._grid_search_optimization(
                        X_processed, y, param_space, config
                    )
                elif optimization_method == 'random_search':
                    optimization_results = self._random_search_optimization(
                        X_processed, y, param_space, config
                    )
                else:
                    raise ValueError(f"Unsupported optimization method: {optimization_method}")
                
                self.logger.info(f"Hyperparameter optimization completed using {optimization_method}")
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return {}

    @timing_decorator()
    def analyze_model_stability(
        self,
        ensemble_model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_iterations: int = 100,
        sample_fraction: float = 0.8
    ) -> Dict[str, Any]:
        """
        Analyze model stability using bootstrap sampling and statistical analysis.
        
        Args:
            ensemble_model: Trained ensemble model
            X: Feature matrix
            y: Target variable
            n_iterations: Number of bootstrap iterations
            sample_fraction: Fraction of data to sample
            
        Returns:
            Model stability analysis results
        """
        try:
            with self.performance_tracker.track_block("analyze_model_stability"):
                stability_results = {
                    'performance_variations': [],
                    'feature_importance_variations': [],
                    'prediction_consistency': [],
                    'stability_metrics': {}
                }
                
                base_predictions = ensemble_model.predict(X)
                
                for iteration in range(n_iterations):
                    # Bootstrap sampling
                    sample_size = int(len(X) * sample_fraction)
                    sample_indices = np.random.choice(len(X), sample_size, replace=True)
                    
                    X_sample = X.iloc[sample_indices]
                    y_sample = y.iloc[sample_indices]
                    
                    # Re-train model on sample
                    sample_model = self._retrain_ensemble_on_sample(
                        ensemble_model, X_sample, y_sample
                    )
                    
                    # Evaluate performance
                    sample_performance = self._evaluate_model_performance(
                        sample_model, X, y
                    )
                    stability_results['performance_variations'].append(sample_performance)
                    
                    # Feature importance analysis
                    if hasattr(sample_model, 'feature_importances_'):
                        importance_dict = dict(zip(X.columns, sample_model.feature_importances_))
                        stability_results['feature_importance_variations'].append(importance_dict)
                    
                    # Prediction consistency
                    sample_predictions = sample_model.predict(X)
                    consistency = np.mean(base_predictions == sample_predictions)
                    stability_results['prediction_consistency'].append(consistency)
                
                # Calculate stability metrics using StatisticalCalculator
                if stability_results['performance_variations']:
                    performance_scores = [perf['accuracy'] for perf in stability_results['performance_variations']]
                    perf_stats = self.stats_calculator.calculate_descriptive_statistics(performance_scores)
                    
                    stability_results['stability_metrics'] = {
                        'performance_mean': perf_stats.get('mean', 0),
                        'performance_std': perf_stats.get('std', 0),
                        'performance_cv': perf_stats.get('coefficient_of_variation', 0),
                        'prediction_consistency_mean': np.mean(stability_results['prediction_consistency']),
                        'prediction_consistency_std': np.std(stability_results['prediction_consistency'])
                    }
                
                self.logger.info(f"Model stability analysis completed over {n_iterations} iterations")
                return stability_results
                
        except Exception as e:
            self.logger.error(f"Error in model stability analysis: {str(e)}")
            return {}

    def _build_individual_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: EnsembleConfig
    ) -> Dict[str, ModelPerformance]:
        """Build individual models using cross-validation."""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            import time
            
            individual_performances = {}
            
            # Define base models
            model_definitions = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(random_state=42, probability=True)
            }
            
            # Cross-validation setup
            cv = StratifiedKFold(n_splits=config.cross_validation_folds, shuffle=True, random_state=42)
            
            for model_name in config.base_models:
                if model_name not in model_definitions:
                    continue
                
                self.logger.info(f"Training {model_name}")
                model = model_definitions[model_name]
                
                # Measure training time
                start_time = time.time()
                
                # Cross-validation evaluation
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                
                # Fit full model for additional metrics
                model.fit(X, y)
                training_time = time.time() - start_time
                
                # Prediction time
                pred_start = time.time()
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                prediction_time = time.time() - pred_start
                
                # Calculate performance metrics
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
                
                try:
                    roc_auc = roc_auc_score(y, y_pred_proba)
                except Exception:
                    roc_auc = 0.5  # Default for multiclass or problematic cases
                
                # Feature importance
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(X.columns, np.abs(model.coef_[0])))
                
                performance = ModelPerformance(
                    model_name=model_name,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    roc_auc=roc_auc,
                    training_time=training_time,
                    prediction_time=prediction_time,
                    feature_importance=feature_importance,
                    cross_val_scores=cv_scores.tolist()
                )
                
                individual_performances[model_name] = performance
            
            return individual_performances
            
        except Exception as e:
            self.logger.error(f"Error building individual models: {str(e)}")
            return {}

    def _create_ensemble_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        individual_performances: Dict[str, ModelPerformance],
        config: EnsembleConfig
    ) -> Any:
        """Create ensemble model based on configuration."""
        try:
            from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            # Define base models
            model_definitions = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(random_state=42, probability=True)
            }
            
            # Create estimators list for ensemble
            estimators = []
            for model_name in config.base_models:
                if model_name in model_definitions and model_name in individual_performances:
                    estimators.append((model_name, model_definitions[model_name]))
            
            if not estimators:
                raise ValueError("No valid base models found for ensemble")
            
            # Create ensemble based on method
            if config.ensemble_method == 'voting':
                ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
            elif config.ensemble_method == 'weighted_voting':
                # Weight by individual model performance
                weights = [individual_performances[name].accuracy for name, _ in estimators]
                ensemble_model = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
            else:
                # Default to simple voting
                ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
            
            # Train ensemble model
            ensemble_model.fit(X, y)
            
            return ensemble_model
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble model: {str(e)}")
            return None

    def _evaluate_ensemble_performance(
        self,
        ensemble_model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]],
        config: EnsembleConfig
    ) -> ModelPerformance:
        """Evaluate ensemble model performance."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            import time
            
            # Use validation data if provided, otherwise use training data
            if validation_data:
                X_eval, y_eval = validation_data
            else:
                X_eval, y_eval = X, y
            
            # Measure prediction time
            pred_start = time.time()
            y_pred = ensemble_model.predict(X_eval)
            y_pred_proba = ensemble_model.predict_proba(X_eval)[:, 1] if hasattr(ensemble_model, 'predict_proba') else y_pred
            prediction_time = time.time() - pred_start
            
            # Calculate performance metrics
            accuracy = accuracy_score(y_eval, y_pred)
            precision = precision_score(y_eval, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_eval, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_eval, y_pred_proba)
            except Exception:
                roc_auc = 0.5
            
            # Cross-validation scores
            cv = StratifiedKFold(n_splits=config.cross_validation_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(ensemble_model, X, y, cv=cv, scoring='accuracy')
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(ensemble_model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, ensemble_model.feature_importances_))
            
            performance = ModelPerformance(
                model_name='ensemble',
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                training_time=0.0,  # Training time measured elsewhere
                prediction_time=prediction_time,
                feature_importance=feature_importance,
                cross_val_scores=cv_scores.tolist()
            )
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating ensemble performance: {str(e)}")
            return ModelPerformance('ensemble', 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, {}, [])

    def _analyze_ensemble_features(
        self,
        ensemble_model: Any,
        individual_performances: Dict[str, ModelPerformance],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze feature importance across ensemble."""
        try:
            analysis = {
                'feature_consistency': {},
                'average_importance': {},
                'importance_std': {},
                'top_features': [],
                'feature_rankings': {}
            }
            
            # Collect feature importances from all models
            all_importances = {}
            for model_name, performance in individual_performances.items():
                if performance.feature_importance:
                    for feature, importance in performance.feature_importance.items():
                        if feature not in all_importances:
                            all_importances[feature] = []
                        all_importances[feature].append(importance)
            
            # Calculate statistics using StatisticalCalculator
            for feature, importances in all_importances.items():
                if len(importances) > 1:
                    stats_dict = self.stats_calculator.calculate_descriptive_statistics(importances)
                    
                    analysis['average_importance'][feature] = stats_dict.get('mean', 0)
                    analysis['importance_std'][feature] = stats_dict.get('std', 0)
                    analysis['feature_consistency'][feature] = 1 / (1 + stats_dict.get('coefficient_of_variation', 1))
                else:
                    analysis['average_importance'][feature] = importances[0] if importances else 0
                    analysis['importance_std'][feature] = 0
                    analysis['feature_consistency'][feature] = 1.0
            
            # Rank features by average importance
            sorted_features = sorted(
                analysis['average_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            analysis['top_features'] = [feature for feature, _ in sorted_features[:20]]
            analysis['feature_rankings'] = {
                feature: rank for rank, (feature, _) in enumerate(sorted_features, 1)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing ensemble features: {str(e)}")
            return {}

    def export_ensemble_results(
        self,
        ensemble_result: EnsembleResult,
        export_directory: str,
        include_model_artifacts: bool = True
    ) -> Dict[str, bool]:
        """Export comprehensive ensemble modeling results."""
        try:
            export_path = self.path_manager.get_exports_path(export_directory)
            
            # Prepare comprehensive export data
            export_data = {
                'ensemble_summary': {
                    'ensemble_performance': {
                        'accuracy': ensemble_result.ensemble_performance.accuracy,
                        'precision': ensemble_result.ensemble_performance.precision,
                        'recall': ensemble_result.ensemble_performance.recall,
                        'f1_score': ensemble_result.ensemble_performance.f1_score,
                        'roc_auc': ensemble_result.ensemble_performance.roc_auc
                    },
                    'individual_models_count': len(ensemble_result.individual_performances),
                    'feature_analysis': ensemble_result.feature_analysis,
                    'recommendations': ensemble_result.recommendations
                },
                'individual_performances': {
                    name: {
                        'accuracy': perf.accuracy,
                        'precision': perf.precision,
                        'recall': perf.recall,
                        'f1_score': perf.f1_score,
                        'roc_auc': perf.roc_auc,
                        'training_time': perf.training_time,
                        'cross_val_mean': np.mean(perf.cross_val_scores),
                        'cross_val_std': np.std(perf.cross_val_scores)
                    }
                    for name, perf in ensemble_result.individual_performances.items()
                },
                'model_interpretability': ensemble_result.model_interpretability
            }
            
            # Export detailed data using DataExporter
            data_export_success = self.data_exporter.export_analysis_dataset(
                {'ensemble_modeling_results': pd.DataFrame([export_data])},
                export_path / "ensemble_modeling_detailed.xlsx"
            )
            
            # Export individual model performances
            performances_df = pd.DataFrame([
                {
                    'model_name': name,
                    'accuracy': perf.accuracy,
                    'precision': perf.precision,
                    'recall': perf.recall,
                    'f1_score': perf.f1_score,
                    'roc_auc': perf.roc_auc,
                    'training_time': perf.training_time,
                    'prediction_time': perf.prediction_time
                }
                for name, perf in ensemble_result.individual_performances.items()
            ])
            
            performances_export_success = self.data_exporter.export_with_metadata(
                performances_df,
                metadata={'analysis_type': 'model_performances', 'generation_timestamp': datetime.now()},
                export_path=export_path / "model_performances.xlsx"
            )
            
            # Export feature importance analysis
            feature_importance_data = []
            for model_name, performance in ensemble_result.individual_performances.items():
                for feature, importance in performance.feature_importance.items():
                    feature_importance_data.append({
                        'model_name': model_name,
                        'feature': feature,
                        'importance': importance
                    })
            
            if feature_importance_data:
                feature_importance_df = pd.DataFrame(feature_importance_data)
                feature_export_success = self.data_exporter.export_with_metadata(
                    feature_importance_df,
                    metadata={'analysis_type': 'feature_importance'},
                    export_path=export_path / "feature_importance_analysis.xlsx"
                )
            else:
                feature_export_success = True
            
            # Export executive report using ReportExporter
            report_success = self.report_exporter.export_executive_report(
                export_data,
                export_path / "ensemble_modeling_executive_report.html",
                format='html',
                include_charts=True
            )
            
            # Export model artifacts if requested
            model_export_success = True
            if include_model_artifacts and ensemble_result.ensemble_model:
                try:
                    import joblib
                    model_path = export_path / "ensemble_model.pkl"
                    joblib.dump(ensemble_result.ensemble_model, model_path)
                    self.logger.info(f"Exported model to {model_path}")
                except Exception as e:
                    self.logger.warning(f"Could not export model artifacts: {str(e)}")
                    model_export_success = False
            
            return {
                'detailed_data': data_export_success,
                'model_performances': performances_export_success,
                'feature_importance': feature_export_success,
                'executive_report': report_success,
                'model_artifacts': model_export_success
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting ensemble results: {str(e)}")
            return {}

    # Helper methods using utility framework
    def _define_parameter_search_space(self, base_models: List[str]) -> Dict[str, Dict[str, Any]]:
        """Define parameter search space for optimization."""
        try:
            param_space = {}
            
            if 'random_forest' in base_models:
                param_space['random_forest'] = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            
            if 'gradient_boosting' in base_models:
                param_space['gradient_boosting'] = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            
            if 'logistic_regression' in base_models:
                param_space['logistic_regression'] = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            
            if 'svm' in base_models:
                param_space['svm'] = {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            
            return param_space
            
        except Exception:
            return {}

    def _bayesian_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Dict[str, Any]],
        config: EnsembleConfig
    ) -> Dict[str, Any]:
        """Perform Bayesian optimization for hyperparameters."""
        try:
            # Simplified Bayesian optimization - in practice would use libraries like optuna
            optimization_results = {
                'method': 'bayesian',
                'best_params': {},
                'best_score': 0.0,
                'optimization_history': []
            }
            
            # For demonstration, use random search as a placeholder
            return self._random_search_optimization(X, y, param_space, config)
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian optimization: {str(e)}")
            return {}

    def _grid_search_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Dict[str, Any]],
        config: EnsembleConfig
    ) -> Dict[str, Any]:
        """Perform grid search optimization."""
        try:
            from sklearn.model_selection import GridSearchCV
            from sklearn.ensemble import RandomForestClassifier
            
            # Simplified grid search for demonstration
            optimization_results = {
                'method': 'grid_search',
                'best_params': {},
                'best_score': 0.0,
                'optimization_history': []
            }
            
            # Example with Random Forest
            if 'random_forest' in param_space:
                rf = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(
                    rf, param_space['random_forest'], 
                    cv=config.cross_validation_folds, 
                    scoring='accuracy'
                )
                grid_search.fit(X, y)
                
                optimization_results['best_params']['random_forest'] = grid_search.best_params_
                optimization_results['best_score'] = grid_search.best_score_
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error in grid search optimization: {str(e)}")
            return {}

    def _random_search_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Dict[str, Any]],
        config: EnsembleConfig
    ) -> Dict[str, Any]:
        """Perform random search optimization."""
        try:
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.ensemble import RandomForestClassifier
            
            optimization_results = {
                'method': 'random_search',
                'best_params': {},
                'best_score': 0.0,
                'optimization_history': []
            }
            
            # Example with Random Forest
            if 'random_forest' in param_space:
                rf = RandomForestClassifier(random_state=42)
                random_search = RandomizedSearchCV(
                    rf, param_space['random_forest'], 
                    cv=config.cross_validation_folds, 
                    scoring='accuracy',
                    n_iter=20,
                    random_state=42
                )
                random_search.fit(X, y)
                
                optimization_results['best_params']['random_forest'] = random_search.best_params_
                optimization_results['best_score'] = random_search.best_score_
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error in random search optimization: {str(e)}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from tracker."""
        return self.performance_tracker.get_performance_summary()

    def get_audit_trail(self, hours: int = 24) -> List[Any]:
        """Get audit trail for ensemble modeling operations."""
        return self.audit_logger.get_audit_trail(
            operation="analysis_execution",
            time_window_hours=hours
        )

    # Additional helper methods would be implemented here...
    def _generate_ensemble_recommendations(
        self,
        individual_performances: Dict[str, ModelPerformance],
        ensemble_performance: ModelPerformance,
        feature_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on ensemble analysis."""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if ensemble_performance.accuracy > 0.9:
                recommendations.append("Excellent ensemble performance achieved - consider deploying to production")
            elif ensemble_performance.accuracy > 0.8:
                recommendations.append("Good ensemble performance - consider further optimization")
            else:
                recommendations.append("Ensemble performance needs improvement - review feature engineering and model selection")
            
            # Individual model recommendations
            best_individual = max(individual_performances.values(), key=lambda x: x.accuracy)
            if best_individual.accuracy > ensemble_performance.accuracy:
                recommendations.append(f"Consider using {best_individual.model_name} as single model - outperforms ensemble")
            
            # Feature recommendations
            top_features = feature_analysis.get('top_features', [])
            if len(top_features) > 10:
                recommendations.append(f"Focus on top {min(10, len(top_features))} features for model optimization")
            
            # Stability recommendations
            feature_consistency = feature_analysis.get('feature_consistency', {})
            if feature_consistency:
                avg_consistency = np.mean(list(feature_consistency.values()))
                if avg_consistency < 0.7:
                    recommendations.append("Feature importance shows high variability - consider feature selection")
            
            return recommendations
            
        except Exception:
            return ["Review ensemble modeling results and optimize based on performance metrics"]

    def _analyze_model_interpretability(
        self,
        ensemble_model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Analyze model interpretability using various techniques."""
        try:
            interpretability = {
                'feature_importance_global': {},
                'model_complexity': {},
                'prediction_explanations': {}
            }
            
            # Global feature importance
            if hasattr(ensemble_model, 'feature_importances_'):
                interpretability['feature_importance_global'] = dict(
                    zip(X.columns, ensemble_model.feature_importances_)
                )
            
            # Model complexity analysis
            interpretability['model_complexity'] = {
                'n_features': len(X.columns),
                'n_estimators': len(ensemble_model.estimators_) if hasattr(ensemble_model, 'estimators_') else 1,
                'interpretability_score': self._calculate_interpretability_score(ensemble_model)
            }
            
            return interpretability
            
        except Exception as e:
            self.logger.error(f"Error analyzing model interpretability: {str(e)}")
            return {}

    def _calculate_interpretability_score(self, model: Any) -> float:
        """Calculate model interpretability score."""
        try:
            # Simplified interpretability scoring
            if hasattr(model, 'estimators_'):
                # Ensemble models are less interpretable
                n_estimators = len(model.estimators_)
                return max(0, 1 - (n_estimators / 100))
            else:
                # Single models are more interpretable
                return 0.8
        except Exception:
            return 0.5

    def _retrain_ensemble_on_sample(self, ensemble_model: Any, X_sample: pd.DataFrame, y_sample: pd.Series) -> Any:
        """Retrain ensemble model on bootstrap sample."""
        try:
            # Clone the ensemble model structure
            from sklearn.base import clone
            sample_model = clone(ensemble_model)
            sample_model.fit(X_sample, y_sample)
            return sample_model
        except Exception:
            return ensemble_model

    def _evaluate_model_performance(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = model.predict(X)
            
            return {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
        except Exception:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
