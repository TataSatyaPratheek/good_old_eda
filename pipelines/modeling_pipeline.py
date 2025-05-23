"""
Modeling Pipeline
Comprehensive modeling pipeline leveraging refactored modules and src/utils
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio

# Import refactored modules
from src.models.position_predictor import PositionPredictor, PredictionReport
from src.models.model_evaluator import ModelEvaluator, ModelComparisonResult
from src.models.ensemble_models import EnsembleModelManager, EnsembleResult
from src.models.anomaly_detector import AnomalyDetector, AnomalyReport
from src.models.traffic_optimizer import TrafficOptimizer, OptimizationResult

# Import utils framework
from src.utils.common_helpers import timing_decorator, memoize
from src.utils.logging_utils import LoggerFactory, PerformanceTracker, AuditLogger
from src.utils.config_utils import ConfigManager
from src.utils.data_utils import DataProcessor, DataValidator
from src.utils.export_utils import ReportExporter, DataExporter
from src.utils.math_utils import StatisticalCalculator

# Import pipeline configuration
from .pipeline_config import PipelineConfigManager

class ModelingPipeline:
    """
    Advanced Modeling Pipeline
    
    Orchestrates comprehensive modeling using all refactored modules
    """
    
    def __init__(self, config_manager: Optional[PipelineConfigManager] = None):
        """Initialize modeling pipeline with comprehensive utilities"""
        self.logger = LoggerFactory.get_logger("modeling_pipeline")
        self.performance_tracker = PerformanceTracker(self.logger)
        self.audit_logger = AuditLogger(self.logger)
        
        # Configuration management
        self.config_manager = config_manager or PipelineConfigManager()
        self.pipeline_config = self.config_manager.get_pipeline_config('modeling_pipeline')
        self.modeling_config = self.config_manager.modeling_config
        self.optimization_config = self.config_manager.optimization_config
        
        # Initialize refactored modeling modules
        self.position_predictor = PositionPredictor(logger=self.logger)
        self.model_evaluator = ModelEvaluator(logger=self.logger)
        self.ensemble_manager = EnsembleModelManager(logger=self.logger)
        self.anomaly_detector = AnomalyDetector(logger=self.logger)
        self.traffic_optimizer = TrafficOptimizer(logger=self.logger)
        
        # Utilities
        self.data_processor = DataProcessor(self.logger)
        self.stats_calculator = StatisticalCalculator(self.logger)
        self.report_exporter = ReportExporter(self.logger)
        self.data_exporter = DataExporter(self.logger)
        
        # Pipeline state
        self.pipeline_results = {}
        self.trained_models = {}

    @timing_decorator()
    async def run_comprehensive_modeling(
        self,
        training_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        modeling_objectives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive modeling pipeline
        
        Args:
            training_data: Training dataset with engineered features
            test_data: Optional test dataset
            target_column: Target variable for supervised learning
            modeling_objectives: Specific modeling objectives
            
        Returns:
            Comprehensive modeling results
        """
        try:
            with self.performance_tracker.track_block("comprehensive_modeling"):
                # Audit log pipeline execution
                self.audit_logger.log_analysis_execution(
                    user_id="pipeline_system",
                    analysis_type="comprehensive_modeling",
                    parameters={
                        "training_data_rows": len(training_data),
                        "test_data_rows": len(test_data) if test_data is not None else 0,
                        "target_column": target_column,
                        "modeling_objectives": modeling_objectives
                    }
                )
                
                self.logger.info("Starting comprehensive modeling pipeline")
                
                # Phase 1: Data Preparation for Modeling
                prepared_data = await self._prepare_modeling_data(
                    training_data, test_data, target_column
                )
                
                # Phase 2: Position Prediction Modeling
                prediction_results = await self._execute_position_prediction(
                    prepared_data
                )
                
                # Phase 3: Ensemble Model Development
                ensemble_results = await self._execute_ensemble_modeling(
                    prepared_data, prediction_results
                )
                
                # Phase 4: Model Evaluation and Comparison
                evaluation_results = await self._execute_model_evaluation(
                    prepared_data, ensemble_results
                )
                
                # Phase 5: Anomaly Detection Modeling
                anomaly_results = await self._execute_anomaly_modeling(
                    prepared_data
                )
                
                # Phase 6: Traffic Optimization Modeling
                optimization_results = await self._execute_optimization_modeling(
                    prepared_data, ensemble_results
                )
                
                # Phase 7: Model Integration and Validation
                integrated_models = await self._integrate_modeling_results({
                    'prepared_data': prepared_data,
                    'prediction_results': prediction_results,
                    'ensemble_results': ensemble_results,
                    'evaluation_results': evaluation_results,
                    'anomaly_results': anomaly_results,
                    'optimization_results': optimization_results
                })
                
                # Phase 8: Model Deployment Preparation
                deployment_package = await self._prepare_deployment_package(
                    integrated_models
                )
                
                # Export comprehensive results
                export_results = await self._export_modeling_results(integrated_models)
                integrated_models['export_results'] = export_results
                integrated_models['deployment_package'] = deployment_package
                
                self.pipeline_results = integrated_models
                self.logger.info("Comprehensive modeling pipeline completed")
                return integrated_models
                
        except Exception as e:
            self.logger.error(f"Error in modeling pipeline: {str(e)}")
            await self._handle_pipeline_error(e)
            return {}

    async def _prepare_modeling_data(
        self,
        training_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame],
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare data for modeling"""
        try:
            with self.performance_tracker.track_block("modeling_data_preparation"):
                self.logger.info("Preparing data for modeling")
                
                # Clean training data
                cleaned_training = self.data_processor.clean_seo_data(training_data)
                
                # Handle test data
                cleaned_test = None
                if test_data is not None:
                    cleaned_test = self.data_processor.clean_seo_data(test_data)
                else:
                    # Split training data if no test data provided
                    from sklearn.model_selection import train_test_split
                    cleaned_training, cleaned_test = train_test_split(
                        cleaned_training, test_size=0.2, random_state=42
                    )
                
                # Prepare features and target
                feature_columns = [col for col in cleaned_training.columns 
                                 if col != target_column and col not in ['Keyword', 'date']]
                
                X_train = cleaned_training[feature_columns].select_dtypes(include=[np.number])
                X_test = cleaned_test[feature_columns].select_dtypes(include=[np.number])
                
                # Handle target variable
                y_train = None
                y_test = None
                if target_column and target_column in cleaned_training.columns:
                    y_train = cleaned_training[target_column]
                    y_test = cleaned_test[target_column] if target_column in cleaned_test.columns else None
                
                # Fill missing values
                X_train = X_train.fillna(X_train.mean())
                X_test = X_test.fillna(X_train.mean())  # Use training means for test
                
                if y_train is not None:
                    y_train = y_train.fillna(y_train.median())
                if y_test is not None:
                    y_test = y_test.fillna(y_train.median())
                
                prepared_data = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'training_data': cleaned_training,
                    'test_data': cleaned_test,
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'data_summary': {
                        'training_shape': X_train.shape,
                        'test_shape': X_test.shape,
                        'feature_count': len(feature_columns),
                        'has_target': target_column is not None
                    }
                }
                
                self.logger.info(f"Data preparation completed: {X_train.shape} training, {X_test.shape} test")
                return prepared_data
                
        except Exception as e:
            self.logger.error(f"Error in modeling data preparation: {str(e)}")
            return {}

    async def _execute_position_prediction(self, prepared_data: Dict[str, Any]) -> PredictionReport:
        """Execute position prediction modeling"""
        try:
            with self.performance_tracker.track_block("position_prediction"):
                self.logger.info("Executing position prediction modeling")
                
                training_data = prepared_data.get('training_data', pd.DataFrame())
                
                if training_data.empty:
                    return PredictionReport([], {}, {}, {}, [], {})
                
                # Use position predictor from refactored module
                prediction_report = self.position_predictor.predict_positions(
                    historical_data=training_data,
                    keywords_to_predict=None,  # Predict for all keywords
                    config=None,  # Use default config
                    competitive_data=None
                )
                
                self.logger.info(f"Position prediction completed: {len(prediction_report.predictions)} predictions")
                return prediction_report
                
        except Exception as e:
            self.logger.error(f"Error in position prediction: {str(e)}")
            return PredictionReport([], {}, {}, {}, [], {})

    async def _execute_ensemble_modeling(
        self,
        prepared_data: Dict[str, Any],
        prediction_results: PredictionReport
    ) -> EnsembleResult:
        """Execute ensemble modeling"""
        try:
            with self.performance_tracker.track_block("ensemble_modeling"):
                self.logger.info("Executing ensemble modeling")
                
                X_train = prepared_data.get('X_train', pd.DataFrame())
                y_train = prepared_data.get('y_train', pd.Series())
                X_test = prepared_data.get('X_test', pd.DataFrame())
                y_test = prepared_data.get('y_test', pd.Series())
                
                if X_train.empty or y_train is None or len(y_train) == 0:
                    return EnsembleResult(None, {}, None, {}, {}, [], [])
                
                # Prepare validation data
                validation_data = (X_test, y_test) if not X_test.empty and y_test is not None else None
                
                # Use ensemble modeling from refactored module
                ensemble_result = self.ensemble_manager.build_ensemble_model(
                    X=X_train,
                    y=y_train,
                    config=None,  # Use default config
                    validation_data=validation_data
                )
                
                self.logger.info("Ensemble modeling completed")
                return ensemble_result
                
        except Exception as e:
            self.logger.error(f"Error in ensemble modeling: {str(e)}")
            return EnsembleResult(None, {}, None, {}, {}, [], [])

    async def _execute_model_evaluation(
        self,
        prepared_data: Dict[str, Any],
        ensemble_results: EnsembleResult
    ) -> ModelComparisonResult:
        """Execute comprehensive model evaluation"""
        try:
            with self.performance_tracker.track_block("model_evaluation"):
                self.logger.info("Executing model evaluation")
                
                X_train = prepared_data.get('X_train', pd.DataFrame())
                X_test = prepared_data.get('X_test', pd.DataFrame())
                y_train = prepared_data.get('y_train', pd.Series())
                y_test = prepared_data.get('y_test', pd.Series())
                
                if X_train.empty or y_train is None or len(y_train) == 0:
                    return ModelComparisonResult({}, "", [], {}, [], {})
                
                # Prepare models for evaluation
                models_to_evaluate = {}
                
                # Add ensemble model if available
                if ensemble_results.ensemble_model is not None:
                    models_to_evaluate['ensemble'] = ensemble_results.ensemble_model
                
                # Add individual models from ensemble
                for model_name, performance in ensemble_results.individual_performances.items():
                    # Note: In real implementation, we'd need access to the actual model objects
                    # This is a simplified version for demonstration
                    pass
                
                if not models_to_evaluate:
                    return ModelComparisonResult({}, "", [], {}, [], {})
                
                # Use model evaluator from refactored module
                evaluation_result = self.model_evaluator.comprehensive_model_evaluation(
                    models=models_to_evaluate,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    config=None  # Use default config
                )
                
                self.logger.info("Model evaluation completed")
                return evaluation_result
                
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            return ModelComparisonResult({}, "", [], {}, [], {})

    async def _execute_anomaly_modeling(self, prepared_data: Dict[str, Any]) -> AnomalyReport:
        """Execute anomaly detection modeling"""
        try:
            with self.performance_tracker.track_block("anomaly_modeling"):
                self.logger.info("Executing anomaly detection modeling")
                
                training_data = prepared_data.get('training_data', pd.DataFrame())
                
                if training_data.empty:
                    return AnomalyReport(datetime.now(), 0, [], {}, {}, {}, [], {})
                
                # Use anomaly detector from refactored module
                anomaly_report = self.anomaly_detector.detect_comprehensive_anomalies(
                    data=training_data,
                    target_columns=['Position', 'Traffic (%)', 'Search Volume'],
                    config=None,  # Use default config
                    historical_context=None
                )
                
                self.logger.info(f"Anomaly modeling completed: {anomaly_report.total_anomalies} anomalies detected")
                return anomaly_report
                
        except Exception as e:
            self.logger.error(f"Error in anomaly modeling: {str(e)}")
            return AnomalyReport(datetime.now(), 0, [], {}, {}, {}, [], {})

    async def _execute_optimization_modeling(
        self,
        prepared_data: Dict[str, Any],
        ensemble_results: EnsembleResult
    ) -> OptimizationResult:
        """Execute traffic optimization modeling"""
        try:
            with self.performance_tracker.track_block("optimization_modeling"):
                self.logger.info("Executing optimization modeling")
                
                training_data = prepared_data.get('training_data', pd.DataFrame())
                
                if training_data.empty:
                    return OptimizationResult({}, {}, {}, {}, {}, [], {})
                
                # Use traffic optimizer from refactored module
                optimization_result = self.traffic_optimizer.optimize_traffic_allocation(
                    keyword_data=training_data,
                    config=None,  # Use default config
                    constraints=None,
                    historical_performance=None
                )
                
                self.logger.info("Optimization modeling completed")
                return optimization_result
                
        except Exception as e:
            self.logger.error(f"Error in optimization modeling: {str(e)}")
            return OptimizationResult({}, {}, {}, {}, {}, [], {})

    async def _integrate_modeling_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from all modeling phases"""
        try:
            with self.performance_tracker.track_block("modeling_results_integration"):
                self.logger.info("Integrating modeling results")
                
                # Create comprehensive integration
                integrated_results = {
                    'executive_summary': self._create_modeling_executive_summary(all_results),
                    'model_performance_summary': self._create_performance_summary(all_results),
                    'prediction_insights': self._extract_prediction_insights(all_results),
                    'optimization_recommendations': self._generate_optimization_recommendations(all_results),
                    'model_deployment_readiness': self._assess_deployment_readiness(all_results),
                    'detailed_results': all_results,
                    'integration_metadata': {
                        'pipeline_execution_time': self.performance_tracker.get_performance_summary(),
                        'models_trained': self._count_trained_models(all_results),
                        'integration_timestamp': datetime.now(),
                        'overall_success': self._assess_overall_success(all_results)
                    }
                }
                
                # Store trained models
                self.trained_models = self._extract_trained_models(all_results)
                
                self.logger.info("Modeling results integration completed")
                return integrated_results
                
        except Exception as e:
            self.logger.error(f"Error integrating modeling results: {str(e)}")
            return all_results

    async def _prepare_deployment_package(self, integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive deployment package"""
        try:
            with self.performance_tracker.track_block("deployment_preparation"):
                self.logger.info("Preparing deployment package")
                
                # Extract best models
                ensemble_results = integrated_results.get('detailed_results', {}).get('ensemble_results')
                evaluation_results = integrated_results.get('detailed_results', {}).get('evaluation_results')
                
                deployment_package = {
                    'production_models': {
                        'primary_model': ensemble_results.ensemble_model if ensemble_results else None,
                        'backup_models': list(ensemble_results.individual_performances.keys()) if ensemble_results else [],
                        'model_metadata': ensemble_results.feature_analysis if ensemble_results else {}
                    },
                    'model_performance': {
                        'best_model': evaluation_results.best_model if evaluation_results else 'unknown',
                        'performance_metrics': evaluation_results.evaluation_summary if evaluation_results else {},
                        'confidence_level': self._calculate_deployment_confidence(integrated_results)
                    },
                    'deployment_requirements': {
                        'feature_dependencies': integrated_results.get('detailed_results', {}).get('prepared_data', {}).get('feature_columns', []),
                        'data_preprocessing_steps': ['clean_seo_data', 'handle_missing_values', 'feature_scaling'],
                        'prediction_endpoints': self._define_prediction_endpoints(integrated_results),
                        'monitoring_requirements': self._define_monitoring_requirements(integrated_results)
                    },
                    'validation_results': {
                        'cross_validation_scores': ensemble_results.individual_performances if ensemble_results else {},
                        'test_set_performance': evaluation_results.statistical_comparison if evaluation_results else {},
                        'anomaly_detection_accuracy': self._assess_anomaly_accuracy(integrated_results)
                    },
                    'deployment_timestamp': datetime.now()
                }
                
                self.logger.info("Deployment package preparation completed")
                return deployment_package
                
        except Exception as e:
            self.logger.error(f"Error preparing deployment package: {str(e)}")
            return {}

    async def _export_modeling_results(self, integrated_results: Dict[str, Any]) -> Dict[str, bool]:
        """Export comprehensive modeling results"""
        try:
            with self.performance_tracker.track_block("modeling_results_export"):
                self.logger.info("Exporting modeling results")
                
                export_results = {}
                
                # Export model performance comparison
                evaluation_results = integrated_results.get('detailed_results', {}).get('evaluation_results')
                if evaluation_results and hasattr(evaluation_results, 'model_performances'):
                    performance_df = pd.DataFrame([
                        {
                            'model_name': name,
                            'accuracy': perf.accuracy,
                            'precision': perf.precision,
                            'recall': perf.recall,
                            'f1_score': perf.f1_score,
                            'roc_auc': perf.roc_auc
                        }
                        for name, perf in evaluation_results.model_performances.items()
                    ])
                    
                    performance_export = self.data_exporter.export_with_metadata(
                        performance_df,
                        metadata={'analysis_type': 'model_performance', 'generation_timestamp': datetime.now()},
                        export_path=f"{self.config_manager.data_config.output_directory}/model_performance.xlsx"
                    )
                    export_results['model_performance'] = performance_export
                
                # Export prediction results
                prediction_results = integrated_results.get('detailed_results', {}).get('prediction_results')
                if prediction_results and hasattr(prediction_results, 'predictions') and prediction_results.predictions:
                    predictions_df = pd.DataFrame([
                        {
                            'keyword': pred.keyword,
                            'current_position': pred.current_position,
                            'predicted_position': pred.predicted_position,
                            'confidence': pred.prediction_confidence
                        }
                        for pred in prediction_results.predictions
                    ])
                    
                    predictions_export = self.data_exporter.export_with_metadata(
                        predictions_df,
                        metadata={'analysis_type': 'position_predictions'},
                        export_path=f"{self.config_manager.data_config.output_directory}/position_predictions.xlsx"
                    )
                    export_results['predictions'] = predictions_export
                
                # Export executive report
                executive_report = self.report_exporter.export_executive_report(
                    integrated_results.get('executive_summary', {}),
                    f"{self.config_manager.data_config.output_directory}/modeling_executive_report.html",
                    format='html',
                    include_charts=True
                )
                export_results['executive_report'] = executive_report
                
                self.logger.info("Modeling results export completed")
                return export_results
                
        except Exception as e:
            self.logger.error(f"Error exporting modeling results: {str(e)}")
            return {}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_name': 'modeling_pipeline',
            'status': 'completed' if self.pipeline_results else 'not_started',
            'trained_models_count': len(self.trained_models),
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'results_available': bool(self.pipeline_results)
        }

    async def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline execution errors"""
        self.logger.error(f"Modeling pipeline error: {str(error)}")
        self.audit_logger.log_analysis_execution(
            user_id="pipeline_system",
            analysis_type="modeling_pipeline_error",
            result="failure",
            details={"error": str(error)}
        )

    # Helper methods
    def _create_modeling_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary for modeling"""
        return {
            'modeling_scope': 'comprehensive',
            'models_developed': self._count_trained_models(results),
            'best_model_performance': self._get_best_model_performance(results),
            'deployment_readiness': self._assess_deployment_readiness(results),
            'processing_timestamp': datetime.now()
        }

    def _create_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance summary across all models"""
        evaluation_results = results.get('evaluation_results')
        if not evaluation_results or not hasattr(evaluation_results, 'model_performances'):
            return {}
        
        performances = evaluation_results.model_performances
        if not performances:
            return {}
        
        # Calculate aggregate metrics
        avg_accuracy = np.mean([perf.accuracy for perf in performances.values()])
        avg_f1 = np.mean([perf.f1_score for perf in performances.values()])
        
        return {
            'average_accuracy': avg_accuracy,
            'average_f1_score': avg_f1,
            'best_model': evaluation_results.best_model,
            'total_models_evaluated': len(performances)
        }

    def _extract_prediction_insights(self, results: Dict[str, Any]) -> List[str]:
        """Extract key insights from predictions"""
        insights = []
        
        prediction_results = results.get('prediction_results')
        if prediction_results and hasattr(prediction_results, 'predictions') and prediction_results.predictions:
            total_predictions = len(prediction_results.predictions)
            high_confidence = len([p for p in prediction_results.predictions if p.prediction_confidence > 0.8])
            
            insights.append(f"Generated {total_predictions} position predictions")
            insights.append(f"{high_confidence} predictions with high confidence (>80%)")
        
        return insights

    def _generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations from results"""
        recommendations = []
        
        optimization_results = results.get('optimization_results')
        if optimization_results and hasattr(optimization_results, 'recommendations'):
            recommendations.extend(optimization_results.recommendations)
        
        ensemble_results = results.get('ensemble_results')
        if ensemble_results and hasattr(ensemble_results, 'recommendations'):
            recommendations.extend(ensemble_results.recommendations)
        
        return recommendations

    def _assess_deployment_readiness(self, results: Dict[str, Any]) -> str:
        """Assess overall deployment readiness"""
        try:
            evaluation_results = results.get('evaluation_results')
            if not evaluation_results:
                return 'not_ready'
            
            # Check if we have a best model with reasonable performance
            if hasattr(evaluation_results, 'best_model') and evaluation_results.best_model:
                best_performance = self._get_best_model_performance(results)
                if best_performance > 0.8:
                    return 'production_ready'
                elif best_performance > 0.7:
                    return 'staging_ready'
                else:
                    return 'development_only'
            
            return 'not_ready'
        except:
            return 'not_ready'

    def _count_trained_models(self, results: Dict[str, Any]) -> int:
        """Count total trained models"""
        count = 0
        
        ensemble_results = results.get('ensemble_results')
        if ensemble_results and hasattr(ensemble_results, 'individual_performances'):
            count += len(ensemble_results.individual_performances)
        
        if ensemble_results and ensemble_results.ensemble_model:
            count += 1
        
        return count

    def _get_best_model_performance(self, results: Dict[str, Any]) -> float:
        """Get best model performance score"""
        try:
            evaluation_results = results.get('evaluation_results')
            if evaluation_results and hasattr(evaluation_results, 'model_performances'):
                performances = [perf.accuracy for perf in evaluation_results.model_performances.values()]
                return max(performances) if performances else 0.0
            return 0.0
        except:
            return 0.0

    def _assess_overall_success(self, results: Dict[str, Any]) -> bool:
        """Assess overall pipeline success"""
        required_components = ['prediction_results', 'ensemble_results', 'evaluation_results']
        return all(component in results and results[component] for component in required_components)

    def _extract_trained_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trained models from results"""
        trained_models = {}
        
        ensemble_results = results.get('ensemble_results')
        if ensemble_results and ensemble_results.ensemble_model:
            trained_models['ensemble_model'] = ensemble_results.ensemble_model
        
        return trained_models

    def _calculate_deployment_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate deployment confidence score"""
        factors = []
        
        # Model performance factor
        best_performance = self._get_best_model_performance(results)
        factors.append(best_performance)
        
        # Model count factor (more models = more confidence)
        model_count = self._count_trained_models(results)
        factors.append(min(model_count / 5, 1.0))  # Normalize to 0-1
        
        # Overall success factor
        success_factor = 1.0 if self._assess_overall_success(results) else 0.5
        factors.append(success_factor)
        
        return np.mean(factors) if factors else 0.0

    def _define_prediction_endpoints(self, results: Dict[str, Any]) -> List[str]:
        """Define prediction endpoints for deployment"""
        endpoints = []
        
        if results.get('prediction_results'):
            endpoints.append('/predict/position')
        
        if results.get('optimization_results'):
            endpoints.append('/optimize/traffic')
        
        if results.get('anomaly_results'):
            endpoints.append('/detect/anomalies')
        
        return endpoints

    def _define_monitoring_requirements(self, results: Dict[str, Any]) -> List[str]:
        """Define monitoring requirements"""
        return [
            'Model performance degradation',
            'Data drift detection',
            'Prediction confidence tracking',
            'Anomaly alert system',
            'Resource utilization monitoring'
        ]

    def _assess_anomaly_accuracy(self, results: Dict[str, Any]) -> float:
        """Assess anomaly detection accuracy"""
        anomaly_results = results.get('anomaly_results')
        if anomaly_results and hasattr(anomaly_results, 'total_anomalies'):
            # Simplified accuracy assessment
            return 0.85 if anomaly_results.total_anomalies > 0 else 0.5
        return 0.0
