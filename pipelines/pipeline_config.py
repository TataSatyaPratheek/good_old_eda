"""
Pipeline Configuration Management
Centralized configuration for all pipeline operations
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from src.utils.config_utils import ConfigManager
from src.utils.logging_utils import LoggerFactory
from src.utils.common_helpers import ensure_list

@dataclass
class PipelineConfig:
    """Base pipeline configuration"""
    pipeline_name: str
    enabled: bool = True
    schedule: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    retry_attempts: int = 3
    timeout_minutes: int = 60
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class DataConfig:
    """Data processing configuration"""
    input_directories: List[str] = field(default_factory=list)
    output_directory: str = "reports"
    file_patterns: Dict[str, str] = field(default_factory=dict)
    date_range_days: int = 30
    include_competitors: List[str] = field(default_factory=lambda: ['dell', 'hp'])
    data_quality_threshold: float = 0.7

@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    enable_competitive_analysis: bool = True
    enable_temporal_analysis: bool = True
    enable_feature_engineering: bool = True
    enable_anomaly_detection: bool = True
    confidence_threshold: float = 0.8
    statistical_significance: float = 0.05

@dataclass
class ModelingConfig:
    """Modeling configuration"""
    model_types: List[str] = field(default_factory=lambda: ['random_forest', 'gradient_boosting', 'ensemble'])
    cross_validation_folds: int = 5
    hyperparameter_optimization: bool = True
    feature_selection: bool = True
    model_interpretability: bool = True

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    optimization_objectives: List[str] = field(default_factory=lambda: ['traffic', 'positions'])
    budget_constraints: Dict[str, float] = field(default_factory=dict)
    risk_tolerance: float = 0.3
    target_roi: float = 2.0
    prediction_horizon_days: int = 30

class PipelineConfigManager:
    """Comprehensive pipeline configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = LoggerFactory.get_logger("pipeline_config")
        self.config_manager = ConfigManager()
        self.config_path = Path(config_path) if config_path else Path("config/pipeline_config.yaml")
        
        # Load configurations
        self.load_configurations()
    
    def load_configurations(self):
        """Load all pipeline configurations"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}
                self.create_default_config()
            
            # Initialize configuration objects
            self.pipeline_configs = {
                name: PipelineConfig(**config) 
                for name, config in config_data.get('pipelines', {}).items()
            }
            
            self.data_config = DataConfig(**config_data.get('data', {}))
            self.analysis_config = AnalysisConfig(**config_data.get('analysis', {}))
            self.modeling_config = ModelingConfig(**config_data.get('modeling', {}))
            self.optimization_config = OptimizationConfig(**config_data.get('optimization', {}))
            
            self.logger.info(f"Loaded configuration for {len(self.pipeline_configs)} pipelines")
            
        except Exception as e:
            self.logger.error(f"Error loading pipeline configurations: {str(e)}")
            self.create_default_config()
    
    def create_default_config(self):
        """Create default pipeline configuration"""
        default_config = {
            'pipelines': {
                'eda_pipeline': {
                    'pipeline_name': 'eda_pipeline',
                    'enabled': True,
                    'schedule': '0 6 * * *',  # Daily at 6 AM
                    'dependencies': [],
                    'timeout_minutes': 30
                },
                'feature_pipeline': {
                    'pipeline_name': 'feature_pipeline',
                    'enabled': True,
                    'schedule': '0 7 * * *',  # Daily at 7 AM
                    'dependencies': ['eda_pipeline'],
                    'timeout_minutes': 45
                },
                'modeling_pipeline': {
                    'pipeline_name': 'modeling_pipeline',
                    'enabled': True,
                    'schedule': '0 8 * * *',  # Daily at 8 AM
                    'dependencies': ['feature_pipeline'],
                    'timeout_minutes': 90
                },
                'competitive_pipeline': {
                    'pipeline_name': 'competitive_pipeline',
                    'enabled': True,
                    'schedule': '0 9 * * *',  # Daily at 9 AM
                    'dependencies': ['eda_pipeline'],
                    'timeout_minutes': 60
                },
                'optimization_pipeline': {
                    'pipeline_name': 'optimization_pipeline',
                    'enabled': True,
                    'schedule': '0 10 * * *',  # Daily at 10 AM
                    'dependencies': ['modeling_pipeline', 'competitive_pipeline'],
                    'timeout_minutes': 120
                }
            },
            'data': {
                'input_directories': ['data'],
                'output_directory': 'reports',
                'file_patterns': {
                    'positions': '*-organic.Positions-*.xlsx',
                    'competitors': '*-organic.Competitors-*.xlsx',
                    'gap_keywords': 'gap.keywords*.xlsx'
                },
                'date_range_days': 7,
                'include_competitors': ['dell', 'hp'],
                'data_quality_threshold': 0.7
            },
            'analysis': {
                'enable_competitive_analysis': True,
                'enable_temporal_analysis': True,
                'enable_feature_engineering': True,
                'enable_anomaly_detection': True,
                'confidence_threshold': 0.8,
                'statistical_significance': 0.05
            },
            'modeling': {
                'model_types': ['random_forest', 'gradient_boosting', 'ensemble'],
                'cross_validation_folds': 5,
                'hyperparameter_optimization': True,
                'feature_selection': True,
                'model_interpretability': True
            },
            'optimization': {
                'optimization_objectives': ['traffic', 'positions'],
                'budget_constraints': {'total_budget': 10000.0},
                'risk_tolerance': 0.3,
                'prediction_horizon_days': 30
            }
        }
        
        # Save default configuration
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Created default pipeline configuration at {self.config_path}")
    
    def get_pipeline_config(self, pipeline_name: str) -> Optional[PipelineConfig]:
        """Get configuration for specific pipeline"""
        return self.pipeline_configs.get(pipeline_name)
    
    def is_pipeline_enabled(self, pipeline_name: str) -> bool:
        """Check if pipeline is enabled"""
        config = self.get_pipeline_config(pipeline_name)
        return config.enabled if config else False
    
    def get_pipeline_dependencies(self, pipeline_name: str) -> List[str]:
        """Get pipeline dependencies"""
        config = self.get_pipeline_config(pipeline_name)
        return config.dependencies if config else []
    
    def update_pipeline_config(self, pipeline_name: str, **kwargs):
        """Update pipeline configuration"""
        if pipeline_name in self.pipeline_configs:
            for key, value in kwargs.items():
                if hasattr(self.pipeline_configs[pipeline_name], key):
                    setattr(self.pipeline_configs[pipeline_name], key, value)
            self.logger.info(f"Updated configuration for {pipeline_name}")
        else:
            self.logger.warning(f"Pipeline {pipeline_name} not found in configuration")
