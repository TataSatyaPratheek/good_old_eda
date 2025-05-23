"""
__init__.py for pipelines folder
Comprehensive pipeline system initialization with proper import handling
"""

import sys
from pathlib import Path

# Add current directory to Python path to handle relative imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Also add parent directory (project root) to path
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    # Import pipeline configuration first
    from .pipeline_config import PipelineConfigManager, PipelineConfig, DataConfig, AnalysisConfig, ModelingConfig, OptimizationConfig
    
    # Import individual pipelines
    from .eda_pipeline import EDAPipeline
    from .feature_pipeline import FeaturePipeline
    from .modeling_pipeline import ModelingPipeline
    from .competitive_pipeline import CompetitivePipeline
    from .optimization_pipeline import OptimizationPipeline
    
    # Import master orchestrator
    from .pipeline_orchestrator import PipelineOrchestrator, PipelineStatus, ExecutionMode, OrchestrationResult
    
    # Define public API
    __all__ = [
        # Configuration classes
        'PipelineConfigManager',
        'PipelineConfig',
        'DataConfig',
        'AnalysisConfig',
        'ModelingConfig',
        'OptimizationConfig',
        
        # Individual pipelines
        'EDAPipeline',
        'FeaturePipeline',
        'ModelingPipeline',
        'CompetitivePipeline',
        'OptimizationPipeline',
        
        # Master orchestrator
        'PipelineOrchestrator',
        'PipelineStatus',
        'ExecutionMode',
        'OrchestrationResult'
    ]
    
    # Pipeline system metadata
    __version__ = "1.0.0"
    __description__ = "Comprehensive SEO Competitive Intelligence Pipeline System"
    __author__ = "SEO Intelligence Team"
    
    # Pipeline system configuration
    PIPELINE_SYSTEM_CONFIG = {
        'version': __version__,
        'supported_pipelines': [
            'eda_pipeline',
            'feature_pipeline', 
            'modeling_pipeline',
            'competitive_pipeline',
            'optimization_pipeline'
        ],
        'execution_modes': ['sequential', 'parallel', 'dependency_based'],
        'default_execution_mode': 'dependency_based',
        'supports_async': True,
        'supports_caching': True,
        'supports_monitoring': True
    }
    
    def get_pipeline_system_info():
        """Get comprehensive pipeline system information"""
        return {
            'version': __version__,
            'description': __description__,
            'configuration': PIPELINE_SYSTEM_CONFIG,
            'available_pipelines': __all__[6:11],  # Pipeline classes
            'orchestrator_available': True,
            'configuration_management': True
        }
    
    def create_pipeline_orchestrator(config_path=None):
        """Factory function to create pipeline orchestrator"""
        return PipelineOrchestrator(config_path=config_path)
    
    def create_individual_pipeline(pipeline_type, config_manager=None):
        """Factory function to create individual pipelines"""
        pipeline_classes = {
            'eda': EDAPipeline,
            'feature': FeaturePipeline,
            'modeling': ModelingPipeline,
            'competitive': CompetitivePipeline,
            'optimization': OptimizationPipeline
        }
        
        if pipeline_type not in pipeline_classes:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {list(pipeline_classes.keys())}")
        
        return pipeline_classes[pipeline_type](config_manager=config_manager)
    
    # Validate imports on module load
    _import_success = True
    _import_errors = []
    
except ImportError as e:
    _import_success = False
    _import_errors = [str(e)]
    
    # Fallback minimal exports
    __all__ = []
    
    def get_pipeline_system_info():
        """Get pipeline system info with import errors"""
        return {
            'version': '1.0.0',
            'status': 'import_failed',
            'errors': _import_errors,
            'available_pipelines': [],
            'orchestrator_available': False
        }
    
    def create_pipeline_orchestrator(config_path=None):
        """Factory function that raises import error"""
        raise ImportError(f"Pipeline orchestrator unavailable due to import errors: {_import_errors}")
    
    def create_individual_pipeline(pipeline_type, config_manager=None):
        """Factory function that raises import error"""
        raise ImportError(f"Individual pipelines unavailable due to import errors: {_import_errors}")

# Module-level validation
def validate_pipeline_system():
    """Validate pipeline system integrity"""
    validation_results = {
        'imports_successful': _import_success,
        'import_errors': _import_errors,
        'orchestrator_available': _import_success,
        'individual_pipelines_available': _import_success,
        'configuration_system_available': _import_success
    }
    
    if _import_success:
        try:
            # Test orchestrator creation
            orchestrator = create_pipeline_orchestrator()
            validation_results['orchestrator_creation'] = True
            
            # Test individual pipeline creation
            test_pipeline = create_individual_pipeline('eda')
            validation_results['individual_pipeline_creation'] = True
            
        except Exception as e:
            validation_results['creation_errors'] = [str(e)]
            validation_results['orchestrator_creation'] = False
            validation_results['individual_pipeline_creation'] = False
    
    return validation_results

# Auto-validate on import
_validation_results = validate_pipeline_system()

if not _validation_results['imports_successful']:
    import warnings
    warnings.warn(
        f"Pipeline system import issues detected: {_validation_results['import_errors']}", 
        ImportWarning
    )
