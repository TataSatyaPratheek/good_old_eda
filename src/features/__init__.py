"""
Feature Engineering Module for SEMrush SEO Competitive Intelligence
Advanced feature creation for bridging analytical gaps and enabling predictive modeling
"""

from .feature_engineer import (
    FeatureEngineeringConfig,
    FeatureEngineeringResult,
    FeatureEngineer
)
from .advanced_features import (
    PredictiveModel,
    CompetitiveIntelligences,
    AdvancedFeaturesEngine,
    AdvancedMetrics
)
from .feature_selector import (
    FeatureSelectionConfig,
    FeatureSelectionResult,
    FeatureImportanceAnalysis,
    FeatureSelector
)
from .feature_validator import (
    FeatureValidationConfig,
    FeatureValidationResult,
    FeatureQualityMetrics,
    FeatureValidator
)   
from .competitive_features import (
    CompetitorProfile,
    MarketAnalysis,
    CompetitiveIntelligence,
    CompetitiveFeatures
)
from .temporal_features import (
    TemporalFeatureEngineer,
    TemporalFeatureConfig,
    TemporalAnalysisResult,
    TemporalFeatureResult
)   

__all__ = [
    'TemporalFeatureEngineer',
    'TemporalFeatureConfig',
    'TemporalAnalysisResult',
    'TemporalFeatureResult',
    
    'FeatureEngineeringConfig',
    'FeatureEngineeringResult',
    'FeatureEngineer',
    
    'PredictiveModel',
    'CompetitiveIntelligences',
    'AdvancedMetrics',
    'AdvancedFeaturesEngine',
    
    'FeatureSelector',
    'FeatureSelectionConfig',
    'FeatureSelectionResult',
    'FeatureImportanceAnalysis',
    
    'FeatureValidator',
    'FeatureValidationConfig',
    'FeatureValidationResult',
    'FeatureQualityMetrics',

    'CompetitorProfile',
    'MarketAnalysis',
    'CompetitiveIntelligence',
    'CompetitiveFeatures'
]

__version__ = "1.0.0"
