"""
Data Loader Module for SEO Competitive Intelligence
Handles loading, validation, and merging of SEMrush data across time periods
"""

from .data_loader import SEMrushDataLoader, DataLoadSummary
from .file_manager import FileManager
from .merge_strategy import (
    MergeStrategy,
    MergeResult,
    GapAnalysis,
    StrategicRecommendations
)
from .schema_validator import SchemaValidator, SEOSchemaValidationResult

__all__ = [
    'SEMrushDataLoader',
    'DataLoadSummary',
    'FileManager',
    'MergeStrategy',
    'MergeResult',
    'GapAnalysis',
    'StrategicRecommendations',
    'SchemaValidator',
    'SEOSchemaValidationResult',
]

__version__ = "1.0.0"
