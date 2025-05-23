"""
Analysis Module for SEO Competitive Intelligence

This module consolidates various analysis components including:
- Position Analysis: Handles ranking analysis, position volatility, and competitive positioning.
- SERP Feature Mapping: Manages SERP feature detection, mapping, and competitive analysis.
- Traffic Comparison: Provides advanced traffic analysis, comparison metrics, and assessment.
"""

from .position_analyzer import PositionAnalyzer, PositionMetrics
from .serp_feature_mapper import SERPFeatureMapper, SERPFeatureAnalysis
from .traffic_comparator import (
    TrafficComparator,
    TrafficMetrics,
    CompetitiveTrafficAnalysis
)

__all__ = [
    'PositionAnalyzer',
    'PositionMetrics',
    'SERPFeatureMapper',
    'SERPFeatureAnalysis',
    'TrafficComparator',
    'TrafficMetrics',
    'CompetitiveTrafficAnalysis',
]

__version__ = "1.0.0"