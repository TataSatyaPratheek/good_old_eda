"""
System Constants for SEO Competitive Intelligence Platform
Comprehensive constants for the entire SEO analysis and competitive intelligence system
"""

from pathlib import Path
from datetime import timedelta

# ============================================================================
# SYSTEM METADATA
# ============================================================================
SYSTEM_NAME = "SEO Competitive Intelligence Platform"
SYSTEM_VERSION = "1.0.0"
SYSTEM_DESCRIPTION = "Advanced SEO competitive intelligence and analysis platform"
SYSTEM_AUTHOR = "SEO Intelligence Team"

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
SUPPORTED_ENVIRONMENTS = ['development', 'staging', 'production']
DEFAULT_ENVIRONMENT = 'development'
CONFIG_FILE_EXTENSIONS = ['.yaml', '.yml', '.json']

# ============================================================================
# DATA SOURCES AND PROVIDERS
# ============================================================================
SUPPORTED_DATA_SOURCES = [
    'semrush',
    'ahrefs', 
    'google_analytics',
    'google_search_console',
    'screaming_frog',
    'custom_csv'
]

# SEMrush specific constants
SEMRUSH_API_BASE_URL = "https://api.semrush.com"
SEMRUSH_RATE_LIMIT = 60  # requests per minute
SEMRUSH_SUPPORTED_DATABASES = [
    'us', 'uk', 'ca', 'ru', 'de', 'fr', 'es', 'it', 'br', 'au', 'jp', 'in'
]

# Ahrefs specific constants
AHREFS_API_BASE_URL = "https://apiv2.ahrefs.com"
AHREFS_RATE_LIMIT = 100  # requests per minute

# ============================================================================
# FILE PATTERNS AND NAMING CONVENTIONS
# ============================================================================
FILE_PATTERNS = {
    'positions': r'.*-organic\.Positions-.*\.xlsx$',
    'competitors': r'.*-organic\.Competitors-.*\.xlsx$',
    'gap_keywords': r'gap\.keywords.*\.xlsx$',
    'backlinks': r'.*-backlinks-.*\.xlsx$',
    'organic_research': r'.*-organic-research-.*\.xlsx$'
}

DATE_FORMATS = [
    '%Y-%m-%d',
    '%d-%m-%Y', 
    '%m-%d-%Y',
    '%Y%m%d',
    '%d-%b-%Y',
    '%b-%d-%Y'
]

SUPPORTED_FILE_EXTENSIONS = ['.xlsx', '.csv', '.json', '.parquet']

# ============================================================================
# DATA PROCESSING CONSTANTS
# ============================================================================
DEFAULT_CHUNK_SIZE = 10000
MAX_MEMORY_USAGE_GB = 4.0
DEFAULT_QUALITY_THRESHOLD = 0.7
MIN_DATA_POINTS_FOR_ANALYSIS = 10

# Column name mappings for different data sources
COLUMN_MAPPINGS = {
    'position': ['Position', 'Rank', 'Ranking', 'pos'],
    'keyword': ['Keyword', 'Query', 'Search Term', 'keyword'],
    'traffic': ['Traffic (%)', 'Traffic', 'Organic Traffic', 'traffic'],
    'search_volume': ['Search Volume', 'Volume', 'Monthly Volume', 'volume'],
    'cpc': ['CPC', 'Cost Per Click', 'cpc'],
    'difficulty': ['Keyword Difficulty', 'KD', 'Difficulty', 'difficulty'],
    'competition': ['Competition', 'Competitive Density', 'comp'],
    'url': ['URL', 'Landing Page', 'Page', 'url'],
    'title': ['Title', 'Page Title', 'title'],
    'serp_features': ['SERP Features by Keyword', 'Features', 'serp_features']
}

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================
# Position analysis constants
POSITION_TIERS = {
    'top_3': (1, 3),
    'top_10': (1, 10), 
    'top_20': (1, 20),
    'page_1': (1, 10),
    'page_2': (11, 20),
    'beyond_page_2': (21, 100)
}

# CTR rates by position (industry averages)
CTR_RATES = {
    1: 0.284, 2: 0.147, 3: 0.094, 4: 0.067, 5: 0.051,
    6: 0.041, 7: 0.034, 8: 0.029, 9: 0.025, 10: 0.022,
    11: 0.018, 12: 0.015, 13: 0.013, 14: 0.011, 15: 0.010
}

# Traffic analysis constants
TRAFFIC_CATEGORIES = {
    'high_traffic': 1000,
    'medium_traffic': 100,
    'low_traffic': 10,
    'minimal_traffic': 1
}

# ============================================================================
# COMPETITIVE ANALYSIS CONSTANTS
# ============================================================================
COMPETITIVE_THREAT_THRESHOLD = 0.7
MARKET_SHARE_THRESHOLDS = {
    'dominant': 0.4,
    'leading': 0.25,
    'competitive': 0.15,
    'emerging': 0.05
}

COMPETITOR_ANALYSIS_METRICS = [
    'organic_keywords',
    'organic_traffic', 
    'avg_position',
    'traffic_share',
    'keyword_overlap',
    'competitive_strength',
    'growth_rate',
    'market_share'
]

# ============================================================================
# SERP FEATURES CONSTANTS
# ============================================================================
SERP_FEATURES = [
    'Featured Snippet',
    'People Also Ask',
    'Image Pack',
    'Video',
    'News',
    'Knowledge Panel',
    'Site Links',
    'Reviews',
    'Shopping Results',
    'Local Pack',
    'Top Stories',
    'Tweets',
    'Recipe',
    'Answer Box',
    'Calculator',
    'Dictionary',
    'Flight Results',
    'Weather'
]

SERP_FEATURE_IMPACT_WEIGHTS = {
    'Featured Snippet': 0.9,
    'People Also Ask': 0.7,
    'Image Pack': 0.6,
    'Video': 0.8,
    'News': 0.5,
    'Knowledge Panel': 0.4,
    'Site Links': 0.3,
    'Reviews': 0.6,
    'Shopping Results': 0.8,
    'Local Pack': 0.9
}

# ============================================================================
# MODELING CONSTANTS
# ============================================================================
MODEL_TYPES = [
    'linear_regression',
    'random_forest',
    'gradient_boosting',
    'svm',
    'neural_network',
    'ensemble'
]

CROSS_VALIDATION_FOLDS = 5
MODEL_PERFORMANCE_THRESHOLD = 0.8
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# Hyperparameter ranges for optimization
HYPERPARAMETER_RANGES = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# ============================================================================
# OPTIMIZATION CONSTANTS
# ============================================================================
OPTIMIZATION_OBJECTIVES = [
    'traffic',
    'positions',
    'roi',
    'conversions',
    'revenue'
]

DEFAULT_BUDGET_CONSTRAINT = 10000.0
DEFAULT_ROI_TARGET = 2.0
DEFAULT_RISK_TOLERANCE = 0.3

# Optimization algorithm parameters
OPTIMIZATION_ALGORITHMS = {
    'genetic': {
        'population_size': 100,
        'generations': 50,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8
    },
    'simulated_annealing': {
        'initial_temperature': 1000,
        'cooling_rate': 0.95,
        'min_temperature': 1
    },
    'particle_swarm': {
        'n_particles': 30,
        'w': 0.7,
        'c1': 1.5,
        'c2': 1.5
    }
}

# ============================================================================
# REPORTING AND EXPORT CONSTANTS
# ============================================================================
REPORT_FORMATS = ['html', 'pdf', 'excel', 'csv', 'json']
CHART_TYPES = [
    'line',
    'bar', 
    'scatter',
    'heatmap',
    'treemap',
    'funnel',
    'waterfall'
]

EXPORT_LIMITS = {
    'max_rows_excel': 1000000,
    'max_rows_csv': 10000000,
    'max_file_size_mb': 100
}

# ============================================================================
# CACHING AND PERFORMANCE CONSTANTS
# ============================================================================
CACHE_TTL = {
    'short': timedelta(minutes=15),
    'medium': timedelta(hours=1),
    'long': timedelta(hours=24),
    'persistent': timedelta(days=7)
}

PERFORMANCE_THRESHOLDS = {
    'query_timeout_seconds': 300,
    'max_concurrent_queries': 10,
    'memory_limit_gb': 8.0,
    'disk_space_limit_gb': 100.0
}

# ============================================================================
# ERROR HANDLING AND LOGGING CONSTANTS
# ============================================================================
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
LOG_FORMATS = {
    'simple': '%(levelname)s - %(message)s',
    'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'comprehensive': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
}

MAX_LOG_FILE_SIZE_MB = 50
LOG_BACKUP_COUNT = 5

# Error codes for different types of failures
ERROR_CODES = {
    'DATA_LOADING_ERROR': 1001,
    'DATA_VALIDATION_ERROR': 1002,
    'PROCESSING_ERROR': 1003,
    'MODEL_TRAINING_ERROR': 2001,
    'MODEL_PREDICTION_ERROR': 2002,
    'OPTIMIZATION_ERROR': 3001,
    'EXPORT_ERROR': 4001,
    'API_ERROR': 5001,
    'CONFIGURATION_ERROR': 6001,
    'PIPELINE_ERROR': 7001
}

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================
VALIDATION_RULES = {
    'position': {
        'min_value': 1,
        'max_value': 100,
        'data_type': 'numeric'
    },
    'traffic': {
        'min_value': 0,
        'max_value': 100,
        'data_type': 'numeric'
    },
    'search_volume': {
        'min_value': 0,
        'data_type': 'numeric'
    },
    'cpc': {
        'min_value': 0,
        'data_type': 'numeric'
    },
    'keyword': {
        'min_length': 1,
        'max_length': 500,
        'data_type': 'string'
    }
}

DATA_QUALITY_THRESHOLDS = {
    'completeness': 0.8,
    'accuracy': 0.9,
    'consistency': 0.85,
    'validity': 0.9,
    'uniqueness': 0.95
}

# ============================================================================
# SECURITY CONSTANTS
# ============================================================================
SECURITY_CONFIG = {
    'max_api_key_age_days': 90,
    'password_min_length': 12,
    'session_timeout_minutes': 60,
    'max_login_attempts': 5,
    'lockout_duration_minutes': 30
}

# Sensitive data patterns (for redaction in logs)
SENSITIVE_PATTERNS = [
    r'api[_-]?key',
    r'password',
    r'secret',
    r'token',
    r'credential'
]

# ============================================================================
# BUSINESS LOGIC CONSTANTS
# ============================================================================
BUSINESS_RULES = {
    'min_keyword_length': 2,
    'max_keyword_length': 100,
    'position_change_significance': 5,
    'traffic_change_significance': 0.1,  # 10%
    'competitor_relevance_threshold': 0.3,
    'opportunity_score_threshold': 0.6
}

# Market analysis constants
MARKET_ANALYSIS = {
    'seasonality_periods': [7, 30, 90, 365],  # days
    'trend_analysis_window': 90,  # days
    'forecast_horizon': 30,  # days
    'confidence_intervals': [0.8, 0.95]
}

# ============================================================================
# INTEGRATION CONSTANTS
# ============================================================================
WEBHOOK_TIMEOUT_SECONDS = 30
MAX_WEBHOOK_RETRIES = 3
API_VERSION = 'v1'

# Third-party service timeouts
SERVICE_TIMEOUTS = {
    'semrush': 30,
    'ahrefs': 45,
    'google_analytics': 60,
    'google_search_console': 60
}

# ============================================================================
# FEATURE ENGINEERING CONSTANTS
# ============================================================================
FEATURE_TYPES = [
    'basic',
    'temporal',
    'competitive',
    'interaction',
    'derived',
    'aggregated'
]

TEMPORAL_FEATURES = {
    'lag_periods': [1, 7, 14, 30],
    'rolling_windows': [7, 14, 30, 90],
    'seasonal_periods': [7, 30, 90, 365]
}

FEATURE_SELECTION_METHODS = [
    'correlation',
    'mutual_information',
    'chi_square',
    'recursive_elimination',
    'lasso',
    'random_forest_importance'
]

# ============================================================================
# SYSTEM LIMITS AND CONSTRAINTS
# ============================================================================
SYSTEM_LIMITS = {
    'max_keywords_per_analysis': 100000,
    'max_competitors_per_analysis': 50,
    'max_date_range_days': 365,
    'max_export_file_size_mb': 500,
    'max_concurrent_pipelines': 5,
    'max_memory_per_process_gb': 8.0
}

# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================
DEFAULT_ANALYSIS_CONFIG = {
    'enable_competitive_analysis': True,
    'enable_temporal_analysis': True,
    'enable_feature_engineering': True,
    'enable_anomaly_detection': True,
    'enable_forecasting': True,
    'confidence_threshold': 0.8,
    'statistical_significance': 0.05
}

DEFAULT_PIPELINE_CONFIG = {
    'retry_attempts': 3,
    'timeout_minutes': 60,
    'enable_caching': True,
    'enable_parallel_processing': True,
    'notification_channels': ['email', 'slack']
}

# ============================================================================
# VERSION COMPATIBILITY
# ============================================================================
MIN_PYTHON_VERSION = '3.8'
REQUIRED_PACKAGES = [
    'pandas>=1.3.0',
    'numpy>=1.21.0',
    'scikit-learn>=1.0.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'plotly>=5.0.0',
    'openpyxl>=3.0.0',
    'pyyaml>=5.4.0',
    'requests>=2.25.0',
    'asyncio',
    'aiohttp>=3.7.0'
]

# ============================================================================
# DEPRECATION WARNINGS
# ============================================================================
DEPRECATED_FEATURES = {
    'old_data_loader': {
        'deprecated_in': '0.9.0',
        'removed_in': '2.0.0',
        'replacement': 'SEMrushDataLoader'
    }
}
