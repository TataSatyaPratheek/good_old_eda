"""
Configuration System for SEO Competitive Intelligence Platform
Comprehensive configuration management with environment support and validation
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Import our utilities for configuration management
try:
    from src.utils.config_utils import ConfigManager
    from src.utils.logging_utils import LoggerFactory
    from src.utils.validation_utils import SchemaValidator
except ImportError:
    # Fallback if utils not available
    ConfigManager = None
    LoggerFactory = None
    SchemaValidator = None

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    debug: bool
    log_level: str
    data_retention_days: int
    cache_ttl_seconds: int
    max_concurrent_operations: int
    enable_detailed_logging: bool
    enable_performance_monitoring: bool

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int
    timeout_seconds: int
    ssl_enabled: bool

@dataclass
class APIConfig:
    """API configuration"""
    semrush_api_key: str
    ahrefs_api_key: str
    google_analytics_credentials: str
    rate_limit_requests_per_minute: int
    timeout_seconds: int
    retry_attempts: int
    enable_caching: bool

@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    chunk_size: int
    max_memory_usage_gb: float
    parallel_processing: bool
    max_workers: int
    enable_data_validation: bool
    quality_threshold: float
    enable_auto_cleaning: bool

class ConfigurationManager:
    """
    Comprehensive Configuration Manager for SEO Platform
    
    Manages all configuration aspects including environment-specific settings,
    API configurations, processing parameters, and system constants.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.logger = self._setup_logger()
        
        # Configuration cache
        self._config_cache = {}
        self._environment = os.getenv('SEO_ENV', 'development')
        
        # Load configurations
        self.constants = self._load_constants()
        self.paths = self._load_paths()
        self.environment_config = self._load_environment_config()
        self.database_config = self._load_database_config()
        self.api_config = self._load_api_config()
        self.processing_config = self._load_processing_config()
        
        # Validate configurations
        self._validate_configurations()
        
        self.logger.info(f"Configuration manager initialized for environment: {self._environment}")

    def _setup_logger(self) -> logging.Logger:
        """Setup basic logger for configuration manager"""
        if LoggerFactory:
            return LoggerFactory.get_logger("config_manager")
        else:
            # Fallback logging setup
            logger = logging.getLogger("config_manager")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            return logger

    def _load_constants(self) -> Dict[str, Any]:
        """Load system constants"""
        try:
            constants_file = self.config_dir / "constants.py"
            if constants_file.exists():
                # Import constants dynamically
                import importlib.util
                spec = importlib.util.spec_from_file_location("constants", constants_file)
                constants_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(constants_module)
                
                # Extract all uppercase attributes as constants
                constants = {
                    name: getattr(constants_module, name)
                    for name in dir(constants_module)
                    if name.isupper() and not name.startswith('_')
                }
                
                self.logger.info(f"Loaded {len(constants)} constants")
                return constants
            else:
                self.logger.warning("Constants file not found, using defaults")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading constants: {str(e)}")
            return {}

    def _load_paths(self) -> Dict[str, str]:
        """Load path configurations"""
        try:
            paths_file = self.config_dir / "paths.yaml"
            if paths_file.exists():
                with open(paths_file, 'r') as f:
                    paths = yaml.safe_load(f) or {}
                
                # Convert relative paths to absolute
                base_dir = self.config_dir.parent
                for key, path in paths.items():
                    if isinstance(path, str) and not os.path.isabs(path):
                        paths[key] = str(base_dir / path)
                
                self.logger.info(f"Loaded {len(paths)} path configurations")
                return paths
            else:
                self.logger.warning("Paths file not found, using defaults")
                return self._get_default_paths()
        except Exception as e:
            self.logger.error(f"Error loading paths: {str(e)}")
            return self._get_default_paths()

    def _load_environment_config(self) -> EnvironmentConfig:
        """Load environment-specific configuration"""
        try:
            env_file = self.config_dir / f"environments/{self._environment}.yaml"
            if env_file.exists():
                with open(env_file, 'r') as f:
                    env_data = yaml.safe_load(f) or {}
                
                return EnvironmentConfig(**env_data.get('environment', {}))
            else:
                self.logger.warning(f"Environment config for {self._environment} not found, using defaults")
                return self._get_default_environment_config()
        except Exception as e:
            self.logger.error(f"Error loading environment config: {str(e)}")
            return self._get_default_environment_config()

    def _load_database_config(self) -> Optional[DatabaseConfig]:
        """Load database configuration"""
        try:
            db_file = self.config_dir / "database.yaml"
            if db_file.exists():
                with open(db_file, 'r') as f:
                    db_data = yaml.safe_load(f) or {}
                
                env_db_config = db_data.get(self._environment, {})
                if env_db_config:
                    return DatabaseConfig(**env_db_config)
            
            self.logger.info("No database configuration found")
            return None
        except Exception as e:
            self.logger.error(f"Error loading database config: {str(e)}")
            return None

    def _load_api_config(self) -> APIConfig:
        """Load API configuration"""
        try:
            api_file = self.config_dir / "api.yaml"
            if api_file.exists():
                with open(api_file, 'r') as f:
                    api_data = yaml.safe_load(f) or {}
                
                # Merge environment variables with file config
                api_config = api_data.get('api', {})
                
                # Override with environment variables if available
                api_config['semrush_api_key'] = os.getenv('SEMRUSH_API_KEY', api_config.get('semrush_api_key', ''))
                api_config['ahrefs_api_key'] = os.getenv('AHREFS_API_KEY', api_config.get('ahrefs_api_key', ''))
                api_config['google_analytics_credentials'] = os.getenv('GA_CREDENTIALS', api_config.get('google_analytics_credentials', ''))
                
                return APIConfig(**api_config)
            else:
                return self._get_default_api_config()
        except Exception as e:
            self.logger.error(f"Error loading API config: {str(e)}")
            return self._get_default_api_config()

    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration"""
        try:
            processing_file = self.config_dir / "processing.yaml"
            if processing_file.exists():
                with open(processing_file, 'r') as f:
                    processing_data = yaml.safe_load(f) or {}
                
                return ProcessingConfig(**processing_data.get('processing', {}))
            else:
                return self._get_default_processing_config()
        except Exception as e:
            self.logger.error(f"Error loading processing config: {str(e)}")
            return self._get_default_processing_config()

    def _validate_configurations(self):
        """Validate all configurations"""
        try:
            # Validate required API keys
            if not self.api_config.semrush_api_key:
                self.logger.warning("SEMrush API key not configured")
            
            # Validate paths exist
            for key, path in self.paths.items():
                if not os.path.exists(path):
                    self.logger.warning(f"Path does not exist: {key} = {path}")
            
            # Validate environment config
            if self.environment_config.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                self.logger.warning(f"Invalid log level: {self.environment_config.log_level}")
            
            self.logger.info("Configuration validation completed")
        except Exception as e:
            self.logger.error(f"Error validating configurations: {str(e)}")

    def _get_default_paths(self) -> Dict[str, str]:
        """Get default path configuration"""
        base_dir = self.config_dir.parent
        return {
            'data_dir': str(base_dir / 'data'),
            'reports_dir': str(base_dir / 'reports'),
            'logs_dir': str(base_dir / 'logs'),
            'exports_dir': str(base_dir / 'reports'),
            'cache_dir': str(base_dir / 'cache'),
            'models_dir': str(base_dir / 'models'),
            'config_dir': str(self.config_dir)
        }

    def _get_default_environment_config(self) -> EnvironmentConfig:
        """Get default environment configuration"""
        return EnvironmentConfig(
            name=self._environment,
            debug=self._environment == 'development',
            log_level='INFO',
            data_retention_days=90,
            cache_ttl_seconds=3600,
            max_concurrent_operations=4,
            enable_detailed_logging=True,
            enable_performance_monitoring=True
        )

    def _get_default_api_config(self) -> APIConfig:
        """Get default API configuration"""
        return APIConfig(
            semrush_api_key=os.getenv('SEMRUSH_API_KEY', ''),
            ahrefs_api_key=os.getenv('AHREFS_API_KEY', ''),
            google_analytics_credentials=os.getenv('GA_CREDENTIALS', ''),
            rate_limit_requests_per_minute=60,
            timeout_seconds=30,
            retry_attempts=3,
            enable_caching=True
        )

    def _get_default_processing_config(self) -> ProcessingConfig:
        """Get default processing configuration"""
        return ProcessingConfig(
            chunk_size=10000,
            max_memory_usage_gb=4.0,
            parallel_processing=True,
            max_workers=4,
            enable_data_validation=True,
            quality_threshold=0.7,
            enable_auto_cleaning=True
        )

    def get_config(self, section: str) -> Any:
        """Get configuration by section"""
        configs = {
            'constants': self.constants,
            'paths': self.paths,
            'environment': self.environment_config,
            'database': self.database_config,
            'api': self.api_config,
            'processing': self.processing_config
        }
        return configs.get(section)

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations"""
        return {
            'constants': self.constants,
            'paths': self.paths,
            'environment': asdict(self.environment_config),
            'database': asdict(self.database_config) if self.database_config else None,
            'api': asdict(self.api_config),
            'processing': asdict(self.processing_config),
            'metadata': {
                'environment': self._environment,
                'loaded_at': datetime.now().isoformat(),
                'config_dir': str(self.config_dir)
            }
        }

    def export_config(self, output_file: str, format: str = 'yaml'):
        """Export configuration to file"""
        try:
            config_data = self.get_all_configs()
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {str(e)}")

# Global configuration instance
_global_config = None

def get_config_manager(config_dir: Optional[str] = None) -> ConfigurationManager:
    """Get or create global configuration manager"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigurationManager(config_dir)
    return _global_config

def get_config(section: str) -> Any:
    """Get configuration by section"""
    return get_config_manager().get_config(section)

def get_constants() -> Dict[str, Any]:
    """Get system constants"""
    return get_config('constants')

def get_paths() -> Dict[str, str]:
    """Get path configurations"""
    return get_config('paths')

def get_environment_config() -> EnvironmentConfig:
    """Get environment configuration"""
    return get_config('environment')

def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_config('api')

def get_processing_config() -> ProcessingConfig:
    """Get processing configuration"""
    return get_config('processing')

# Export public API
__all__ = [
    'ConfigurationManager',
    'EnvironmentConfig',
    'DatabaseConfig', 
    'APIConfig',
    'ProcessingConfig',
    'get_config_manager',
    'get_config',
    'get_constants',
    'get_paths',
    'get_environment_config',
    'get_api_config',
    'get_processing_config'
]
