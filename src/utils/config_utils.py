"""
Configuration Management Utilities for SEO Competitive Intelligence

Advanced configuration management, environment handling, and path management
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import configparser
from enum import Enum

class Environment(Enum):
    """Environment enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "require"

@dataclass
class APIConfig:
    """API configuration"""
    semrush_api_key: str
    google_api_key: str
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    retry_attempts: int = 3

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    default_lookback_days: int = 30
    position_volatility_threshold: float = 2.0
    traffic_anomaly_threshold: float = 0.3
    competitive_threat_threshold: float = 0.7
    min_search_volume: int = 100
    max_keywords_per_analysis: int = 10000

class ConfigManager:
    """
    Advanced configuration management for SEO competitive intelligence.
    Handles environment-specific configurations, secrets management,
    dynamic configuration updates, and multiple config file formats.
    """

    def __init__(self, config_path: Optional[str] = None, config_dir: Optional[Union[str, Path]] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Handle both config_path (single file) and config_dir (directory) for backward compatibility
        if config_dir:
            self.config_dir = Path(config_dir)
            self.config_path = self.config_dir
        elif config_path:
            self.config_path = Path(config_path)
            self.config_dir = self.config_path if self.config_path.is_dir() else self.config_path.parent
        else:
            self.config_path = Path("config")
            self.config_dir = self.config_path
            
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        self._config_cache = {}
        self._config_watchers = {}

        # Initialize configuration
        self._load_configurations()
        self._load_default_configs()

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration for current environment."""
        try:
            db_config = self._get_config_section("database")
            return DatabaseConfig(
                host=db_config.get("host", "localhost"),
                port=int(db_config.get("port", 5432)),
                database=db_config.get("database", "seo_intelligence"),
                username=db_config.get("username", ""),
                password=self._get_secret("database_password", ""),
                ssl_mode=db_config.get("ssl_mode", "require")
            )
        except Exception as e:
            self.logger.error(f"Error loading database config: {str(e)}")
            return DatabaseConfig("localhost", 5432, "seo_intelligence", "", "")

    def get_api_config(self) -> APIConfig:
        """Get API configuration for current environment."""
        try:
            api_config = self._get_config_section("api")
            return APIConfig(
                semrush_api_key=self._get_secret("semrush_api_key", ""),
                google_api_key=self._get_secret("google_api_key", ""),
                rate_limit_per_minute=int(api_config.get("rate_limit_per_minute", 60)),
                timeout_seconds=int(api_config.get("timeout_seconds", 30)),
                retry_attempts=int(api_config.get("retry_attempts", 3))
            )
        except Exception as e:
            self.logger.error(f"Error loading API config: {str(e)}")
            return APIConfig("", "", 60, 30, 3)

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration for current environment."""
        try:
            logging_config = self._get_config_section("logging")
            return LoggingConfig(
                level=logging_config.get("level", "INFO"),
                format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                file_path=logging_config.get("file_path"),
                max_file_size=int(logging_config.get("max_file_size", 10485760)),
                backup_count=int(logging_config.get("backup_count", 5))
            )
        except Exception as e:
            self.logger.error(f"Error loading logging config: {str(e)}")
            return LoggingConfig()

    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration for current environment."""
        try:
            analysis_config = self._get_config_section("analysis")
            return AnalysisConfig(
                default_lookback_days=int(analysis_config.get("default_lookback_days", 30)),
                position_volatility_threshold=float(analysis_config.get("position_volatility_threshold", 2.0)),
                traffic_anomaly_threshold=float(analysis_config.get("traffic_anomaly_threshold", 0.3)),
                competitive_threat_threshold=float(analysis_config.get("competitive_threat_threshold", 0.7)),
                min_search_volume=int(analysis_config.get("min_search_volume", 100)),
                max_keywords_per_analysis=int(analysis_config.get("max_keywords_per_analysis", 10000))
            )
        except Exception as e:
            self.logger.error(f"Error loading analysis config: {str(e)}")
            return AnalysisConfig()

    def update_config(self, section: str, key: str, value: Any) -> bool:
        """
        Update configuration value dynamically.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
            
        Returns:
            True if update successful
        """
        try:
            if section not in self._config_cache:
                self._config_cache[section] = {}
            self._config_cache[section][key] = value

            # Persist to file if needed
            self._persist_config_update(section, key, value)
            self.logger.info(f"Updated config: {section}.{key} = {value}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating config: {str(e)}")
            return False

    def reload_configurations(self) -> bool:
        """Reload all configurations from files."""
        try:
            self._config_cache.clear()
            self._load_configurations()
            self._load_default_configs()
            self.logger.info("Configurations reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error reloading configurations: {str(e)}")
            return False

    def reload_config(self, config_name: Optional[str] = None):
        """
        Reload configuration(s)
        
        Args:
            config_name: Specific config to reload, or None to reload all
        """
        if config_name:
            config_path = self.config_dir / f"{config_name}.yaml"
            if config_path.exists():
                self._load_yaml_config(config_path)
            else:
                config_path = self.config_dir / f"{config_name}.json"
                if config_path.exists():
                    self._load_json_config(config_path)
        else:
            self.reload_configurations()

    def get_config(self, config_name: str, default: Any = None) -> Any:
        """
        Get configuration by name
        
        Args:
            config_name: Name of the configuration
            default: Default value if config not found
            
        Returns:
            Configuration data
        """
        return self._config_cache.get(config_name, default)

    def get_nested_config(self, path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation
        
        Args:
            path: Dot-separated path (e.g., 'api.semrush.api_key')
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        parts = path.split('.')
        config = self._config_cache
        
        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default
        
        return config

    def set_config(self, config_name: str, value: Any):
        """
        Set configuration value
        
        Args:
            config_name: Name of the configuration
            value: Configuration value
        """
        self._config_cache[config_name] = value

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all loaded configurations"""
        return self._config_cache.copy()

    def validate_config(self, config_name: str, schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a schema
        
        Args:
            config_name: Name of the configuration to validate
            schema: Validation schema
            
        Returns:
            True if valid, False otherwise
        """
        config = self.get_config(config_name)
        if not config:
            return False
        
        # Simple validation - can be extended with jsonschema
        for key, expected_type in schema.items():
            if key not in config:
                self.logger.warning(f"Missing required config key: {key}")
                return False
            
            if not isinstance(config[key], expected_type):
                self.logger.warning(f"Invalid type for {key}: expected {expected_type}, got {type(config[key])}")
                return False
        
        return True

    def _load_configurations(self):
        """Load configurations from files."""
        try:
            # Load main configuration
            main_config_path = self.config_path / f"{self.environment.value}.yaml"
            if main_config_path.exists():
                with open(main_config_path, 'r') as f:
                    self._config_cache.update(yaml.safe_load(f) or {})

            # Load environment-specific overrides
            env_config_path = self.config_path / f"{self.environment.value}_override.yaml"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_overrides = yaml.safe_load(f) or {}
                    self._merge_configs(self._config_cache, env_overrides)

            # Load secrets from environment variables
            self._load_secrets_from_env()

        except Exception as e:
            self.logger.error(f"Error loading configurations: {str(e)}")

    def _load_default_configs(self):
        """Load default configuration files"""
        config_files = [
            'constants.py',
            'api.yaml',
            'database.yaml',
            'paths.yaml',
            'pipeline.yaml',
            'processing.yaml'
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                try:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        self._load_yaml_config(config_path)
                    elif config_file.endswith('.json'):
                        self._load_json_config(config_path)
                except Exception as e:
                    self.logger.warning(f"Failed to load config {config_file}: {str(e)}")

    def _load_yaml_config(self, config_path: Path):
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
                config_name = config_path.stem
                
                # Merge with existing config if it exists
                if config_name in self._config_cache:
                    self._merge_configs(self._config_cache[config_name], config_data)
                else:
                    self._config_cache[config_name] = config_data
                    
                self.logger.info(f"Loaded configuration: {config_name}")
        except Exception as e:
            self.logger.error(f"Error loading YAML config {config_path}: {str(e)}")

    def _load_json_config(self, config_path: Path):
        """Load JSON configuration file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                config_name = config_path.stem
                
                # Merge with existing config if it exists
                if config_name in self._config_cache:
                    self._merge_configs(self._config_cache[config_name], config_data)
                else:
                    self._config_cache[config_name] = config_data
                    
                self.logger.info(f"Loaded configuration: {config_name}")
        except Exception as e:
            self.logger.error(f"Error loading JSON config {config_path}: {str(e)}")

    def _get_config_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section with fallbacks."""
        return self._config_cache.get(section, {})

    def _get_secret(self, key: str, default: str = "") -> str:
        """Get secret value from environment or config."""
        # Try environment variable first (uppercase)
        env_key = key.upper()
        env_value = os.getenv(env_key)
        if env_value:
            return env_value

        # Try secrets section in config
        secrets = self._config_cache.get("secrets", {})
        return secrets.get(key, default)

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _load_secrets_from_env(self):
        """Load secrets from environment variables."""
        secrets = {}
        for key, value in os.environ.items():
            if key.endswith('_SECRET') or key.endswith('_PASSWORD') or key.endswith('_KEY'):
                secrets[key.lower()] = value
        
        if secrets:
            self._config_cache['secrets'] = self._config_cache.get('secrets', {})
            self._config_cache['secrets'].update(secrets)

    def _persist_config_update(self, section: str, key: str, value: Any):
        """Persist configuration update to file (optional implementation)."""
        # This could be implemented to save changes back to config files
        # For now, it's a placeholder for future enhancement
        pass

class PathManager:
    """
    Advanced path management for SEO competitive intelligence.
    Handles dynamic path resolution, directory creation,
    and path validation across different environments.
    """

    def __init__(self, base_path: Optional[str] = None, logger: Optional[logging.Logger] = None, config_manager: Optional['ConfigManager'] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = config_manager
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.path_cache = {}
        
        # Load paths from config if available
        if self.config_manager:
            try:
                paths_config = self.config_manager.get_config('paths')
                if paths_config:
                    # Use configured base path if not explicitly provided
                    if not base_path and 'project_root' in paths_config:
                        self.base_path = Path(paths_config['project_root'])
                    # Store configured paths for easy access
                    self._configured_paths = paths_config
            except Exception as e:
                self.logger.warning(f"Could not load paths from config: {str(e)}")
                self._configured_paths = {}
        else:
            self._configured_paths = {}
        
        # Standard directory structure
        self.standard_dirs = {
            'data': 'data',
            'reports': 'reports', 
            'logs': 'logs',
            'config': 'config',
            'cache': 'cache',
            'exports': 'exports',
            'temp': 'temp'
        }


    def get_data_path(self, subfolder: Optional[str] = None) -> Path:
        """Get data directory path with optional subfolder."""
        return self._get_standard_path('data', subfolder)

    def get_reports_path(self, subfolder: Optional[str] = None) -> Path:
        """Get reports directory path with optional subfolder."""
        return self._get_standard_path('reports', subfolder)

    def get_logs_path(self, subfolder: Optional[str] = None) -> Path:
        """Get logs directory path with optional subfolder."""
        return self._get_standard_path('logs', subfolder)

    def get_cache_path(self, subfolder: Optional[str] = None) -> Path:
        """Get cache directory path with optional subfolder."""
        return self._get_standard_path('cache', subfolder)

    def get_exports_path(self, subfolder: Optional[str] = None) -> Path:
        """Get exports directory path with optional subfolder."""
        return self._get_standard_path('exports', subfolder)

    def get_temp_path(self, subfolder: Optional[str] = None) -> Path:
        """Get temporary directory path with optional subfolder."""
        return self._get_standard_path('temp', subfolder)

    def create_dated_folder(self, base_folder: str, date: Optional[datetime] = None) -> Path:
        """
        Create dated folder structure.
        
        Args:
            base_folder: Base folder name
            date: Date for folder (defaults to today)
            
        Returns:
            Path to created dated folder
        """
        try:
            if date is None:
                date = datetime.now()

            dated_path = (
                self.base_path / base_folder /
                str(date.year) /
                f"{date.month:02d}" /
                f"{date.day:02d}"
            )

            dated_path.mkdir(parents=True, exist_ok=True)
            return dated_path

        except Exception as e:
            self.logger.error(f"Error creating dated folder: {str(e)}")
            return self.base_path / base_folder

    def ensure_directory_exists(self, path: Union[str, Path]) -> bool:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            path: Directory path to ensure
            
        Returns:
            True if directory exists or was created
        """
        try:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Error ensuring directory exists: {str(e)}")
            return False

    def get_file_with_timestamp(self, directory: Union[str, Path], filename: str, extension: str = "") -> Path:
        """
        Get file path with timestamp to avoid conflicts.
        
        Args:
            directory: Directory for the file
            filename: Base filename
            extension: File extension
            
        Returns:
            Path with timestamp
        """
        try:
            directory_path = Path(directory)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if extension and not extension.startswith('.'):
                extension = f".{extension}"
            timestamped_filename = f"{filename}_{timestamp}{extension}"
            return directory_path / timestamped_filename

        except Exception as e:
            self.logger.error(f"Error creating timestamped file path: {str(e)}")
            return Path(directory) / f"{filename}{extension}"

    def _get_standard_path(self, dir_type: str, subfolder: Optional[str] = None) -> Path:
        """Get standard directory path with optional subfolder."""
        try:
            cache_key = f"{dir_type}_{subfolder or ''}"
            if cache_key in self.path_cache:
                return self.path_cache[cache_key]

            base_dir = self.base_path / self.standard_dirs.get(dir_type, dir_type)
            if subfolder:
                final_path = base_dir / subfolder
            else:
                final_path = base_dir

            # Ensure directory exists
            final_path.mkdir(parents=True, exist_ok=True)

            # Cache the path
            self.path_cache[cache_key] = final_path
            return final_path

        except Exception as e:
            self.logger.error(f"Error getting standard path: {str(e)}")
            return self.base_path

class EnvironmentManager:
    """
    Environment management for SEO competitive intelligence.
    Handles environment detection, validation, and environment-specific
    configuration and behavior.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.current_environment = self._detect_environment()
        self.environment_configs = self._load_environment_configs()

    def get_current_environment(self) -> Environment:
        """Get current environment."""
        return self.current_environment

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.current_environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.current_environment == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.current_environment == Environment.TESTING

    def get_environment_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable with environment-specific prefixes.
        
        Args:
            key: Variable key
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        try:
            # Try environment-specific variable first
            env_specific_key = f"{self.current_environment.value.upper()}_{key.upper()}"
            value = os.getenv(env_specific_key)
            if value is not None:
                return value

            # Try general variable
            value = os.getenv(key.upper())
            if value is not None:
                return value

            return default

        except Exception as e:
            self.logger.error(f"Error getting environment variable: {str(e)}")
            return default

    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate current environment setup.
        
        Returns:
            Validation results
        """
        try:
            validation_results = {
                'environment': self.current_environment.value,
                'valid': True,
                'issues': [],
                'warnings': []
            }

            # Check required environment variables
            required_vars = self._get_required_environment_variables()
            for var in required_vars:
                if not os.getenv(var):
                    validation_results['issues'].append(f"Missing required environment variable: {var}")
                    validation_results['valid'] = False

            # Check directory permissions
            permission_checks = self._check_directory_permissions()
            validation_results['directory_permissions'] = permission_checks

            if not all(permission_checks.values()):
                validation_results['issues'].append("Directory permission issues detected")
                validation_results['valid'] = False

            # Environment-specific validations
            env_validation = self._validate_environment_specific()
            validation_results.update(env_validation)

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating environment: {str(e)}")
            return {'environment': 'unknown', 'valid': False, 'issues': [str(e)]}

    def _detect_environment(self) -> Environment:
        """Detect current environment."""
        try:
            # Check environment variable
            env_var = os.getenv("ENVIRONMENT", "").lower()
            if env_var in [e.value for e in Environment]:
                return Environment(env_var)

            # Check for common environment indicators
            if os.getenv("PRODUCTION") or os.getenv("PROD"):
                return Environment.PRODUCTION

            if os.getenv("TESTING") or os.getenv("TEST"):
                return Environment.TESTING

            if os.getenv("STAGING"):
                return Environment.STAGING

            # Default to development
            return Environment.DEVELOPMENT

        except Exception as e:
            self.logger.error(f"Error detecting environment: {str(e)}")
            return Environment.DEVELOPMENT

    def _load_environment_configs(self) -> Dict[str, Any]:
        """Load environment-specific configurations."""
        return {}

    def _get_required_environment_variables(self) -> List[str]:
        """Get list of required environment variables for current environment."""
        base_vars = ["ENVIRONMENT"]

        if self.current_environment == Environment.PRODUCTION:
            return base_vars + [
                "DATABASE_PASSWORD",
                "SEMRUSH_API_KEY",
                "GOOGLE_API_KEY"
            ]
        elif self.current_environment == Environment.STAGING:
            return base_vars + [
                "DATABASE_PASSWORD",
                "SEMRUSH_API_KEY"
            ]
        else:
            return base_vars

    def _check_directory_permissions(self) -> Dict[str, bool]:
        """Check directory permissions."""
        directories_to_check = ['data', 'reports', 'logs', 'cache', 'exports']
        permissions = {}

        for directory in directories_to_check:
            try:
                dir_path = Path(directory)
                dir_path.mkdir(exist_ok=True)

                # Test write permission
                test_file = dir_path / "permission_test.tmp"
                test_file.write_text("test")
                test_file.unlink()
                permissions[directory] = True

            except Exception:
                permissions[directory] = False

        return permissions

    def _validate_environment_specific(self) -> Dict[str, Any]:
        """Perform environment-specific validations."""
        return {'environment_specific_checks': 'passed'}
