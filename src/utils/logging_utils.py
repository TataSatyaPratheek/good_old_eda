"""
Logging Utilities for SEO Competitive Intelligence

Advanced logging, performance tracking, and audit trail capabilities
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import time
import functools
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback

@dataclass
class PerformanceMetric:
    """Performance metric data"""
    operation: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class AuditLogEntry:
    """Audit log entry"""
    timestamp: datetime
    user_id: str
    operation: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any]

class LoggerFactory:
    """
    Advanced logger factory for SEO competitive intelligence.
    Creates specialized loggers with appropriate handlers, formatters,
    and configuration for different components and environments.
    """

    _loggers = {}
    _handlers = {}
    _log_level = logging.INFO
    _log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        include_console: bool = True,
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """
        Get or create logger with specified configuration.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            include_console: Whether to include console output
            format_string: Custom format string
            
        Returns:
            Configured logger instance
        """
        try:
            if name in cls._loggers:
                return cls._loggers[name]

            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, level.upper(), logging.INFO))
            
            # Clear existing handlers
            logger.handlers.clear()

            # Default format
            if format_string is None:
                format_string = cls._log_format

            formatter = logging.Formatter(format_string)

            # Console handler
            if include_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            # File handler
            if log_file:
                file_handler = cls._create_file_handler(log_file, formatter)
                logger.addHandler(file_handler)

            # Store logger
            cls._loggers[name] = logger
            return logger

        except Exception as e:
            # Fallback to basic logger
            fallback_logger = logging.getLogger(name)
            fallback_logger.error(f"Error creating logger: {str(e)}")
            return fallback_logger

    @classmethod
    def set_log_level(cls, level: str):
        """Set global log level for all loggers"""
        cls._log_level = getattr(logging, level.upper(), logging.INFO)
        # Update existing loggers
        for logger in cls._loggers.values():
            logger.setLevel(cls._log_level)

    @classmethod
    def get_structured_logger(
        cls,
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO"
    ) -> logging.Logger:
        """
        Get logger with structured (JSON) output.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            
        Returns:
            Logger with JSON formatter
        """
        try:
            logger_name = f"{name}_structured"
            if logger_name in cls._loggers:
                return cls._loggers[logger_name]

            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level.upper(), logging.INFO))
            logger.handlers.clear()

            # JSON formatter
            json_formatter = JSONFormatter()

            # Console handler with JSON format
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(json_formatter)
            logger.addHandler(console_handler)

            # File handler with JSON format
            if log_file:
                file_handler = cls._create_file_handler(log_file, json_formatter)
                logger.addHandler(file_handler)

            cls._loggers[logger_name] = logger
            return logger

        except Exception as e:
            return cls.get_logger(name, level, log_file)

    @classmethod
    def _create_file_handler(
        cls,
        log_file: str,
        formatter: logging.Formatter,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ) -> logging.Handler:
        """Create rotating file handler."""
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            return file_handler

        except Exception as e:
            # Fallback to basic file handler
            return logging.FileHandler(log_file)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'thread': record.thread,
                'process': record.process
            }

            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)

            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                              'filename', 'module', 'lineno', 'funcName', 'created',
                              'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                    log_entry[key] = value

            return json.dumps(log_entry)

        except Exception as e:
            # Fallback to standard formatting
            return super().format(record)

class PerformanceTracker:
    """
    Advanced performance tracking for SEO competitive intelligence.
    Tracks execution time, memory usage, and other performance metrics
    for operations and functions.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or LoggerFactory.get_logger("performance_tracker")
        self.metrics = []
        self.active_operations = {}
        self._lock = threading.Lock()
        # Additional metrics for simpler interface compatibility
        self.simple_metrics: Dict[str, Dict[str, Any]] = {}
        self.active_blocks: Dict[str, float] = {}

    def track_operation(self, operation_name: str):
        """
        Decorator to track operation performance.
        
        Args:
            operation_name: Name of the operation to track
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_tracking(operation_name, func, *args, **kwargs)
            return wrapper
        return decorator

    @contextmanager
    def track_block(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager to track code block performance.
        
        Args:
            operation_name: Name of the operation
            metadata: Additional metadata to track
            
        Yields:
            Performance tracking context
        """
        start_time = datetime.now()
        start_time_float = time.time()
        start_memory = self._get_memory_usage()
        success = True
        error_message = None
        
        # Track for simple metrics compatibility
        self.active_blocks[operation_name] = start_time_float

        try:
            yield

        except Exception as e:
            success = False
            error_message = str(e)
            raise

        finally:
            end_time = datetime.now()
            end_time_float = time.time()
            end_memory = self._get_memory_usage()
            duration = (end_time - start_time).total_seconds()
            execution_time = end_time_float - start_time_float

            # Create comprehensive metric
            metric = PerformanceMetric(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                memory_usage_mb=end_memory - start_memory,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )
            self._record_metric(metric)
            
            # Update simple metrics for compatibility
            if operation_name not in self.simple_metrics:
                self.simple_metrics[operation_name] = {
                    'count': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0,
                    'avg_time': 0
                }
            
            simple_metric = self.simple_metrics[operation_name]
            simple_metric['count'] += 1
            simple_metric['total_time'] += execution_time
            simple_metric['min_time'] = min(simple_metric['min_time'], execution_time)
            simple_metric['max_time'] = max(simple_metric['max_time'], execution_time)
            simple_metric['avg_time'] = simple_metric['total_time'] / simple_metric['count']
            
            self.logger.debug(f"Block '{operation_name}' executed in {execution_time:.3f} seconds")
            
            if operation_name in self.active_blocks:
                del self.active_blocks[operation_name]

    def get_performance_summary(
        self,
        operation_filter: Optional[str] = None,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            operation_filter: Filter by operation name
            time_window_minutes: Time window for analysis
            
        Returns:
            Performance summary
        """
        try:
            with self._lock:
                metrics_to_analyze = self.metrics.copy()

            # Apply filters
            if time_window_minutes:
                cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
                metrics_to_analyze = [
                    m for m in metrics_to_analyze if m.start_time >= cutoff_time
                ]

            if operation_filter:
                metrics_to_analyze = [
                    m for m in metrics_to_analyze if operation_filter in m.operation
                ]

            if not metrics_to_analyze:
                return self.simple_metrics.copy()  # Return simple metrics as fallback

            # Calculate statistics
            durations = [m.duration_seconds for m in metrics_to_analyze]
            memory_usage = [m.memory_usage_mb for m in metrics_to_analyze]
            success_rate = sum(1 for m in metrics_to_analyze if m.success) / len(metrics_to_analyze)

            summary = {
                'total_operations': len(metrics_to_analyze),
                'success_rate': success_rate,
                'duration_stats': {
                    'mean': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'median': sorted(durations)[len(durations) // 2]
                },
                'memory_stats': {
                    'mean': sum(memory_usage) / len(memory_usage),
                    'min': min(memory_usage),
                    'max': max(memory_usage)
                },
                'slowest_operations': self._get_slowest_operations(metrics_to_analyze, 5),
                'failed_operations': [
                    {'operation': m.operation, 'error': m.error_message}
                    for m in metrics_to_analyze if not m.success
                ]
            }
            return summary

        except Exception as e:
            self.logger.error(f"Error generating performance summary: {str(e)}")
            return self.simple_metrics.copy()  # Return simple metrics as fallback

    def get_total_execution_time(self) -> float:
        """Get total execution time across all blocks"""
        return sum(m['total_time'] for m in self.simple_metrics.values())

    def _execute_with_tracking(self, operation_name: str, func: Callable, *args, **kwargs):
        """Execute function with performance tracking."""
        start_time = datetime.now()
        start_memory = self._get_memory_usage()
        success = True
        error_message = None
        result = None

        try:
            result = func(*args, **kwargs)
            return result

        except Exception as e:
            success = False
            error_message = str(e)
            raise

        finally:
            end_time = datetime.now()
            end_memory = self._get_memory_usage()
            duration = (end_time - start_time).total_seconds()

            metric = PerformanceMetric(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                memory_usage_mb=end_memory - start_memory,
                success=success,
                error_message=error_message,
                metadata={'function_name': func.__name__}
            )
            self._record_metric(metric)

    def _record_metric(self, metric: PerformanceMetric):
        """Record performance metric."""
        try:
            with self._lock:
                self.metrics.append(metric)
                # Keep only recent metrics to prevent memory bloat
                if len(self.metrics) > 10000:
                    self.metrics = self.metrics[-5000:]

            # Log performance metric
            self.logger.info(
                f"Performance: {metric.operation} - "
                f"Duration: {metric.duration_seconds:.3f}s - "
                f"Memory: {metric.memory_usage_mb:.2f}MB - "
                f"Success: {metric.success}"
            )

        except Exception as e:
            self.logger.error(f"Error recording metric: {str(e)}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_slowest_operations(self, metrics: List[PerformanceMetric], count: int) -> List[Dict[str, Any]]:
        """Get slowest operations from metrics."""
        try:
            sorted_metrics = sorted(metrics, key=lambda m: m.duration_seconds, reverse=True)
            return [
                {
                    'operation': m.operation,
                    'duration_seconds': m.duration_seconds,
                    'timestamp': m.start_time.isoformat()
                }
                for m in sorted_metrics[:count]
            ]
        except Exception:
            return []

class AuditLogger:
    """
    Advanced audit logging for SEO competitive intelligence.
    Provides comprehensive audit trail capabilities for compliance,
    security, and operational monitoring.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, audit_file: Optional[str] = None):
        self.logger = logger or LoggerFactory.get_structured_logger("audit_logger")
        self.audit_entries = []
        self._lock = threading.Lock()
        self.audit_file = audit_file or "audit_trail.jsonl"
        self.audit_path = Path(self.audit_file)
        
        # Ensure audit directory exists
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log data access event.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            result: Result of the action
            details: Additional details
        """
        self._log_audit_event("data_access", user_id, resource, action, result, details or {})

    def log_analysis_execution(
        self,
        user_id: str,
        analysis_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log analysis execution event.
        
        Args:
            user_id: User identifier
            analysis_type: Type of analysis
            parameters: Analysis parameters
            result: Result of the analysis
            details: Additional details
        """
        # Enhanced parameters handling
        if isinstance(analysis_type, str):
            audit_details = {
                'analysis_type': analysis_type,
                'parameters': parameters or {}
            }
        else:
            # Backward compatibility
            audit_details = {
                'analysis_type': str(analysis_type),
                'parameters': parameters or {}
            }
        
        audit_details.update(details or {})
        
        # Write to file for persistence
        audit_entry_file = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'analysis_type': analysis_type,
            'parameters': parameters or {},
            'result': result,
            'details': details or {}
        }
        
        try:
            with open(self.audit_path, 'a') as f:
                f.write(json.dumps(audit_entry_file) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {str(e)}")
        
        self._log_audit_event("analysis_execution", user_id, analysis_type, "execute", result, audit_details)

    def log_configuration_change(
        self,
        user_id: str,
        configuration_section: str,
        old_value: Any,
        new_value: Any,
        result: str = "success"
    ):
        """
        Log configuration change event.
        
        Args:
            user_id: User identifier
            configuration_section: Configuration section changed
            old_value: Previous value
            new_value: New value
            result: Result of the change
        """
        details = {
            'old_value': str(old_value),
            'new_value': str(new_value),
            'section': configuration_section
        }
        self._log_audit_event("configuration_change", user_id, configuration_section, "modify", result, details)

    def log_export_event(
        self,
        user_id: str,
        export_type: str,
        data_scope: str,
        destination: str,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log data export event.
        
        Args:
            user_id: User identifier
            export_type: Type of export
            data_scope: Scope of exported data
            destination: Export destination
            result: Result of the export
            details: Additional details
        """
        audit_details = {
            'export_type': export_type,
            'data_scope': data_scope,
            'destination': destination
        }
        audit_details.update(details or {})
        self._log_audit_event("data_export", user_id, export_type, "export", result, audit_details)

    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        time_window_hours: int = 24,
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail entries from both memory and file.
        
        Args:
            user_id: Filter by user ID
            operation: Filter by operation type
            time_window_hours: Time window for entries
            start_date: Optional start date filter
            
        Returns:
            List of audit log entries
        """
        entries = []
        
        # Get entries from file first
        if self.audit_path.exists():
            try:
                with open(self.audit_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_date = datetime.fromisoformat(entry['timestamp'])
                            
                            if start_date is None or entry_date >= start_date:
                                entries.append(entry)
                        except:
                            continue
            except Exception as e:
                self.logger.error(f"Failed to read audit trail: {str(e)}")
        
        # Get entries from memory
        try:
            with self._lock:
                memory_entries = self.audit_entries.copy()

            # Apply time filter
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            memory_entries = [e for e in memory_entries if e.timestamp >= cutoff_time]

            # Apply user filter
            if user_id:
                memory_entries = [e for e in memory_entries if e.user_id == user_id]

            # Apply operation filter
            if operation:
                memory_entries = [e for e in memory_entries if e.operation == operation]

            # Convert to dict format and add to entries
            for entry in memory_entries:
                entries.append(asdict(entry))

            return sorted(entries, key=lambda e: e.get('timestamp', ''), reverse=True)

        except Exception as e:
            self.logger.error(f"Error retrieving audit trail: {str(e)}")
            return entries

    def _log_audit_event(
        self,
        operation: str,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        details: Dict[str, Any]
    ):
        """Log audit event."""
        try:
            audit_entry = AuditLogEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                operation=operation,
                resource=resource,
                action=action,
                result=result,
                details=details
            )

            with self._lock:
                self.audit_entries.append(audit_entry)
                # Keep only recent entries
                if len(self.audit_entries) > 50000:
                    self.audit_entries = self.audit_entries[-25000:]

            # Log to structured logger
            self.logger.info(
                "Audit Event",
                extra={
                    'audit_operation': operation,
                    'audit_user_id': user_id,
                    'audit_resource': resource,
                    'audit_action': action,
                    'audit_result': result,
                    'audit_details': details
                }
            )

        except Exception as e:
            self.logger.error(f"Error logging audit event: {str(e)}")
