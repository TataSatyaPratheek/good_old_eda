"""
Common Helper Utilities for SEO Competitive Intelligence
Frequently used utility functions, string helpers, date utilities, and caching mechanisms
"""

import re
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta, timezone
from functools import wraps, lru_cache
import logging
from dataclasses import dataclass
import json
import threading
from collections import defaultdict, OrderedDict
import gc

class StringHelper:
    """
    Advanced string manipulation utilities for SEO competitive intelligence.
    
    Provides comprehensive string processing capabilities optimized
    for SEO data cleaning and analysis.
    """
    
    @staticmethod
    def clean_keyword(keyword: str) -> str:
        """
        Clean and normalize keyword string.
        
        Args:
            keyword: Raw keyword string
            
        Returns:
            Cleaned keyword string
        """
        try:
            if not isinstance(keyword, str):
                keyword = str(keyword)
            
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', keyword.strip())
            
            # Remove special characters (keep only alphanumeric, spaces, hyphens, apostrophes)
            cleaned = re.sub(r'[^\w\s\-\'\"]', ' ', cleaned)
            
            # Normalize quotation marks
            cleaned = re.sub(r'[""''`]', '"', cleaned)
            
            # Remove multiple consecutive spaces again
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            return cleaned.lower()
            
        except Exception:
            return ""

    @staticmethod
    def extract_domain_from_url(url: str) -> Optional[str]:
        """
        Extract domain from URL string.
        
        Args:
            url: URL string
            
        Returns:
            Domain string or None if invalid
        """
        try:
            if not url or not isinstance(url, str):
                return None
            
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Extract domain using regex
            pattern = r'https?://(?:www\.)?([^/\?#]+)'
            match = re.search(pattern, url.lower())
            
            if match:
                domain = match.group(1)
                # Remove port if present
                domain = domain.split(':')[0]
                return domain
            
            return None
            
        except Exception:
            return None

    @staticmethod
    def normalize_serp_features(features_string: str) -> List[str]:
        """
        Normalize SERP features string into standardized list.
        
        Args:
            features_string: Raw SERP features string
            
        Returns:
            List of normalized feature names
        """
        try:
            if not features_string or pd.isna(features_string):
                return []
            
            # Split by common delimiters
            features = re.split(r'[,;|]', str(features_string))
            
            # Clean and normalize each feature
            normalized_features = []
            feature_mappings = {
                'featured snippet': 'featured_snippet',
                'people also ask': 'people_also_ask',
                'knowledge panel': 'knowledge_panel',
                'image pack': 'image_pack',
                'video': 'video_carousel',
                'shopping': 'shopping_results',
                'ads': 'ads_top',
                'related searches': 'related_searches',
                'site links': 'sitelinks',
                'reviews': 'reviews'
            }
            
            for feature in features:
                feature_clean = feature.strip().lower()
                
                # Map common variations to standard names
                for pattern, standard_name in feature_mappings.items():
                    if pattern in feature_clean:
                        if standard_name not in normalized_features:
                            normalized_features.append(standard_name)
                        break
                else:
                    # If no mapping found, use cleaned version
                    if feature_clean and feature_clean not in normalized_features:
                        normalized_features.append(feature_clean.replace(' ', '_'))
            
            return sorted(normalized_features)
            
        except Exception:
            return []

    @staticmethod
    def calculate_keyword_similarity(keyword1: str, keyword2: str) -> float:
        """
        Calculate similarity between two keywords using multiple methods.
        
        Args:
            keyword1: First keyword
            keyword2: Second keyword
            
        Returns:
            Similarity score (0-1)
        """
        try:
            if not keyword1 or not keyword2:
                return 0.0
            
            # Clean keywords
            k1_clean = StringHelper.clean_keyword(keyword1)
            k2_clean = StringHelper.clean_keyword(keyword2)
            
            if k1_clean == k2_clean:
                return 1.0
            
            # Word-level Jaccard similarity
            words1 = set(k1_clean.split())
            words2 = set(k2_clean.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_similarity = intersection / union if union > 0 else 0
            
            # Character-level similarity (Levenshtein-based)
            char_similarity = StringHelper._calculate_levenshtein_similarity(k1_clean, k2_clean)
            
            # Combined similarity (weighted average)
            combined_similarity = (jaccard_similarity * 0.7) + (char_similarity * 0.3)
            
            return combined_similarity
            
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_levenshtein_similarity(s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity between two strings."""
        try:
            if len(s1) == 0 or len(s2) == 0:
                return 0.0
            
            # Dynamic programming approach for Levenshtein distance
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Initialize base cases
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            # Fill the dp table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            # Convert distance to similarity
            max_length = max(len(s1), len(s2))
            distance = dp[m][n]
            similarity = 1 - (distance / max_length) if max_length > 0 else 0
            
            return similarity
            
        except Exception:
            return 0.0

    @staticmethod
    def extract_brand_keywords(keywords: List[str], brand_terms: List[str]) -> Tuple[List[str], List[str]]:
        """
        Separate branded and non-branded keywords.
        
        Args:
            keywords: List of keywords
            brand_terms: List of brand terms to identify
            
        Returns:
            Tuple of (branded_keywords, non_branded_keywords)
        """
        try:
            branded = []
            non_branded = []
            
            brand_patterns = [re.compile(f'\\b{term.lower()}\\b') for term in brand_terms]
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                is_branded = any(pattern.search(keyword_lower) for pattern in brand_patterns)
                
                if is_branded:
                    branded.append(keyword)
                else:
                    non_branded.append(keyword)
            
            return branded, non_branded
            
        except Exception:
            return [], keywords


class DateHelper:
    """
    Advanced date and time utilities for SEO competitive intelligence.
    
    Provides comprehensive date handling capabilities including
    timezone management, date parsing, and business date calculations.
    """
    
    @staticmethod
    def parse_flexible_date(date_input: Union[str, datetime, int]) -> Optional[datetime]:
        """
        Parse date from various input formats.
        
        Args:
            date_input: Date in various formats
            
        Returns:
            Parsed datetime object or None if invalid
        """
        try:
            if isinstance(date_input, datetime):
                return date_input
            
            if isinstance(date_input, int):
                # Assume Unix timestamp
                return datetime.fromtimestamp(date_input)
            
            if not isinstance(date_input, str):
                return None
            
            # Common date formats to try
            date_formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S.%f',
                '%b %d, %Y',
                '%B %d, %Y',
                '%d %b %Y',
                '%d %B %Y'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_input.strip(), fmt)
                except ValueError:
                    continue
            
            # Try pandas to_datetime as fallback
            try:
                import pandas as pd
                return pd.to_datetime(date_input).to_pydatetime()
            except:
                pass
            
            return None
            
        except Exception:
            return None

    @staticmethod
    def get_date_range(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        frequency: str = 'D'
    ) -> List[datetime]:
        """
        Generate list of dates between start and end dates.
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Frequency ('D', 'W', 'M' for daily, weekly, monthly)
            
        Returns:
            List of datetime objects
        """
        try:
            start_dt = DateHelper.parse_flexible_date(start_date)
            end_dt = DateHelper.parse_flexible_date(end_date)
            
            if not start_dt or not end_dt:
                return []
            
            dates = []
            current_date = start_dt
            
            if frequency == 'D':
                delta = timedelta(days=1)
            elif frequency == 'W':
                delta = timedelta(weeks=1)
            elif frequency == 'M':
                delta = timedelta(days=30)  # Approximate month
            else:
                delta = timedelta(days=1)
            
            while current_date <= end_dt:
                dates.append(current_date)
                current_date += delta
            
            return dates
            
        except Exception:
            return []

    @staticmethod
    def get_business_days_between(
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> int:
        """
        Calculate number of business days between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of business days
        """
        try:
            start_dt = DateHelper.parse_flexible_date(start_date)
            end_dt = DateHelper.parse_flexible_date(end_date)
            
            if not start_dt or not end_dt:
                return 0
            
            business_days = 0
            current_date = start_dt
            
            while current_date <= end_dt:
                # Monday = 0, Sunday = 6
                if current_date.weekday() < 5:  # Monday to Friday
                    business_days += 1
                current_date += timedelta(days=1)
            
            return business_days
            
        except Exception:
            return 0

    @staticmethod
    def format_relative_time(target_date: datetime, reference_date: Optional[datetime] = None) -> str:
        """
        Format date as relative time string.
        
        Args:
            target_date: Target date to format
            reference_date: Reference date (defaults to now)
            
        Returns:
            Relative time string
        """
        try:
            if reference_date is None:
                reference_date = datetime.now()
            
            delta = reference_date - target_date
            
            if delta.days > 365:
                years = delta.days // 365
                return f"{years} year{'s' if years > 1 else ''} ago"
            elif delta.days > 30:
                months = delta.days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"
            elif delta.days > 7:
                weeks = delta.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            elif delta.days > 0:
                return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
            elif delta.seconds > 3600:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif delta.seconds > 60:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            else:
                return "Just now"
                
        except Exception:
            return "Unknown"

    @staticmethod
    def get_week_boundaries(date_input: Union[str, datetime]) -> Tuple[datetime, datetime]:
        """
        Get start and end of week for given date.
        
        Args:
            date_input: Input date
            
        Returns:
            Tuple of (week_start, week_end)
        """
        try:
            target_date = DateHelper.parse_flexible_date(date_input)
            if not target_date:
                return None, None
            
            # Get Monday of the week (start of week)
            days_since_monday = target_date.weekday()
            week_start = target_date - timedelta(days=days_since_monday)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get Sunday of the week (end of week)
            week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            
            return week_start, week_end
            
        except Exception:
            return None, None


class CacheManager:
    """
    Advanced caching utilities for SEO competitive intelligence.
    
    Provides flexible caching mechanisms with TTL support,
    memory management, and cache statistics.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.access_times = {}
        self.creation_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with TTL check.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        try:
            with self.lock:
                current_time = time.time()
                
                if key in self.cache:
                    # Check TTL
                    if current_time - self.creation_times[key] < self.default_ttl:
                        # Update access time and move to end (LRU)
                        self.access_times[key] = current_time
                        self.cache.move_to_end(key)
                        self.hit_count += 1
                        return self.cache[key]
                    else:
                        # Expired, remove from cache
                        self._remove_key(key)
                
                self.miss_count += 1
                return default
                
        except Exception:
            self.miss_count += 1
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successfully cached
        """
        try:
            with self.lock:
                current_time = time.time()
                
                # Remove expired entries if at max size
                if len(self.cache) >= self.max_size:
                    self._cleanup_expired()
                    
                    # If still at max size, remove oldest
                    if len(self.cache) >= self.max_size:
                        oldest_key = next(iter(self.cache))
                        self._remove_key(oldest_key)
                
                # Add/update cache entry
                self.cache[key] = value
                self.access_times[key] = current_time
                self.creation_times[key] = current_time
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                return True
                
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        try:
            with self.lock:
                if key in self.cache:
                    self._remove_key(key)
                    return True
                return False
                
        except Exception:
            return False

    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if cache was cleared
        """
        try:
            with self.lock:
                self.cache.clear()
                self.access_times.clear()
                self.creation_times.clear()
                self.hit_count = 0
                self.miss_count = 0
                return True
                
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        try:
            with self.lock:
                total_requests = self.hit_count + self.miss_count
                hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
                
                return {
                    'size': len(self.cache),
                    'max_size': self.max_size,
                    'hit_count': self.hit_count,
                    'miss_count': self.miss_count,
                    'hit_rate': hit_rate,
                    'total_requests': total_requests
                }
                
        except Exception:
            return {}

    def cache_function(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live for cached results
            key_prefix: Prefix for cache keys
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                cache_key = self._generate_function_cache_key(
                    func.__name__, args, kwargs, key_prefix
                )
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl or self.default_ttl)
                
                return result
            
            return wrapper
        return decorator

    def _remove_key(self, key: str):
        """Remove key and associated metadata."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.creation_times:
            del self.creation_times[key]

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, creation_time in self.creation_times.items():
            if current_time - creation_time >= self.default_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_key(key)

    def _generate_function_cache_key(
        self,
        func_name: str,
        args: Tuple,
        kwargs: Dict,
        prefix: str
    ) -> str:
        """Generate cache key for function call."""
        try:
            # Create a string representation of arguments
            args_str = str(args) + str(sorted(kwargs.items()))
            
            # Hash the arguments to create a consistent key
            args_hash = hashlib.md5(args_str.encode()).hexdigest()
            
            return f"{prefix}{func_name}_{args_hash}"
            
        except Exception:
            # Fallback to simple key
            return f"{prefix}{func_name}_{len(args)}_{len(kwargs)}"


# Global cache instance
global_cache = CacheManager(max_size=5000, default_ttl=3600)


def memoize(ttl: int = 3600, cache_instance: Optional[CacheManager] = None):
    """
    Memoization decorator using cache manager.
    
    Args:
        ttl: Time to live for cached results
        cache_instance: Specific cache instance to use
        
    Returns:
        Decorator function
    """
    cache = cache_instance or global_cache
    return cache.cache_function(ttl=ttl)


def timing_decorator(logger: Optional[logging.Logger] = None):
    """
    Decorator to measure function execution time.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if logger:
                    logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                
                if logger:
                    logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}")
                
                raise
        
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff_factor: Backoff multiplier
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
        return wrapper
    return decorator


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """
    Safely divide two numbers with default for division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value for division by zero
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default


def ensure_list(value: Any) -> List[Any]:
    """
    Ensure value is a list.
    
    Args:
        value: Input value
        
    Returns:
        List containing the value(s)
    """
    if value is None:
        return []
    elif isinstance(value, list):
        return value
    elif isinstance(value, (tuple, set)):
        return list(value)
    else:
        return [value]


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    try:
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
        
    except Exception:
        return dict1.copy()


def memory_efficient_chunk_processor(
    data: List[Any],
    chunk_size: int = 1000,
    processor_func: Callable = None
) -> List[Any]:
    """
    Process large datasets in memory-efficient chunks.
    
    Args:
        data: Large dataset to process
        chunk_size: Size of each chunk
        processor_func: Function to process each chunk
        
    Returns:
        List of processed results
    """
    try:
        if processor_func is None:
            processor_func = lambda x: x
        
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            processed_chunk = processor_func(chunk)
            results.extend(ensure_list(processed_chunk))
            
            # Force garbage collection after each chunk
            gc.collect()
        
        return results
        
    except Exception:
        return []
