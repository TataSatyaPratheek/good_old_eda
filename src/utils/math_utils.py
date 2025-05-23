"""
Mathematical Utilities for SEO Competitive Intelligence
Advanced statistical calculations, optimization helpers, and time series analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from datetime import datetime, timedelta
from scipy import stats, optimize, signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class StatisticalCalculator:
    """
    Advanced statistical calculations for SEO competitive intelligence.
    
    Provides comprehensive statistical analysis including descriptive statistics,
    hypothesis testing, and advanced statistical modeling.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_descriptive_statistics(
        self,
        data: Union[pd.Series, np.ndarray, List[float]],
        include_advanced: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            data: Input data
            include_advanced: Whether to include advanced statistics
            
        Returns:
            Dictionary of statistical measures
        """
        try:
            if isinstance(data, list):
                data = np.array(data)
            elif isinstance(data, pd.Series):
                data = data.dropna().values
            
            if len(data) == 0:
                return {}
            
            # Basic statistics
            stats_dict = {
                'count': len(data),
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data, ddof=1),
                'variance': np.var(data, ddof=1),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'iqr': np.percentile(data, 75) - np.percentile(data, 25)
            }
            
            if include_advanced:
                # Advanced statistics
                stats_dict.update({
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'coefficient_of_variation': stats_dict['std'] / abs(stats_dict['mean']) if stats_dict['mean'] != 0 else np.inf,
                    'mad': np.median(np.abs(data - stats_dict['median'])),  # Median Absolute Deviation
                    'harmonic_mean': stats.hmean(data[data > 0]) if np.any(data > 0) else 0,
                    'geometric_mean': stats.gmean(data[data > 0]) if np.any(data > 0) else 0,
                    'trim_mean_10': stats.trim_mean(data, 0.1),
                    'mode': stats.mode(data, keepdims=True).mode[0] if len(data) > 0 else np.nan,
                    'entropy': self._calculate_entropy(data)
                })
            
            return stats_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating descriptive statistics: {str(e)}")
            return {}

    def perform_normality_tests(
        self,
        data: Union[pd.Series, np.ndarray],
        alpha: float = 0.05
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform multiple normality tests.
        
        Args:
            data: Input data
            alpha: Significance level
            
        Returns:
            Dictionary of test results
        """
        try:
            if isinstance(data, pd.Series):
                data = data.dropna().values
            
            if len(data) < 8:
                return {'error': 'Insufficient data for normality tests'}
            
            results = {}
            
            # Shapiro-Wilk test
            if len(data) <= 5000:  # Shapiro-Wilk has sample size limitations
                shapiro_stat, shapiro_p = stats.shapiro(data)
                results['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > alpha
                }
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > alpha
            }
            
            # Anderson-Darling test
            ad_result = stats.anderson(data, dist='norm')
            results['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_level.tolist()
            }
            
            # D'Agostino-Pearson omnibus test
            if len(data) >= 20:
                dp_stat, dp_p = stats.normaltest(data)
                results['dagostino_pearson'] = {
                    'statistic': dp_stat,
                    'p_value': dp_p,
                    'is_normal': dp_p > alpha
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing normality tests: {str(e)}")
            return {}

    def calculate_correlation_matrix(
        self,
        data: pd.DataFrame,
        method: str = 'pearson',
        min_periods: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation matrix with significance testing.
        
        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_periods: Minimum periods for correlation calculation
            
        Returns:
            Tuple of (correlation matrix, p-values matrix)
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Calculate correlations
            corr_matrix = numeric_data.corr(method=method, min_periods=min_periods)
            
            # Calculate p-values
            p_values = pd.DataFrame(
                index=corr_matrix.index,
                columns=corr_matrix.columns,
                dtype=float
            )
            
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i <= j:
                        if i == j:
                            p_values.iloc[i, j] = 0.0
                        else:
                            data1 = numeric_data[col1].dropna()
                            data2 = numeric_data[col2].dropna()
                            
                            # Find common indices
                            common_idx = data1.index.intersection(data2.index)
                            if len(common_idx) >= min_periods:
                                if method == 'pearson':
                                    _, p_val = stats.pearsonr(data1[common_idx], data2[common_idx])
                                elif method == 'spearman':
                                    _, p_val = stats.spearmanr(data1[common_idx], data2[common_idx])
                                elif method == 'kendall':
                                    _, p_val = stats.kendalltau(data1[common_idx], data2[common_idx])
                                else:
                                    p_val = np.nan
                                
                                p_values.iloc[i, j] = p_val
                                p_values.iloc[j, i] = p_val
                            else:
                                p_values.iloc[i, j] = np.nan
                                p_values.iloc[j, i] = np.nan
            
            return corr_matrix, p_values
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def perform_hypothesis_test(
        self,
        sample1: Union[pd.Series, np.ndarray],
        sample2: Optional[Union[pd.Series, np.ndarray]] = None,
        test_type: str = 'ttest',
        alternative: str = 'two-sided',
        alpha: float = 0.05,
        expected_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform various hypothesis tests.
        
        Args:
            sample1: First sample
            sample2: Second sample (for two-sample tests)
            test_type: Type of test ('ttest', 'mannwhitney', 'wilcoxon', 'chi2')
            alternative: Alternative hypothesis
            alpha: Significance level
            expected_value: Expected value for one-sample tests
            
        Returns:
            Test results
        """
        try:
            if isinstance(sample1, pd.Series):
                sample1 = sample1.dropna().values
            if sample2 is not None and isinstance(sample2, pd.Series):
                sample2 = sample2.dropna().values
            
            results = {
                'test_type': test_type,
                'alternative': alternative,
                'alpha': alpha
            }
            
            if test_type == 'ttest':
                if sample2 is None:
                    # One-sample t-test
                    if expected_value is None:
                        expected_value = 0
                    statistic, p_value = stats.ttest_1samp(sample1, expected_value)
                else:
                    # Two-sample t-test
                    statistic, p_value = stats.ttest_ind(sample1, sample2)
                
                results.update({
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < alpha
                })
            
            elif test_type == 'mannwhitney':
                if sample2 is None:
                    raise ValueError("Mann-Whitney test requires two samples")
                
                statistic, p_value = stats.mannwhitneyu(
                    sample1, sample2, alternative=alternative
                )
                
                results.update({
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < alpha
                })
            
            elif test_type == 'wilcoxon':
                if sample2 is None:
                    # One-sample Wilcoxon signed-rank test
                    statistic, p_value = stats.wilcoxon(sample1)
                else:
                    # Paired Wilcoxon signed-rank test
                    statistic, p_value = stats.wilcoxon(sample1, sample2)
                
                results.update({
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < alpha
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing hypothesis test: {str(e)}")
            return {'error': str(e)}

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        try:
            # Bin the data for entropy calculation
            hist, _ = np.histogram(data, bins=10)
            hist = hist[hist > 0]  # Remove zero bins
            probabilities = hist / hist.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy
        except Exception:
            return 0.0


class OptimizationHelper:
    """
    Advanced optimization utilities for SEO competitive intelligence.
    
    Provides mathematical optimization capabilities for traffic optimization,
    resource allocation, and strategic planning.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def optimize_traffic_allocation(
        self,
        keyword_data: pd.DataFrame,
        budget_constraint: float,
        effort_function: Optional[Callable] = None,
        traffic_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Optimize traffic allocation across keywords using mathematical optimization.
        
        Args:
            keyword_data: DataFrame with keyword metrics
            budget_constraint: Total budget constraint
            effort_function: Function to calculate effort required
            traffic_function: Function to calculate traffic potential
            
        Returns:
            Optimization results
        """
        try:
            if effort_function is None:
                effort_function = self._default_effort_function
            
            if traffic_function is None:
                traffic_function = self._default_traffic_function
            
            # Prepare optimization variables
            keywords = keyword_data['Keyword'].tolist()
            n_keywords = len(keywords)
            
            # Define objective function (maximize traffic)
            def objective(x):
                total_traffic = 0
                for i, keyword in enumerate(keywords):
                    row = keyword_data.iloc[i]
                    traffic_gain = traffic_function(x[i], row)
                    total_traffic += traffic_gain
                return -total_traffic  # Negative for minimization
            
            # Define constraint function (budget constraint)
            def constraint(x):
                total_effort = 0
                for i, keyword in enumerate(keywords):
                    row = keyword_data.iloc[i]
                    effort = effort_function(x[i], row)
                    total_effort += effort
                return budget_constraint - total_effort
            
            # Set bounds (0 to max improvement possible)
            bounds = []
            for i, keyword in enumerate(keywords):
                row = keyword_data.iloc[i]
                max_improvement = min(row.get('Position', 100) - 1, 20)  # Max 20 position improvement
                bounds.append((0, max_improvement))
            
            # Initial guess
            x0 = [5.0] * n_keywords  # Start with 5 position improvement for all
            
            # Define constraints
            constraints = {'type': 'ineq', 'fun': constraint}
            
            # Perform optimization
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Process results
            optimization_results = {
                'success': result.success,
                'optimal_improvements': result.x.tolist(),
                'total_traffic_gain': -result.fun,
                'budget_utilization': sum(
                    effort_function(result.x[i], keyword_data.iloc[i])
                    for i in range(n_keywords)
                ),
                'keyword_allocations': {
                    keywords[i]: {
                        'improvement': result.x[i],
                        'effort': effort_function(result.x[i], keyword_data.iloc[i]),
                        'traffic_gain': traffic_function(result.x[i], keyword_data.iloc[i])
                    }
                    for i in range(n_keywords)
                }
            }
            
            self.logger.info(f"Traffic allocation optimization completed. Success: {result.success}")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error in traffic allocation optimization: {str(e)}")
            return {}

    def find_optimal_bid_strategy(
        self,
        performance_data: pd.DataFrame,
        target_metric: str = 'traffic',
        constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal bidding strategy for keyword portfolios.
        
        Args:
            performance_data: Historical performance data
            target_metric: Target metric to optimize
            constraints: Budget and other constraints
            
        Returns:
            Optimal bidding strategy
        """
        try:
            if constraints is None:
                constraints = {'max_cpc': 10.0, 'total_budget': 10000.0}
            
            # Prepare data
            keywords = performance_data['Keyword'].unique()
            
            # Define optimization problem
            def objective(bids):
                total_value = 0
                for i, keyword in enumerate(keywords):
                    keyword_data = performance_data[performance_data['Keyword'] == keyword]
                    if not keyword_data.empty:
                        # Estimate performance based on bid
                        estimated_performance = self._estimate_performance_from_bid(
                            bids[i], keyword_data.iloc[0]
                        )
                        total_value += estimated_performance.get(target_metric, 0)
                return -total_value  # Negative for minimization
            
            # Budget constraint
            def budget_constraint(bids):
                total_spend = sum(bids) * 30  # Monthly spend estimate
                return constraints['total_budget'] - total_spend
            
            # Set bounds
            bounds = [(0, constraints.get('max_cpc', 10.0)) for _ in keywords]
            
            # Initial guess
            x0 = [1.0] * len(keywords)
            
            # Define constraints
            constraint_defs = [{'type': 'ineq', 'fun': budget_constraint}]
            
            # Optimize
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_defs
            )
            
            # Process results
            bid_strategy = {
                'success': result.success,
                'optimal_bids': {
                    keywords[i]: result.x[i] for i in range(len(keywords))
                },
                'expected_performance': -result.fun,
                'total_budget_used': sum(result.x) * 30
            }
            
            return bid_strategy
            
        except Exception as e:
            self.logger.error(f"Error finding optimal bid strategy: {str(e)}")
            return {}

    def solve_portfolio_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_tolerance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Solve portfolio optimization problem for keyword investments.
        
        Args:
            expected_returns: Expected returns for each keyword
            covariance_matrix: Covariance matrix of returns
            risk_tolerance: Risk tolerance parameter
            
        Returns:
            Optimal portfolio weights
        """
        try:
            n_assets = len(expected_returns)
            
            # Objective function (maximize utility = return - risk_penalty * risk)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                utility = portfolio_return - risk_tolerance * portfolio_risk
                return -utility  # Negative for minimization
            
            # Constraint: weights sum to 1
            def weight_constraint(weights):
                return np.sum(weights) - 1.0
            
            # Bounds: weights between 0 and 1
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            x0 = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': weight_constraint}]
            
            # Optimize
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Calculate portfolio metrics
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            portfolio_results = {
                'success': result.success,
                'optimal_weights': optimal_weights.tolist(),
                'expected_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'utility': -result.fun
            }
            
            return portfolio_results
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            return {}

    def _default_effort_function(self, improvement: float, keyword_row: pd.Series) -> float:
        """Default effort calculation function."""
        base_effort = improvement * 10
        difficulty_multiplier = (keyword_row.get('Keyword Difficulty', 50) / 100) + 1
        volume_multiplier = np.log10(max(keyword_row.get('Search Volume', 100), 100)) / 5
        return base_effort * difficulty_multiplier * volume_multiplier

    def _default_traffic_function(self, improvement: float, keyword_row: pd.Series) -> float:
        """Default traffic calculation function."""
        current_position = keyword_row.get('Position', 50)
        new_position = max(1, current_position - improvement)
        search_volume = keyword_row.get('Search Volume', 100)
        
        # CTR model
        ctr_rates = {1: 0.284, 2: 0.147, 3: 0.094, 4: 0.067, 5: 0.051,
                    6: 0.041, 7: 0.034, 8: 0.029, 9: 0.025, 10: 0.022}
        
        current_ctr = ctr_rates.get(min(current_position, 10), 0.01)
        new_ctr = ctr_rates.get(min(new_position, 10), 0.01)
        
        traffic_gain = search_volume * (new_ctr - current_ctr)
        return max(0, traffic_gain)

    def _estimate_performance_from_bid(self, bid: float, keyword_data: pd.Series) -> Dict[str, float]:
        """Estimate performance metrics from bid amount."""
        # Simplified model - in practice, this would be more sophisticated
        base_traffic = keyword_data.get('Traffic', 0)
        search_volume = keyword_data.get('Search Volume', 100)
        competition = keyword_data.get('Competition', 0.5)
        
        # Estimate position improvement from bid
        bid_factor = min(bid / max(keyword_data.get('CPC', 1), 0.1), 3)
        position_improvement = bid_factor * (1 - competition) * 5
        
        # Estimate traffic improvement
        traffic_multiplier = 1 + (position_improvement / 20)
        estimated_traffic = base_traffic * traffic_multiplier
        
        return {
            'traffic': estimated_traffic,
            'position_improvement': position_improvement,
            'estimated_cost': bid * search_volume * 0.01  # Simplified cost model
        }


class TimeSeriesAnalyzer:
    """
    Advanced time series analysis for SEO competitive intelligence.
    
    Provides comprehensive time series analysis including trend detection,
    seasonality analysis, and forecasting capabilities.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def decompose_time_series(
        self,
        data: pd.Series,
        period: int = 7,
        model: str = 'additive'
    ) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            data: Time series data
            period: Seasonal period
            model: Decomposition model ('additive' or 'multiplicative')
            
        Returns:
            Dictionary of decomposed components
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            if len(data) < 2 * period:
                raise ValueError(f"Need at least {2 * period} observations for decomposition")
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                data.dropna(),
                model=model,
                period=period,
                extrapolate_trend='freq'
            )
            
            components = {
                'original': data,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
            
            # Calculate component statistics
            trend_strength = 1 - (decomposition.resid.var() / (decomposition.trend + decomposition.resid).var())
            seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())
            
            components['trend_strength'] = pd.Series([trend_strength])
            components['seasonal_strength'] = pd.Series([seasonal_strength])
            
            self.logger.info(f"Time series decomposition completed. Trend strength: {trend_strength:.3f}")
            return components
            
        except Exception as e:
            self.logger.error(f"Error in time series decomposition: {str(e)}")
            return {}

    def detect_changepoints(
        self,
        data: pd.Series,
        method: str = 'ruptures',
        min_size: int = 5
    ) -> List[int]:
        """
        Detect change points in time series.
        
        Args:
            data: Time series data
            method: Detection method
            min_size: Minimum segment size
            
        Returns:
            List of change point indices
        """
        try:
            data_clean = data.dropna()
            
            if len(data_clean) < min_size * 2:
                return []
            
            if method == 'ruptures':
                try:
                    import ruptures as rpt
                    
                    # Use PELT algorithm for change point detection
                    algo = rpt.Pelt(model='rbf').fit(data_clean.values)
                    changepoints = algo.predict(pen=10)
                    
                    # Remove the last point (end of series) and adjust indices
                    changepoints = [cp for cp in changepoints[:-1] if cp < len(data_clean)]
                    
                except ImportError:
                    # Fallback to simple variance-based method
                    changepoints = self._simple_changepoint_detection(data_clean, min_size)
            
            elif method == 'variance':
                changepoints = self._simple_changepoint_detection(data_clean, min_size)
            
            else:
                changepoints = []
            
            self.logger.info(f"Detected {len(changepoints)} change points")
            return changepoints
            
        except Exception as e:
            self.logger.error(f"Error detecting change points: {str(e)}")
            return []

    def calculate_autocorrelation(
        self,
        data: pd.Series,
        max_lags: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate autocorrelation function.
        
        Args:
            data: Time series data
            max_lags: Maximum number of lags
            
        Returns:
            Tuple of (autocorrelations, lags)
        """
        try:
            data_clean = data.dropna()
            
            if max_lags is None:
                max_lags = min(len(data_clean) // 4, 40)
            
            # Calculate autocorrelation
            autocorr = []
            for lag in range(max_lags + 1):
                if lag == 0:
                    autocorr.append(1.0)
                elif lag < len(data_clean):
                    corr = np.corrcoef(data_clean[:-lag], data_clean[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorr.append(0)
            
            lags = np.arange(len(autocorr))
            
            return np.array(autocorr), lags
            
        except Exception as e:
            self.logger.error(f"Error calculating autocorrelation: {str(e)}")
            return np.array([]), np.array([])

    def fit_trend_model(
        self,
        data: pd.Series,
        model_type: str = 'linear'
    ) -> Dict[str, Any]:
        """
        Fit trend model to time series data.
        
        Args:
            data: Time series data
            model_type: Type of trend model ('linear', 'polynomial', 'exponential')
            
        Returns:
            Fitted model parameters and statistics
        """
        try:
            data_clean = data.dropna()
            x = np.arange(len(data_clean))
            y = data_clean.values
            
            if len(data_clean) < 3:
                return {}
            
            model_results = {}
            
            if model_type == 'linear':
                # Linear trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                fitted_values = slope * x + intercept
                
                model_results = {
                    'model_type': 'linear',
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'std_error': std_err,
                    'fitted_values': fitted_values
                }
            
            elif model_type == 'polynomial':
                # Polynomial trend (degree 2)
                coeffs = np.polyfit(x, y, 2)
                fitted_values = np.polyval(coeffs, x)
                
                # Calculate R-squared
                ss_res = np.sum((y - fitted_values) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                model_results = {
                    'model_type': 'polynomial',
                    'coefficients': coeffs.tolist(),
                    'r_squared': r_squared,
                    'fitted_values': fitted_values
                }
            
            elif model_type == 'exponential':
                # Exponential trend
                if np.all(y > 0):
                    log_y = np.log(y)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_y)
                    fitted_values = np.exp(slope * x + intercept)
                    
                    model_results = {
                        'model_type': 'exponential',
                        'growth_rate': slope,
                        'initial_value': np.exp(intercept),
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'fitted_values': fitted_values
                    }
                else:
                    # Fallback to linear if data contains non-positive values
                    return self.fit_trend_model(data, 'linear')
            
            # Calculate residuals and diagnostics
            residuals = y - model_results['fitted_values']
            model_results.update({
                'residuals': residuals,
                'rmse': np.sqrt(np.mean(residuals ** 2)),
                'mae': np.mean(np.abs(residuals)),
                'aic': self._calculate_aic(residuals, len(model_results.get('coefficients', [2]))),
                'durbin_watson': self._durbin_watson_statistic(residuals)
            })
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error fitting trend model: {str(e)}")
            return {}

    def detect_anomalies_in_series(
        self,
        data: pd.Series,
        method: str = 'isolation_forest',
        contamination: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in time series data.
        
        Args:
            data: Time series data
            method: Detection method
            contamination: Expected proportion of anomalies
            
        Returns:
            Tuple of (anomaly_mask, anomaly_scores)
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            data_clean = data.dropna()
            
            if len(data_clean) < 10:
                return np.array([]), np.array([])
            
            # Prepare features (value, lag1, moving average, etc.)
            features = []
            for i in range(len(data_clean)):
                feature_vector = [data_clean.iloc[i]]
                
                # Add lag features
                if i > 0:
                    feature_vector.append(data_clean.iloc[i-1])
                else:
                    feature_vector.append(data_clean.iloc[i])
                
                # Add moving average
                window_start = max(0, i-6)
                moving_avg = data_clean.iloc[window_start:i+1].mean()
                feature_vector.append(moving_avg)
                
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            if method == 'isolation_forest':
                # Scale features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_array)
                
                # Fit Isolation Forest
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42
                )
                anomaly_labels = iso_forest.fit_predict(features_scaled)
                anomaly_scores = iso_forest.decision_function(features_scaled)
                
                anomaly_mask = anomaly_labels == -1
                
            elif method == 'statistical':
                # Statistical method using z-score
                z_scores = np.abs(stats.zscore(features_array[:, 0]))
                threshold = stats.norm.ppf(1 - contamination/2)
                
                anomaly_mask = z_scores > threshold
                anomaly_scores = z_scores
            
            else:
                anomaly_mask = np.array([])
                anomaly_scores = np.array([])
            
            self.logger.info(f"Detected {np.sum(anomaly_mask)} anomalies in time series")
            return anomaly_mask, anomaly_scores
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return np.array([]), np.array([])

    def _simple_changepoint_detection(self, data: pd.Series, min_size: int) -> List[int]:
        """Simple variance-based change point detection."""
        changepoints = []
        
        for i in range(min_size, len(data) - min_size):
            # Calculate variance before and after potential change point
            var_before = data.iloc[:i].var()
            var_after = data.iloc[i:].var()
            
            # Calculate F-statistic for variance change
            if var_before > 0 and var_after > 0:
                f_stat = max(var_before, var_after) / min(var_before, var_after)
                
                # Simple threshold for change detection
                if f_stat > 4:  # Threshold for significant variance change
                    changepoints.append(i)
        
        return changepoints

    def _calculate_aic(self, residuals: np.ndarray, k: int) -> float:
        """Calculate Akaike Information Criterion."""
        try:
            n = len(residuals)
            sse = np.sum(residuals ** 2)
            aic = n * np.log(sse / n) + 2 * k
            return aic
        except Exception:
            return np.inf

    def _durbin_watson_statistic(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation in residuals."""
        try:
            diff = np.diff(residuals)
            dw = np.sum(diff ** 2) / np.sum(residuals ** 2)
            return dw
        except Exception:
            return 2.0  # No autocorrelation
