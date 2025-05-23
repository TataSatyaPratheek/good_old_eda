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
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
        input_data: Union[pd.Series, np.ndarray, List[float]],
        include_advanced: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            input_data: Input data
            include_advanced: Whether to include advanced statistics
            
        Returns:
            Dictionary of statistical measures
        """
        try:
            num_nans = 0
            data_for_stats: np.ndarray

            if isinstance(input_data, list):
                current_data_arr = np.array(input_data, dtype=float) # Ensure float for np.nan
                num_nans = np.sum(np.isnan(current_data_arr))
                data_for_stats = current_data_arr[~np.isnan(current_data_arr)]
            elif isinstance(input_data, pd.Series):
                num_nans = input_data.isnull().sum()
                data_for_stats = input_data.dropna().values
            elif isinstance(input_data, np.ndarray):
                current_data_arr = input_data.astype(float) # Ensure float for np.nan
                if current_data_arr.ndim > 1:
                    self.logger.warning(
                        "Input np.ndarray is not 1D. Flattening the array."
                    )
                    current_data_arr = current_data_arr.flatten()
                num_nans = np.sum(np.isnan(current_data_arr))
                data_for_stats = current_data_arr[~np.isnan(current_data_arr)]
            else:
                self.logger.error(f"Unsupported data type for descriptive statistics: {type(input_data)}")
                return {}

            if len(data_for_stats) == 0:
                return {'count': 0, 'null_count': num_nans}

            # Basic statistics
            stats_dict = {
                'count': len(data_for_stats),
                'null_count': num_nans,
                'mean': np.mean(data_for_stats),
                'median': np.median(data_for_stats),
                'std': np.std(data_for_stats, ddof=1) if len(data_for_stats) >= 2 else np.nan,
                'variance': np.var(data_for_stats, ddof=1) if len(data_for_stats) >= 2 else np.nan,
                'min': np.min(data_for_stats),
                'max': np.max(data_for_stats),
                'range': np.max(data_for_stats) - np.min(data_for_stats),
                'q25': np.percentile(data_for_stats, 25),
                'q75': np.percentile(data_for_stats, 75),
                'iqr': np.percentile(data_for_stats, 75) - np.percentile(data_for_stats, 25)
            }

            # Coefficient of Variation
            mean_val = stats_dict['mean']
            std_val = stats_dict['std']
            cv = np.nan
            if not (pd.isna(mean_val) or pd.isna(std_val)):
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                elif std_val == 0:  # Mean is 0, Std is 0
                    cv = 0.0
                else:  # Mean is 0, Std is > 0
                    cv = np.inf

            stats_dict.update({
                'q1': stats_dict['q25'],  # Alias for compatibility
                'q3': stats_dict['q75'],  # Alias for compatibility
                'cv': cv  # For compatibility
            })

            if include_advanced:
                # Advanced statistics
                stats_dict.update({
                    'skewness': stats.skew(data_for_stats) if len(data_for_stats) >= 3 else np.nan,
                    'kurtosis': stats.kurtosis(data_for_stats) if len(data_for_stats) >= 4 else np.nan, # Fisher kurtosis (default)
                    'coefficient_of_variation': cv,
                    'mad': np.median(np.abs(data_for_stats - stats_dict['median'])),  # Median Absolute Deviation
                    'harmonic_mean': stats.hmean(data_for_stats[data_for_stats > 0]) if np.any(data_for_stats > 0) else 0,
                    'geometric_mean': stats.gmean(data_for_stats[data_for_stats > 0]) if np.any(data_for_stats > 0) else 0,
                    'trim_mean_10': stats.trim_mean(data_for_stats, 0.1) if len(data_for_stats) > 0 else np.nan,
                    'mode': stats.mode(data_for_stats, keepdims=True).mode[0] if len(data_for_stats) > 0 else np.nan,
                    'entropy': self._calculate_entropy(data_for_stats),
                    # Aliases for paste file compatibility
                    'skew': stats_dict['skewness'],
                })

            return stats_dict

        except Exception as e:
            self.logger.error(f"Error calculating descriptive statistics: {str(e)}")
            return {}

    def calculate_correlation_matrix(
        self,
        data: Union[pd.DataFrame, pd.Series],
        method: str = 'pearson',
        min_periods: int = 10
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Calculate correlation matrix with significance testing.
        
        Args:
            data: Input DataFrame or Series
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_periods: Minimum periods for correlation calculation
            
        Returns:
            Tuple of (correlation matrix, p-values matrix) or simple correlation matrix
        """
        try:
            # Handle simple case from paste file
            if isinstance(data, pd.DataFrame):
                numeric_data = data.select_dtypes(include=[np.number])
                if numeric_data.empty:
                    return pd.DataFrame()

                # Simple correlation matrix (paste file compatibility)
                if min_periods == 10 and method == 'pearson':
                    simple_corr = numeric_data.corr(method=method)
                    return simple_corr

            # Advanced correlation analysis (from uploaded file)
            if isinstance(data, pd.Series):
                # Convert series to dataframe for consistency
                numeric_data = pd.DataFrame({'data': data})
            else:
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

    def detect_outliers(
        self,
        data: pd.Series,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers in data (from paste file)
        
        Args:
            data: Input data series
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        try:
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                return (data < lower_bound) | (data > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data.dropna()))
                outlier_mask = z_scores > threshold
                # Align with original series index
                result = pd.Series(False, index=data.index)
                result.loc[data.dropna().index] = outlier_mask
                return result
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {str(e)}")
            return pd.Series(False, index=data.index)

    def calculate_trend_strength(
        self,
        data: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate trend strength and direction (from paste file)
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            if len(data) < 2:
                return {'trend': 'insufficient_data', 'strength': 0}
            
            # Simple linear regression
            x = np.arange(len(data))
            y = data.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            
            if len(x) < 2:
                return {'trend': 'insufficient_data', 'strength': 0}
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
            trend_strength = abs(r_value)
            
            return {
                'trend': trend_direction,
                'strength': trend_strength,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err,
                'intercept': intercept
            }
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return {'trend': 'error', 'strength': 0}

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
            test_type: Type of test ('ttest', 'mannwhitney', 'wilcoxon')
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

            elif test_type == 'chi2':
                raise ValueError("Chi-squared test ('chi2') is not implemented in this method.")

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

    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to 0-1 range
        
        Args:
            value: Value to normalize
            min_val: Minimum value in range
            max_val: Maximum value in range
            
        Returns:
            Normalized value between 0 and 1
        """
        try:
            if max_val == min_val:
                return 0.0 if value == min_val else np.nan # Or 0.5, or raise error
            return (value - min_val) / (max_val - min_val)
        except Exception as e:
            self.logger.error(f"Error normalizing value: {str(e)}")
            return np.nan # Return NaN on error

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

    def calculate_pareto_frontier(
        self,
        objectives: pd.DataFrame
    ) -> List[int]:
        """
        Calculate Pareto frontier for multi-objective optimization (from paste file)
        
        Args:
            objectives: DataFrame with objective values
            
        Returns:
            List of indices representing Pareto frontier
        """
        try:
            if objectives.empty:
                return []
            
            # Simple Pareto frontier calculation
            pareto_frontier = []
            
            for i, row_i in objectives.iterrows():
                is_dominated = False
                for j, row_j in objectives.iterrows():
                    if i != j:
                        # Check if row_j dominates row_i (all objectives better or equal, at least one strictly better)
                        dominates = True
                        strictly_better = False
                        
                        for col in objectives.columns:
                            if row_j[col] < row_i[col]:  # Assuming minimization
                                dominates = False
                                break
                            elif row_j[col] > row_i[col]:
                                strictly_better = True
                        
                        if dominates and strictly_better:
                            is_dominated = True
                            break
                
                if not is_dominated:
                    pareto_frontier.append(i)
            
            self.logger.info(f"Pareto frontier calculated with {len(pareto_frontier)} solutions")
            return pareto_frontier[:5]  # Return top 5 for compatibility
        
        except Exception as e:
            self.logger.error(f"Error calculating Pareto frontier: {str(e)}")
            return []

    def optimize_resource_allocation(
        self,
        resources: float,
        options: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Simple resource allocation optimization (from paste file)
        
        Args:
            resources: Total resources available
            options: DataFrame with options
            constraints: Optional constraints
            
        Returns:
            Dictionary with resource allocations
        """
        try:
            if options.empty:
                return {}
            
            # Allocate proportionally to value
            if 'value' in options.columns:
                total_value = options['value'].sum()
                if total_value > 0:
                    allocations = {}
                    for idx, row in options.iterrows():
                        allocations[str(idx)] = (row['value'] / total_value) * resources
                    return allocations
            
            # Equal allocation as fallback
            allocation_per_option = resources / len(options)
            return {str(idx): allocation_per_option for idx in options.index}
        
        except Exception as e:
            self.logger.error(f"Error in resource allocation: {str(e)}")
            return {}

    def calculate_optimization_metrics(
        self,
        actual: pd.Series,
        predicted: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate optimization performance metrics (from paste file)
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Align series and remove NaN values
            aligned_actual, aligned_predicted = actual.align(predicted, join='inner')
            mask = ~(pd.isna(aligned_actual) | pd.isna(aligned_predicted))
            clean_actual = aligned_actual[mask]
            clean_predicted = aligned_predicted[mask]
            
            if len(clean_actual) == 0:
                return {'error': 'No valid data points for comparison'}
            
            metrics = {
                'mse': mean_squared_error(clean_actual, clean_predicted),
                'mae': mean_absolute_error(clean_actual, clean_predicted),
                'rmse': np.sqrt(mean_squared_error(clean_actual, clean_predicted))
            }
            
            # MAPE calculation with zero-division protection
            non_zero_mask = clean_actual != 0
            if non_zero_mask.any():
                mape = np.mean(np.abs((clean_actual[non_zero_mask] - clean_predicted[non_zero_mask]) / clean_actual[non_zero_mask])) * 100
                metrics['mape'] = mape
            else:
                metrics['mape'] = np.inf
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error calculating optimization metrics: {str(e)}")
            return {'error': str(e)}

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
            # Try advanced decomposition first
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

            except ImportError:
                # Fallback to simple decomposition (from paste file)
                self.logger.warning("statsmodels not available, using simple decomposition")
                return {
                    'trend': data.rolling(window=period, center=True).mean(),
                    'seasonal': pd.Series(index=data.index, data=0),
                    'residual': data - data.rolling(window=period, center=True).mean(),
                    'original': data
                }

        except Exception as e:
            self.logger.error(f"Error in time series decomposition: {str(e)}")
            return {}

    def calculate_moving_averages(
        self,
        data: pd.Series,
        windows: List[int] = [7, 14, 30]
    ) -> Dict[str, pd.Series]:
        """
        Calculate multiple moving averages (from paste file)
        
        Args:
            data: Time series data
            windows: List of window sizes
            
        Returns:
            Dictionary of moving averages
        """
        try:
            mas = {}
            for window in windows:
                mas[f'ma_{window}'] = data.rolling(window=window).mean()
            return mas
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {str(e)}")
            return {}

    def detect_seasonality(
        self,
        data: pd.Series,
        freq: str = 'D'
    ) -> Dict[str, Any]:
        """
        Detect seasonality patterns (enhanced from paste file)
        
        Args:
            data: Time series data
            freq: Frequency string
            
        Returns:
            Dictionary with seasonality information
        """
        try:
            if len(data) < 14:  # Need at least 2 weeks of data
                return {
                    'has_seasonality': False,
                    'period': None,
                    'strength': 0.0
                }

            # Calculate autocorrelation to detect periodicity
            autocorr, lags = self.calculate_autocorrelation(data, max_lags=min(len(data)//2, 50))
            
            # Find peaks in autocorrelation
            if len(autocorr) > 2:
                # Skip lag 0 (always 1) and find maximum autocorrelation
                max_autocorr_idx = np.argmax(autocorr[1:]) + 1
                max_autocorr = autocorr[max_autocorr_idx]
                
                # Consider seasonality if autocorrelation > 0.3
                has_seasonality = max_autocorr > 0.3
                period = lags[max_autocorr_idx] if has_seasonality else None
                strength = max_autocorr if has_seasonality else 0.0
            else:
                has_seasonality = False
                period = None
                strength = 0.0

            return {
                'has_seasonality': has_seasonality,
                'period': period,
                'strength': strength,
                'autocorrelation': autocorr,
                'lags': lags
            }

        except Exception as e:
            self.logger.error(f"Error detecting seasonality: {str(e)}")
            return {
                'has_seasonality': False,
                'period': None,
                'strength': 0.0
            }

    def forecast_simple(
        self,
        data: pd.Series,
        periods: int = 7,
        method: str = 'naive'
    ) -> pd.Series:
        """
        Simple forecasting methods (from paste file)
        
        Args:
            data: Time series data
            periods: Number of periods to forecast
            method: Forecasting method
            
        Returns:
            Forecasted series
        """
        try:
            if method == 'naive':
                # Last value forecast
                last_value = data.iloc[-1]
                if hasattr(data.index, 'freq') and data.index.freq:
                    forecast_index = pd.date_range(
                        start=data.index[-1] + data.index.freq,
                        periods=periods,
                        freq=data.index.freq
                    )
                else:
                    forecast_index = pd.date_range(
                        start=data.index[-1] + pd.Timedelta(days=1),
                        periods=periods,
                        freq='D'
                    )
                return pd.Series(index=forecast_index, data=last_value)
            
            elif method == 'mean':
                # Mean forecast
                mean_value = data.mean()
                if hasattr(data.index, 'freq') and data.index.freq:
                    forecast_index = pd.date_range(
                        start=data.index[-1] + data.index.freq,
                        periods=periods,
                        freq=data.index.freq
                    )
                else:
                    forecast_index = pd.date_range(
                        start=data.index[-1] + pd.Timedelta(days=1),
                        periods=periods,
                        freq='D'
                    )
                return pd.Series(index=forecast_index, data=mean_value)
            
            elif method == 'trend':
                # Linear trend forecast
                x = np.arange(len(data))
                y = data.values
                slope, intercept, _, _, _ = stats.linregress(x, y)
                
                forecast_values = []
                for i in range(periods):
                    forecast_x = len(data) + i
                    forecast_y = slope * forecast_x + intercept
                    forecast_values.append(forecast_y)
                
                if hasattr(data.index, 'freq') and data.index.freq:
                    forecast_index = pd.date_range(
                        start=data.index[-1] + data.index.freq,
                        periods=periods,
                        freq=data.index.freq
                    )
                else:
                    forecast_index = pd.date_range(
                        start=data.index[-1] + pd.Timedelta(days=1),
                        periods=periods,
                        freq='D'
                    )
                return pd.Series(index=forecast_index, data=forecast_values)
            
            else:
                raise ValueError(f"Unknown forecast method: {method}")

        except Exception as e:
            self.logger.error(f"Error in simple forecasting: {str(e)}")
            return pd.Series()

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

    def analyze_trend_patterns(self, data, time_column=None):
        """
        Analyze trend patterns in time series data
        
        Args:
            data: DataFrame with time series data
            time_column: Name of the time column (optional)
            
        Returns:
            Dict containing trend analysis results
        """
        try:
            self.logger.info("Analyzing trend patterns in time series data")
            
            if data.empty:
                return {
                    'trend_direction': 'no_data',
                    'trend_strength': 0.0,
                    'trend_consistency': 0.0,
                    'pattern_type': 'insufficient_data'
                }
            
            # Identify time column
            time_col = self._identify_time_column(data, time_column)
            
            if time_col is None:
                return self._analyze_trends_without_time(data)
            
            # Prepare time series
            ts_data = self._prepare_time_series(data, time_col)
            
            # Linear trend analysis
            trend_analysis = self._calculate_linear_trend(ts_data)
            
            # Seasonal decomposition if enough data
            seasonal_analysis = self._analyze_seasonal_patterns(ts_data)
            
            # Pattern recognition
            pattern_analysis = self._identify_patterns(ts_data)
            
            # Trend consistency analysis
            consistency_analysis = self._analyze_trend_consistency(ts_data)
            
            # Combine all analyses
            comprehensive_analysis = {
                'trend_direction': trend_analysis['direction'],
                'trend_strength': trend_analysis['strength'],
                'trend_slope': trend_analysis['slope'],
                'trend_r_squared': trend_analysis['r_squared'],
                'trend_consistency': consistency_analysis['consistency_score'],
                'pattern_type': pattern_analysis['primary_pattern'],
                'seasonal_component': seasonal_analysis['has_seasonality'],
                'volatility': pattern_analysis['volatility'],
                'change_points': pattern_analysis['change_points'],
                'trend_confidence': self._calculate_trend_confidence(trend_analysis, consistency_analysis),
                'forecast_reliability': self._assess_forecast_reliability(ts_data),
                'analysis_metadata': {
                    'data_points': len(ts_data),
                    'time_span_days': self._calculate_time_span(ts_data, time_col),
                    'missing_values': ts_data.isnull().sum().sum(),
                    'analysis_timestamp': datetime.now()
                }
            }
            
            self.logger.info(f"Trend analysis completed: {comprehensive_analysis['trend_direction']} trend with {comprehensive_analysis['trend_strength']:.3f} strength")
            
            return comprehensive_analysis
            
        except Exception as e:
            self.logger.error(f"Error in trend pattern analysis: {str(e)}")
            return {
                'trend_direction': 'unknown',
                'trend_strength': 0.0,
                'trend_consistency': 0.0,
                'pattern_type': 'error',
                'error': str(e)
            }

    def calculate_growth_trends(self, data):
        """
        Calculate growth trends and growth rates
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Dict containing growth trend analysis
        """
        try:
            self.logger.info("Calculating growth trends")
            
            if data.empty:
                return {
                    'growth_rate': 0.0,
                    'growth_acceleration': 0.0,
                    'growth_stability': 0.0,
                    'growth_pattern': 'no_data'
                }
            
            # Find numeric columns for growth analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                return {
                    'growth_rate': 0.0,
                    'growth_acceleration': 0.0,
                    'growth_stability': 0.0,
                    'growth_pattern': 'no_numeric_data'
                }
            
            growth_results = {}
            
            for col in numeric_cols[:3]:  # Analyze top 3 numeric columns
                col_growth = self._calculate_column_growth(data[col])
                growth_results[col] = col_growth
            
            # Overall growth analysis
            overall_growth = self._calculate_overall_growth(growth_results)
            
            # Growth pattern classification
            growth_pattern = self._classify_growth_pattern(overall_growth)
            
            # Growth stability metrics
            stability_metrics = self._calculate_growth_stability(growth_results)
            
            growth_analysis = {
                'growth_rate': overall_growth['average_growth_rate'],
                'growth_acceleration': overall_growth['growth_acceleration'],
                'growth_stability': stability_metrics['stability_score'],
                'growth_pattern': growth_pattern,
                'growth_volatility': stability_metrics['volatility'],
                'growth_consistency': stability_metrics['consistency'],
                'column_specific_growth': growth_results,
                'growth_forecasts': self._generate_growth_forecasts(overall_growth),
                'growth_insights': self._extract_growth_insights(overall_growth, growth_pattern),
                'analysis_metadata': {
                    'columns_analyzed': list(growth_results.keys()),
                    'data_points': len(data),
                    'analysis_method': 'compound_annual_growth_rate',
                    'confidence_level': stability_metrics.get('confidence', 0.7)
                }
            }
            
            self.logger.info(f"Growth analysis completed: {growth_analysis['growth_rate']:.3f} average growth rate")
            
            return growth_analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating growth trends: {str(e)}")
            return {
                'growth_rate': 0.0,
                'growth_acceleration': 0.0,
                'growth_stability': 0.0,
                'growth_pattern': 'error',
                'error': str(e)
            }

    def detect_seasonal_patterns(self, data):
        """
        Detect seasonal patterns in time series data
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Dict containing seasonality analysis
        """
        try:
            self.logger.info("Detecting seasonal patterns")
            
            if data.empty or len(data) < 14:  # Need at least 2 weeks of data
                return {
                    'has_seasonality': False,
                    'seasonal_strength': 0.0,
                    'seasonal_periods': [],
                    'dominant_period': None,
                    'pattern_confidence': 0.0
                }
            
            # Find time column
            time_col = self._identify_time_column(data)
            
            if time_col is None:
                return self._detect_index_based_seasonality(data)
            
            # Prepare time series data
            ts_data = self._prepare_seasonal_analysis(data, time_col)
            
            # Test for different seasonal periods
            seasonal_periods = [7, 14, 30, 90, 365]  # Daily, bi-weekly, monthly, quarterly, yearly
            seasonal_tests = {}
            
            for period in seasonal_periods:
                if len(ts_data) >= period * 2:  # Need at least 2 cycles
                    seasonal_test = self._test_seasonal_period(ts_data, period)
                    seasonal_tests[period] = seasonal_test
            
            # Find dominant seasonal pattern
            dominant_season = self._find_dominant_seasonality(seasonal_tests)
            
            # Seasonal decomposition for the dominant period
            decomposition = self._perform_seasonal_decomposition(ts_data, dominant_season)
            
            # Calculate seasonal strength
            seasonal_strength = self._calculate_seasonal_strength(decomposition)
            
            # Generate seasonal insights
            seasonal_insights = self._generate_seasonal_insights(decomposition, dominant_season)
            
            seasonality_analysis = {
                'has_seasonality': seasonal_strength > 0.3,
                'seasonal_strength': seasonal_strength,
                'seasonal_periods': list(seasonal_tests.keys()),
                'dominant_period': dominant_season['period'] if dominant_season else None,
                'pattern_confidence': dominant_season['confidence'] if dominant_season else 0.0,
                'seasonal_decomposition': decomposition,
                'seasonal_tests': seasonal_tests,
                'peak_season': seasonal_insights.get('peak_season', 'unknown'),
                'low_season': seasonal_insights.get('low_season', 'unknown'),
                'seasonal_amplitude': seasonal_insights.get('amplitude', 0.0),
                'seasonal_forecasts': self._generate_seasonal_forecasts(decomposition, dominant_season),
                'analysis_metadata': {
                    'data_points': len(ts_data),
                    'periods_tested': len(seasonal_tests),
                    'decomposition_method': 'additive',
                    'analysis_timestamp': datetime.now()
                }
            }
            
            self.logger.info(f"Seasonality analysis completed: {'Seasonal' if seasonality_analysis['has_seasonality'] else 'Non-seasonal'} pattern detected")
            
            return seasonality_analysis
            
        except Exception as e:
            self.logger.error(f"Error detecting seasonal patterns: {str(e)}")
            return {
                'has_seasonality': False,
                'seasonal_strength': 0.0,
                'seasonal_periods': [],
                'dominant_period': None,
                'pattern_confidence': 0.0,
                'error': str(e)
            }

    # Helper methods for TimeSeriesAnalyzer

    def _identify_time_column(self, data, time_column=None):
        """Identify the time column in the dataset"""
        if time_column and time_column in data.columns:
            return time_column
        
        # Look for common time column names
        time_column_names = ['date', 'time', 'timestamp', 'datetime', 'period']
        for col in data.columns:
            if col.lower() in time_column_names:
                return col
        
        # Check for datetime-like columns
        for col in data.columns:
            try:
                pd.to_datetime(data[col].dropna().iloc[:5])
                return col
            except:
                continue
        
        return None

    def _prepare_time_series(self, data, time_col):
        """Prepare time series data for analysis"""
        ts_data = data.copy()
        ts_data[time_col] = pd.to_datetime(ts_data[time_col])
        ts_data = ts_data.sort_values(time_col)
        ts_data = ts_data.set_index(time_col)
        
        # Fill missing values with interpolation
        numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
        ts_data[numeric_cols] = ts_data[numeric_cols].interpolate()
        
        return ts_data

    def _calculate_linear_trend(self, ts_data):
        """Calculate linear trend statistics"""
        from scipy import stats
        
        # Use the first numeric column or create aggregate
        numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {'direction': 'no_data', 'strength': 0.0, 'slope': 0.0, 'r_squared': 0.0}
        
        # Create time index for regression
        x = np.arange(len(ts_data))
        y = ts_data[numeric_cols[0]].values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        x, y = x[mask], y[mask]
        
        if len(x) < 2:
            return {'direction': 'insufficient_data', 'strength': 0.0, 'slope': 0.0, 'r_squared': 0.0}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if abs(slope) < 0.001:
            direction = 'stable'
        elif slope > 0:
            direction = 'positive'
        else:
            direction = 'negative'
        
        return {
            'direction': direction,
            'strength': abs(r_value),
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significance': p_value < 0.05
        }

    def _analyze_trends_without_time(self, data):
        """Analyze trends when no time column is available"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                'trend_direction': 'no_numeric_data',
                'trend_strength': 0.0,
                'trend_consistency': 0.0,
                'pattern_type': 'non_numeric'
            }
        
        # Use row index as proxy for time
        trends = []
        for col in numeric_cols[:3]:  # Analyze first 3 columns
            values = data[col].dropna()
            if len(values) >= 3:
                x = np.arange(len(values))
                slope, _, r_value, _, _ = stats.linregress(x, values)
                trends.append({'slope': slope, 'strength': abs(r_value)})
        
        if not trends:
            return {
                'trend_direction': 'insufficient_data',
                'trend_strength': 0.0,
                'trend_consistency': 0.0,
                'pattern_type': 'insufficient_data'
            }
        
        # Aggregate trends
        avg_slope = np.mean([t['slope'] for t in trends])
        avg_strength = np.mean([t['strength'] for t in trends])
        
        direction = 'positive' if avg_slope > 0.001 else 'negative' if avg_slope < -0.001 else 'stable'
        
        return {
            'trend_direction': direction,
            'trend_strength': avg_strength,
            'trend_consistency': avg_strength,  # Simplified
            'pattern_type': 'index_based_analysis'
        }

    def _calculate_column_growth(self, series):
        """Calculate growth metrics for a single column"""
        values = series.dropna()
        
        if len(values) < 2:
            return {
                'growth_rate': 0.0,
                'volatility': 0.0,
                'trend': 'insufficient_data'
            }
        
        # Calculate period-over-period growth rates
        pct_changes = values.pct_change().dropna()
        
        if len(pct_changes) == 0:
            return {
                'growth_rate': 0.0,
                'volatility': 0.0,
                'trend': 'no_change'
            }
        
        # Growth metrics
        avg_growth = pct_changes.mean()
        growth_volatility = pct_changes.std()
        
        # Compound annual growth rate (simplified)
        if len(values) > 1 and values.iloc[0] != 0:
            total_return = values.iloc[-1] / values.iloc[0] - 1
            periods = len(values) - 1
            cagr = (1 + total_return) ** (1/periods) - 1 if periods > 0 else 0
        else:
            cagr = 0
        
        return {
            'growth_rate': avg_growth,
            'volatility': growth_volatility,
            'cagr': cagr,
            'total_return': total_return if 'total_return' in locals() else 0,
            'trend': 'positive' if avg_growth > 0.01 else 'negative' if avg_growth < -0.01 else 'stable'
        }

    def _calculate_overall_growth(self, growth_results):
        """Calculate overall growth metrics across all columns"""
        if not growth_results:
            return {
                'average_growth_rate': 0.0,
                'growth_acceleration': 0.0,
                'growth_consistency': 0.0
            }
        
        growth_rates = [result['growth_rate'] for result in growth_results.values()]
        
        return {
            'average_growth_rate': np.mean(growth_rates),
            'growth_acceleration': np.std(growth_rates),  # Higher std indicates more acceleration/deceleration
            'growth_consistency': 1.0 / (1.0 + np.std(growth_rates)),  # Inverse relationship with volatility
            'median_growth_rate': np.median(growth_rates),
            'max_growth_rate': np.max(growth_rates),
            'min_growth_rate': np.min(growth_rates)
        }
