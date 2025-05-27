"""
High-Performance Correlation Analysis Engine
Optimized for large-scale SEO data processing with proper indentation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CorrelationEngine:
    """Blazing fast correlation analysis for large SEO datasets"""
    
    def __init__(self, chunk_size: int = 1000, n_jobs: int = 4, top_k: int = 100):
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size  # Process data in chunks
        self.n_jobs = n_jobs  # Parallel processing
        self.top_k = top_k  # Only keep top correlations
        
    def analyze_metric_correlations(self, data: Dict[str, pd.DataFrame], 
                                  historical_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        BLAZING FAST correlation analysis using vectorized operations
        """
        
        print("🚀 Analyzing Metric Correlations (High-Performance Mode)...")
        start_time = datetime.now()
        
        analysis = {
            'primary_correlations': {},
            'advanced_correlations': {},
            'temporal_correlations': {},
            'competitive_correlations': {},
            'predictive_correlations': {},
            'correlation_insights': [],
            'performance_metrics': {
                'processing_time': 0,
                'data_points_processed': 0,
                'correlations_computed': 0
            }
        }
        
        try:
            # Primary correlations - VECTORIZED
            analysis['primary_correlations'] = self._fast_primary_correlations(data)
            
            # Advanced correlations - CHUNKED
            analysis['advanced_correlations'] = self._fast_advanced_correlations(data)
            
            # Temporal correlations with error handling
            if historical_data:
                analysis['temporal_correlations'] = self._analyze_temporal_correlations(data, historical_data)
            
            # Competitive correlations
            analysis['competitive_correlations'] = self._analyze_competitive_correlations(data)
            
            # Predictive correlations
            analysis['predictive_correlations'] = self._analyze_predictive_correlations(data)
            
            # Generate insights
            analysis['correlation_insights'] = self._generate_correlation_insights(analysis)
            
            # Performance metrics
            end_time = datetime.now()
            analysis['performance_metrics']['processing_time'] = (end_time - start_time).total_seconds()
            
            print(f"🎉 Correlation analysis completed in {analysis['performance_metrics']['processing_time']:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Correlation analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _fast_primary_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """ULTRA-FAST primary correlations using vectorized NumPy operations"""
        
        gap_keywords_df = data.get('gap_keywords', pd.DataFrame())
        
        correlations = {
            'position_traffic_correlation': {},
            'cross_brand_correlations': {},
            'organic_traffic_featured_snippets': {},
            'keyword_difficulty_conversion_rates': {},
            'backlink_velocity_ranking_improvements': {},
            'serp_features_traffic_correlation': {},
            'processing_stats': {}
        }
        
        # Process all brands in parallel for position-traffic correlation
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_brand = {
                executor.submit(self._vectorized_brand_correlation, brand, df): brand 
                for brand, df in data.items() if brand != 'gap_keywords' and not df.empty
            }
            
            for future in as_completed(future_to_brand):
                brand = future_to_brand[future]
                try:
                    result = future.get(timeout=30)  # 30 second timeout per brand
                    correlations['position_traffic_correlation'][brand] = result
                except Exception as e:
                    correlations['position_traffic_correlation'][brand] = {'error': str(e)}
        
        # Cross-brand correlations using vectorized operations
        correlations['cross_brand_correlations'] = self._fast_cross_brand_correlations(data)
        
        # Advanced correlation analyses
        try:
            correlations['organic_traffic_featured_snippets'] = self._analyze_featured_snippet_correlation(data)
        except Exception as e:
            correlations['organic_traffic_featured_snippets'] = {'error': str(e)}
        
        try:
            correlations['keyword_difficulty_conversion_rates'] = self._analyze_difficulty_conversion_correlation(data, gap_keywords_df)
        except Exception as e:
            correlations['keyword_difficulty_conversion_rates'] = {'error': str(e)}
        
        try:
            correlations['backlink_velocity_ranking_improvements'] = self._analyze_backlink_ranking_correlation(data)
        except Exception as e:
            correlations['backlink_velocity_ranking_improvements'] = {'error': str(e)}
        
        # SERP features analysis for each brand
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            try:
                serp_correlation = self._analyze_serp_features_correlation(df)
                correlations['serp_features_traffic_correlation'][brand] = serp_correlation
            except Exception as e:
                correlations['serp_features_traffic_correlation'][brand] = {'error': str(e)}
        
        return correlations
    
    def _vectorized_brand_correlation(self, brand: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Vectorized correlation calculation for a single brand"""
        
        if df.empty or 'Position' not in df.columns or 'Traffic (%)' not in df.columns:
            return {'correlation': 0, 'significance': 'no_data'}
        
        # Convert to NumPy arrays for speed
        positions = df['Position'].values.astype(np.float64)
        traffic = df['Traffic (%)'].values.astype(np.float64)
        
        # Remove NaN values vectorized
        valid_mask = ~(np.isnan(positions) | np.isnan(traffic))
        positions_clean = positions[valid_mask]
        traffic_clean = traffic[valid_mask]
        
        if len(positions_clean) < 3:
            return {'correlation': 0, 'significance': 'insufficient_data'}
        
        # Check for constant arrays
        if np.std(positions_clean) == 0 or np.std(traffic_clean) == 0:
            return {'correlation': 0, 'significance': 'constant_data'}
        
        # BLAZING FAST correlation using NumPy
        correlation = self._fast_pearson_correlation(positions_clean, traffic_clean)
        
        # Fast significance estimation
        n = len(positions_clean)
        if abs(correlation) >= 0.999999:
            t_stat = np.inf if correlation != 0 else 0
        elif (1 - correlation**2) <= 0:
            t_stat = np.inf if correlation != 0 else 0
        else:
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        
        # Determine significance quickly
        significance = ('highly_significant' if abs(t_stat) > 2.576 else 
                       'significant' if abs(t_stat) > 1.96 else 
                       'not_significant')
        
        return {
            'correlation': correlation,
            'significance': significance,
            'sample_size': n,
            'correlation_strength': self._classify_correlation_strength(abs(correlation))
        }
    
    def _fast_pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """ULTRA-FAST Pearson correlation using pure NumPy"""
        
        # Center the data
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        
        # Calculate correlation
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _fast_cross_brand_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """BLAZING FAST cross-brand correlation matrix using vectorized operations"""
        
        cross_correlations = {}
        brands = [brand for brand in data.keys() if brand != 'gap_keywords' and not data[brand].empty]
        
        if len(brands) < 2:
            return cross_correlations
        
        # Extract traffic vectors for all brands at once
        brand_traffic_vectors = {}
        common_keywords_set = None
        
        for brand in brands:
            df = data[brand]
            if 'Keyword' in df.columns and 'Traffic (%)' in df.columns:
                df_cleaned = df.dropna(subset=['Keyword', 'Traffic (%)'])
                keyword_traffic = df_cleaned.set_index('Keyword')['Traffic (%)'].to_dict()
                brand_traffic_vectors[brand] = keyword_traffic
                
                brand_keywords = set(keyword_traffic.keys())
                if common_keywords_set is None:
                    common_keywords_set = brand_keywords
                else:
                    common_keywords_set = common_keywords_set.intersection(brand_keywords)
        
        if not common_keywords_set or len(common_keywords_set) < 2:
            return {'status': 'insufficient_common_keywords_or_data_points'}
        
        # Convert to aligned numpy arrays for BLAZING FAST computation
        common_keywords_list = list(common_keywords_set)
        traffic_matrix = np.zeros((len(brands), len(common_keywords_list)))
        
        for i, brand in enumerate(brands):
            if brand in brand_traffic_vectors:
                for j, keyword in enumerate(common_keywords_list):
                    traffic_matrix[i, j] = brand_traffic_vectors[brand].get(keyword, 0)
        
        # Filter out rows with no variance
        valid_rows_mask = np.std(traffic_matrix, axis=1) > 0
        if np.sum(valid_rows_mask) < 2:
            return {'status': 'insufficient_brands_with_variance_for_correlation'}
        
        traffic_matrix_filtered = traffic_matrix[valid_rows_mask]
        brands_filtered = [brands[i] for i, is_valid in enumerate(valid_rows_mask) if is_valid]
        
        if traffic_matrix_filtered.shape[1] < 2:
            return {'status': 'insufficient_common_keywords_with_data_for_correlation'}
        
        # VECTORIZED correlation matrix computation
        correlation_matrix = np.corrcoef(traffic_matrix_filtered)
        
        # Extract pairwise correlations
        for i, brand1 in enumerate(brands_filtered):
            for j, brand2 in enumerate(brands_filtered[i+1:], i+1):
                if i < correlation_matrix.shape[0] and j < correlation_matrix.shape[1]:
                    correlation = correlation_matrix[i, j]
                    if not np.isnan(correlation):
                        cross_correlations[f"{brand1}_vs_{brand2}"] = {
                            'correlation': correlation,
                            'common_keywords': len(common_keywords_list),
                            'significance': 'significant' if abs(correlation) > 0.3 else 'not_significant'
                        }
        
        return cross_correlations
    
    def _fast_advanced_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """CHUNKED advanced correlations for memory efficiency"""
        
        advanced = {
            'keyword_cluster_correlations': {},
            'content_type_correlations': {},
            'processing_stats': {}
        }
        
        total_processed = 0
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            # Process in chunks for memory efficiency
            chunk_results = []
            if not df.empty:
                for chunk_start in range(0, len(df), self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, len(df))
                    chunk_df = df.iloc[chunk_start:chunk_end].copy()
                    
                    chunk_result = self._process_chunk_correlations(chunk_df)
                    chunk_results.append(chunk_result)
                    total_processed += len(chunk_df)
            
            # Aggregate chunk results for the current brand
            if chunk_results:
                advanced['keyword_cluster_correlations'][brand] = self._aggregate_chunk_results(chunk_results)
            else:
                advanced['keyword_cluster_correlations'][brand] = {'status': 'no_data_to_process_in_chunks'}
        
        advanced['processing_stats']['total_keywords_processed'] = total_processed
        return advanced
    
    def _process_chunk_correlations(self, chunk_df: pd.DataFrame) -> Dict[str, Any]:
        """Process correlations for a data chunk"""
        
        if (chunk_df.empty or 'Keyword' not in chunk_df.columns or 
            'Position' not in chunk_df.columns or 'Traffic (%)' not in chunk_df.columns):
            return {'chunk_size': 0, 'content_performance': {}}
        
        # Ensure 'Keyword' is string type and handle potential NaNs
        keywords_series = chunk_df['Keyword'].astype(str).str.lower()
        keywords = keywords_series.values
        positions = chunk_df['Position'].values.astype(np.float64)
        traffic = chunk_df['Traffic (%)'].values.astype(np.float64)
        
        # FAST content type classification using vectorized operations
        product_terms = ['laptop', 'computer', 'pc', 'notebook']
        informational_terms = ['how', 'what', 'guide', 'tutorial']
        commercial_terms = ['buy', 'price', 'deal', 'review']
        
        # Create masks using vectorized operations
        check_product = np.vectorize(lambda x: any(term in x for term in product_terms))
        check_informational = np.vectorize(lambda x: any(term in x for term in informational_terms))
        check_commercial = np.vectorize(lambda x: any(term in x for term in commercial_terms))
        
        product_mask = check_product(keywords)
        informational_mask = check_informational(keywords)
        commercial_mask = check_commercial(keywords)
        
        content_performance = {}
        
        for mask, category_name in [(product_mask, 'product'),
                                   (informational_mask, 'informational'),
                                   (commercial_mask, 'commercial')]:
            if np.any(mask):
                cat_positions = positions[mask]
                cat_traffic = traffic[mask]
                
                if cat_positions.size > 0 and cat_traffic.size > 0:
                    content_performance[category_name] = {
                        'keyword_count': np.sum(mask),
                        'avg_position': np.mean(cat_positions[~np.isnan(cat_positions)]),
                        'avg_traffic': np.mean(cat_traffic[~np.isnan(cat_traffic)]),
                        'total_traffic': np.sum(cat_traffic[~np.isnan(cat_traffic)])
                    }
                else:
                    content_performance[category_name] = {
                        'keyword_count': 0,
                        'avg_position': 0,
                        'avg_traffic': 0,
                        'total_traffic': 0
                    }
        
        return {
            'chunk_size': len(chunk_df),
            'content_performance': content_performance
        }
    
    def _aggregate_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple chunks"""
        
        aggregated = {
            'total_chunks_processed': len(chunk_results),
            'total_keywords': sum(result.get('chunk_size', 0) for result in chunk_results),
            'content_categories': {}
        }
        
        # Aggregate content performance across chunks
        category_stats_accumulator = {}
        
        for result in chunk_results:
            content_perf = result.get('content_performance', {})
            for category, stats in content_perf.items():
                if category not in category_stats_accumulator:
                    category_stats_accumulator[category] = {
                        'total_keywords': 0, 
                        'sum_position_x_keywords': 0, 
                        'total_traffic': 0
                    }
                
                current_cat_stats = category_stats_accumulator[category]
                current_cat_stats['total_keywords'] += stats.get('keyword_count', 0)
                current_cat_stats['sum_position_x_keywords'] += (
                    stats.get('avg_position', 0) * stats.get('keyword_count', 0)
                )
                current_cat_stats['total_traffic'] += stats.get('total_traffic', 0)
        
        # Calculate final aggregated stats
        for category, acc_stats in category_stats_accumulator.items():
            total_keywords = acc_stats['total_keywords']
            weighted_avg_position = (acc_stats['sum_position_x_keywords'] / total_keywords 
                                   if total_keywords > 0 else 0)
            total_traffic = acc_stats['total_traffic']
            
            aggregated['content_categories'][category] = {
                'total_keywords': total_keywords,
                'avg_position': weighted_avg_position,
                'total_traffic': total_traffic,
                'efficiency_score': total_traffic / total_keywords if total_keywords > 0 else 0
            }
        
        return aggregated
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Fast correlation strength classification"""
        
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    # Featured Snippet Analysis
    def _analyze_featured_snippet_correlation(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze actual featured snippet correlation using real traffic patterns"""
        
        correlation_analysis = {
            'brands_analyzed': {},
            'overall_snippet_impact': 0,
            'confidence_level': 'low'
        }
        
        total_brands_analyzed = 0
        total_impact_score = 0
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            snippet_candidates = self._identify_featured_snippet_candidates(df)
            
            if not snippet_candidates.empty:
                snippet_analysis = self._analyze_snippet_performance(snippet_candidates, df)
                correlation_analysis['brands_analyzed'][brand] = snippet_analysis
                
                total_brands_analyzed += 1
                total_impact_score += snippet_analysis.get('traffic_impact_score', 0)
        
        if total_brands_analyzed > 0:
            correlation_analysis['overall_snippet_impact'] = total_impact_score / total_brands_analyzed
            correlation_analysis['confidence_level'] = ('high' if total_brands_analyzed >= 2 else 'medium')
        
        return correlation_analysis
    
    def _identify_featured_snippet_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify keywords likely to have featured snippets based on real patterns"""
        
        if df.empty or 'Position' not in df.columns or 'Traffic (%)' not in df.columns:
            return pd.DataFrame()
        
        candidates = pd.DataFrame()
        
        # Pattern 1: High traffic for top positions (potential snippet)
        top_positions = df[(df['Position'] <= 3)]
        if not top_positions.empty:
            for _, row in top_positions.iterrows():
                expected_traffic = self._estimate_expected_traffic(row['Position'])
                actual_traffic = row['Traffic (%)']
                
                if actual_traffic > expected_traffic * 1.3:  # 30% boost threshold
                    candidates = pd.concat([candidates, row.to_frame().T])
        
        # Pattern 2: Question-based keywords
        if 'Keyword' in df.columns:
            question_keywords = df[
                df['Keyword'].str.contains(
                    r'\b(?:what|how|why|when|where|which|who|is|are|can|will|should|do|does)\b',
                    case=False, na=False, regex=True
                )
            ]
            candidates = pd.concat([candidates, question_keywords]).drop_duplicates()
        
        return candidates.reset_index(drop=True)
    
    def _estimate_expected_traffic(self, position: float) -> float:
        """Estimate expected traffic based on position using industry CTR data"""
        
        ctr_by_position = {
            1: 28.5, 2: 15.7, 3: 11.0, 4: 8.0, 5: 7.2,
            6: 5.1, 7: 4.0, 8: 3.2, 9: 2.8, 10: 2.5
        }
        
        pos_int = int(position)
        return ctr_by_position.get(pos_int, 1.0)
    
    def _analyze_snippet_performance(self, snippet_candidates: pd.DataFrame, 
                                   all_keywords: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance of potential featured snippet keywords"""
        
        analysis = {
            'snippet_candidates_count': len(snippet_candidates),
            'total_keywords': len(all_keywords),
            'snippet_penetration': 0,
            'avg_traffic_boost': 0,
            'traffic_impact_score': 0,
            'top_snippet_keywords': []
        }
        
        if snippet_candidates.empty or all_keywords.empty:
            return analysis
        
        analysis['snippet_penetration'] = (len(snippet_candidates) / len(all_keywords)) * 100
        
        # Calculate average traffic boost
        traffic_boosts = []
        for _, row in snippet_candidates.iterrows():
            expected = self._estimate_expected_traffic(row['Position'])
            actual = row['Traffic (%)']
            if expected > 0:
                boost = (actual / expected) - 1
                traffic_boosts.append(boost)
        
        if traffic_boosts:
            analysis['avg_traffic_boost'] = sum(traffic_boosts) / len(traffic_boosts)
            analysis['traffic_impact_score'] = min(analysis['avg_traffic_boost'] * 50, 100)
        
        if 'Keyword' in snippet_candidates.columns:
            top_snippets = snippet_candidates.nlargest(5, 'Traffic (%)')
            analysis['top_snippet_keywords'] = top_snippets[['Keyword', 'Position', 'Traffic (%)']].to_dict('records')
        
        return analysis
    
    # Difficulty Analysis
    def _analyze_difficulty_conversion_correlation(self, data: Dict[str, pd.DataFrame],
                                                 gap_keywords: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze actual keyword difficulty vs performance correlation using real data"""
        
        correlation_analysis = {
            'brands_analyzed': {},
            'overall_correlation': 0,
            'correlation_strength': 'weak',
            'sample_size': 0
        }
        
        if gap_keywords is not None and not gap_keywords.empty and 'Keyword Difficulty' in gap_keywords.columns:
            difficulty_analysis = self._analyze_gap_keywords_difficulty(gap_keywords)
            correlation_analysis['gap_keywords_analysis'] = difficulty_analysis
        
        total_correlations = []
        total_sample_size = 0
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            difficulty_correlation = self._estimate_difficulty_from_performance(df)
            
            if difficulty_correlation['sample_size'] > 10:
                correlation_analysis['brands_analyzed'][brand] = difficulty_correlation
                total_correlations.append(difficulty_correlation['correlation'])
                total_sample_size += difficulty_correlation['sample_size']
        
        if total_correlations:
            correlation_analysis['overall_correlation'] = sum(total_correlations) / len(total_correlations)
            correlation_analysis['sample_size'] = total_sample_size
            
            abs_corr = abs(correlation_analysis['overall_correlation'])
            if abs_corr > 0.5:
                correlation_analysis['correlation_strength'] = 'strong'
            elif abs_corr > 0.3:
                correlation_analysis['correlation_strength'] = 'moderate'
            else:
                correlation_analysis['correlation_strength'] = 'weak'
        
        return correlation_analysis
    
    def _analyze_gap_keywords_difficulty(self, gap_keywords: pd.DataFrame) -> Dict[str, Any]:
        """Analyze keyword difficulty from gap keywords data"""
        
        analysis = {
            'total_gap_keywords': len(gap_keywords),
            'avg_difficulty': 0,
            'difficulty_distribution': {},
            'opportunity_score': 0
        }
        
        if 'Keyword Difficulty' not in gap_keywords.columns:
            return analysis
        
        difficulties = gap_keywords['Keyword Difficulty']
        analysis['avg_difficulty'] = difficulties.mean()
        
        analysis['difficulty_distribution'] = {
            'easy': len(difficulties[difficulties <= 30]),
            'medium': len(difficulties[(difficulties > 30) & (difficulties <= 60)]),
            'hard': len(difficulties[difficulties > 60])
        }
        
        easy_ratio = analysis['difficulty_distribution']['easy'] / len(gap_keywords) if len(gap_keywords) > 0 else 0
        analysis['opportunity_score'] = easy_ratio * 100
        
        return analysis
    
    def _estimate_difficulty_from_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate keyword difficulty patterns from current performance"""
        
        difficulty_analysis = {
            'correlation': 0,
            'sample_size': len(df),
            'performance_patterns': {}
        }
        
        if df.empty or 'Position' not in df.columns or 'Traffic (%)' not in df.columns:
            return difficulty_analysis
        
        estimated_difficulty = []
        performance_scores = []
        
        for _, row in df.iterrows():
            position = row['Position']
            traffic = row['Traffic (%)']
            
            if position <= 10:
                expected_traffic = self._estimate_expected_traffic(position)
                traffic_efficiency = traffic / max(expected_traffic, 0.1)
                
                estimated_diff = max(0, 100 - (traffic_efficiency * 50))
                estimated_difficulty.append(estimated_diff)
                
                perf_score = (10 - position) * 10 + traffic * 2
                performance_scores.append(perf_score)
        
        if len(estimated_difficulty) >= 5 and len(performance_scores) == len(estimated_difficulty):
            try:
                correlation, p_value = pearsonr(estimated_difficulty, performance_scores)
                difficulty_analysis['correlation'] = correlation
                difficulty_analysis['p_value'] = p_value
            except ValueError:
                difficulty_analysis['correlation'] = 0
                difficulty_analysis['p_value'] = 1.0
        
        difficulty_analysis['performance_patterns'] = {
            'avg_estimated_difficulty': np.mean(estimated_difficulty) if estimated_difficulty else 0,
            'avg_performance_score': np.mean(performance_scores) if performance_scores else 0,
            'high_difficulty_keywords': len([d for d in estimated_difficulty if d > 70]),
            'low_difficulty_keywords': len([d for d in estimated_difficulty if d < 30])
        }
        
        return difficulty_analysis
    
    # Backlink Analysis
    def _analyze_backlink_ranking_correlation(self, data: Dict[str, pd.DataFrame],
                                            historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze actual backlink patterns and ranking correlation using real data"""
        
        backlink_analysis = {
            'position_authority_correlation': {},
            'traffic_authority_correlation': {},
            'ranking_improvement_patterns': {},
            'overall_correlation': 0
        }
        
        correlations = []
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            authority_analysis = self._estimate_authority_from_performance(df, brand)
            
            if authority_analysis['sample_size'] > 10:
                backlink_analysis['position_authority_correlation'][brand] = authority_analysis
                correlations.append(authority_analysis.get('position_correlation', 0))
        
        if historical_data:
            improvement_analysis = self._analyze_ranking_improvements(data, historical_data)
            backlink_analysis['ranking_improvement_patterns'] = improvement_analysis
        
        if correlations:
            backlink_analysis['overall_correlation'] = sum(correlations) / len(correlations)
        
        return backlink_analysis
    
    def _estimate_authority_from_performance(self, df: pd.DataFrame, brand: str) -> Dict[str, Any]:
        """Estimate domain authority patterns from performance data"""
        
        authority_analysis = {
            'sample_size': len(df),
            'position_correlation': 0,
            'authority_indicators': {},
            'competitive_strength': 0
        }
        
        if df.empty or 'Position' not in df.columns:
            return authority_analysis
        
        # Authority indicators from performance patterns
        top_10_count = len(df[df['Position'] <= 10])
        top_3_count = len(df[df['Position'] <= 3])
        total_traffic = df['Traffic (%)'].sum() if 'Traffic (%)' in df.columns else 0
        
        authority_analysis['authority_indicators'] = {
            'top_10_ratio': top_10_count / len(df) if len(df) > 0 else 0,
            'top_3_ratio': top_3_count / len(df) if len(df) > 0 else 0,
            'avg_position': df['Position'].mean(),
            'total_traffic': total_traffic,
            'traffic_efficiency': total_traffic / len(df) if len(df) > 0 else 0
        }
        
        # Competitive strength score
        strength_score = (
            authority_analysis['authority_indicators']['top_10_ratio'] * 40 +
            authority_analysis['authority_indicators']['top_3_ratio'] * 30 +
            min(authority_analysis['authority_indicators']['traffic_efficiency'], 10) * 3
        )
        
        authority_analysis['competitive_strength'] = strength_score
        
        # Position correlation calculation
        positions = df['Position'].values
        
        if len(positions) >= 5:
            try:
                if 'Traffic (%)' in df.columns:
                    traffic_values = df['Traffic (%)'].values
                    
                    authority_proxy = []
                    for i, pos in enumerate(positions):
                        traffic = traffic_values[i] if i < len(traffic_values) else 0
                        expected_traffic = self._estimate_expected_traffic(pos)
                        
                        if expected_traffic > 0:
                            efficiency = traffic / expected_traffic
                        else:
                            efficiency = 0
                        
                        authority_proxy.append(efficiency)
                    
                    if (len(set(positions)) > 1 and len(set(authority_proxy)) > 1):
                        correlation, _ = pearsonr(positions, authority_proxy)
                        authority_analysis['position_correlation'] = correlation
                    else:
                        authority_analysis['position_correlation'] = 0
                        authority_analysis['correlation_note'] = 'constant_data'
                else:
                    position_std = df['Position'].std()
                    if position_std > 0:
                        consistency_score = 1 / (1 + position_std)
                        authority_analysis['position_correlation'] = -0.5 * consistency_score
                    else:
                        authority_analysis['position_correlation'] = 0
                        authority_analysis['correlation_note'] = 'no_position_variance'
                        
            except Exception as e:
                authority_analysis['position_correlation'] = 0
                authority_analysis['correlation_error'] = str(e)
        
        return authority_analysis
    
    def _analyze_ranking_improvements(self, current_data: Dict[str, pd.DataFrame],
                                    historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze ranking improvement patterns"""
        
        improvement_analysis = {
            'brands_with_improvements': {},
            'improvement_velocity': {},
            'correlation_with_authority': 0
        }
        
        for brand in current_data.keys():
            if brand == 'gap_keywords':
                continue
            
            current_df = current_data[brand]
            historical_df = historical_data.get(brand, pd.DataFrame())
            
            if current_df.empty or historical_df.empty:
                continue
            
            improvements = self._calculate_position_improvements(current_df, historical_df)
            
            if improvements:
                improvement_analysis['brands_with_improvements'][brand] = improvements
                avg_improvement = sum(imp['improvement'] for imp in improvements) / len(improvements)
                improvement_analysis['improvement_velocity'][brand] = avg_improvement
        
        return improvement_analysis
    
    def _calculate_position_improvements(self, current_df: pd.DataFrame,
                                       historical_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate position improvements between datasets"""
        
        if 'Keyword' not in current_df.columns or 'Keyword' not in historical_df.columns:
            return []
        
        merged = pd.merge(
            current_df[['Keyword', 'Position', 'Traffic (%)']],
            historical_df[['Keyword', 'Position', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_current', '_historical')
        )
        
        merged = merged.dropna(subset=[
            'Position_current', 'Position_historical', 
            'Traffic (%)_current', 'Traffic (%)_historical'
        ])
        
        merged['Position_current'] = pd.to_numeric(merged['Position_current'], errors='coerce')
        merged['Position_historical'] = pd.to_numeric(merged['Position_historical'], errors='coerce')
        merged['Traffic (%)_current'] = pd.to_numeric(merged['Traffic (%)_current'], errors='coerce')
        merged['Traffic (%)_historical'] = pd.to_numeric(merged['Traffic (%)_historical'], errors='coerce')
        
        merged = merged.dropna(subset=[
            'Position_current', 'Position_historical', 
            'Traffic (%)_current', 'Traffic (%)_historical'
        ])
        
        improvements = []
        for _, row in merged.iterrows():
            position_change = row['Position_historical'] - row['Position_current']  # Positive = improvement
            
            if position_change > 2:  # Significant improvement
                improvements.append({
                    'keyword': row['Keyword'],
                    'improvement': position_change,
                    'current_position': row['Position_current'],
                    'historical_position': row['Position_historical'],
                    'traffic_change': row['Traffic (%)_current'] - row['Traffic (%)_historical']
                })
        
        return improvements
    
    # SERP Features Analysis
    def _analyze_serp_features_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze actual SERP features correlation using real traffic patterns"""
        
        if 'Traffic (%)' not in df.columns or 'Position' not in df.columns:
            return {'correlation': 0, 'significance': 'no_data'}
        
        serp_features_analysis = self._detect_serp_features_from_traffic(df)
        traffic = df['Traffic (%)'].values
        feature_indicators = serp_features_analysis['feature_indicators']
        
        if len(feature_indicators) == 0:
            return {'correlation': 0, 'significance': 'no_features_detected'}
        
        features_present_traffic = [
            traffic[i] for i in range(len(traffic)) 
            if i < len(feature_indicators) and feature_indicators[i] == 1
        ]
        features_absent_traffic = [
            traffic[i] for i in range(len(traffic)) 
            if i < len(feature_indicators) and feature_indicators[i] == 0
        ]
        
        if len(features_present_traffic) == 0 or len(features_absent_traffic) == 0:
            return {'correlation': 0, 'significance': 'insufficient_data'}
        
        mean_with_features = np.mean(features_present_traffic)
        mean_without_features = np.mean(features_absent_traffic)
        
        if (mean_with_features + mean_without_features) > 0:
            correlation_effect = ((mean_with_features - mean_without_features) / 
                                (mean_with_features + mean_without_features))
        else:
            correlation_effect = 0
        
        return {
            'correlation_effect': correlation_effect,
            'mean_traffic_with_features': mean_with_features,
            'mean_traffic_without_features': mean_without_features,
            'features_detected_count': len(features_present_traffic),
            'non_features_count': len(features_absent_traffic),
            'significance': ('strong' if abs(correlation_effect) > 0.3 else 
                           'moderate' if abs(correlation_effect) > 0.1 else 'weak'),
            'detection_method': 'traffic_pattern_analysis'
        }
    
    def _detect_serp_features_from_traffic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect likely SERP features from traffic patterns"""
        
        feature_detection = {
            'feature_indicators': [],
            'detection_confidence': 'low',
            'total_features_detected': 0
        }
        
        if df.empty or 'Position' not in df.columns or 'Traffic (%)' not in df.columns:
            return feature_detection
        
        feature_indicators = []
        
        for _, row in df.iterrows():
            position = row['Position']
            traffic = row['Traffic (%)']
            
            expected = self._estimate_expected_traffic(position)
            
            if traffic > expected * 1.2:  # 20% boost threshold
                feature_indicators.append(1)
            else:
                feature_indicators.append(0)
        
        feature_detection['feature_indicators'] = feature_indicators
        feature_detection['total_features_detected'] = sum(feature_indicators)
        
        if len(feature_indicators) > 0:
            feature_ratio = sum(feature_indicators) / len(feature_indicators)
            if 0.1 <= feature_ratio <= 0.4:
                feature_detection['detection_confidence'] = 'high'
            elif 0.05 <= feature_ratio <= 0.6:
                feature_detection['detection_confidence'] = 'medium'
        
        return feature_detection
    
    # Temporal and Competitive Analysis
    def _analyze_temporal_correlations(self, current_data: Dict[str, pd.DataFrame],
                                     historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze temporal correlations and trends"""
        
        if not historical_data:
            return {'status': 'no_historical_data'}
        
        temporal_analysis = {
            'trend_correlations': {},
            'momentum_analysis': {},
            'cyclical_patterns': {}
        }
        
        for brand in current_data.keys():
            if brand == 'gap_keywords':
                continue
            
            current_df = current_data[brand]
            historical_df = historical_data.get(brand, pd.DataFrame())
            
            if current_df.empty or historical_df.empty:
                continue
            
            trend_analysis = self._analyze_brand_trends(current_df, historical_df)
            temporal_analysis['trend_correlations'][brand] = trend_analysis
        
        return temporal_analysis
    
    def _analyze_brand_trends(self, current_df: pd.DataFrame, historical_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends for a specific brand"""
        
        if 'Keyword' not in current_df.columns or 'Keyword' not in historical_df.columns:
            return {'status': 'insufficient_data'}
        
        merged = pd.merge(
            current_df[['Keyword', 'Position', 'Traffic (%)']],
            historical_df[['Keyword', 'Position', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_current', '_historical')
        )
        
        if merged.empty:
            return {'status': 'no_common_keywords'}
        
        merged['position_change'] = merged['Position_current'] - merged['Position_historical']
        merged['traffic_change'] = merged['Traffic (%)_current'] - merged['Traffic (%)_historical']
        
        try:
            position_trend_correlation, _ = pearsonr(merged['Position_historical'], merged['position_change'])
            traffic_trend_correlation, _ = pearsonr(merged['Traffic (%)_historical'], merged['traffic_change'])
        except:
            position_trend_correlation = 0
            traffic_trend_correlation = 0
        
        return {
            'common_keywords': len(merged),
            'avg_position_change': merged['position_change'].mean(),
            'avg_traffic_change': merged['traffic_change'].mean(),
            'position_trend_correlation': position_trend_correlation,
            'traffic_trend_correlation': traffic_trend_correlation,
            'trend_direction': 'improving' if merged['position_change'].mean() < 0 else 'declining',
            'momentum_strength': abs(merged['position_change'].mean())
        }
    
    def _analyze_competitive_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze correlations in competitive dynamics"""
        
        competitive_analysis = {
            'market_correlation': {},
            'inverse_relationships': {},
            'competitive_pressure': {}
        }
        
        competitors = [brand for brand in data.keys() if brand != 'gap_keywords']
        
        market_metrics = self._extract_market_metrics(data)
        competitive_analysis['market_correlation'] = self._analyze_market_correlations(market_metrics)
        competitive_analysis['inverse_relationships'] = self._identify_inverse_relationships(data)
        
        return competitive_analysis
    
    def _extract_market_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Extract market-level metrics"""
        
        metrics = {}
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            metrics[brand] = {
                'total_traffic': df['Traffic (%)'].sum() if 'Traffic (%)' in df.columns else 0,
                'avg_position': df['Position'].mean() if 'Position' in df.columns else 0,
                'keyword_count': len(df),
                'top_10_share': (len(df[df['Position'] <= 10]) / len(df) 
                               if 'Position' in df.columns and len(df) > 0 else 0)
            }
        
        return metrics
    
    def _analyze_market_correlations(self, market_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze correlations in market metrics"""
        
        if len(market_metrics) < 2:
            return {'status': 'insufficient_competitors'}
        
        traffic_values = [metrics['total_traffic'] for metrics in market_metrics.values()]
        position_values = [metrics['avg_position'] for metrics in market_metrics.values()]
        
        try:
            traffic_position_corr, _ = pearsonr(traffic_values, position_values)
        except:
            traffic_position_corr = 0
        
        return {
            'traffic_position_correlation': traffic_position_corr,
            'market_concentration': (np.std(traffic_values) / np.mean(traffic_values) 
                                   if np.mean(traffic_values) > 0 else 0),
            'competitive_balance': ('balanced' if np.std(traffic_values) / np.mean(traffic_values) < 0.5 
                                  else 'concentrated')
        }
    
    def _identify_inverse_relationships(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Identify inverse competitive relationships"""
        
        inverse_relationships = []
        competitors = [brand for brand in data.keys() if brand != 'gap_keywords']
        
        for i, comp1 in enumerate(competitors):
            for comp2 in competitors[i+1:]:
                relationship = self._analyze_competitive_relationship(data[comp1], data[comp2])
                if relationship['relationship_type'] == 'inverse':
                    inverse_relationships.append({
                        'competitor_1': comp1,
                        'competitor_2': comp2,
                        'correlation_strength': relationship['correlation_strength'],
                        'evidence': relationship['evidence']
                    })
        
        return inverse_relationships
    
    def _analyze_competitive_relationship(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """Analyze competitive relationship between two brands"""
        
        if (df1.empty or df2.empty or 'Keyword' not in df1.columns or 'Keyword' not in df2.columns):
            return {'relationship_type': 'unknown', 'correlation_strength': 0}
        
        common_keywords = pd.merge(
            df1[['Keyword', 'Position', 'Traffic (%)']],
            df2[['Keyword', 'Position', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_1', '_2')
        )
        
        if len(common_keywords) < 5:
            return {'relationship_type': 'insufficient_overlap', 'correlation_strength': 0}
        
        try:
            traffic_correlation, _ = pearsonr(
                common_keywords['Traffic (%)_1'],
                common_keywords['Traffic (%)_2']
            )
        except:
            traffic_correlation = 0
        
        if traffic_correlation < -0.3:
            relationship_type = 'inverse'
        elif traffic_correlation > 0.3:
            relationship_type = 'positive'
        else:
            relationship_type = 'neutral'
        
        return {
            'relationship_type': relationship_type,
            'correlation_strength': abs(traffic_correlation),
            'common_keywords_count': len(common_keywords),
            'evidence': f"Traffic correlation: {traffic_correlation:.3f}"
        }
    
    def _analyze_predictive_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze predictive correlations for forecasting"""
        
        predictive_analysis = {
            'leading_indicators': {},
            'lagging_indicators': {},
            'predictive_models': {}
        }
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            leading = self._identify_leading_indicators(df)
            predictive_analysis['leading_indicators'][brand] = leading
            
            model = self._build_predictive_model(df)
            predictive_analysis['predictive_models'][brand] = model
        
        return predictive_analysis
    
    def _identify_leading_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify metrics that predict future performance"""
        
        if df.empty or 'Position' not in df.columns:
            return {'status': 'insufficient_data'}
        
        indicators = {
            'position_momentum': self._calculate_position_momentum(df),
            'traffic_velocity': self._calculate_traffic_velocity(df),
            'ranking_stability': self._calculate_ranking_stability(df)
        }
        
        return indicators
    
    def _calculate_position_momentum(self, df: pd.DataFrame) -> float:
        """Calculate position momentum indicator"""
        
        page_1_ratio = len(df[df['Position'] <= 10]) / len(df) if len(df) > 0 else 0
        avg_position = df['Position'].mean()
        momentum = (page_1_ratio * 50) + max(0, (20 - avg_position))
        return min(momentum, 100)
    
    def _calculate_traffic_velocity(self, df: pd.DataFrame) -> float:
        """Calculate traffic velocity indicator"""
        
        if 'Traffic (%)' not in df.columns:
            return 0
        
        total_traffic = df['Traffic (%)'].sum()
        high_traffic_keywords = len(df[df['Traffic (%)'] > 1.0])
        velocity = (total_traffic * 2) + (high_traffic_keywords * 5)
        return min(velocity, 100)
    
    def _calculate_ranking_stability(self, df: pd.DataFrame) -> float:
        """Calculate ranking stability indicator"""
        
        position_std = df['Position'].std()
        stability = max(0, 100 - (position_std * 2))
        return stability
    
    def _build_predictive_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build simple predictive model"""
        
        if df.empty or len(df) < 10:
            return {'status': 'insufficient_data'}
        
        try:
            X = df[['Position']].values
            y = df['Traffic (%)'].values if 'Traffic (%)' in df.columns else np.zeros(len(df))
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            r2_score = model.score(X_scaled, y)
            
            return {
                'model_type': 'linear_regression',
                'r2_score': r2_score,
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_,
                'feature_importance': {
                    'position': abs(model.coef_[0]) if len(model.coef_) > 0 else 0
                },
                'model_quality': ('good' if r2_score > 0.5 else 
                                'fair' if r2_score > 0.3 else 'poor')
            }
            
        except Exception as e:
            return {'status': 'model_failed', 'error': str(e)}
    
    def _generate_correlation_insights(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from correlation analysis"""
        
        insights = []
        
        # Primary correlations insights
        primary = analysis.get('primary_correlations', {})
        position_traffic = primary.get('position_traffic_correlation', {})
        
        for brand, correlation_data in position_traffic.items():
            if isinstance(correlation_data, dict):
                correlation = correlation_data.get('correlation', 0)
                significance = correlation_data.get('significance', 'unknown')
                
                if significance in ['significant', 'highly_significant']:
                    insights.append({
                        'type': 'primary_correlation',
                        'brand': brand,
                        'insight': f'Strong position-traffic correlation for {brand} (r={correlation:.3f})',
                        'implication': 'Position improvements will likely increase traffic',
                        'action': 'Focus on ranking improvements for high-potential keywords'
                    })
        
        # Advanced correlations insights
        advanced = analysis.get('advanced_correlations', {})
        cross_competitor = advanced.get('cross_competitor_correlations', {})
        
        for comparison, correlation_data in cross_competitor.items():
            if isinstance(correlation_data, dict):
                avg_correlation = correlation_data.get('avg_correlation', 0)
                if abs(avg_correlation) > 0.5:
                    comp1, comp2 = comparison.split('_vs_')
                    insights.append({
                        'type': 'competitive_correlation',
                        'brands': [comp1, comp2],
                        'insight': f'High correlation between {comp1} and {comp2} performance',
                        'implication': 'Similar SEO strategies or market dynamics',
                        'action': 'Analyze successful strategies from both competitors'
                    })
        
        # Predictive correlations insights
        predictive = analysis.get('predictive_correlations', {})
        for brand, predictive_data in predictive.items():
            if isinstance(predictive_data, dict):
                leading_indicators = predictive_data.get('leading_indicators', {})
                position_momentum = leading_indicators.get('position_momentum', 0)
                if position_momentum > 70:
                    insights.append({
                        'type': 'predictive_indicator',
                        'brand': brand,
                        'insight': f'Strong position momentum for {brand} indicates future growth',
                        'implication': 'Likely to see continued ranking improvements',
                        'action': 'Maintain current SEO strategy and scale successful tactics'
                    })
        
        return insights

# Optional GPU-accelerated version
try:
    import cupy as cp
    
    class GPUCorrelationEngine(CorrelationEngine):
        """GPU-accelerated correlation analysis (requires CUDA GPU and CuPy)"""
        
        def __init__(self, chunk_size: int = 1000, n_jobs: int = 4, top_k: int = 100):
            super().__init__(chunk_size, n_jobs, top_k)
            self.logger.info("GPUCorrelationEngine initialized. CuPy will be used for Pearson correlation.")
        
        def _fast_pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
            """GPU-accelerated Pearson correlation using CuPy"""
            
            if x.size == 0 or y.size == 0:
                return 0.0
            
            x_gpu = cp.asarray(x)
            y_gpu = cp.asarray(y)
            
            if cp.std(x_gpu) == 0 or cp.std(y_gpu) == 0:
                return 0.0
            
            try:
                correlation_matrix = cp.corrcoef(x_gpu, y_gpu)
                correlation = correlation_matrix[0, 1]
                return float(cp.asnumpy(correlation) if isinstance(correlation, cp.ndarray) else correlation)
            except Exception as e:
                self.logger.warning(f"CuPy corrcoef failed: {e}. Falling back to 0.")
                return 0.0
    
    print("✅ CuPy found! GPUCorrelationEngine is available for GPU acceleration.")
    
except ImportError:
    print("💡 Install CuPy for GPU acceleration: pip install cupy-cuda11x")
    GPUCorrelationEngine = None
