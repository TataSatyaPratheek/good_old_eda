"""
High-Performance TOFU (Top of Funnel) Analysis Module
Optimized for blazing fast performance with large datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TOFUMetrics:
    """TOFU-specific metrics structure"""
    non_branded_keywords: int = 0
    non_branded_traffic: float = 0.0
    brand_awareness_score: float = 0.0
    customer_acquisition_potential: float = 0.0
    funnel_efficiency: float = 0.0
    competitive_vulnerability: float = 0.0

class TOFUAnalyzer:
    """🚀 BLAZING FAST Enhanced TOFU analysis for customer acquisition insights"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_tofu_performance(self, data: Dict[str, pd.DataFrame], 
                                timeframe_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        🚀 BLAZING FAST Comprehensive TOFU analysis focusing on customer acquisition
        """
        
        print("🎯 Analyzing TOFU Performance (High-Performance Mode)...")
        start_time = datetime.now()
        
        # Pre-process all data for vectorized operations
        processed_data = self._preprocess_data_for_vectorization(data)
        
        analysis = {
            'non_branded_analysis': self._fast_analyze_non_branded_performance(processed_data),
            'customer_acquisition_funnel': self._fast_analyze_customer_acquisition_funnel(processed_data),
            'tofu_competitive_analysis': self._analyze_tofu_competitive_landscape(processed_data),
            'acquisition_opportunities': self._identify_acquisition_opportunities(processed_data),
            'tofu_risk_assessment': self._assess_tofu_risks(processed_data),
            'funnel_optimization': self._generate_funnel_optimization_recommendations(processed_data)
        }
        
        # Add temporal analysis if historical data available
        if timeframe_data:
            processed_historical = self._preprocess_data_for_vectorization(timeframe_data)
            analysis['tofu_trends'] = self._analyze_tofu_trends(processed_data, processed_historical)
        
        # Calculate overall TOFU score
        analysis['tofu_score'] = self._calculate_tofu_score(analysis)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"🎉 TOFU analysis completed in {processing_time:.2f} seconds")
        
        return analysis
    
    def _preprocess_data_for_vectorization(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """🚀 Pre-process data for maximum vectorization efficiency"""
        
        processed = {}
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                processed[brand] = df
                continue
            
            # Ensure required columns exist and are properly typed
            if 'Keyword' in df.columns:
                df_processed = df.copy()
                
                # Vectorized string preprocessing
                df_processed['Keyword_lower'] = df_processed['Keyword'].str.lower()
                df_processed['Keyword_length'] = df_processed['Keyword'].str.len()
                df_processed['Word_count'] = df_processed['Keyword'].str.split().str.len()
                
                # Pre-compute brand identification
                df_processed['Is_branded'] = self._vectorized_brand_identification(df_processed, brand)
                
                # Pre-compute intent classification
                df_processed = self._vectorized_intent_classification(df_processed)
                
                # Ensure numeric columns
                numeric_cols = ['Position', 'Traffic (%)']
                for col in numeric_cols:
                    if col in df_processed.columns:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                processed[brand] = df_processed
            else:
                processed[brand] = df
        
        return processed
    
    def _vectorized_brand_identification(self, df: pd.DataFrame, brand: str) -> pd.Series:
        """🚀 BLAZING FAST vectorized brand keyword identification"""
        
        brand_terms = [brand.lower(), brand.replace('.com', '')]
        
        # Vectorized string contains operation
        brand_pattern = '|'.join(brand_terms)
        return df['Keyword_lower'].str.contains(brand_pattern, na=False, regex=True)
    
    def _vectorized_intent_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """🚀 BLAZING FAST vectorized search intent classification"""
        
        intent_patterns = {
            'informational': r'\b(?:how|what|why|guide|tutorial|tips|best practices)\b',
            'commercial': r'\b(?:best|vs|compare|review|top|alternatives)\b',
            'navigational': r'\b(?:login|support|contact|download)\b',
            'transactional': r'\b(?:buy|price|cost|deal|offer|discount)\b'
        }
        
        # Vectorized intent classification
        for intent, pattern in intent_patterns.items():
            df[f'Intent_{intent}'] = df['Keyword_lower'].str.contains(pattern, na=False, regex=True)
        
        return df
    
    def _fast_analyze_non_branded_performance(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """🚀 BLAZING FAST vectorized non-branded keyword performance analysis"""
        
        non_branded_analysis = {}
        
        # Process all brands in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_brand = {
                executor.submit(self._process_brand_non_branded, brand, df): brand
                for brand, df in data.items()
                if brand != 'gap_keywords' and not df.empty and 'Is_branded' in df.columns
            }
            
            for future in as_completed(future_to_brand):
                brand = future_to_brand[future]
                try:
                    result = future.result()
                    if result:
                        non_branded_analysis[brand] = result
                except Exception as e:
                    self.logger.error(f"Error processing brand {brand}: {e}")
        
        # Competitive comparison
        if non_branded_analysis:
            non_branded_analysis['competitive_comparison'] = self._compare_non_branded_performance(non_branded_analysis)
        
        return non_branded_analysis
    
    def _process_brand_non_branded(self, brand: str, df: pd.DataFrame) -> Dict[str, Any]:
        """🚀 Process single brand non-branded analysis with vectorization"""
        
        # BLAZING FAST vectorized filtering
        non_branded_mask = ~df['Is_branded']
        non_branded_keywords = df[non_branded_mask]
        
        if non_branded_keywords.empty:
            return None
        
        # VECTORIZED CALCULATIONS
        
        analysis = {
            'total_non_branded_keywords': non_branded_keywords.shape[0],
            'avg_non_branded_position': non_branded_keywords['Position'].mean() if 'Position' in non_branded_keywords.columns else 0,
            'non_branded_traffic_share': non_branded_keywords['Traffic (%)'].sum() if 'Traffic (%)' in non_branded_keywords.columns else 0,
            'top_10_non_branded': (non_branded_keywords['Position'] <= 10).sum() if 'Position' in non_branded_keywords.columns else 0,
            'page_1_non_branded': (non_branded_keywords['Position'] <= 10).sum() if 'Position' in non_branded_keywords.columns else 0,
            'customer_acquisition_potential': self._calculate_acquisition_potential(non_branded_keywords),
            'intent_distribution': self._analyze_search_intent(non_branded_keywords),
            'difficulty_vs_opportunity': self._analyze_difficulty_opportunity_matrix(non_branded_keywords),
            'top_acquisition_keywords': self._identify_top_acquisition_keywords(non_branded_keywords)
        }
        
        return analysis

    def _identify_non_branded_keywords(self, df: pd.DataFrame, brand: str) -> pd.DataFrame:
        """Identify non-branded keywords"""
        
        if 'Keyword' not in df.columns:
            return pd.DataFrame()
        
        brand_terms = [brand.lower(), brand.replace('.com', '')]
        
        # Filter out branded keywords
        non_branded = df[~df['Keyword'].str.lower().str.contains('|'.join(brand_terms), na=False)]
        
        return non_branded
    
    def _calculate_acquisition_potential(self, keywords_df: pd.DataFrame) -> float:

        """Calculate customer acquisition potential score"""
        if keywords_df.empty:
            return 0.0
        
        # Factors: search volume, position, commercial intent, competition
        factors = []
        
        # Position factor (lower position = higher potential)
        avg_position = keywords_df['Position'].mean()
        position_factor = max(0, (20 - avg_position) / 20)  # Normalize to 0-1
        factors.append(position_factor * 0.3)
        
        # Traffic potential factor
        traffic_share = keywords_df['Traffic (%)'].sum()
        traffic_factor = min(traffic_share / 10, 1.0)  # Normalize, cap at 1
        factors.append(traffic_factor * 0.4)
    
        """Calculate customer acquisition potential score"""
        
        if keywords_df.empty:
            return 0.0
        
        # Factors: search volume, position, commercial intent, competition
        factors = []
        
        # Position factor (lower position = higher potential)
        avg_position = keywords_df['Position'].mean()
        position_factor = max(0, (20 - avg_position) / 20)  # Normalize to 0-1
        factors.append(position_factor * 0.3)
        
        # Traffic potential factor
        traffic_share = keywords_df['Traffic (%)'].sum()
        traffic_factor = min(traffic_share / 10, 1.0)  # Normalize, cap at 1
        factors.append(traffic_factor * 0.4)
        
        # Keywords count factor (more keywords = more potential)
        keyword_count_factor = min(len(keywords_df) / 100, 1.0)  # Normalize
        factors.append(keyword_count_factor * 0.3)
        
        return sum(factors) * 100  # Convert to 0-100 score
    
    def _analyze_search_intent(self, keywords_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze search intent distribution for TOFU keywords"""
        
        if keywords_df.empty or 'Keyword' not in keywords_df.columns:
            return {}
        
        intent_patterns = {
            'informational': ['how', 'what', 'why', 'guide', 'tutorial', 'tips', 'best practices'],
            'commercial': ['best', 'vs', 'compare', 'review', 'top', 'alternatives'],
            'navigational': ['login', 'support', 'contact', 'download'],
            'transactional': ['buy', 'price', 'cost', 'deal', 'offer', 'discount']
        }
        
        intent_counts = {intent: 0 for intent in intent_patterns.keys()}
        
        for _, row in keywords_df.iterrows():
            keyword = str(row['Keyword']).lower() # Ensure keyword is a string
            classified = False
            
            for intent, patterns in intent_patterns.items():
                if any(pattern in keyword for pattern in patterns):
                    intent_counts[intent] += 1
                    classified = True
                    break
            
            if not classified:
                intent_counts.setdefault('other', 0)
                intent_counts['other'] += 1
        
        return intent_counts
    

    def _analyze_difficulty_opportunity_matrix(self, keywords_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze difficulty vs opportunity matrix"""
        
        if keywords_df.empty:
            return {}
        
        # Simulate difficulty analysis (would normally come from keyword tools)
        keywords_df_copy = keywords_df.copy()
        keywords_df_copy['Estimated_Difficulty'] = np.random.randint(20, 80, len(keywords_df_copy))
        keywords_df_copy['Opportunity_Score'] = (
            (keywords_df_copy['Traffic (%)'] * 2) +  # Traffic weight
            ((21 - keywords_df_copy['Position']) / 20 * 50) +  # Position weight
            ((100 - keywords_df_copy['Estimated_Difficulty']) / 100 * 30)  # Difficulty weight
        )
        
        # Categorize opportunities
        high_opportunity = keywords_df_copy[
            (keywords_df_copy['Opportunity_Score'] > 60) & 
            (keywords_df_copy['Estimated_Difficulty'] < 50)
        ]
        
        medium_opportunity = keywords_df_copy[
            (keywords_df_copy['Opportunity_Score'] > 40) & 
            (keywords_df_copy['Opportunity_Score'] <= 60)
        ]
        
        return {
            'high_opportunity_keywords': len(high_opportunity),
            'medium_opportunity_keywords': len(medium_opportunity),
            'avg_difficulty': keywords_df_copy['Estimated_Difficulty'].mean(),
            'avg_opportunity_score': keywords_df_copy['Opportunity_Score'].mean(),
            'top_opportunities': high_opportunity.nlargest(10, 'Opportunity_Score')[
                ['Keyword', 'Position', 'Traffic (%)', 'Opportunity_Score']
            ].to_dict('records')
        }
    
    def _identify_top_acquisition_keywords(self, keywords_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify top customer acquisition keywords"""
        
        if keywords_df.empty:
            return []
        
        # Score keywords for acquisition potential
        scored_keywords = keywords_df.copy()
        scored_keywords['Acquisition_Score'] = (
            (scored_keywords['Traffic (%)'] * 3) +  # High traffic weight
            ((21 - scored_keywords['Position']) / 20 * 40) +  # Position potential
            (scored_keywords['Keyword'].str.len() > 10).astype(int) * 10  # Long-tail bonus
        )
        
        # Get top acquisition opportunities
        top_keywords = scored_keywords.nlargest(15, 'Acquisition_Score')
        
        return top_keywords[['Keyword', 'Position', 'Traffic (%)', 'Acquisition_Score']].to_dict('records')
    
    def _compare_non_branded_performance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare non-branded performance across competitors"""
        
        comparison = {
            'performance_ranking': {},
            'share_of_voice': {},
            'opportunity_gaps': {}
        }
        
        brands = [brand for brand in analysis.keys() if brand != 'competitive_comparison']
        
        if not brands:
            return comparison
        
        # Performance ranking
        for brand in brands:
            brand_data = analysis[brand]
            performance_score = (
                brand_data.get('non_branded_traffic_share', 0) * 0.4 +
                brand_data.get('customer_acquisition_potential', 0) * 0.3 +
                (100 - brand_data.get('avg_non_branded_position', 100)) * 0.3
            )
            comparison['performance_ranking'][brand] = performance_score
        
        # Share of voice
        total_traffic = sum(analysis[brand].get('non_branded_traffic_share', 0) for brand in brands)
        if total_traffic > 0:
            for brand in brands:
                brand_traffic = analysis[brand].get('non_branded_traffic_share', 0)
                comparison['share_of_voice'][brand] = (brand_traffic / total_traffic) * 100
        
        return comparison
    
    def _fast_analyze_customer_acquisition_funnel(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze customer acquisition funnel performance"""
        
        funnel_analysis = {
            'awareness_stage': {},
            'consideration_stage': {},
            'decision_stage': {},
            'funnel_efficiency': {}
        }
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            non_branded = self._identify_non_branded_keywords(df, brand)
            if non_branded.empty or non_branded.shape[0] == 0:
                continue
            
            # Awareness stage (informational keywords)
            awareness_keywords = non_branded[
                non_branded['Keyword'].str.contains(
                    '|'.join(['how', 'what', 'guide', 'tutorial', 'tips']), 
                    case=False, na=False
                )
            ]
            
            # Consideration stage (comparison keywords)
            consideration_keywords = non_branded[
                non_branded['Keyword'].str.contains(
                    '|'.join(['vs', 'compare', 'best', 'review', 'alternatives']), 
                    case=False, na=False
                )
            ]
            
            # Decision stage (transactional keywords)
            decision_keywords = non_branded[
                non_branded['Keyword'].str.contains(
                    '|'.join(['buy', 'price', 'deal', 'discount', 'purchase']), 
                    case=False, na=False
                )
            ]
            
            funnel_analysis['awareness_stage'][brand] = {
                'keyword_count': len(awareness_keywords),
                'traffic_share': awareness_keywords['Traffic (%)'].sum(),
                'avg_position': awareness_keywords['Position'].mean() if not awareness_keywords.empty else 0
            }
            
            funnel_analysis['consideration_stage'][brand] = {
                'keyword_count': len(consideration_keywords),
                'traffic_share': consideration_keywords['Traffic (%)'].sum(),
                'avg_position': consideration_keywords['Position'].mean() if not consideration_keywords.empty else 0
            }
            
            funnel_analysis['decision_stage'][brand] = {
                'keyword_count': len(decision_keywords),
                'traffic_share': decision_keywords['Traffic (%)'].sum(),
                'avg_position': decision_keywords['Position'].mean() if not decision_keywords.empty else 0
            }
            
            # Calculate funnel efficiency
            total_funnel_traffic = (
                funnel_analysis['awareness_stage'][brand]['traffic_share'] +
                funnel_analysis['consideration_stage'][brand]['traffic_share'] +
                funnel_analysis['decision_stage'][brand]['traffic_share']
            )
            
            if total_funnel_traffic > 0:
                decision_weight = funnel_analysis['decision_stage'][brand]['traffic_share'] / total_funnel_traffic
                consideration_weight = funnel_analysis['consideration_stage'][brand]['traffic_share'] / total_funnel_traffic
                
                funnel_analysis['funnel_efficiency'][brand] = (decision_weight * 0.5 + consideration_weight * 0.3) * 100
        
        return funnel_analysis
    
    def _analyze_tofu_competitive_landscape(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze competitive landscape for TOFU"""
        
        competitive_analysis = {
            'market_penetration': {},
            'competitive_gaps': {},
            'acquisition_threats': {}
        }
        
        brands = [brand for brand in data.keys() if brand != 'gap_keywords']
        
        for brand in brands:
            df = data[brand]
            if df.empty:
                continue
            
            non_branded = self._identify_non_branded_keywords(df, brand)
            
            # Market penetration analysis
            competitive_analysis['market_penetration'][brand] = {
                'non_branded_coverage': len(non_branded),
                'top_10_share': len(non_branded[non_branded['Position'] <= 10]),
                'traffic_concentration': non_branded['Traffic (%)'].sum(),
                'penetration_score': self._calculate_penetration_score(non_branded)
            }
        
        # Identify competitive gaps and threats
        competitive_analysis['competitive_gaps'] = self._identify_tofu_gaps(data)
        competitive_analysis['acquisition_threats'] = self._identify_acquisition_threats(data)
        
        return competitive_analysis
    
    def _calculate_penetration_score(self, keywords_df: pd.DataFrame) -> float:
        """Calculate market penetration score"""
        
        if keywords_df.empty:
            return 0.0
        
        # Factors: keyword coverage, position strength, traffic share
        coverage_score = min(len(keywords_df) / 50, 1.0) * 30  # Normalize to 30 points
        position_score = max(0, (20 - keywords_df['Position'].mean()) / 20) * 40  # 40 points
        traffic_score = min(keywords_df['Traffic (%)'].sum() / 10, 1.0) * 30  # 30 points
        
        return coverage_score + position_score + traffic_score
    
    def _identify_tofu_gaps(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Identify TOFU-specific competitive gaps"""
        
        gaps = []
        
        # Get Lenovo's non-branded keywords
        lenovo_df = data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return gaps
        
        lenovo_non_branded = self._identify_non_branded_keywords(lenovo_df, 'lenovo')
        lenovo_keywords = set(lenovo_non_branded['Keyword'].str.lower())
        
        # Compare with competitors
        for competitor in ['dell', 'hp']:
            competitor_df = data.get(competitor, pd.DataFrame())
            if competitor_df.empty:
                continue
            
            competitor_non_branded = self._identify_non_branded_keywords(competitor_df, competitor)
            
            # Find keywords where competitor is stronger
            for _, row in competitor_non_branded.iterrows():
                keyword = row['Keyword']
                keyword_lower = str(keyword).lower()
                
                if keyword_lower in lenovo_keywords:
                    # Compare positions
                    lenovo_row = lenovo_non_branded[lenovo_non_branded['Keyword'].str.lower() == keyword_lower]
                    if not lenovo_row.empty:
                        lenovo_position = lenovo_row.iloc[0]['Position']
                        competitor_position = row['Position']
                        
                        if competitor_position < lenovo_position - 5:  # Significant gap
                            gaps.append({
                                'keyword': keyword,
                                'competitor': competitor,
                                'lenovo_position': lenovo_position,
                                'competitor_position': competitor_position,
                                'position_gap': lenovo_position - competitor_position,
                                'competitor_traffic': row['Traffic (%)'],
                                'opportunity_type': 'position_improvement',
                                'acquisition_potential': 'high' if competitor_position <= 10 else 'medium'
                            })
                else:
                    # Keyword not in Lenovo's portfolio
                    if row['Position'] <= 20 and row['Traffic (%)'] > 0.1:  # Meaningful positions
                        gaps.append({
                            'keyword': keyword,
                            'competitor': competitor,
                            'lenovo_position': None,
                            'competitor_position': row['Position'],
                            'position_gap': float('inf'),
                            'competitor_traffic': row['Traffic (%)'],
                            'opportunity_type': 'keyword_gap',
                            'acquisition_potential': 'high' if row['Position'] <= 10 else 'medium'
                        })
        
        # Sort by acquisition potential and traffic
        gaps.sort(key=lambda x: (x['acquisition_potential'] == 'high', x['competitor_traffic']), reverse=True)
        
        return gaps[:50]  # Top 50 gaps
    
    def _identify_acquisition_threats(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Identify competitive threats to customer acquisition"""
        
        threats = []
        
        # Analyze competitor movements in high-value acquisition keywords
        lenovo_df = data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return threats
        
        lenovo_non_branded = self._identify_non_branded_keywords(lenovo_df, 'lenovo')
        
        # High-value acquisition keywords (high traffic, informational/commercial intent)
        high_value_keywords = lenovo_non_branded[
            (lenovo_non_branded['Traffic (%)'] > 0.5) &
            (lenovo_non_branded['Position'] <= 20)
        ]
        
        for competitor in ['dell', 'hp']:
            competitor_df = data.get(competitor, pd.DataFrame())
            if competitor_df.empty:
                continue
            
            competitor_non_branded = self._identify_non_branded_keywords(competitor_df, competitor)
            
            # Check for threats in high-value keywords
            for _, lenovo_row in high_value_keywords.iterrows():
                keyword = lenovo_row['Keyword']
                lenovo_position = lenovo_row['Position']
                
                # Find competitor's position for same keyword
                competitor_row = competitor_non_branded[
                    competitor_non_branded['Keyword'].str.lower() == str(keyword).lower()
                ]
                
                if not competitor_row.empty:
                    competitor_position = competitor_row.iloc[0]['Position']
                    
                    # Threat conditions
                    if (competitor_position < lenovo_position and 
                        competitor_position <= 10 and 
                        lenovo_position > 10):
                        
                        threat_level = 'critical' if competitor_position <= 3 else 'high'
                        
                        threats.append({
                            'keyword': keyword,
                            'competitor': competitor,
                            'threat_level': threat_level,
                            'lenovo_position': lenovo_position,
                            'competitor_position': competitor_position,
                            'traffic_at_risk': lenovo_row['Traffic (%)'],
                            'acquisition_impact': 'high',
                            'recommended_action': 'immediate_optimization' if threat_level == 'critical' else 'monitor_and_improve'
                        })
        
        # Sort by threat level and traffic impact
        threats.sort(key=lambda x: (x['threat_level'] == 'critical', x['traffic_at_risk']), reverse=True)
        
        return threats[:25]  # Top 25 threats
    
    def _identify_acquisition_opportunities(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Identify top customer acquisition opportunities"""
        
        opportunities = {
            'immediate_wins': [],
            'short_term_opportunities': [],
            'long_term_potential': [],
            'content_gaps': []
        }
        
        lenovo_df = data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return opportunities
        
        lenovo_non_branded = self._identify_non_branded_keywords(lenovo_df, 'lenovo')
        
        # Immediate wins (page 2 keywords with high acquisition potential)
        page_2_keywords = lenovo_non_branded[
            (lenovo_non_branded['Position'] >= 11) & 
            (lenovo_non_branded['Position'] <= 20) &
            (lenovo_non_branded['Traffic (%)'] > 0.2)
        ]
        
        for _, row in page_2_keywords.iterrows():
            opportunities['immediate_wins'].append({
                'keyword': row['Keyword'],
                'current_position': row['Position'],
                'traffic_potential': self._estimate_traffic_potential(row['Position'], row['Traffic (%)']),
                'acquisition_score': self._calculate_acquisition_score(row),
                'optimization_difficulty': 'medium',
                'expected_timeline': '1-3 months'
            })
        
        # Sort and limit
        opportunities['immediate_wins'].sort(key=lambda x: x['acquisition_score'], reverse=True)
        opportunities['immediate_wins'] = opportunities['immediate_wins'][:20]
        
        # Short-term opportunities (positions 21-50 with potential)
        short_term_keywords = lenovo_non_branded[
            (lenovo_non_branded['Position'] >= 21) & 
            (lenovo_non_branded['Position'] <= 50) &
            (lenovo_non_branded['Traffic (%)'] > 0.1)
        ]
        
        for _, row in short_term_keywords.iterrows():
            opportunities['short_term_opportunities'].append({
                'keyword': row['Keyword'],
                'current_position': row['Position'],
                'traffic_potential': self._estimate_traffic_potential(row['Position'], row['Traffic (%)']),
                'acquisition_score': self._calculate_acquisition_score(row),
                'optimization_difficulty': 'high',
                'expected_timeline': '3-6 months'
            })
        
        opportunities['short_term_opportunities'].sort(key=lambda x: x['acquisition_score'], reverse=True)
        opportunities['short_term_opportunities'] = opportunities['short_term_opportunities'][:15]
        
        # Analyze content gaps
        opportunities['content_gaps'] = self._identify_content_gaps(data)
        
        return opportunities
    
    def _estimate_traffic_potential(self, current_position: float, current_traffic: float) -> float:
        """Estimate traffic potential if keyword reaches page 1"""
        
        # CTR estimates for different positions
        ctr_estimates = {
            1: 0.284, 2: 0.147, 3: 0.103, 4: 0.073, 5: 0.053,
            6: 0.040, 7: 0.031, 8: 0.025, 9: 0.020, 10: 0.017
        }
        
        if current_position <= 10:
            # Already on page 1, estimate potential for position 5
            target_ctr = ctr_estimates.get(5, 0.053)
            current_ctr = ctr_estimates.get(int(current_position), 0.017)
        else:
            # Estimate potential for position 10
            target_ctr = ctr_estimates.get(10, 0.017)
            current_ctr = 0.005  # Estimate for page 2+
        
        if current_ctr == 0:
            return 0
        
        traffic_multiplier = target_ctr / current_ctr
        return current_traffic * traffic_multiplier
    
    def _calculate_acquisition_score(self, keyword_row: pd.Series) -> float:
        """Calculate acquisition score for a keyword"""
        
        position = keyword_row['Position']
        traffic = keyword_row['Traffic (%)']
        keyword = keyword_row['Keyword']
        
        # Base score from traffic and position
        base_score = traffic * 2 + (50 - position) / 50 * 30
        
        # Intent bonuses
        intent_bonus = 0
        keyword_lower = str(keyword).lower()
        
        if any(term in keyword_lower for term in ['how', 'what', 'guide', 'tutorial']):
            intent_bonus += 10  # Informational intent
        if any(term in keyword_lower for term in ['best', 'vs', 'compare', 'review']):
            intent_bonus += 15  # Commercial intent
        if len(keyword.split()) >= 3:
            intent_bonus += 5   # Long-tail bonus
        
        return base_score + intent_bonus
    
    def _identify_content_gaps(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Identify content gaps for acquisition"""
        
        content_gaps = []
        
        # Analyze gap keywords for content opportunities
        gap_keywords = data.get('gap_keywords', pd.DataFrame())
        if gap_keywords.empty:
            return content_gaps
        
        # Filter for high-volume, low-difficulty keywords
        if 'Volume' in gap_keywords.columns and 'Keyword Difficulty' in gap_keywords.columns:
            content_opportunities = gap_keywords[
                (gap_keywords['Volume'] > 500) &
                (gap_keywords['Keyword Difficulty'] < 40)
            ]
            
            for _, row in content_opportunities.iterrows():
                content_gaps.append({
                    'keyword': row['Keyword'],
                    'search_volume': row['Volume'],
                    'difficulty': row['Keyword Difficulty'],
                    'content_type': self._suggest_content_type(row['Keyword']),
                    'acquisition_potential': 'high' if row['Volume'] > 2000 else 'medium',
                    'priority_score': (row['Volume'] / 100) + (100 - row['Keyword Difficulty'])
                })
        
        content_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
        return content_gaps[:30]
    
    def _suggest_content_type(self, keyword: str) -> str:
        """Suggest content type based on keyword"""
        
        keyword_lower = str(keyword).lower()
        
        if any(term in keyword_lower for term in ['how', 'tutorial', 'guide', 'step']):
            return 'tutorial_guide'
        elif any(term in keyword_lower for term in ['vs', 'compare', 'difference']):
            return 'comparison_article'
        elif any(term in keyword_lower for term in ['best', 'top', 'review']):
            return 'review_roundup'
        elif any(term in keyword_lower for term in ['what', 'definition', 'meaning']):
            return 'educational_content'
        else:
            return 'informational_article'
    
    def _assess_tofu_risks(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess risks to TOFU performance"""
        
        risks = {
            'traffic_concentration_risk': {},
            'competitive_vulnerability': {},
            'keyword_cannibalization': {},
            'content_gaps_risk': {}
        }
        
        lenovo_df = data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return risks
        
        lenovo_non_branded = self._identify_non_branded_keywords(lenovo_df, 'lenovo')
        
        # Traffic concentration risk
        total_traffic = lenovo_non_branded['Traffic (%)'].sum()
        if total_traffic > 0:
            top_10_traffic = lenovo_non_branded.nlargest(10, 'Traffic (%)')['Traffic (%)'].sum()
            concentration_ratio = top_10_traffic / total_traffic
            
            risks['traffic_concentration_risk'] = {
                'concentration_ratio': concentration_ratio,
                'risk_level': 'high' if concentration_ratio > 0.7 else 'medium' if concentration_ratio > 0.5 else 'low',
                'top_traffic_keywords': lenovo_non_branded.nlargest(5, 'Traffic (%)')[['Keyword', 'Traffic (%)']].to_dict('records')
            }
        
        # Competitive vulnerability assessment
        vulnerability_score = self._calculate_vulnerability_score(data)
        risks['competitive_vulnerability'] = {
            'vulnerability_score': vulnerability_score,
            'risk_level': 'high' if vulnerability_score > 70 else 'medium' if vulnerability_score > 40 else 'low'
        }
        
        return risks
    
    def _calculate_vulnerability_score(self, data: Dict[str, pd.DataFrame]) -> float:
        """Calculate competitive vulnerability score"""
        
        lenovo_df = data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return 0
        
        lenovo_non_branded = self._identify_non_branded_keywords(lenovo_df, 'lenovo')
        vulnerability_factors = []
        
        # Factor 1: Average position vulnerability
        avg_position = lenovo_non_branded['Position'].mean()
        position_vulnerability = min(avg_position / 20, 1.0) * 30
        vulnerability_factors.append(position_vulnerability)
        
        # Factor 2: Competitor pressure
        competitor_pressure = 0
        for competitor in ['dell', 'hp']:
            competitor_df = data.get(competitor, pd.DataFrame())
            if not competitor_df.empty:
                competitor_non_branded = self._identify_non_branded_keywords(competitor_df, competitor)
                competitor_avg_position = competitor_non_branded['Position'].mean()
                if competitor_avg_position < avg_position:
                    competitor_pressure += 20
        
        vulnerability_factors.append(min(competitor_pressure, 40))
        
        # Factor 3: Traffic concentration
        total_traffic = lenovo_non_branded['Traffic (%)'].sum()
        if total_traffic > 0:
            top_5_traffic = lenovo_non_branded.nlargest(5, 'Traffic (%)')['Traffic (%)'].sum()
            concentration_vulnerability = (top_5_traffic / total_traffic) * 30
            vulnerability_factors.append(concentration_vulnerability)
        
        return sum(vulnerability_factors)
    
    def _generate_funnel_optimization_recommendations(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate TOFU funnel optimization recommendations"""
        
        recommendations = []
        
        lenovo_df = data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return recommendations
        
        lenovo_non_branded = self._identify_non_branded_keywords(lenovo_df, 'lenovo')
        
        # Analyze funnel stages
        awareness_keywords = lenovo_non_branded[
            lenovo_non_branded['Keyword'].str.contains(
                '|'.join(['how', 'what', 'guide', 'tutorial']), case=False, na=False
            )
        ]
        
        consideration_keywords = lenovo_non_branded[
            lenovo_non_branded['Keyword'].str.contains(
                '|'.join(['vs', 'compare', 'best', 'review']), case=False, na=False
            )
        ]
        
        # Awareness stage recommendations
        if len(awareness_keywords) < len(lenovo_non_branded) * 0.3:
            recommendations.append({
                'stage': 'awareness',
                'priority': 'high',
                'recommendation': 'Increase informational content creation for top-of-funnel awareness',
                'action': 'Create how-to guides and educational content',
                'expected_impact': 'Expand awareness stage keyword coverage by 40%',
                'resources_needed': ['content_team', 'seo_specialist']
            })
        
        # Consideration stage recommendations
        if len(consideration_keywords) < len(lenovo_non_branded) * 0.2:
            recommendations.append({
                'stage': 'consideration',
                'priority': 'high',
                'recommendation': 'Develop comparison and review content',
                'action': 'Create competitive comparison pages and product reviews',
                'expected_impact': 'Improve consideration stage presence by 30%',
                'resources_needed': ['content_team', 'product_team']
            })
        
        # Position optimization recommendations
        page_2_keywords = lenovo_non_branded[
            (lenovo_non_branded['Position'] >= 11) & (lenovo_non_branded['Position'] <= 20)
        ]
        
        if len(page_2_keywords) > 10:
            recommendations.append({
                'stage': 'optimization',
                'priority': 'immediate',
                'recommendation': f'Optimize {len(page_2_keywords)} page 2 keywords for quick wins',
                'action': 'Technical SEO and content optimization for near-page-1 keywords',
                'expected_impact': f'Move 30-50% of page 2 keywords to page 1',
                'resources_needed': ['seo_specialist', 'technical_team']
            })
        
        return recommendations
    
    def _analyze_tofu_trends(self, current_data: Dict[str, pd.DataFrame], 
                           historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze TOFU trends over time"""
        # """Analyze TOFU trends over time using actual 3-day data"""
        
        trends = {
            'traffic_trends': {},
            'position_trends': {},
            'acquisition_trends': {},
            'competitive_trends': {},
            'trend_summary': {}
        }
        
        if not historical_data:
            self.logger.warning("Historical data not provided for trend analysis. Skipping.")
            return trends
        
        # Get all available dates and sort them
        all_dates = self._get_sorted_dates_from_data(historical_data, current_data)
        
        if len(all_dates) < 2:
            self.logger.warning(f"Not enough data points ({len(all_dates)}) for trend analysis. Need at least 2.")
            return trends
        
        self.logger.info(f"📈 Analyzing trends across {len(all_dates)} days: {', '.join(all_dates)}")
        
        for brand in current_data.keys():
            if brand == 'gap_keywords':
                continue
                
            # Get daily dataframes for this brand
            daily_data = self._get_daily_brand_data(brand, historical_data, current_data, all_dates)
            
            if len(daily_data) < 2:
                self.logger.warning(f"Not enough daily data for brand '{brand}' for trend analysis.")
                continue
            
            # Calculate traffic trends
            traffic_trends = self._calculate_traffic_trends(daily_data, brand)
            trends['traffic_trends'][brand] = traffic_trends
            
            # Calculate position trends  
            position_trends = self._calculate_position_trends(daily_data, brand)
            trends['position_trends'][brand] = position_trends
            
            # Calculate acquisition trends (non-branded focus)
            acquisition_trends = self._calculate_acquisition_trends(daily_data, brand)
            trends['acquisition_trends'][brand] = acquisition_trends
            
            # Calculate competitive trends
            competitive_trends = self._calculate_competitive_trends(daily_data, brand, current_data, historical_data, all_dates)
            trends['competitive_trends'][brand] = competitive_trends
        
        # Generate trend summary
        trends['trend_summary'] = self._generate_trend_summary(trends, all_dates)
        
        return trends

    def _get_sorted_dates_from_data(self, historical_data: Dict[str, pd.DataFrame],
                                   current_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Extract and sort dates from data structure"""
        
        dates = set()
        
        # Extract dates from historical_data keys (assuming format like 'May-19-2025')
        # historical_data is expected to be Dict[date_str, Dict[brand, pd.DataFrame]]
        for date_key in historical_data.keys():
            if isinstance(date_key, str): # and 'may' in date_key.lower(): # Making it more generic
                try:
                    self._parse_date_from_string(date_key) # Validate if parsable
                    dates.add(date_key)
                except ValueError:
                    self.logger.warning(f"Could not parse date from historical_data key: {date_key}")

        # Add current date (assume it's the latest)
        # This needs a more robust way to determine the "current" date.
        # For now, let's assume a convention or pass it explicitly.
        # Using a placeholder for current date determination.
        # If current_data is associated with a specific date, that should be used.
        # For this implementation, we'll rely on the historical_data keys and a fixed "current" date.
        
        # A more robust way would be to have dates associated with each dataframe in current_data
        # For now, using a fixed string as per the provided example.
        # This should ideally be derived or passed in.
        current_date_placeholder = "May-21-2025" # Placeholder, as in the example
        try:
            self._parse_date_from_string(current_date_placeholder)
            dates.add(current_date_placeholder)
        except ValueError:
            self.logger.error(f"Could not parse placeholder current date: {current_date_placeholder}")

        
        # Sort dates chronologically
        date_list = list(dates)
        date_list.sort(key=lambda x: self._parse_date_from_string(x))
        
        return date_list

    def _parse_date_from_string(self, date_str: str) -> datetime:
        """Parse date from string format like 'May-19-2025'"""
        try:
            # Attempt to parse common date formats
            return datetime.strptime(date_str, '%b-%d-%Y')
        except ValueError:
            try:
                return datetime.strptime(date_str, '%B-%d-%Y')
            except ValueError:
                self.logger.error(f"Failed to parse date string: {date_str}. Falling back to current time.")
                # Fallback to current date might not be ideal for sorting historical data.
                # Raising an error or returning a sentinel None might be better.
                # For now, adhering to the original fallback.
                # Consider raising ValueError if parsing fails and it's critical.
                raise ValueError(f"Invalid date format: {date_str}")

    def _get_daily_brand_data(self, brand: str, historical_data: Dict[str, Dict[str, pd.DataFrame]],
                             current_data: Dict[str, pd.DataFrame], all_dates: List[str]) -> Dict[str, pd.DataFrame]:
        """Get daily dataframes for a specific brand"""
        
        daily_data = {}
        
        # Get historical data
        # historical_data is Dict[date_str, Dict[brand, pd.DataFrame]]
        for date_str in all_dates:
            if date_str in historical_data:
                if brand in historical_data[date_str]:
                    daily_data[date_str] = historical_data[date_str][brand].copy()
                else:
                    self.logger.debug(f"Brand '{brand}' not found in historical data for date '{date_str}'.")
            # Check if the date_str corresponds to the "current_data" period
            # This logic assumes the last date in all_dates is the "current" one.
            elif date_str == all_dates[-1] and brand in current_data:
                 daily_data[date_str] = current_data[brand].copy()
            else:
                self.logger.debug(f"Data for date '{date_str}' not found for brand '{brand}'.")

        return daily_data

    def _calculate_traffic_trends(self, daily_data: Dict[str, pd.DataFrame], brand: str) -> Dict[str, Any]:
        """Calculate traffic trends over the period"""
        
        traffic_trends = {
            'daily_total_traffic': {},
            'daily_traffic_change_percent': {}, # Renamed for clarity
            'traffic_velocity_percent': 0, # Renamed for clarity
            'top_gaining_keywords': [],
            'top_losing_keywords': [],
            'overall_trend': 'stable'
        }
        
        dates = sorted(daily_data.keys(), key=self._parse_date_from_string)
        
        if not dates:
            return traffic_trends

        # Calculate daily total traffic
        for date in dates:
            df = daily_data.get(date)
            if df is not None and 'Traffic (%)' in df.columns:
                total_traffic = df['Traffic (%)'].sum()
                traffic_trends['daily_total_traffic'][date] = total_traffic
        
        # Calculate day-over-day changes
        daily_totals_map = traffic_trends['daily_total_traffic']
        sorted_daily_totals = [daily_totals_map[d] for d in dates if d in daily_totals_map]

        if len(sorted_daily_totals) >= 2:
            changes = []
            for i in range(1, len(sorted_daily_totals)):
                if sorted_daily_totals[i-1] > 0:
                    change = ((sorted_daily_totals[i] - sorted_daily_totals[i-1]) / sorted_daily_totals[i-1]) * 100
                    changes.append(change)
                    # Store change against the date it changed TO
                    traffic_trends['daily_traffic_change_percent'][dates[i]] = change 
            
            if changes:
                traffic_trends['traffic_velocity_percent'] = np.mean(changes) # Use numpy mean
                
                if traffic_trends['traffic_velocity_percent'] > 5:
                    traffic_trends['overall_trend'] = 'growing'
                elif traffic_trends['traffic_velocity_percent'] < -5:
                    traffic_trends['overall_trend'] = 'declining'
        
        if len(dates) >= 2:
            first_date_data = daily_data.get(dates[0])
            last_date_data = daily_data.get(dates[-1])

            if first_date_data is not None and last_date_data is not None:
                keyword_changes = self._calculate_keyword_traffic_changes(first_date_data, last_date_data)
                sorted_changes = sorted(keyword_changes, key=lambda x: x['traffic_change_percent'], reverse=True)
                
                traffic_trends['top_gaining_keywords'] = [k for k in sorted_changes if k['traffic_change_percent'] > 0][:5]
                traffic_trends['top_losing_keywords'] = [k for k in sorted_changes if k['traffic_change_percent'] < 0][-5:]
        
        return traffic_trends

    def _calculate_position_trends(self, daily_data: Dict[str, pd.DataFrame], brand: str) -> Dict[str, Any]:
        """Calculate position trends over the period"""
        
        position_trends = {
            'avg_position_by_day': {},
            'position_improvement_velocity': 0, # This is absolute change, not percentage
            'keywords_moved_to_page1': [],
            'keywords_dropped_from_page1': [],
            'position_volatility_stddev': 0, # More specific name
            'overall_ranking_trend': 'stable'
        }
        
        dates = sorted(daily_data.keys(), key=self._parse_date_from_string)
        if not dates:
            return position_trends

        for date in dates:
            df = daily_data.get(date)
            if df is not None and 'Position' in df.columns and not df['Position'].empty:
                avg_position = df['Position'].mean()
                position_trends['avg_position_by_day'][date] = avg_position
        
        daily_positions_map = position_trends['avg_position_by_day']
        sorted_daily_positions = [daily_positions_map[d] for d in dates if d in daily_positions_map]
        
        if len(sorted_daily_positions) >= 2:
            position_changes = [] # Higher value means worse ranking, so improvement is negative change
            for i in range(1, len(sorted_daily_positions)):
                change = sorted_daily_positions[i] - sorted_daily_positions[i-1] # current - previous
                position_changes.append(change)
            
            if position_changes:
                # Velocity: negative means improving (e.g., pos 5 to pos 3 = -2)
                position_trends['position_improvement_velocity'] = np.mean(position_changes) 
                
                # Trend: if velocity is negative, it's improving
                if position_trends['position_improvement_velocity'] < -0.5: # Avg drop of 0.5 pos
                    position_trends['overall_ranking_trend'] = 'improving'
                elif position_trends['position_improvement_velocity'] > 0.5: # Avg gain of 0.5 pos
                    position_trends['overall_ranking_trend'] = 'declining'
        
        if len(dates) >= 2:
            first_date_data = daily_data.get(dates[0])
            last_date_data = daily_data.get(dates[-1])

            if first_date_data is not None and last_date_data is not None:
                page1_movements = self._calculate_page1_movements(first_date_data, last_date_data)
                position_trends['keywords_moved_to_page1'] = page1_movements['gained']
                position_trends['keywords_dropped_from_page1'] = page1_movements['lost']
            
            position_trends['position_volatility_stddev'] = self._calculate_position_volatility(daily_data)
        
        return position_trends

    def _calculate_acquisition_trends(self, daily_data: Dict[str, pd.DataFrame], brand: str) -> Dict[str, Any]:
        """Calculate acquisition-focused trends (non-branded keywords)"""
        
        acquisition_trends = {
            'non_branded_traffic_by_day': {},
            'acquisition_velocity_percent': 0, # Renamed for clarity
            'new_acquisition_keywords': [], # Keywords that appeared and are non-branded
            'lost_acquisition_keywords': [], # Non-branded keywords that disappeared
            'acquisition_efficiency_trend': 'stable'
        }
        
        dates = sorted(daily_data.keys(), key=self._parse_date_from_string)
        if not dates: return acquisition_trends

        daily_nb_dfs = {}
        for date in dates:
            df = daily_data.get(date)
            if df is not None:
                non_branded_df = self._identify_non_branded_keywords(df, brand)
                daily_nb_dfs[date] = non_branded_df
                if not non_branded_df.empty and 'Traffic (%)' in non_branded_df.columns:
                    total_nb_traffic = non_branded_df['Traffic (%)'].sum()
                    acquisition_trends['non_branded_traffic_by_day'][date] = total_nb_traffic
        
        daily_nb_traffic_map = acquisition_trends['non_branded_traffic_by_day']
        sorted_daily_nb_traffic = [daily_nb_traffic_map[d] for d in dates if d in daily_nb_traffic_map]
        
        if len(sorted_daily_nb_traffic) >= 2:
            changes = []
            for i in range(1, len(sorted_daily_nb_traffic)):
                if sorted_daily_nb_traffic[i-1] > 0:
                    change = ((sorted_daily_nb_traffic[i] - sorted_daily_nb_traffic[i-1]) / sorted_daily_nb_traffic[i-1]) * 100
                    changes.append(change)
            
            if changes:
                acquisition_trends['acquisition_velocity_percent'] = np.mean(changes)
                if acquisition_trends['acquisition_velocity_percent'] > 3:
                    acquisition_trends['acquisition_efficiency_trend'] = 'improving'
                elif acquisition_trends['acquisition_velocity_percent'] < -3:
                    acquisition_trends['acquisition_efficiency_trend'] = 'declining'
        
        if len(dates) >= 2:
            first_nb_df = daily_nb_dfs.get(dates[0])
            last_nb_df = daily_nb_dfs.get(dates[-1])

            if first_nb_df is not None and last_nb_df is not None and \
               'Keyword' in first_nb_df.columns and 'Keyword' in last_nb_df.columns:
                
                first_keywords = set(first_nb_df['Keyword'].str.lower())
                last_keywords = set(last_nb_df['Keyword'].str.lower())
                
                acquisition_trends['new_acquisition_keywords'] = list(last_keywords - first_keywords)[:10]
                acquisition_trends['lost_acquisition_keywords'] = list(first_keywords - last_keywords)[:10]
        
        return acquisition_trends

    def _calculate_competitive_trends(self, daily_data_brand: Dict[str, pd.DataFrame], brand: str,
                                     current_data_all_brands: Dict[str, pd.DataFrame],
                                     historical_data_all_brands: Dict[str, Dict[str, pd.DataFrame]],
                                     all_dates: List[str]) -> Dict[str, Any]:
        """Calculate competitive trends for the specified brand."""
        competitive_trends = {
            'market_share_trend_percent': {}, # Renamed
            'competitive_pressure_score': 0, # This might be better as a trend too
            'share_of_voice_trend': 'stable', # Based on market_share_trend_percent velocity
        }

        dates = sorted(daily_data_brand.keys(), key=self._parse_date_from_string)
        if not dates: return competitive_trends

        competitors = [b for b in current_data_all_brands.keys() if b != brand and b != 'gap_keywords']
        market_share_values = []

        for date_str in dates:
            brand_df_for_date = daily_data_brand.get(date_str)
            if brand_df_for_date is None or 'Traffic (%)' not in brand_df_for_date.columns:
                continue
            
            brand_traffic = brand_df_for_date['Traffic (%)'].sum()
            total_market_traffic_for_date = brand_traffic

            for comp in competitors:
                comp_df_for_date = None
                # Check historical first, then current if it's the last date
                if date_str in historical_data_all_brands and comp in historical_data_all_brands[date_str]:
                    comp_df_for_date = historical_data_all_brands[date_str][comp]
                elif date_str == all_dates[-1] and comp in current_data_all_brands: # Assuming last date is current
                    comp_df_for_date = current_data_all_brands[comp]

                if comp_df_for_date is not None and 'Traffic (%)' in comp_df_for_date.columns:
                    total_market_traffic_for_date += comp_df_for_date['Traffic (%)'].sum()
            
            if total_market_traffic_for_date > 0:
                market_share = (brand_traffic / total_market_traffic_for_date) * 100
                competitive_trends['market_share_trend_percent'][date_str] = market_share
                market_share_values.append(market_share)

        if len(market_share_values) >= 2:
            market_share_changes = [(market_share_values[i] - market_share_values[i-1]) for i in range(1, len(market_share_values))]
            avg_change = np.mean(market_share_changes) if market_share_changes else 0
            if avg_change > 1: # Gaining more than 1% share on avg
                competitive_trends['share_of_voice_trend'] = 'increasing'
            elif avg_change < -1: # Losing more than 1% share on avg
                competitive_trends['share_of_voice_trend'] = 'decreasing'

        # Simplified competitive pressure score based on the latest data
        latest_brand_df = daily_data_brand.get(dates[-1])
        if latest_brand_df is not None and 'Traffic (%)' in latest_brand_df.columns:
            latest_brand_traffic = latest_brand_df['Traffic (%)'].sum()
            latest_competitor_total_traffic = 0
            for comp in competitors:
                comp_df_latest = current_data_all_brands.get(comp) # Use current data for latest snapshot
                if comp_df_latest is not None and 'Traffic (%)' in comp_df_latest.columns:
                    latest_competitor_total_traffic += comp_df_latest['Traffic (%)'].sum()
            
            if latest_brand_traffic > 0:
                pressure_ratio = latest_competitor_total_traffic / latest_brand_traffic
                competitive_trends['competitive_pressure_score'] = min(pressure_ratio * 50, 100) # Capped at 100

        return competitive_trends

    def _calculate_keyword_traffic_changes(self, first_df: pd.DataFrame, last_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate traffic changes for individual keywords between two dataframes."""
        changes = []
        if first_df is None or last_df is None or \
           'Keyword' not in first_df.columns or 'Traffic (%)' not in first_df.columns or \
           'Keyword' not in last_df.columns or 'Traffic (%)' not in last_df.columns:
            return changes

        merged_df = pd.merge(
            first_df[['Keyword', 'Traffic (%)']],
            last_df[['Keyword', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_first', '_last'),
            how='outer' # Use outer to capture keywords present in one but not the other
        ).fillna({'Traffic (%)_first': 0, 'Traffic (%)_last': 0}) # Fill NaNs for keywords not in both

        for _, row in merged_df.iterrows():
            first_traffic = row['Traffic (%)_first']
            last_traffic = row['Traffic (%)_last']
            
            change_absolute = last_traffic - first_traffic
            change_percent = 0
            if first_traffic > 0:
                change_percent = (change_absolute / first_traffic) * 100
            elif last_traffic > 0: # Gained from 0
                change_percent = float('inf') # Or a large number like 99999
            
            changes.append({
                'keyword': row['Keyword'],
                'first_traffic': first_traffic,
                'last_traffic': last_traffic,
                'traffic_change_percent': change_percent,
                'traffic_change_absolute': change_absolute
            })
        return changes

    def _calculate_page1_movements(self, first_df: pd.DataFrame, last_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Calculate keywords that moved to/from page 1 between two dataframes."""
        movements = {'gained': [], 'lost': []}
        if first_df is None or last_df is None or \
           'Keyword' not in first_df.columns or 'Position' not in first_df.columns or \
           'Keyword' not in last_df.columns or 'Position' not in last_df.columns:
            return movements

        merged_df = pd.merge(
            first_df[['Keyword', 'Position']],
            last_df[['Keyword', 'Position']],
            on='Keyword',
            suffixes=('_first', '_last'),
            how='inner' # Only consider keywords present in both for position change
        )

        for _, row in merged_df.iterrows():
            first_pos = row['Position_first']
            last_pos = row['Position_last']
            
            if pd.isna(first_pos) or pd.isna(last_pos): # Skip if position is NaN
                continue

            # Moved to page 1 (position 1-10)
            if first_pos > 10 and last_pos <= 10:
                movements['gained'].append(row['Keyword'])
            # Dropped from page 1
            elif first_pos <= 10 and last_pos > 10:
                movements['lost'].append(row['Keyword'])
        return movements

    def _calculate_position_volatility(self, daily_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate average position standard deviation across days for common keywords."""
        dates = sorted(daily_data.keys(), key=self._parse_date_from_string)
        if len(dates) < 2:
            return 0.0

        keyword_positions_over_time = {} # Dict[keyword, List[positions]]

        for date_str in dates:
            df = daily_data.get(date_str)
            if df is not None and 'Keyword' in df.columns and 'Position' in df.columns:
                for _, row in df.iterrows():
                    keyword = row['Keyword']
                    position = row['Position']
                    if pd.notna(position):
                        keyword_positions_over_time.setdefault(keyword, []).append(position)
        
        keyword_std_devs = []
        for keyword, positions in keyword_positions_over_time.items():
            if len(positions) >= 2: # Need at least 2 data points to calculate std dev
                keyword_std_devs.append(np.std(positions))
        
        return np.mean(keyword_std_devs) if keyword_std_devs else 0.0

    def _generate_trend_summary(self, trends_data: Dict[str, Any], all_dates: List[str]) -> Dict[str, Any]:
        """Generate overall trend summary, focusing on 'lenovo' if present."""
        summary = {
            'analysis_period': f"{all_dates[0]} to {all_dates[-1]}" if all_dates else "N/A",
            'total_days_analyzed': len(all_dates),
            'overall_performance_trend': 'stable', # Default
            'key_insights': [],
            'recommended_actions': []
        }

        # Prioritize 'lenovo' for summary, or use the first available brand
        target_brand = 'lenovo' if 'lenovo' in trends_data.get('traffic_trends', {}) else None
        if not target_brand and trends_data.get('traffic_trends'):
            target_brand = next(iter(trends_data['traffic_trends']), None)

        if not target_brand:
            summary['key_insights'].append("No specific brand data found for detailed trend summary.")
            return summary

        summary['brand_summarized'] = target_brand
        brand_traffic = trends_data.get('traffic_trends', {}).get(target_brand, {})
        brand_position = trends_data.get('position_trends', {}).get(target_brand, {})
        brand_acquisition = trends_data.get('acquisition_trends', {}).get(target_brand, {})
        brand_competitive = trends_data.get('competitive_trends', {}).get(target_brand, {})

        insights = []
        recommendations = []
        performance_signals = [] # Collect signals to determine overall trend

        if brand_traffic:
            velocity = brand_traffic.get('traffic_velocity_percent', 0)
            trend = brand_traffic.get('overall_trend', 'stable')
            insights.append(f"Traffic for {target_brand} is {trend} (velocity: {velocity:.1f}%).")
            if trend == 'growing': performance_signals.append(1)
            elif trend == 'declining': performance_signals.append(-1)
            if trend == 'declining': recommendations.append(f"Investigate causes for {target_brand} traffic decline.")
            if brand_traffic.get('top_gaining_keywords'):
                insights.append(f"Top gaining traffic keywords for {target_brand}: {len(brand_traffic['top_gaining_keywords'])} found.")
            if brand_traffic.get('top_losing_keywords'):
                insights.append(f"Top losing traffic keywords for {target_brand}: {len(brand_traffic['top_losing_keywords'])} found.")
                recommendations.append(f"Address traffic loss for {target_brand}'s declining keywords.")

        if brand_position:
            pos_velocity = brand_position.get('position_improvement_velocity', 0) # Negative is good
            pos_trend = brand_position.get('overall_ranking_trend', 'stable')
            insights.append(f"Avg. ranking for {target_brand} is {pos_trend} (velocity: {pos_velocity:.2f} positions).")
            if pos_trend == 'improving': performance_signals.append(1)
            elif pos_trend == 'declining': performance_signals.append(-1)
            if pos_trend == 'declining': recommendations.append(f"Focus on improving {target_brand}'s declining keyword rankings.")
            
            moved_to_p1 = len(brand_position.get('keywords_moved_to_page1', []))
            dropped_from_p1 = len(brand_position.get('keywords_dropped_from_page1', []))
            if moved_to_p1 > 0: insights.append(f"{moved_to_p1} keywords for {target_brand} moved to Page 1.")
            if dropped_from_p1 > 0: insights.append(f"{dropped_from_p1} keywords for {target_brand} dropped from Page 1.")
            if dropped_from_p1 > 0 : recommendations.append(f"Analyze why {target_brand} keywords dropped from Page 1.")

            volatility = brand_position.get('position_volatility_stddev', 0)
            insights.append(f"Position volatility (avg std dev) for {target_brand}: {volatility:.2f}.")
            if volatility > 5: # Arbitrary threshold for high volatility
                recommendations.append(f"Investigate and stabilize high-volatility keywords for {target_brand}.")

        if brand_acquisition:
            acq_velocity = brand_acquisition.get('acquisition_velocity_percent', 0)
            acq_trend = brand_acquisition.get('acquisition_efficiency_trend', 'stable')
            insights.append(f"Non-branded (acquisition) traffic for {target_brand} is {acq_trend} (velocity: {acq_velocity:.1f}%).")
            if acq_trend == 'improving': performance_signals.append(1)
            elif acq_trend == 'declining': performance_signals.append(-1)
            if acq_trend == 'declining': recommendations.append(f"Strengthen TOFU content for {target_brand} to improve acquisition.")
            
            new_acq_kw = len(brand_acquisition.get('new_acquisition_keywords', []))
            lost_acq_kw = len(brand_acquisition.get('lost_acquisition_keywords', []))
            if new_acq_kw > 0: insights.append(f"{target_brand} gained {new_acq_kw} new non-branded keywords.")
            if lost_acq_kw > 0: insights.append(f"{target_brand} lost {lost_acq_kw} non-branded keywords.")

        if brand_competitive:
            sov_trend = brand_competitive.get('share_of_voice_trend', 'stable')
            insights.append(f"Share of Voice for {target_brand} is {sov_trend}.")
            if sov_trend == 'increasing': performance_signals.append(0.5) # Less weight than direct metrics
            elif sov_trend == 'decreasing': performance_signals.append(-0.5)
            if sov_trend == 'decreasing': recommendations.append(f"Address declining Share of Voice for {target_brand}.")
            pressure = brand_competitive.get('competitive_pressure_score', 0)
            insights.append(f"Competitive pressure score for {target_brand}: {pressure:.1f} (0-100 scale).")

        if performance_signals:
            avg_signal = np.mean(performance_signals)
            if avg_signal > 0.3: summary['overall_performance_trend'] = 'improving'
            elif avg_signal < -0.3: summary['overall_performance_trend'] = 'declining'
        
        summary['key_insights'] = insights
        summary['recommended_actions'] = recommendations if recommendations else ["Monitor performance and continue current strategies."]
        
        return summary
    
    def _calculate_tofu_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall TOFU performance score"""
        
        score_components = {
            'non_branded_performance': 0,
            'customer_acquisition_potential': 0,
            'competitive_position': 0,
            'risk_assessment': 0
        }
        
        # Non-branded performance score
        non_branded_data = analysis.get('non_branded_analysis', {}).get('lenovo', {})
        if non_branded_data:
            nb_score = (
                min(non_branded_data.get('customer_acquisition_potential', 0) / 100, 1) * 40 +
                min(non_branded_data.get('non_branded_traffic_share', 0) / 20, 1) * 30 +
                max(0, (20 - non_branded_data.get('avg_non_branded_position', 20)) / 20) * 30
            )
            score_components['non_branded_performance'] = nb_score
        
        # Risk assessment score (inverse - lower risk = higher score)
        risk_data = analysis.get('tofu_risk_assessment', {})
        vulnerability = risk_data.get('competitive_vulnerability', {}).get('vulnerability_score', 50)
        score_components['risk_assessment'] = max(0, 100 - vulnerability)
        
        # Overall TOFU score
        overall_score = sum(score_components.values()) / len(score_components)
        
        return {
            'overall_score': overall_score,
            'score_components': score_components,
            'performance_grade': self._get_performance_grade(overall_score),
            'improvement_areas': self._identify_improvement_areas(score_components)
        }
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert score to performance grade"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 50:
            return 'C+'
        elif score >= 40:
            return 'C'
        else:
            return 'D'
    
    def _identify_improvement_areas(self, score_components: Dict[str, float]) -> List[str]:
        """Identify areas for improvement"""
        
        areas = []
        for component, score in score_components.items():
            if score < 60:
                areas.append(component.replace('_', ' ').title())
        
        return areas
