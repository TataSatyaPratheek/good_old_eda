"""
Brand Analysis Module
Comprehensive branded vs non-branded analysis for brand awareness insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging

class BrandAnalyzer:
    """Analyze branded vs non-branded performance and brand awareness"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.brand_terms = {
            'lenovo': ['lenovo', 'thinkpad', 'ideapad', 'legion'],
            'dell': ['dell', 'alienware', 'inspiron', 'latitude', 'precision'],
            'hp': ['hp', 'hewlett packard', 'pavilion', 'envy', 'omen', 'elitebook']
        }
    
    def analyze_brand_performance(self, data: Dict[str, pd.DataFrame], 
                                 historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive brand analysis including awareness trends
        """
        
        print("🏷️ Analyzing Brand Performance & Awareness...")
        
        analysis = {
            'branded_vs_nonbranded': self._analyze_branded_nonbranded_split(data),
            'brand_awareness_metrics': self._calculate_brand_awareness_metrics(data),
            'brand_protection_analysis': self._analyze_brand_protection(data),
            'competitor_brand_analysis': self._analyze_competitor_brand_performance(data),
            'brand_opportunity_analysis': self._identify_brand_opportunities(data)
        }
        
        # Historical trend analysis if data available
        if historical_data:
            analysis['brand_trends'] = self._analyze_brand_trends(data, historical_data)
        
        # Overall brand health score
        analysis['brand_health_score'] = self._calculate_brand_health_score(analysis)
        
        return analysis
    
    def _analyze_branded_nonbranded_split(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze branded vs non-branded keyword performance"""
        
        brand_analysis = {}
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            if 'Keyword' not in df.columns:
                continue
            
            # Identify branded keywords
            branded_keywords = self._identify_branded_keywords(df, brand)
            non_branded_keywords = self._identify_non_branded_keywords(df, brand)
            
            brand_analysis[brand] = {
                'branded_keywords': {
                    'count': len(branded_keywords),
                    'avg_position': branded_keywords['Position'].mean() if not branded_keywords.empty else 0,
                    'traffic_share': branded_keywords['Traffic (%)'].sum() if not branded_keywords.empty else 0,
                    'top_10_count': len(branded_keywords[branded_keywords['Position'] <= 10]) if not branded_keywords.empty else 0,
                    'page_1_count': len(branded_keywords[branded_keywords['Position'] <= 10]) if not branded_keywords.empty else 0
                },
                'non_branded_keywords': {
                    'count': len(non_branded_keywords),
                    'avg_position': non_branded_keywords['Position'].mean() if not non_branded_keywords.empty else 0,
                    'traffic_share': non_branded_keywords['Traffic (%)'].sum() if not non_branded_keywords.empty else 0,
                    'top_10_count': len(non_branded_keywords[non_branded_keywords['Position'] <= 10]) if not non_branded_keywords.empty else 0,
                    'page_1_count': len(non_branded_keywords[non_branded_keywords['Position'] <= 10]) if not non_branded_keywords.empty else 0
                }
            }
            
            # Calculate ratios and insights
            total_keywords = len(df)
            total_traffic = df['Traffic (%)'].sum()
            
            if total_keywords > 0:
                brand_analysis[brand]['branded_ratio'] = len(branded_keywords) / total_keywords
                brand_analysis[brand]['non_branded_ratio'] = len(non_branded_keywords) / total_keywords
            
            if total_traffic > 0:
                brand_analysis[brand]['branded_traffic_ratio'] = branded_keywords['Traffic (%)'].sum() / total_traffic if not branded_keywords.empty else 0
                brand_analysis[brand]['non_branded_traffic_ratio'] = non_branded_keywords['Traffic (%)'].sum() / total_traffic if not non_branded_keywords.empty else 0
            
            # Performance efficiency
            brand_analysis[brand]['branded_efficiency'] = self._calculate_efficiency(branded_keywords)
            brand_analysis[brand]['non_branded_efficiency'] = self._calculate_efficiency(non_branded_keywords)
            
            # Brand strength indicators
            brand_analysis[brand]['brand_strength'] = self._calculate_brand_strength(branded_keywords, brand)
        
        return brand_analysis
    
    def _identify_branded_keywords(self, df: pd.DataFrame, brand: str) -> pd.DataFrame:
        """Identify branded keywords for a specific brand"""
        
        if 'Keyword' not in df.columns:
            return pd.DataFrame()
        
        brand_terms = self.brand_terms.get(brand, [brand])
        pattern = '|'.join(brand_terms)
        
        branded = df[df['Keyword'].str.lower().str.contains(pattern, na=False)]
        return branded
    
    def _identify_non_branded_keywords(self, df: pd.DataFrame, brand: str) -> pd.DataFrame:
        """Identify non-branded keywords for a specific brand"""
        
        if 'Keyword' not in df.columns:
            return pd.DataFrame()
        
        brand_terms = self.brand_terms.get(brand, [brand])
        pattern = '|'.join(brand_terms)
        
        non_branded = df[~df['Keyword'].str.lower().str.contains(pattern, na=False)]
        return non_branded
    
    def _calculate_efficiency(self, keywords_df: pd.DataFrame) -> float:
        """Calculate traffic efficiency (traffic per keyword)"""
        
        if keywords_df.empty:
            return 0.0
        
        total_traffic = keywords_df['Traffic (%)'].sum()
        keyword_count = len(keywords_df)
        
        return total_traffic / keyword_count if keyword_count > 0 else 0
    
    def _calculate_brand_strength(self, branded_keywords: pd.DataFrame, brand: str) -> Dict[str, Any]:
        """Calculate brand strength indicators"""
        
        if branded_keywords.empty:
            return {'strength_score': 0, 'dominance_level': 'weak'}
        
        # Brand strength factors
        avg_position = branded_keywords['Position'].mean()
        top_3_count = len(branded_keywords[branded_keywords['Position'] <= 3])
        total_traffic = branded_keywords['Traffic (%)'].sum()
        
        # Calculate strength score
        position_score = max(0, (10 - avg_position) / 10) * 40  # 40 points max
        dominance_score = (top_3_count / len(branded_keywords)) * 30  # 30 points max
        traffic_score = min(total_traffic / 10, 1.0) * 30  # 30 points max
        
        strength_score = position_score + dominance_score + traffic_score
        
        # Determine dominance level
        if strength_score >= 80:
            dominance_level = 'dominant'
        elif strength_score >= 60:
            dominance_level = 'strong'
        elif strength_score >= 40:
            dominance_level = 'moderate'
        else:
            dominance_level = 'weak'
        
        return {
            'strength_score': strength_score,
            'dominance_level': dominance_level,
            'position_score': position_score,
            'dominance_score': dominance_score,
            'traffic_score': traffic_score
        }
    
    def _calculate_brand_awareness_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate brand awareness metrics"""
        
        awareness_metrics = {}
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            branded_keywords = self._identify_branded_keywords(df, brand)
            
            if branded_keywords.empty:
                continue
            
            # Core brand keywords (exact brand match)
            core_brand_keywords = branded_keywords[
                branded_keywords['Keyword'].str.lower().str.contains(f'^{brand}', na=False)
            ]
            
            # Brand awareness indicators
            awareness_metrics[brand] = {
                'brand_search_volume': branded_keywords['Traffic (%)'].sum(),
                'core_brand_performance': {
                    'keyword_count': len(core_brand_keywords),
                    'avg_position': core_brand_keywords['Position'].mean() if not core_brand_keywords.empty else 0,
                    'traffic_share': core_brand_keywords['Traffic (%)'].sum() if not core_brand_keywords.empty else 0
                },
                'brand_variant_performance': self._analyze_brand_variants(branded_keywords, brand),
                'brand_awareness_score': self._calculate_awareness_score(branded_keywords, brand),
                'share_of_voice': 0  # Will be calculated after all brands processed
            }
        
        # Calculate share of voice
        total_brand_traffic = sum(
            metrics['brand_search_volume'] 
            for metrics in awareness_metrics.values()
        )
        
        if total_brand_traffic > 0:
            for brand in awareness_metrics:
                brand_traffic = awareness_metrics[brand]['brand_search_volume']
                awareness_metrics[brand]['share_of_voice'] = (brand_traffic / total_brand_traffic) * 100
        
        return awareness_metrics
    
    def _analyze_brand_variants(self, branded_keywords: pd.DataFrame, brand: str) -> Dict[str, Any]:
        """Analyze performance of brand variants and sub-brands"""
        
        variants = {}
        brand_terms = self.brand_terms.get(brand, [brand])
        
        for term in brand_terms:
            variant_keywords = branded_keywords[
                branded_keywords['Keyword'].str.lower().str.contains(term, na=False)
            ]
            
            if not variant_keywords.empty:
                variants[term] = {
                    'keyword_count': len(variant_keywords),
                    'avg_position': variant_keywords['Position'].mean(),
                    'traffic_share': variant_keywords['Traffic (%)'].sum(),
                    'top_keywords': variant_keywords.nlargest(5, 'Traffic (%)')['Keyword'].tolist()
                }
        
        return variants
    
    def _calculate_awareness_score(self, branded_keywords: pd.DataFrame, brand: str) -> float:
        """Calculate brand awareness score"""
        
        if branded_keywords.empty:
            return 0.0
        
        # Factors for awareness score
        total_traffic = branded_keywords['Traffic (%)'].sum()
        avg_position = branded_keywords['Position'].mean()
        keyword_diversity = len(branded_keywords)
        
        # Score components
        traffic_component = min(total_traffic / 20, 1.0) * 40  # Traffic weight
        position_component = max(0, (10 - avg_position) / 10) * 35  # Position weight
        diversity_component = min(keyword_diversity / 50, 1.0) * 25  # Diversity weight
        
        return traffic_component + position_component + diversity_component
    
    def _analyze_brand_protection(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze brand protection and competitive brand threats"""
        
        protection_analysis = {}
        
        for brand, df in data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            branded_keywords = self._identify_branded_keywords(df, brand)
            
            # Analyze how well the brand protects its own terms
            brand_protection = {
                'own_brand_control': self._calculate_brand_control(branded_keywords, brand),
                'vulnerable_brand_terms': self._identify_vulnerable_brand_terms(branded_keywords, brand),
                'brand_cannibalization': self._detect_brand_cannibalization(branded_keywords),
                'competitive_threats': []
            }
            
            # Check competitive threats (other brands ranking for this brand's terms)
            for competitor in data.keys():
                if competitor != brand and competitor != 'gap_keywords':
                    competitor_df = data[competitor]
                    threats = self._identify_brand_threats(competitor_df, brand, competitor)
                    if threats:
                        brand_protection['competitive_threats'].extend(threats)
            
            protection_analysis[brand] = brand_protection
        
        return protection_analysis
    
    def _calculate_brand_control(self, branded_keywords: pd.DataFrame, brand: str) -> Dict[str, Any]:
        """Calculate how well a brand controls its own branded terms"""
        
        if branded_keywords.empty:
            return {'control_score': 0, 'control_level': 'poor'}
        
        # Control factors
        top_3_positions = len(branded_keywords[branded_keywords['Position'] <= 3])
        top_10_positions = len(branded_keywords[branded_keywords['Position'] <= 10])
        total_keywords = len(branded_keywords)
        
        # Control score calculation
        top_3_ratio = top_3_positions / total_keywords if total_keywords > 0 else 0
        top_10_ratio = top_10_positions / total_keywords if total_keywords > 0 else 0
        
        control_score = (top_3_ratio * 70) + (top_10_ratio * 30)
        
        # Control level
        if control_score >= 80:
            control_level = 'excellent'
        elif control_score >= 60:
            control_level = 'good'
        elif control_score >= 40:
            control_level = 'fair'
        else:
            control_level = 'poor'
        
        return {
            'control_score': control_score,
            'control_level': control_level,
            'top_3_ratio': top_3_ratio,
            'top_10_ratio': top_10_ratio,
            'total_brand_keywords': total_keywords
        }
    
    def _identify_vulnerable_brand_terms(self, branded_keywords: pd.DataFrame, brand: str) -> List[Dict[str, Any]]:
        """Identify branded terms where the brand is not performing well"""
        
        vulnerable_terms = []
        
        if branded_keywords.empty:
            return vulnerable_terms
        
        # Terms not in top 10
        weak_positions = branded_keywords[branded_keywords['Position'] > 10]
        
        for _, row in weak_positions.iterrows():
            vulnerable_terms.append({
                'keyword': row['Keyword'],
                'position': row['Position'],
                'traffic': row['Traffic (%)'],
                'vulnerability_level': 'high' if row['Position'] > 20 else 'medium',
                'recommended_action': 'immediate_optimization' if row['Position'] > 20 else 'monitor_and_improve'
            })
        
        # Sort by position (worst first)
        vulnerable_terms.sort(key=lambda x: x['position'], reverse=True)
        
        return vulnerable_terms[:20]  # Top 20 vulnerable terms
    
    def _detect_brand_cannibalization(self, branded_keywords: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential keyword cannibalization within branded terms"""
        
        cannibalization_issues = []
        
        if branded_keywords.empty or 'Keyword' not in branded_keywords.columns:
            return cannibalization_issues
        
        # Group similar keywords and check for position conflicts
        keyword_groups = {}
        
        for _, row in branded_keywords.iterrows():
            keyword = str(row['Keyword']).lower() # Ensure keyword is a string
            base_keyword = ' '.join(keyword.split()[:3])  # First 3 words as base
            
            if base_keyword not in keyword_groups:
                keyword_groups[base_keyword] = []
            
            keyword_groups[base_keyword].append({
                'keyword': row['Keyword'],
                'position': row['Position'],
                'traffic': row['Traffic (%)']
            })
        
        # Identify groups with potential cannibalization
        for base_keyword, keywords in keyword_groups.items():
            if len(keywords) > 1:
                # Check if multiple keywords are competing for similar positions
                positions = [kw['position'] for kw in keywords]
                if max(positions) - min(positions) > 10:  # Significant position spread
                    cannibalization_issues.append({
                        'base_keyword': base_keyword,
                        'competing_keywords': keywords,
                        'position_spread': max(positions) - min(positions),
                        'total_traffic': sum(kw['traffic'] for kw in keywords),
                        'issue_severity': 'high' if len(keywords) > 3 else 'medium'
                    })
        
        return cannibalization_issues
    
    def _identify_brand_threats(self, competitor_df: pd.DataFrame, target_brand: str, competitor: str) -> List[Dict[str, Any]]:
        """Identify when competitors rank for target brand's terms"""
        
        threats = []
        
        if competitor_df.empty or 'Keyword' not in competitor_df.columns:
            return threats
        
        brand_terms = self.brand_terms.get(target_brand, [target_brand])
        
        for term in brand_terms:
            # Find competitor keywords containing the target brand term
            competitor_brand_keywords = competitor_df[
                competitor_df['Keyword'].str.lower().str.contains(term, na=False)
            ]
            
            for _, row in competitor_brand_keywords.iterrows():
                if row['Position'] <= 20:  # Only meaningful positions
                    threats.append({
                        'competitor': competitor,
                        'keyword': row['Keyword'],
                        'competitor_position': row['Position'],
                        'competitor_traffic': row['Traffic (%)'],
                        'threat_level': 'critical' if row['Position'] <= 10 else 'medium',
                        'brand_term_targeted': term
                    })
        
        return threats
    
    def _analyze_competitor_brand_performance(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze competitor brand performance comparatively"""
        
        competitor_analysis = {}
        
        brands = [brand for brand in data.keys() if brand != 'gap_keywords']
        
        for brand in brands:
            df = data[brand]
            if df.empty:
                continue
            
            branded_keywords = self._identify_branded_keywords(df, brand)
            
            competitor_analysis[brand] = {
                'brand_visibility': branded_keywords['Traffic (%)'].sum() if not branded_keywords.empty else 0,
                'brand_strength': self._calculate_brand_strength(branded_keywords, brand),
                'brand_keyword_count': len(branded_keywords),
                'brand_avg_position': branded_keywords['Position'].mean() if not branded_keywords.empty else 0
            }
        
        # Competitive rankings
        competitor_analysis['competitive_rankings'] = self._rank_brand_performance(competitor_analysis)
        
        return competitor_analysis
    
    def _rank_brand_performance(self, brand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rank brands by various performance metrics"""
        
        rankings = {
            'by_visibility': {},
            'by_strength': {},
            'by_keyword_count': {},
            'overall_ranking': {}
        }
        
        brands = [brand for brand in brand_data.keys() if brand != 'competitive_rankings']
        
        # Visibility ranking
        visibility_scores = {brand: data['brand_visibility'] for brand, data in brand_data.items() if brand in brands}
        rankings['by_visibility'] = dict(sorted(visibility_scores.items(), key=lambda x: x[1], reverse=True))
        
        # Strength ranking
        strength_scores = {brand: data['brand_strength']['strength_score'] for brand, data in brand_data.items() if brand in brands}
        rankings['by_strength'] = dict(sorted(strength_scores.items(), key=lambda x: x[1], reverse=True))
        
        # Keyword count ranking
        keyword_counts = {brand: data['brand_keyword_count'] for brand, data in brand_data.items() if brand in brands}
        rankings['by_keyword_count'] = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Overall ranking (weighted combination)
        overall_scores = {}
        for brand in brands:
            if brand in brand_data:
                data = brand_data[brand]
                overall_score = (
                    data['brand_visibility'] * 0.4 +
                    data['brand_strength']['strength_score'] * 0.4 +
                    data['brand_keyword_count'] * 0.2
                )
                overall_scores[brand] = overall_score
        
        rankings['overall_ranking'] = dict(sorted(overall_scores.items(), key=lambda x: x[1], reverse=True))
        
        return rankings
    
    def _identify_brand_opportunities(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Identify brand-related opportunities"""
        
        opportunities = {
            'brand_expansion_opportunities': [],
            'competitive_brand_opportunities': [],
            'brand_protection_priorities': []
        }
        
        lenovo_df = data.get('lenovo', pd.DataFrame())
        if lenovo_df.empty:
            return opportunities
        
        lenovo_branded = self._identify_branded_keywords(lenovo_df, 'lenovo')
        
        # Brand expansion opportunities (weak brand terms that could be improved)
        weak_brand_terms = lenovo_branded[
            (lenovo_branded['Position'] > 10) & 
            (lenovo_branded['Traffic (%)'] > 0.1)
        ]
        
        for _, row in weak_brand_terms.iterrows():
            opportunities['brand_expansion_opportunities'].append({
                'keyword': row['Keyword'],
                'current_position': row['Position'],
                'traffic_potential': self._estimate_brand_traffic_potential(row),
                'optimization_difficulty': 'low',  # Brand terms easier to optimize
                'priority': 'high' if row['Traffic (%)'] > 0.5 else 'medium'
            })
        
        # Competitive brand opportunities (ranking for competitor brand terms)
        for competitor in ['dell', 'hp']:
            competitor_df = data.get(competitor, pd.DataFrame())
            if competitor_df.empty:
                continue
            
            # Find where Lenovo could potentially rank for competitor brand + product terms
            competitor_branded = self._identify_branded_keywords(competitor_df, competitor)
            product_terms = competitor_branded[
                competitor_branded['Keyword'].str.contains('laptop|computer|pc|notebook', case=False, na=False)
            ]
            
            for _, row in product_terms.iterrows():
                if row['Position'] > 5:  # Opportunity if competitor not dominating
                    opportunities['competitive_brand_opportunities'].append({
                        'keyword': row['Keyword'],
                        'competitor': competitor,
                        'competitor_position': row['Position'],
                        'traffic_opportunity': row['Traffic (%)'],
                        'strategy': 'competitive_content'
                    })
        
        # Sort opportunities
        opportunities['brand_expansion_opportunities'].sort(
            key=lambda x: x['traffic_potential'], reverse=True
        )
        opportunities['competitive_brand_opportunities'].sort(
            key=lambda x: x['traffic_opportunity'], reverse=True
        )
        
        return opportunities
    
    def _estimate_brand_traffic_potential(self, keyword_row: pd.Series) -> float:
        """Estimate traffic potential for brand keyword improvement"""
        
        current_position = keyword_row['Position']
        current_traffic = keyword_row['Traffic (%)']
        
        # Brand terms typically have higher CTR
        brand_ctr_estimates = {
            1: 0.35, 2: 0.20, 3: 0.15, 4: 0.10, 5: 0.08,
            6: 0.06, 7: 0.05, 8: 0.04, 9: 0.03, 10: 0.025
        }
        
        if current_position <= 10:
            target_ctr = brand_ctr_estimates.get(3, 0.15)  # Target position 3
            current_ctr = brand_ctr_estimates.get(int(current_position), 0.025)
        else:
            target_ctr = brand_ctr_estimates.get(5, 0.08)  # Target position 5
            current_ctr = 0.01  # Estimate for page 2+
        
        if current_ctr == 0:
            return 0
        
        traffic_multiplier = target_ctr / current_ctr
        return current_traffic * traffic_multiplier
    
    def _analyze_brand_trends(self, current_data: Dict[str, pd.DataFrame],
                            historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze brand performance trends over time using actual 3-day data"""
    
        trends = {
            'brand_awareness_trends': {},
            'brand_share_trends': {},
            'brand_position_trends': {},
            'trend_summary': {}
        }
        
        if not historical_data:
            return trends
        
        # Get sorted dates from available data
        available_dates = self._get_available_dates(historical_data, current_data)
        
        if len(available_dates) < 2:
            return trends
        
        print(f"📊 Analyzing brand trends across {len(available_dates)} days: {', '.join(available_dates)}")
        
        for brand in current_data.keys():
            if brand == 'gap_keywords':
                continue
            
            # Get daily brand data
            daily_brand_data = self._get_daily_brand_data(brand, historical_data, current_data, available_dates)
            
            if len(daily_brand_data) < 2:
                continue
            
            # Calculate actual brand awareness trends
            awareness_trends = self._calculate_brand_awareness_trends(daily_brand_data, brand)
            trends['brand_awareness_trends'][brand] = awareness_trends
            
            # Calculate brand share trends
            share_trends = self._calculate_brand_share_trends(daily_brand_data, brand, current_data)
            trends['brand_share_trends'][brand] = share_trends
            
            # Calculate brand position trends
            position_trends = self._calculate_brand_position_trends(daily_brand_data, brand)
            trends['brand_position_trends'][brand] = position_trends
        
        # Generate overall trend summary
        trends['trend_summary'] = self._generate_brand_trend_summary(trends, available_dates)
        
        return trends

    def _get_available_dates(self, historical_data: Dict[str, pd.DataFrame], 
                            current_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Get available dates from the data structure"""
        
        dates = set()
        
        # Extract dates from historical_data keys (assuming date-based structure)
        if isinstance(historical_data, dict):
            for key in historical_data.keys():
                if isinstance(key, str) and 'may' in key.lower(): # Specific to example "May-DD-YYYY"
                    dates.add(key)
        
        # Add current date (latest data)
        # This assumes current_data corresponds to a specific date not explicitly in its keys
        # For consistency, let's assume a convention or pass the current date if available
        # Using a placeholder based on the example in the request.
        dates.add('May-21-2025')  # Current data represents latest day
        
        # Sort dates chronologically
        date_list = list(dates)
        date_list.sort(key=lambda x: self._parse_date_string(x))
        
        return date_list

    def _parse_date_string(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        try:
            # Example format: "May-19-2025"
            parts = date_str.split('-')
            if len(parts) == 3:
                month_str, day_str, year_str = parts
                month_num = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }.get(month_str.lower()[:3], None) # Use first 3 chars for month
                
                if month_num is not None:
                    return datetime(int(year_str), month_num, int(day_str))
        except ValueError: # Catches errors from int() conversion
            self.logger.warning(f"Could not parse date string: {date_str}. Using current time as fallback.")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing date string {date_str}: {e}")
        
        # Fallback if parsing fails, though this might not be ideal for chronological sorting
        return datetime.now()

    def _get_daily_brand_data(self, brand: str, historical_data: Dict[str, pd.DataFrame],
                             current_data: Dict[str, pd.DataFrame], dates: List[str]) -> Dict[str, pd.DataFrame]:
        """Get daily brand dataframes"""
        
        daily_data = {}
        
        # Get historical data
        # Assumes historical_data is structured like: {'May-19-2025': {'lenovo': pd.DataFrame, ...}, ...}
        for date_str in dates[:-1]:  # Exclude last date (current)
            if date_str in historical_data and isinstance(historical_data[date_str], dict) and brand in historical_data[date_str]:
                daily_data[date_str] = historical_data[date_str][brand].copy()
        
        # Get current data (latest date)
        if brand in current_data:
            latest_date = dates[-1]
            daily_data[latest_date] = current_data[brand].copy()
        
        return daily_data

    def _calculate_brand_awareness_trends(self, daily_data: Dict[str, pd.DataFrame], brand: str) -> Dict[str, Any]:
        """Calculate actual brand awareness trends from data"""
        
        awareness_trends = {
            'daily_brand_traffic': {},
            'daily_brand_keywords': {},
            'awareness_velocity': 0.0,
            'trend_direction': 'stable',
            'awareness_change_percent': 0.0,
            'volatility_coefficient': 0.0, # Coefficient of Variation
            'volatility_level': 'low',
            'brand_strength_progression': {}
        }
        
        dates = sorted(daily_data.keys(), key=self._parse_date_string)
        
        # Calculate daily brand metrics
        for date_str in dates:
            df = daily_data[date_str]
            branded_keywords = self._identify_branded_keywords(df, brand)
            
            if not branded_keywords.empty:
                brand_traffic = branded_keywords['Traffic (%)'].sum()
                brand_keyword_count = len(branded_keywords)
                brand_strength_metrics = self._calculate_brand_strength(branded_keywords, brand)
                
                awareness_trends['daily_brand_traffic'][date_str] = brand_traffic
                awareness_trends['daily_brand_keywords'][date_str] = brand_keyword_count
                awareness_trends['brand_strength_progression'][date_str] = brand_strength_metrics['strength_score']
        
        # Calculate awareness velocity and trend
        daily_traffic_values = [awareness_trends['daily_brand_traffic'][d] for d in dates if d in awareness_trends['daily_brand_traffic']]
        
        if len(daily_traffic_values) >= 2:
            changes = []
            for i in range(1, len(daily_traffic_values)):
                if daily_traffic_values[i-1] > 0:
                    change = ((daily_traffic_values[i] - daily_traffic_values[i-1]) / daily_traffic_values[i-1]) * 100
                    changes.append(change)
            
            if changes:
                awareness_trends['awareness_velocity'] = np.mean(changes)
                
                if awareness_trends['awareness_velocity'] > 5:
                    awareness_trends['trend_direction'] = 'growing'
                elif awareness_trends['awareness_velocity'] < -5:
                    awareness_trends['trend_direction'] = 'declining'
                
            if daily_traffic_values[0] > 0:
                total_change = ((daily_traffic_values[-1] - daily_traffic_values[0]) / daily_traffic_values[0]) * 100
                awareness_trends['awareness_change_percent'] = total_change
        
        # Calculate volatility
        if len(daily_traffic_values) >= 2: # Need at least 2 points for std dev
            std_dev = np.std(daily_traffic_values)
            mean_traffic = np.mean(daily_traffic_values)
            if mean_traffic > 0:
                cv = (std_dev / mean_traffic) * 100
                awareness_trends['volatility_coefficient'] = cv
                if cv > 20:
                    awareness_trends['volatility_level'] = 'high'
                elif cv > 10:
                    awareness_trends['volatility_level'] = 'medium'
        
        return awareness_trends

    def _calculate_brand_share_trends(self, daily_data: Dict[str, pd.DataFrame], brand: str,
                                    all_brands_current_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate brand share trends over time"""
        
        share_trends = {
            'daily_market_share': {},
            'share_velocity': 0.0,
            'competitive_position_change': 'stable',
            'share_volatility_coefficient': 0.0,
            'share_volatility_level': 'low'
        }
        
        dates = sorted(daily_data.keys(), key=self._parse_date_string)
        
        for date_str in dates:
            df_brand = daily_data[date_str]
            branded_keywords_brand = self._identify_branded_keywords(df_brand, brand)
            
            if not branded_keywords_brand.empty:
                brand_traffic = branded_keywords_brand['Traffic (%)'].sum()
                total_market_traffic_for_date = 0
                
                # Estimate total market traffic for this date
                # This is an approximation using current data for competitors if historical isn't perfectly aligned
                for competitor_brand_name, competitor_df in all_brands_current_data.items():
                    if competitor_brand_name == 'gap_keywords' or competitor_df.empty:
                        continue
                    # For the current brand, use its traffic for the specific date
                    if competitor_brand_name == brand:
                        total_market_traffic_for_date += brand_traffic
                    else:
                        # For competitors, we might need a more sophisticated way to get their historical traffic
                        # For now, using their current branded traffic as a proxy if specific historical not available for them
                        # This part needs careful consideration of data structure for historical competitor data
                        comp_branded_kws = self._identify_branded_keywords(competitor_df, competitor_brand_name)
                        if not comp_branded_kws.empty:
                             total_market_traffic_for_date += comp_branded_kws['Traffic (%)'].sum()

                if total_market_traffic_for_date > 0:
                    market_share = (brand_traffic / total_market_traffic_for_date) * 100
                    share_trends['daily_market_share'][date_str] = market_share
        
        daily_shares_values = [share_trends['daily_market_share'][d] for d in dates if d in share_trends['daily_market_share']]
        
        if len(daily_shares_values) >= 2:
            changes = []
            for i in range(1, len(daily_shares_values)):
                if daily_shares_values[i-1] > 0: # Avoid division by zero
                    change = ((daily_shares_values[i] - daily_shares_values[i-1]) / daily_shares_values[i-1]) * 100
                    changes.append(change)
            
            if changes:
                share_trends['share_velocity'] = np.mean(changes)
                if share_trends['share_velocity'] > 3:
                    share_trends['competitive_position_change'] = 'gaining'
                elif share_trends['share_velocity'] < -3:
                    share_trends['competitive_position_change'] = 'losing'

        if len(daily_shares_values) >= 2:
            std_dev_share = np.std(daily_shares_values)
            mean_share = np.mean(daily_shares_values)
            if mean_share > 0:
                share_cv = (std_dev_share / mean_share) * 100
                share_trends['share_volatility_coefficient'] = share_cv
                if share_cv > 15:
                    share_trends['share_volatility_level'] = 'high'
                elif share_cv > 8:
                    share_trends['share_volatility_level'] = 'medium'
        
        return share_trends

    def _calculate_brand_position_trends(self, daily_data: Dict[str, pd.DataFrame], brand: str) -> Dict[str, Any]:
        """Calculate brand position trends over time"""
        
        position_trends = {
            'daily_avg_brand_position': {},
            'position_improvement_velocity': 0.0, # Avg change in position points per day
            'brand_ranking_trend': 'stable',
            'position_consistency_stddev': 0.0,
            'position_consistency_level': 'stable'
        }
        
        dates = sorted(daily_data.keys(), key=self._parse_date_string)
        
        for date_str in dates:
            df = daily_data[date_str]
            branded_keywords = self._identify_branded_keywords(df, brand)
            
            if not branded_keywords.empty and 'Position' in branded_keywords.columns and not branded_keywords['Position'].empty:
                avg_position = branded_keywords['Position'].mean()
                position_trends['daily_avg_brand_position'][date_str] = avg_position
        
        daily_positions_values = [position_trends['daily_avg_brand_position'][d] for d in dates if d in position_trends['daily_avg_brand_position']]
        
        if len(daily_positions_values) >= 2:
            position_changes = [] # Higher value means worse rank, so improvement is negative change
            for i in range(1, len(daily_positions_values)):
                change = daily_positions_values[i] - daily_positions_values[i-1] # current - previous
                position_changes.append(change)
            
            if position_changes:
                # Velocity: average of daily changes. Negative means improving.
                avg_daily_change = np.mean(position_changes)
                position_trends['position_improvement_velocity'] = avg_daily_change 
                
                if avg_daily_change < -0.5: # Avg position improved by more than 0.5 points daily
                    position_trends['brand_ranking_trend'] = 'improving'
                elif avg_daily_change > 0.5: # Avg position worsened by more than 0.5 points daily
                    position_trends['brand_ranking_trend'] = 'declining'
        
        if len(daily_positions_values) >= 2:
            position_std = np.std(daily_positions_values)
            position_trends['position_consistency_stddev'] = position_std
            if position_std > 5:
                position_trends['position_consistency_level'] = 'volatile'
            elif position_std > 2:
                position_trends['position_consistency_level'] = 'moderate'
        
        return position_trends

    def _generate_brand_trend_summary(self, trends_data: Dict[str, Any], dates: List[str]) -> Dict[str, Any]:
        """Generate overall brand trend summary, focusing on 'lenovo' if available"""
        
        summary = {
            'analysis_period': f"{dates[0]} to {dates[-1]}" if dates else "N/A",
            'total_days_analyzed': len(dates),
            'overall_brand_health_trend': 'stable', # Default
            'key_brand_insights': [],
            'brand_recommendations': []
        }
        
        # Prioritize 'lenovo' for summary, or the first available brand
        target_brand = 'lenovo' if 'lenovo' in trends_data.get('brand_awareness_trends', {}) else None
        if not target_brand and trends_data.get('brand_awareness_trends'):
            target_brand = list(trends_data['brand_awareness_trends'].keys())[0]

        if not target_brand:
            summary['key_brand_insights'].append("No specific brand data found for trend summary.")
            return summary

        summary['target_brand_analyzed'] = target_brand
        awareness_data = trends_data.get('brand_awareness_trends', {}).get(target_brand, {})
        share_data = trends_data.get('brand_share_trends', {}).get(target_brand, {})
        position_data = trends_data.get('brand_position_trends', {}).get(target_brand, {})
        
        insights = []
        recommendations = []
        health_signals = [] # Store 'improving', 'declining', 'stable'

        if awareness_data:
            velocity = awareness_data.get('awareness_velocity', 0)
            direction = awareness_data.get('trend_direction', 'stable')
            volatility = awareness_data.get('volatility_level', 'low')
            insights.append(f"{target_brand.title()} brand awareness trend: {direction} (velocity: {velocity:.1f}%). Volatility: {volatility}.")
            if direction == 'growing': health_signals.append('improving')
            elif direction == 'declining': health_signals.append('declining'); recommendations.append(f"Address declining brand awareness for {target_brand.title()}.")
            if volatility == 'high': recommendations.append(f"Investigate high brand awareness volatility for {target_brand.title()}.")

        if share_data:
            share_velocity = share_data.get('share_velocity', 0)
            share_change = share_data.get('competitive_position_change', 'stable')
            insights.append(f"{target_brand.title()} market share trend: {share_change} (velocity: {share_velocity:.1f}%).")
            if share_change == 'gaining': health_signals.append('improving')
            elif share_change == 'losing': health_signals.append('declining'); recommendations.append(f"Counteract market share loss for {target_brand.title()}.")

        if position_data:
            pos_velocity = position_data.get('position_improvement_velocity', 0)
            pos_trend = position_data.get('brand_ranking_trend', 'stable')
            insights.append(f"{target_brand.title()} average brand position trend: {pos_trend} (avg daily change: {pos_velocity:.1f} points).")
            if pos_trend == 'improving': health_signals.append('improving')
            elif pos_trend == 'declining': health_signals.append('declining'); recommendations.append(f"Focus on improving declining brand keyword positions for {target_brand.title()}.")

        # Determine overall health trend
        if 'improving' in health_signals and 'declining' not in health_signals:
            summary['overall_brand_health_trend'] = 'improving'
        elif 'declining' in health_signals and 'improving' not in health_signals:
            summary['overall_brand_health_trend'] = 'declining'
        elif 'improving' in health_signals and 'declining' in health_signals:
             summary['overall_brand_health_trend'] = 'mixed'
        
        summary['key_brand_insights'] = insights if insights else ["No specific trend insights generated."]
        summary['brand_recommendations'] = recommendations if recommendations else ["Monitor trends and continue current strategies."]
        
        return summary

    
    def _calculate_brand_health_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall brand health score"""
        
        health_components = {
            'brand_awareness': 0,
            'brand_protection': 0,
            'competitive_position': 0,
            'growth_potential': 0
        }
        
        # Brand awareness score
        awareness_data = analysis.get('brand_awareness_metrics', {}).get('lenovo', {})
        if awareness_data:
            health_components['brand_awareness'] = awareness_data.get('brand_awareness_score', 0)
        
        # Brand protection score
        protection_data = analysis.get('brand_protection_analysis', {}).get('lenovo', {})
        if protection_data:
            control_score = protection_data.get('own_brand_control', {}).get('control_score', 0)
            threat_penalty = len(protection_data.get('competitive_threats', [])) * 5
            health_components['brand_protection'] = max(0, control_score - threat_penalty)
        
        # Competitive position score
        competitor_data = analysis.get('competitor_brand_analysis', {})
        if competitor_data and 'competitive_rankings' in competitor_data:
            rankings = competitor_data['competitive_rankings'].get('overall_ranking', {})
            if 'lenovo' in rankings:
                position = list(rankings.keys()).index('lenovo') + 1
                health_components['competitive_position'] = max(0, 100 - (position - 1) * 30)
        
        # Growth potential score
        opportunities = analysis.get('brand_opportunity_analysis', {})
        expansion_opps = len(opportunities.get('brand_expansion_opportunities', []))
        health_components['growth_potential'] = min(expansion_opps * 5, 100)
        
        # Overall health score
        overall_health = sum(health_components.values()) / len(health_components)
        
        return {
            'overall_health_score': overall_health,
            'health_components': health_components,
            'health_grade': self._get_health_grade(overall_health),
            'key_strengths': self._identify_brand_strengths(health_components),
            'improvement_priorities': self._identify_brand_improvement_priorities(health_components)
        }
    
    def _get_health_grade(self, score: float) -> str:
        """Convert health score to grade"""
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
        else:
            return 'C'
    
    def _identify_brand_strengths(self, components: Dict[str, float]) -> List[str]:
        """Identify brand strength areas"""
        strengths = []
        for component, score in components.items():
            if score >= 75:
                strengths.append(component.replace('_', ' ').title())
        return strengths
    
    def _identify_brand_improvement_priorities(self, components: Dict[str, float]) -> List[str]:
        """Identify brand improvement priorities"""
        priorities = []
        for component, score in components.items():
            if score < 60:
                priorities.append(component.replace('_', ' ').title())
        return priorities
