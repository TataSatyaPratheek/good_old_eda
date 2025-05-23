"""
Enhanced SEO Analyzer
Includes Advanced Keyword Analysis (Brick 1) and Gap Analysis (Brick 2)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import re

class SEOAnalyzer:
    """Enhanced SEO analysis with advanced features"""
    
    def analyze_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run complete SEO analysis"""
        
        print("🔍 Running core SEO analysis...")
        
        results = {
            'summary': self._create_summary(data),
            'competitive': self._competitive_analysis(data),
            'keyword_analysis': self._keyword_analysis(data),
            'opportunity_analysis': self._opportunity_analysis(data)
        }
        
        return results
    
    # =========================================================================
    # BRICK 1: ADVANCED KEYWORD ANALYSIS
    # =========================================================================
    
    def run_advanced_keyword_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Advanced keyword analysis with competitive insights (Brick 1)"""
        
        print("🎯 Running advanced keyword analysis...")
        
        lenovo_data = data.get('lenovo', pd.DataFrame())
        
        if lenovo_data.empty:
            return {}
        
        # Intent classification
        intent_analysis = self._classify_keyword_intent(lenovo_data)
        
        # SERP feature analysis
        serp_analysis = self._analyze_serp_features(lenovo_data)
        
        # Position momentum analysis
        position_trends = self._analyze_position_trends(lenovo_data)
        
        # Branded vs non-branded analysis
        branded_analysis = self._analyze_branded_performance(lenovo_data)
        
        # Keyword clustering analysis
        clustering_analysis = self._analyze_keyword_clusters(lenovo_data)
        
        return {
            'intent_distribution': intent_analysis,
            'serp_features': serp_analysis,
            'position_trends': position_trends,
            'branded_vs_nonbranded': branded_analysis,
            'keyword_clusters': clustering_analysis
        }
    
    def _classify_keyword_intent(self, df: pd.DataFrame) -> Dict[str, int]:
        """Classify keywords by search intent"""
        
        if 'Keyword' not in df.columns:
            return {}
        
        # Intent classification patterns
        commercial_terms = ['buy', 'price', 'cost', 'cheap', 'deal', 'sale', 'shop', 'purchase', 'discount', 'coupon']
        informational_terms = ['how', 'what', 'why', 'guide', 'tutorial', 'tips', 'learn', 'compare', 'vs', 'review']
        branded_terms = ['lenovo', 'thinkpad', 'yoga', 'legion', 'ideapad', 'thinkbook', 'thinkcentre']
        navigational_terms = ['login', 'support', 'driver', 'download', 'manual', 'warranty', 'contact']
        
        intent_counts = {
            'commercial': 0, 
            'informational': 0, 
            'branded': 0, 
            'navigational': 0,
            'other': 0
        }
        
        for keyword in df['Keyword'].str.lower():
            keyword_lower = str(keyword).lower()
            
            if any(term in keyword_lower for term in commercial_terms):
                intent_counts['commercial'] += 1
            elif any(term in keyword_lower for term in informational_terms):
                intent_counts['informational'] += 1
            elif any(term in keyword_lower for term in branded_terms):
                intent_counts['branded'] += 1
            elif any(term in keyword_lower for term in navigational_terms):
                intent_counts['navigational'] += 1
            else:
                intent_counts['other'] += 1
        
        return intent_counts
    
    def _analyze_serp_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze SERP features presence"""
        
        if 'SERP Features by Keyword' not in df.columns:
            return {'total_with_features': 0, 'feature_breakdown': {}, 'coverage_percentage': 0}
        
        feature_counts = {}
        total_with_features = 0
        
        for features_str in df['SERP Features by Keyword'].dropna():
            if features_str and str(features_str) != 'nan':
                total_with_features += 1
                features = [f.strip() for f in str(features_str).split(',')]
                for feature in features:
                    if feature:  # Skip empty features
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Calculate feature opportunities
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_with_features': total_with_features,
            'feature_breakdown': feature_counts,
            'coverage_percentage': (total_with_features / len(df)) * 100 if len(df) > 0 else 0,
            'top_features': top_features,
            'feature_opportunities': self._identify_serp_opportunities(feature_counts)
        }
    
    def _identify_serp_opportunities(self, feature_counts: Dict[str, int]) -> List[str]:
        """Identify SERP feature optimization opportunities"""
        
        # Features that indicate good optimization potential
        high_value_features = [
            'Featured Snippet', 'People Also Ask', 'Video Carousel', 
            'Shopping Results', 'Local Pack', 'Knowledge Panel'
        ]
        
        opportunities = []
        for feature in high_value_features:
            if feature in feature_counts and feature_counts[feature] > 5:
                opportunities.append(f"Optimize for {feature} ({feature_counts[feature]} keywords)")
        
        return opportunities
    
    def _analyze_position_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze position trends over time"""
        
        if 'source_date' not in df.columns or 'Position' not in df.columns:
            return {'trend_available': False}
        
        # Group by date and calculate average positions
        daily_positions = df.groupby('source_date')['Position'].agg(['mean', 'count']).reset_index()
        daily_positions = daily_positions.sort_values('source_date')
        
        if len(daily_positions) < 2:
            return {'trend_available': False}
        
        # Calculate trend
        position_changes = daily_positions['mean'].diff().dropna()
        
        trend_direction = 'stable'
        if position_changes.mean() < -0.5:  # Improving positions (lower numbers)
            trend_direction = 'improving'
        elif position_changes.mean() > 0.5:  # Worsening positions
            trend_direction = 'declining'
        
        return {
            'trend_available': True,
            'trend_direction': trend_direction,
            'average_daily_change': position_changes.mean(),
            'position_volatility': position_changes.std(),
            'daily_data': daily_positions.to_dict('records')
        }
    
    def _analyze_branded_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze branded vs non-branded keyword performance"""
        
        if 'Keyword' not in df.columns:
            return {}
        
        branded_terms = ['lenovo', 'thinkpad', 'yoga', 'legion', 'ideapad', 'thinkbook', 'thinkcentre']
        
        # Classify keywords
        df_copy = df.copy()
        df_copy['is_branded'] = df_copy['Keyword'].str.lower().apply(
            lambda x: any(term in str(x).lower() for term in branded_terms)
        )
        
        branded_df = df_copy[df_copy['is_branded'] == True]
        non_branded_df = df_copy[df_copy['is_branded'] == False]
        
        analysis = {
            'branded_count': len(branded_df),
            'non_branded_count': len(non_branded_df),
            'branded_avg_position': branded_df['Position'].mean() if not branded_df.empty and 'Position' in branded_df.columns else 0,
            'non_branded_avg_position': non_branded_df['Position'].mean() if not non_branded_df.empty and 'Position' in non_branded_df.columns else 0,
            'branded_traffic_share': branded_df['Traffic (%)'].sum() if not branded_df.empty and 'Traffic (%)' in branded_df.columns else 0,
            'non_branded_traffic_share': non_branded_df['Traffic (%)'].sum() if not non_branded_df.empty and 'Traffic (%)' in non_branded_df.columns else 0
        }
        
        # Performance insights
        insights = []
        if analysis['branded_avg_position'] < analysis['non_branded_avg_position']:
            insights.append("Branded keywords perform better than non-branded")
        
        if analysis['branded_traffic_share'] > analysis['non_branded_traffic_share']:
            insights.append("Heavy reliance on branded traffic")
        else:
            insights.append("Good balance of branded vs non-branded traffic")
        
        analysis['insights'] = insights
        
        return analysis
    
    def _analyze_keyword_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze keyword clusters and themes"""
        
        if 'Keyword' not in df.columns:
            return {}
        
        # Extract topics from keywords (simple word frequency approach)
        all_words = []
        for keyword in df['Keyword'].dropna():
            words = re.findall(r'\b\w+\b', str(keyword).lower())
            # Filter out common stop words and single characters
            filtered_words = [word for word in words if len(word) > 2 and word not in 
                            ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'but', 'but', 'can']]
            all_words.extend(filtered_words)
        
        # Count word frequency
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Get top themes
        top_themes = word_counts.most_common(20)
        
        return {
            'total_unique_words': len(word_counts),
            'top_themes': top_themes,
            'theme_distribution': dict(top_themes[:10])
        }
    
    # =========================================================================
    # BRICK 2: DETAILED GAP ANALYSIS
    # =========================================================================
    
    def run_detailed_gap_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detailed competitive gap analysis (Brick 2)"""
        
        print("📉 Running detailed gap analysis...")
        
        lenovo_data = data.get('lenovo', pd.DataFrame())
        dell_data = data.get('dell', pd.DataFrame())
        hp_data = data.get('hp', pd.DataFrame())
        gap_data = data.get('gap_keywords', pd.DataFrame())
        
        gaps = {
            'position_gaps': self._find_position_gaps(lenovo_data, dell_data, hp_data),
            'keyword_gaps': self._find_keyword_gaps(lenovo_data, dell_data, hp_data),
            'traffic_opportunities': self._find_traffic_opportunities(gap_data),
            'quick_wins': self._identify_quick_wins(lenovo_data, gap_data),
            'content_gaps': self._analyze_content_gaps(lenovo_data, dell_data, hp_data)
        }
        
        return gaps
    
    def _find_position_gaps(self, lenovo_data, dell_data, hp_data) -> List[Dict]:
        """Find keywords where competitors rank significantly better"""
        
        gaps = []
        
        if lenovo_data.empty or 'Keyword' not in lenovo_data.columns:
            return gaps
        
        # Create lookup dictionaries for competitor positions
        dell_positions = dict(zip(dell_data['Keyword'].str.lower(), dell_data['Position'])) if not dell_data.empty and 'Keyword' in dell_data.columns else {}
        hp_positions = dict(zip(hp_data['Keyword'].str.lower(), hp_data['Position'])) if not hp_data.empty and 'Keyword' in hp_data.columns else {}
        
        for _, row in lenovo_data.iterrows():
            keyword = str(row['Keyword']).lower()
            lenovo_pos = row.get('Position', 100)
            
            # Check Dell gaps
            if keyword in dell_positions and dell_positions[keyword] < lenovo_pos - 5:  # 5+ position gap
                gaps.append({
                    'keyword': row['Keyword'],
                    'lenovo_position': lenovo_pos,
                    'competitor': 'Dell',
                    'competitor_position': dell_positions[keyword],
                    'gap': lenovo_pos - dell_positions[keyword],
                    'traffic_potential': row.get('Traffic (%)', 0),
                    'priority': 'high' if lenovo_pos - dell_positions[keyword] > 10 else 'medium'
                })
            
            # Check HP gaps
            if keyword in hp_positions and hp_positions[keyword] < lenovo_pos - 5:
                gaps.append({
                    'keyword': row['Keyword'],
                    'lenovo_position': lenovo_pos,
                    'competitor': 'HP', 
                    'competitor_position': hp_positions[keyword],
                    'gap': lenovo_pos - hp_positions[keyword],
                    'traffic_potential': row.get('Traffic (%)', 0),
                    'priority': 'high' if lenovo_pos - hp_positions[keyword] > 10 else 'medium'
                })
        
        # Sort by largest gaps and traffic potential
        gaps.sort(key=lambda x: (x['gap'], x['traffic_potential']), reverse=True)
        
        return gaps[:20]  # Top 20 gaps
    
    def _find_keyword_gaps(self, lenovo_data, dell_data, hp_data) -> Dict[str, List]:
        """Find keywords where competitors rank but Lenovo doesn't"""
        
        if lenovo_data.empty:
            return {'dell_exclusive': [], 'hp_exclusive': [], 'competitor_shared': []}
        
        lenovo_keywords = set(lenovo_data['Keyword'].str.lower()) if 'Keyword' in lenovo_data.columns else set()
        dell_keywords = set(dell_data['Keyword'].str.lower()) if not dell_data.empty and 'Keyword' in dell_data.columns else set()
        hp_keywords = set(hp_data['Keyword'].str.lower()) if not hp_data.empty and 'Keyword' in hp_data.columns else set()
        
        # Find exclusive keywords
        dell_exclusive = dell_keywords - lenovo_keywords - hp_keywords
        hp_exclusive = hp_keywords - lenovo_keywords - dell_keywords
        competitor_shared = (dell_keywords & hp_keywords) - lenovo_keywords
        
        return {
            'dell_exclusive': list(dell_exclusive)[:50],  # Limit to top 50
            'hp_exclusive': list(hp_exclusive)[:50],
            'competitor_shared': list(competitor_shared)[:50]
        }
    
    def _find_traffic_opportunities(self, gap_data) -> List[Dict]:
        """Find high-traffic opportunities from gap keywords"""
        
        if gap_data.empty:
            return []
        
        opportunities = []
        
        # Filter for high-value opportunities
        if all(col in gap_data.columns for col in ['Volume', 'Keyword Difficulty']):
            # High volume, medium difficulty (achievable)
            good_opportunities = gap_data[
                (gap_data['Volume'] > 1000) & 
                (gap_data['Keyword Difficulty'] >= 30) &
                (gap_data['Keyword Difficulty'] <= 60)
            ].copy()
            
            # Sort by volume and add priority
            good_opportunities = good_opportunities.sort_values('Volume', ascending=False)
            
            for _, row in good_opportunities.head(15).iterrows():
                opportunities.append({
                    'keyword': row['Keyword'],
                    'volume': row['Volume'],
                    'difficulty': row['Keyword Difficulty'],
                    'potential_traffic': row['Volume'] * 0.05,  # Estimated 5% CTR
                    'priority': 'high' if row['Volume'] > 5000 else 'medium'
                })
        
        return opportunities
    
    def _identify_quick_wins(self, lenovo_data, gap_data) -> List[Dict]:
        """Identify quick win opportunities"""
        
        quick_wins = []
        
        # Page 2 keywords (positions 11-20) - easy to move to page 1
        if not lenovo_data.empty and 'Position' in lenovo_data.columns:
            page_2_keywords = lenovo_data[
                (lenovo_data['Position'] >= 11) & 
                (lenovo_data['Position'] <= 20)
            ].copy()
            
            if not page_2_keywords.empty:
                # Sort by traffic potential
                if 'Traffic (%)' in page_2_keywords.columns:
                    page_2_keywords = page_2_keywords.sort_values('Traffic (%)', ascending=False)
                
                for _, row in page_2_keywords.head(10).iterrows():
                    quick_wins.append({
                        'type': 'page_2_keywords',
                        'keyword': row['Keyword'],
                        'current_position': row['Position'],
                        'traffic_potential': row.get('Traffic (%)', 0),
                        'effort': 'low',
                        'expected_improvement': '5-10 positions'
                    })
        
        # Low-hanging fruit from gap keywords
        if not gap_data.empty and all(col in gap_data.columns for col in ['Volume', 'Keyword Difficulty']):
            easy_gaps = gap_data[
                (gap_data['Volume'] > 500) & 
                (gap_data['Keyword Difficulty'] < 30)
            ].copy()
            
            for _, row in easy_gaps.head(10).iterrows():
                quick_wins.append({
                    'type': 'easy_gap_keywords',
                    'keyword': row['Keyword'],
                    'volume': row['Volume'],
                    'difficulty': row['Keyword Difficulty'],
                    'effort': 'low',
                    'expected_improvement': 'new ranking opportunity'
                })
        
        return quick_wins
    
    def _analyze_content_gaps(self, lenovo_data, dell_data, hp_data) -> Dict[str, Any]:
        """Analyze content topic gaps"""
        
        gaps = {'topic_gaps': [], 'content_opportunities': []}
        
        if lenovo_data.empty:
            return gaps
        
        # Extract topics from all datasets
        lenovo_topics = self._extract_topics_from_keywords(lenovo_data)
        dell_topics = self._extract_topics_from_keywords(dell_data)
        hp_topics = self._extract_topics_from_keywords(hp_data)
        
        # Find topics where competitors are strong but Lenovo is weak
        competitor_topics = set(dell_topics.keys()) | set(hp_topics.keys())
        
        for topic in competitor_topics:
            lenovo_count = lenovo_topics.get(topic, 0)
            dell_count = dell_topics.get(topic, 0)
            hp_count = hp_topics.get(topic, 0)
            competitor_max = max(dell_count, hp_count)
            
            if competitor_max > lenovo_count * 2 and competitor_max > 5:  # Significant gap
                gaps['topic_gaps'].append({
                    'topic': topic,
                    'lenovo_keywords': lenovo_count,
                    'competitor_max_keywords': competitor_max,
                    'gap_size': competitor_max - lenovo_count,
                    'opportunity_level': 'high' if competitor_max > 20 else 'medium'
                })
        
        # Sort by gap size
        gaps['topic_gaps'].sort(key=lambda x: x['gap_size'], reverse=True)
        gaps['topic_gaps'] = gaps['topic_gaps'][:15]  # Top 15 gaps
        
        return gaps
    
    def _extract_topics_from_keywords(self, df: pd.DataFrame) -> Dict[str, int]:
        """Extract topics from keywords for content gap analysis"""
        
        if df.empty or 'Keyword' not in df.columns:
            return {}
        
        topics = {}
        
        for keyword in df['Keyword'].dropna():
            # Simple topic extraction - first meaningful word
            words = str(keyword).lower().split()
            if words:
                # Skip very common words
                topic_word = None
                for word in words:
                    if len(word) > 3 and word not in ['laptop', 'computer', 'best', 'top', 'review']:
                        topic_word = word
                        break
                
                if topic_word:
                    topics[topic_word] = topics.get(topic_word, 0) + 1
        
        return topics
    
    # =========================================================================
    # ORIGINAL METHODS (Enhanced)
    # =========================================================================
    
    def _create_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create enhanced data summary"""
        
        summary = {}
        
        for company, df in data.items():
            if df.empty:
                continue
                
            base_summary = {
                'total_keywords': len(df),
                'avg_position': df.get('Position', pd.Series()).mean(),
                'total_traffic': df.get('Traffic (%)', pd.Series()).sum(),
                'top_10_count': len(df[df.get('Position', 100) <= 10]) if 'Position' in df.columns else 0,
                'date_range': f"{df['source_date'].min()} to {df['source_date'].max()}"
            }
            
            # Add enhanced metrics
            if 'Position' in df.columns:
                positions = df['Position'].dropna()
                base_summary.update({
                    'median_position': positions.median(),
                    'top_3_count': len(df[df['Position'] <= 3]),
                    'page_1_count': len(df[df['Position'] <= 10]),
                    'page_2_count': len(df[(df['Position'] >= 11) & (df['Position'] <= 20)])
                })
            
            summary[company] = base_summary
            
        return summary
    
    def _competitive_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced competitive analysis"""
        
        lenovo_data = data.get('lenovo', pd.DataFrame())
        dell_data = data.get('dell', pd.DataFrame())
        hp_data = data.get('hp', pd.DataFrame())
        
        # Traffic comparison
        traffic_comparison = {
            'lenovo': lenovo_data.get('Traffic (%)', pd.Series()).sum(),
            'dell': dell_data.get('Traffic (%)', pd.Series()).sum(),
            'hp': hp_data.get('Traffic (%)', pd.Series()).sum()
        }
        
        total_traffic = sum(traffic_comparison.values())
        market_share = {
            company: (traffic / total_traffic * 100) if total_traffic > 0 else 0
            for company, traffic in traffic_comparison.items()
        }
        
        # Enhanced keyword overlap analysis
        keyword_overlap = {}
        common_keywords_analysis = {}
        
        if not lenovo_data.empty and 'Keyword' in lenovo_data.columns:
            lenovo_keywords = set(lenovo_data['Keyword'].str.lower())
            
            for competitor, comp_data in [('dell', dell_data), ('hp', hp_data)]:
                if not comp_data.empty and 'Keyword' in comp_data.columns:
                    comp_keywords = set(comp_data['Keyword'].str.lower())
                    overlap = lenovo_keywords.intersection(comp_keywords)
                    
                    keyword_overlap[competitor] = len(overlap)
                    common_keywords_analysis[competitor] = {
                        'overlap_count': len(overlap),
                        'overlap_percentage': (len(overlap) / len(lenovo_keywords)) * 100 if lenovo_keywords else 0,
                        'unique_to_competitor': len(comp_keywords - lenovo_keywords),
                        'unique_to_lenovo': len(lenovo_keywords - comp_keywords)
                    }
        
        return {
            'traffic_comparison': traffic_comparison,
            'market_share': market_share,
            'keyword_overlap': keyword_overlap,
            'detailed_overlap_analysis': common_keywords_analysis,
            'competitive_summary': self._generate_competitive_summary(market_share, keyword_overlap)
        }
    
    def _generate_competitive_summary(self, market_share, keyword_overlap):
        """Generate competitive summary insights"""
        
        insights = []
        
        lenovo_share = market_share.get('lenovo', 0)
        if lenovo_share > 40:
            insights.append("Lenovo has dominant market position")
        elif lenovo_share > 30:
            insights.append("Lenovo has strong market position")
        else:
            insights.append("Lenovo has opportunity to increase market share")
        
        # Keyword overlap insights
        total_overlap = sum(keyword_overlap.values())
        if total_overlap > 10000:
            insights.append("High keyword competition with both Dell and HP")
        elif total_overlap > 5000:
            insights.append("Moderate keyword competition")
        else:
            insights.append("Low keyword overlap - potential for differentiation")
        
        return insights
    
    def _keyword_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced keyword performance analysis"""
        
        lenovo_data = data.get('lenovo', pd.DataFrame())
        
        if lenovo_data.empty or 'Position' not in lenovo_data.columns:
            return {}
        
        # Enhanced position distribution
        position_ranges = {
            'top_3': len(lenovo_data[lenovo_data['Position'] <= 3]),
            'top_10': len(lenovo_data[lenovo_data['Position'] <= 10]),
            'top_20': len(lenovo_data[lenovo_data['Position'] <= 20]),
            'top_50': len(lenovo_data[lenovo_data['Position'] <= 50]),
            'beyond_50': len(lenovo_data[lenovo_data['Position'] > 50])
        }
        
        # Performance analysis
        best_keywords = lenovo_data.nsmallest(10, 'Position')[['Keyword', 'Position', 'Traffic (%)']].to_dict('records') if not lenovo_data.empty else []
        worst_keywords = lenovo_data.nlargest(10, 'Position')[['Keyword', 'Position', 'Traffic (%)']].to_dict('records') if not lenovo_data.empty else []
        
        # Traffic efficiency analysis
        traffic_efficiency = []
        if 'Traffic (%)' in lenovo_data.columns:
            # Keywords with high traffic relative to position
            lenovo_data['traffic_per_position'] = lenovo_data['Traffic (%)'] / (lenovo_data['Position'] + 1)
            high_efficiency = lenovo_data.nlargest(10, 'traffic_per_position')[['Keyword', 'Position', 'Traffic (%)', 'traffic_per_position']].to_dict('records')
            traffic_efficiency = high_efficiency
        
        return {
            'position_distribution': position_ranges,
            'best_keywords': best_keywords,
            'worst_keywords': worst_keywords,
            'avg_position': lenovo_data['Position'].mean(),
            'median_position': lenovo_data['Position'].median(),
            'traffic_efficiency_leaders': traffic_efficiency,
            'performance_insights': self._generate_keyword_insights(position_ranges, lenovo_data)
        }
    
    def _generate_keyword_insights(self, position_ranges, lenovo_data):
        """Generate keyword performance insights"""
        
        insights = []
        total_keywords = len(lenovo_data)
        
        # Position distribution insights
        page_1_percentage = (position_ranges['top_10'] / total_keywords) * 100 if total_keywords > 0 else 0
        if page_1_percentage > 20:
            insights.append(f"Strong page 1 presence: {page_1_percentage:.1f}% of keywords")
        else:
            insights.append(f"Opportunity to improve page 1 presence: only {page_1_percentage:.1f}%")
        
        # Traffic concentration
        if 'Traffic (%)' in lenovo_data.columns:
            top_10_traffic = lenovo_data[lenovo_data['Position'] <= 10]['Traffic (%)'].sum()
            total_traffic = lenovo_data['Traffic (%)'].sum()
            
            if total_traffic > 0:
                top_10_share = (top_10_traffic / total_traffic) * 100
                if top_10_share > 80:
                    insights.append(f"High traffic concentration in top 10: {top_10_share:.1f}%")
                else:
                    insights.append(f"Opportunity to optimize lower-ranking keywords for more traffic")
        
        return insights
    
    def _opportunity_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced opportunity analysis"""
        
        gap_data = data.get('gap_keywords', pd.DataFrame())
        
        if gap_data.empty:
            return {}
        
        opportunities = []
        
        if 'Volume' in gap_data.columns and 'Keyword Difficulty' in gap_data.columns:
            # Categorize opportunities by difficulty and volume
            easy_high_volume = gap_data[(gap_data['Volume'] > 1000) & (gap_data['Keyword Difficulty'] < 30)]
            medium_high_volume = gap_data[(gap_data['Volume'] > 1000) & (gap_data['Keyword Difficulty'].between(30, 60))]
            
            opportunities = {
                'easy_wins': easy_high_volume.head(10)[['Keyword', 'Volume', 'Keyword Difficulty']].to_dict('records'),
                'medium_effort': medium_high_volume.head(10)[['Keyword', 'Volume', 'Keyword Difficulty']].to_dict('records'),
                'total_easy_wins': len(easy_high_volume),
                'total_medium_effort': len(medium_high_volume)
            }
        
        return {
            'total_gap_keywords': len(gap_data),
            'categorized_opportunities': opportunities,
            'avg_difficulty': gap_data.get('Keyword Difficulty', pd.Series()).mean(),
            'avg_volume': gap_data.get('Volume', pd.Series()).mean(),
            'opportunity_insights': self._generate_opportunity_insights(gap_data)
        }
    
    def _generate_opportunity_insights(self, gap_data):
        """Generate opportunity insights"""
        
        insights = []
        
        if 'Volume' in gap_data.columns:
            high_volume_count = len(gap_data[gap_data['Volume'] > 1000])
            total_keywords = len(gap_data)
            
            high_volume_percentage = (high_volume_count / total_keywords) * 100 if total_keywords > 0 else 0
            insights.append(f"{high_volume_percentage:.1f}% of gap keywords have high search volume (>1000)")
        
        if 'Keyword Difficulty' in gap_data.columns:
            easy_keywords = len(gap_data[gap_data['Keyword Difficulty'] < 30])
            if easy_keywords > 100:
                insights.append(f"{easy_keywords} easy-to-rank keywords identified")
        
        return insights

    def calculate_advanced_competitive_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate advanced competitive metrics based on academic research"""
        
        print("📊 Calculating advanced competitive metrics...")
        
        lenovo_data = data.get('lenovo', pd.DataFrame())
        dell_data = data.get('dell', pd.DataFrame())
        hp_data = data.get('hp', pd.DataFrame())
        
        metrics = {}
        
        # 1. HHI (Herfindahl-Hirschman Index) for traffic concentration
        if not lenovo_data.empty and 'Traffic (%)' in lenovo_data.columns:
            traffic_values = lenovo_data['Traffic (%)'].dropna()
            if len(traffic_values) > 0:
                total_traffic = traffic_values.sum()
                if total_traffic > 0:
                    shares = traffic_values / total_traffic
                    hhi = (shares ** 2).sum()
                    metrics['lenovo_hhi'] = hhi
                    metrics['traffic_concentration_risk'] = 'High' if hhi > 0.25 else 'Medium' if hhi > 0.15 else 'Low'
        
        # 2. Position Velocity Index (PVI) for competitive threat detection
        position_velocity = {}
        for company, data_df in [('lenovo', lenovo_data), ('dell', dell_data), ('hp', hp_data)]:
            if not data_df.empty and 'Position' in data_df.columns and 'source_date' in data_df.columns:
                # Calculate daily position changes
                daily_avg = data_df.groupby('source_date')['Position'].mean().sort_index()
                if len(daily_avg) > 1:
                    position_changes = daily_avg.diff().dropna()
                    avg_change = position_changes.mean()
                    
                    # Weight by search volume if available
                    if 'Volume' in data_df.columns:
                        avg_volume = data_df['Volume'].mean()
                        pvi = avg_change * np.log(max(avg_volume, 1))
                    else:
                        pvi = avg_change
                    
                    position_velocity[company] = {
                        'pvi_score': pvi,
                        'trend': 'Improving' if pvi < -0.5 else 'Declining' if pvi > 0.5 else 'Stable',
                        'velocity_magnitude': abs(pvi)
                    }
        
        metrics['position_velocity'] = position_velocity
        
        # 3. Market Dominance Score
        if lenovo_data.empty == False and dell_data.empty == False and hp_data.empty == False:
            # Calculate relative market positions
            companies_traffic = {
                'lenovo': lenovo_data.get('Traffic (%)', pd.Series()).sum(),
                'dell': dell_data.get('Traffic (%)', pd.Series()).sum(),
                'hp': hp_data.get('Traffic (%)', pd.Series()).sum()
            }
            
            total_market = sum(companies_traffic.values())
            if total_market > 0:
                dominance_scores = {}
                for company, traffic in companies_traffic.items():
                    market_share = traffic / total_market
                    # Dominance score considers both share and concentration
                    dominance_scores[company] = {
                        'market_share': market_share,
                        'dominance_score': market_share * (1 + market_share),  # Squared advantage
                        'competitive_status': 'Leader' if market_share > 0.4 else 'Strong' if market_share > 0.3 else 'Challenger'
                    }
                
                metrics['market_dominance'] = dominance_scores
        
        # 4. Competitive Threat Level Assessment
        threat_indicators = []
        
        # Check for rapid competitor improvements
        for company, velocity_data in position_velocity.items():
            if company != 'lenovo' and velocity_data['pvi_score'] < -1.0:  # Significant improvement
                threat_indicators.append(f"{company.title()} showing rapid ranking improvements")
        
        # Check for market share threats
        if 'market_dominance' in metrics:
            lenovo_share = metrics['market_dominance']['lenovo']['market_share']
            for company, dominance in metrics['market_dominance'].items():
                if company != 'lenovo' and dominance['market_share'] > lenovo_share * 0.8:
                    threat_indicators.append(f"{company.title()} approaching Lenovo's market share")
        
        metrics['threat_assessment'] = {
            'active_threats': threat_indicators,
            'threat_level': 'High' if len(threat_indicators) > 2 else 'Medium' if len(threat_indicators) > 0 else 'Low',
            'monitoring_priority': threat_indicators[:3] if threat_indicators else ['Maintain current monitoring']
        }
        
        # 5. SERP Feature Dominance Score
        serp_dominance = {}
        for company, data_df in [('lenovo', lenovo_data), ('dell', dell_data), ('hp', hp_data)]:
            if not data_df.empty and 'SERP Features by Keyword' in data_df.columns:
                feature_presence = 0
                total_keywords = 0
                
                for features_str in data_df['SERP Features by Keyword'].dropna():
                    total_keywords += 1
                    if features_str and str(features_str) != 'nan':
                        features = [f.strip() for f in str(features_str).split(',')]
                        # Weight high-value features more
                        high_value_features = ['Featured Snippet', 'Video Carousel', 'Shopping Results']
                        feature_score = len(features) + sum(2 for f in features if any(hvf in f for hvf in high_value_features))
                        feature_presence += feature_score
                
                if total_keywords > 0:
                    serp_dominance[company] = {
                        'dominance_score': feature_presence / total_keywords,
                        'feature_coverage': feature_presence,
                        'optimization_potential': 'High' if feature_presence / total_keywords < 2 else 'Medium'
                    }
        
        metrics['serp_dominance'] = serp_dominance
        
        return metrics

    def generate_predictive_insights(self, data: Dict[str, pd.DataFrame], advanced_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictive insights and forecasts"""
        
        print("🔮 Generating predictive insights...")
        
        predictions = {}
        
        # 1. Traffic Forecast based on current trends
        lenovo_data = data.get('lenovo', pd.DataFrame())
        
        if not lenovo_data.empty and 'source_date' in lenovo_data.columns and 'Traffic (%)' in lenovo_data.columns:
            # Group by date and calculate daily traffic
            daily_traffic = lenovo_data.groupby('source_date')['Traffic (%)'].sum().sort_index()
            
            if len(daily_traffic) >= 2:
                # Simple linear trend calculation
                x = np.arange(len(daily_traffic))
                y = daily_traffic.values
                
                # Linear regression for trend
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Forecast next 30 days
                future_days = 30
                forecast_x = np.arange(len(daily_traffic), len(daily_traffic) + future_days)
                forecast_traffic = slope * forecast_x + intercept
                
                predictions['traffic_forecast'] = {
                    'current_trend': 'Positive' if slope > 0 else 'Negative' if slope < 0 else 'Stable',
                    'slope': slope,
                    'confidence': r_value ** 2,
                    'forecast_30_days': forecast_traffic[-1],
                    'forecast_change': ((forecast_traffic[-1] - daily_traffic.iloc[-1]) / daily_traffic.iloc[-1]) * 100 if daily_traffic.iloc[-1] != 0 else 0,
                    'forecast_reliability': 'High' if r_value ** 2 > 0.7 else 'Medium' if r_value ** 2 > 0.5 else 'Low'
                }
        
        # 2. Competitive Threat Prediction
        position_velocity = advanced_metrics.get('position_velocity', {})
        threat_predictions = []
        
        for competitor, velocity_data in position_velocity.items():
            if competitor != 'lenovo':
                pvi_score = velocity_data.get('pvi_score', 0)
                
                # Predict threat level based on velocity
                if pvi_score < -2.0:  # Rapid improvement
                    threat_predictions.append({
                        'competitor': competitor,
                        'threat_level': 'Critical',
                        'predicted_impact': 'High',
                        'timeline': '30-60 days',
                        'recommendation': f'Immediate competitive response needed against {competitor}'
                    })
                elif pvi_score < -1.0:  # Moderate improvement
                    threat_predictions.append({
                        'competitor': competitor,
                        'threat_level': 'Medium',
                        'predicted_impact': 'Medium',
                        'timeline': '60-90 days',
                        'recommendation': f'Monitor {competitor} closely and prepare defensive strategies'
                    })
        
        predictions['competitive_threats'] = threat_predictions
        
        # 3. Market Share Trajectory
        market_dominance = advanced_metrics.get('market_dominance', {})
        if market_dominance:
            lenovo_share = market_dominance.get('lenovo', {}).get('market_share', 0)
            
            # Calculate trajectory based on threat indicators
            threat_level = advanced_metrics.get('threat_assessment', {}).get('threat_level', 'Low')
            
            if threat_level == 'High':
                projected_change = -5  # 5% potential loss
            elif threat_level == 'Medium':
                projected_change = -2  # 2% potential loss
            else:
                projected_change = 2   # 2% potential gain
            
            predictions['market_share_trajectory'] = {
                'current_share': lenovo_share * 100,
                'projected_3_month_change': projected_change,
                'projected_share': (lenovo_share * 100) + projected_change,
                'confidence_level': 'Medium',
                'key_factors': [
                    f"Current threat level: {threat_level}",
                    f"Competitive velocity trends",
                    f"Market dominance position"
                ]
            }
        
        # 4. Keyword Opportunity Scoring
        gap_data = data.get('gap_keywords', pd.DataFrame())
        if not gap_data.empty and all(col in gap_data.columns for col in ['Volume', 'Keyword Difficulty']):
            
            # Score opportunities based on volume/difficulty ratio
            gap_data_copy = gap_data.copy()
            gap_data_copy['opportunity_score'] = (gap_data_copy['Volume'] / 1000) / (gap_data_copy['Keyword Difficulty'] / 10 + 1)
            gap_data_copy['success_probability'] = np.clip(1 - (gap_data_copy['Keyword Difficulty'] / 100), 0.1, 0.9)
            gap_data_copy['effort_estimate'] = gap_data_copy['Keyword Difficulty'].apply(
                lambda x: 'Low' if x < 30 else 'Medium' if x < 60 else 'High'
            )
            
            top_opportunities = gap_data_copy.nlargest(15, 'opportunity_score')
            
            predictions['keyword_opportunities'] = {
                'total_scored': len(gap_data_copy),
                'high_probability_wins': len(gap_data_copy[gap_data_copy['success_probability'] > 0.7]),
                'top_opportunities': top_opportunities[['Keyword', 'Volume', 'Keyword Difficulty', 'opportunity_score', 'success_probability', 'effort_estimate']].to_dict('records'),
                'expected_traffic_gain': top_opportunities['Volume'].sum() * 0.05,  # Assume 5% CTR
                'implementation_priority': 'Immediate' if len(top_opportunities) > 0 else 'None'
            }
        
        # 5. Risk Assessment Matrix
        risk_factors = []
        
        # Traffic concentration risk
        hhi = advanced_metrics.get('lenovo_hhi', 0)
        if hhi > 0.25:
            risk_factors.append({
                'risk_type': 'Traffic Concentration',
                'severity': 'High',
                'description': 'Heavy reliance on few keywords for traffic',
                'mitigation': 'Diversify keyword portfolio across more terms'
            })
        
        # Competitive pressure risk
        threat_level = advanced_metrics.get('threat_assessment', {}).get('threat_level', 'Low')
        if threat_level in ['High', 'Medium']:
            risk_factors.append({
                'risk_type': 'Competitive Pressure',
                'severity': threat_level,
                'description': 'Competitors gaining market position',
                'mitigation': 'Accelerate optimization efforts and competitive monitoring'
            })
        
        # SERP feature vulnerability
        serp_dominance = advanced_metrics.get('serp_dominance', {})
        lenovo_serp_score = serp_dominance.get('lenovo', {}).get('dominance_score', 0)
        if lenovo_serp_score < 1.5:  # Low SERP feature presence
            risk_factors.append({
                'risk_type': 'SERP Feature Gap',
                'severity': 'Medium',
                'description': 'Limited presence in enhanced search features',
                'mitigation': 'Optimize content for featured snippets and rich results'
            })
        
        predictions['risk_assessment'] = {
            'identified_risks': risk_factors,
            'overall_risk_level': 'High' if len([r for r in risk_factors if r['severity'] == 'High']) > 0 else 'Medium' if len(risk_factors) > 1 else 'Low',
            'priority_mitigations': [r['mitigation'] for r in risk_factors if r['severity'] == 'High'][:3]
        }
        
        return predictions

    def generate_strategic_action_plan(self, results: Dict[str, Any], advanced_metrics: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic action plan based on comprehensive analysis"""
        
        print("📋 Generating strategic action plan...")
        
        action_plan = {
            'immediate_actions': [],      # 0-30 days
            'short_term_actions': [],     # 1-3 months  
            'long_term_actions': [],      # 3-12 months
            'defensive_actions': [],      # Competitive defense
            'offensive_actions': []       # Market expansion
        }
        
        # === IMMEDIATE ACTIONS (0-30 days) ===
        
        # Quick wins from predictions
        keyword_opportunities = predictions.get('keyword_opportunities', {})
        top_opportunities = keyword_opportunities.get('top_opportunities', [])
        
        high_prob_opportunities = [opp for opp in top_opportunities if opp.get('success_probability', 0) > 0.8 and opp.get('effort_estimate') == 'Low']
        
        if high_prob_opportunities:
            action_plan['immediate_actions'].append({
                'action': 'Target High-Probability Keywords',
                'description': f"Optimize for {len(high_prob_opportunities)} high-success-probability, low-effort keywords",
                'keywords': [opp['Keyword'] for opp in high_prob_opportunities[:5]],
                'expected_impact': 'Traffic increase 15-25%',
                'resources_needed': ['Content team', 'SEO specialist'],
                'timeline': '2-4 weeks'
            })
        
        # Page 2 optimization
        gap_results = results.get('competitive_gaps', {})
        quick_wins = gap_results.get('quick_wins', [])
        page_2_keywords = [qw for qw in quick_wins if qw.get('type') == 'page_2_keywords']
        
        if page_2_keywords:
            action_plan['immediate_actions'].append({
                'action': 'Page 2 Keywords Optimization',
                'description': f"Push {len(page_2_keywords)} keywords from page 2 to page 1",
                'expected_impact': 'Traffic increase 20-30%',
                'resources_needed': ['Technical SEO', 'Content optimization'],
                'timeline': '3-4 weeks'
            })
        
        # === DEFENSIVE ACTIONS ===
        
        # Competitive threat response
        competitive_threats = predictions.get('competitive_threats', [])
        critical_threats = [t for t in competitive_threats if t.get('threat_level') == 'Critical']
        
        for threat in critical_threats:
            action_plan['defensive_actions'].append({
                'action': f'Counter {threat["competitor"]} Threat',
                'description': f"Immediate response to {threat['competitor']} rapid improvements",
                'threat_level': threat['threat_level'],
                'timeline': threat['timeline'],
                'recommendation': threat['recommendation'],
                'resources_needed': ['SEO team', 'Content team', 'Analytics']
            })
        
        # Traffic concentration risk mitigation
        hhi = advanced_metrics.get('lenovo_hhi', 0)
        if hhi > 0.25:
            action_plan['defensive_actions'].append({
                'action': 'Diversify Traffic Sources',
                'description': 'Reduce dependency on top-performing keywords',
                'risk_level': 'High',
                'expected_impact': 'Reduced traffic volatility risk',
                'resources_needed': ['Keyword research', 'Content expansion'],
                'timeline': '6-8 weeks'
            })
        
        # === OFFENSIVE ACTIONS ===
        
        # Market share expansion
        market_dominance = advanced_metrics.get('market_dominance', {})
        lenovo_status = market_dominance.get('lenovo', {}).get('competitive_status', 'Unknown')
        
        if lenovo_status in ['Challenger', 'Strong']:
            action_plan['offensive_actions'].append({
                'action': 'Aggressive Market Share Capture',
                'description': 'Target competitor weaknesses for market share gains',
                'current_status': lenovo_status,
                'target_improvement': '5-10% market share increase',
                'resources_needed': ['Full SEO team', 'Content creation', 'Technical optimization'],
                'timeline': '3-6 months'
            })
        
        # SERP feature domination
        serp_dominance = advanced_metrics.get('serp_dominance', {})
        lenovo_serp_score = serp_dominance.get('lenovo', {}).get('dominance_score', 0)
        
        if lenovo_serp_score < 2.0:
            action_plan['offensive_actions'].append({
                'action': 'SERP Feature Optimization Campaign',
                'description': 'Aggressive optimization for featured snippets and rich results',
                'current_score': lenovo_serp_score,
                'target_score': 3.0,
                'expected_impact': 'Increased visibility and CTR',
                'resources_needed': ['Content strategy', 'Technical SEO', 'Schema markup'],
                'timeline': '2-4 months'
            })
        
        # === SHORT-TERM ACTIONS (1-3 months) ===
        
        # Content gap filling
        content_gaps = gap_results.get('content_gaps', {}).get('topic_gaps', [])
        high_opportunity_gaps = [gap for gap in content_gaps if gap.get('opportunity_level') == 'high']
        
        if high_opportunity_gaps:
            action_plan['short_term_actions'].append({
                'action': 'Strategic Content Gap Filling',
                'description': f"Create content for {len(high_opportunity_gaps)} high-opportunity topic gaps",
                'topics': [gap['topic'] for gap in high_opportunity_gaps[:5]],
                'expected_impact': 'New keyword rankings and traffic',
                'resources_needed': ['Content team', 'Subject matter experts'],
                'timeline': '6-12 weeks'
            })
        
        # Technical SEO improvements
        position_velocity = advanced_metrics.get('position_velocity', {})
        lenovo_velocity = position_velocity.get('lenovo', {})
        
        if lenovo_velocity.get('trend') in ['Declining', 'Stable']:
            action_plan['short_term_actions'].append({
                'action': 'Technical SEO Audit & Optimization',
                'description': 'Comprehensive technical improvements to boost ranking velocity',
                'current_trend': lenovo_velocity.get('trend', 'Unknown'),
                'expected_impact': 'Improved ranking momentum',
                'resources_needed': ['Technical SEO specialist', 'Development team'],
                'timeline': '8-10 weeks'
            })
        
        # === LONG-TERM ACTIONS (3-12 months) ===
        
        # Predictive optimization
        traffic_forecast = predictions.get('traffic_forecast', {})
        if traffic_forecast.get('current_trend') in ['Negative', 'Stable']:
            action_plan['long_term_actions'].append({
                'action': 'Strategic Portfolio Restructuring',
                'description': 'Long-term keyword portfolio optimization based on predictive insights',
                'current_trend': traffic_forecast.get('current_trend'),
                'expected_impact': 'Sustained traffic growth',
                'resources_needed': ['Strategic planning', 'Full optimization team'],
                'timeline': '6-12 months'
            })
        
        # Market leadership positioning
        if lenovo_status == 'Strong':
            action_plan['long_term_actions'].append({
                'action': 'Market Leadership Campaign',
                'description': 'Comprehensive strategy to achieve market leadership position',
                'target_status': 'Leader',
                'expected_impact': 'Market dominance and sustained competitive advantage',
                'resources_needed': ['Cross-functional teams', 'Significant investment'],
                'timeline': '9-12 months'
            })
        
        # === ACTION PRIORITIZATION ===
        
        # Calculate priority scores
        all_actions = []
        for category, actions in action_plan.items():
            for action in actions:
                priority_score = 0
                
                # Immediate actions get highest priority
                if category == 'immediate_actions':
                    priority_score += 10
                elif category == 'defensive_actions':
                    priority_score += 8
                elif category == 'short_term_actions':
                    priority_score += 6
                elif category == 'offensive_actions':
                    priority_score += 7
                else:  # long_term_actions
                    priority_score += 4
                
                # Boost priority for high-impact actions
                if 'Traffic increase' in action.get('expected_impact', ''):
                    priority_score += 3
                
                action['priority_score'] = priority_score
                action['category'] = category
                all_actions.append(action)
        
        # Sort by priority
        all_actions.sort(key=lambda x: x['priority_score'], reverse=True)
        
        action_plan['prioritized_actions'] = all_actions[:10]  # Top 10 priority actions
        
        # === RESOURCE ALLOCATION ===
        
        resource_summary = {}
        for action in all_actions:
            for resource in action.get('resources_needed', []):
                if resource not in resource_summary:
                    resource_summary[resource] = 0
                resource_summary[resource] += 1
        
        action_plan['resource_requirements'] = {
            'most_needed_resources': sorted(resource_summary.items(), key=lambda x: x[1], reverse=True)[:5],
            'total_actions': len(all_actions),
            'immediate_priority_actions': len([a for a in all_actions if a.get('category') == 'immediate_actions']),
            'defensive_priority_actions': len([a for a in all_actions if a.get('category') == 'defensive_actions'])
        }
        
        return action_plan
