"""
Traffic Diagnostics Module
Comprehensive traffic drop diagnosis and pattern analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy.stats import pearsonr

@dataclass
class TrafficPattern:
    """Traffic pattern data structure"""
    pattern_type: str
    severity: str
    affected_keywords: int
    traffic_impact: float
    date_detected: datetime
    potential_causes: List[str]

class TrafficDiagnostics:
    """Advanced traffic drop diagnosis and pattern recognition"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.google_updates = self._load_google_updates()
        self.seasonal_patterns = self._load_seasonal_patterns()
    
    def diagnose_traffic_patterns(self, current_data: Dict[str, pd.DataFrame],
                                historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive traffic pattern diagnosis
        """
        
        print("🔍 Diagnosing Traffic Patterns & Drops...")
        
        diagnosis = {
            'recent_changes': self._analyze_recent_changes(current_data, historical_data),
            'traffic_drops': self._identify_traffic_drops(current_data, historical_data),
            'seasonal_analysis': self._analyze_seasonal_patterns(current_data),
            'algorithm_correlation': self._analyze_algorithm_correlation(current_data, historical_data),
            'competitor_correlation': self._analyze_competitor_correlation(current_data, historical_data),
            'recovery_recommendations': [],
            'alert_triggers': []
        }
        
        # Generate recovery recommendations
        diagnosis['recovery_recommendations'] = self._generate_recovery_recommendations(diagnosis)
        
        # Set up alert triggers
        diagnosis['alert_triggers'] = self._setup_alert_triggers(diagnosis)
        
        return diagnosis
    
    def _load_google_updates(self) -> List[Dict[str, Any]]:
        """Load known Google algorithm updates"""
        
        # In a real implementation, this would load from a database or API
        return [
            {
                'name': 'May 2025 Core Update',
                'date': datetime(2025, 5, 15),
                'type': 'core',
                'impact_areas': ['content_quality', 'user_experience', 'eeat']
            },
            {
                'name': 'April 2025 Product Reviews Update',
                'date': datetime(2025, 4, 20),
                'type': 'product_reviews',
                'impact_areas': ['review_content', 'affiliate_sites', 'comparison_pages']
            }
        ]
    
    def _load_seasonal_patterns(self) -> Dict[str, List[str]]:
        """Load seasonal pattern definitions"""
        
        return {
            'back_to_school': ['august', 'september'],
            'holiday_shopping': ['november', 'december'],
            'new_year_planning': ['january', 'february'],
            'spring_refresh': ['march', 'april'],
            'summer_slowdown': ['june', 'july']
        }
    
    def _analyze_recent_changes(self, current_data: Dict[str, pd.DataFrame],
                              historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze recent traffic changes (last 7-14 days)"""
        
        recent_changes = {
            'significant_drops': [],
            'significant_gains': [],
            'volatility_alerts': [],
            'change_summary': {}
        }
        
        if not historical_data:
            return recent_changes
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords' or current_df.empty:
                continue
            
            historical_df = historical_data.get(brand, pd.DataFrame())
            if historical_df.empty:
                continue
            
            # Compare traffic changes
            changes = self._calculate_traffic_changes(current_df, historical_df)
            
            # Identify significant changes
            significant_drops = [
                change for change in changes 
                if change['change_percent'] < -20  # 20% drop threshold
            ]
            
            significant_gains = [
                change for change in changes 
                if change['change_percent'] > 50  # 50% gain threshold
            ]
            
            recent_changes['significant_drops'].extend(significant_drops)
            recent_changes['significant_gains'].extend(significant_gains)
            
            # Calculate brand-level summary
            total_current_traffic = current_df['Traffic (%)'].sum()
            total_historical_traffic = historical_df['Traffic (%)'].sum()
            
            if total_historical_traffic > 0:
                overall_change = ((total_current_traffic - total_historical_traffic) / 
                                total_historical_traffic) * 100
            else:
                overall_change = 0
            
            recent_changes['change_summary'][brand] = {
                'overall_change_percent': overall_change,
                'keywords_dropped': len(significant_drops),
                'keywords_gained': len(significant_gains),
                'total_keywords_analyzed': len(changes)
            }
        
        return recent_changes
    
    def _calculate_traffic_changes(self, current_df: pd.DataFrame, 
                                 historical_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate traffic changes for individual keywords"""
        
        changes = []
        
        if 'Keyword' not in current_df.columns or 'Keyword' not in historical_df.columns:
            return changes
        
        # Merge dataframes on keyword
        merged = pd.merge(
            current_df[['Keyword', 'Position', 'Traffic (%)']],
            historical_df[['Keyword', 'Position', 'Traffic (%)']],
            on='Keyword',
            suffixes=('_current', '_historical'),
            how='outer'
        )
        
        for _, row in merged.iterrows():
            current_traffic = row.get('Traffic (%)_current', 0)
            historical_traffic = row.get('Traffic (%)_historical', 0)
            current_position = row.get('Position_current', 100)
            historical_position = row.get('Position_historical', 100)
            
            # Calculate changes
            if historical_traffic > 0:
                traffic_change_percent = ((current_traffic - historical_traffic) / 
                                        historical_traffic) * 100
            else:
                traffic_change_percent = 100 if current_traffic > 0 else 0
            
            position_change = current_position - historical_position
            
            changes.append({
                'keyword': row['Keyword'],
                'current_traffic': current_traffic,
                'historical_traffic': historical_traffic,
                'traffic_change_absolute': current_traffic - historical_traffic,
                'change_percent': traffic_change_percent,
                'current_position': current_position,
                'historical_position': historical_position,
                'position_change': position_change,
                'change_type': self._classify_change_type(traffic_change_percent, position_change)
            })
        
        return changes
    
    def _classify_change_type(self, traffic_change: float, position_change: float) -> str:
        """Classify the type of change observed"""
        
        if traffic_change < -20:
            if position_change > 5:
                return 'ranking_drop'
            else:
                return 'traffic_drop_stable_ranking'
        elif traffic_change > 50:
            if position_change < -5:
                return 'ranking_improvement'
            else:
                return 'traffic_gain_stable_ranking'
        elif abs(position_change) > 10:
            return 'high_volatility'
        else:
            return 'stable'
    
    def _identify_traffic_drops(self, current_data: Dict[str, pd.DataFrame],
                              historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Identify and categorize traffic drops"""
        
        traffic_drops = {
            'critical_drops': [],
            'moderate_drops': [],
            'minor_drops': [],
            'drop_patterns': {},
            'affected_categories': {}
        }
        
        if not historical_data:
            return traffic_drops
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords' or current_df.empty:
                continue
            
            historical_df = historical_data.get(brand, pd.DataFrame())
            if historical_df.empty:
                continue
            
            changes = self._calculate_traffic_changes(current_df, historical_df)
            
            # Categorize drops by severity
            for change in changes:
                drop_percent = change['change_percent']
                
                if drop_percent < -50:
                    severity = 'critical'
                    traffic_drops['critical_drops'].append({
                        'brand': brand,
                        'keyword': change['keyword'],
                        'drop_percent': drop_percent,
                        'traffic_lost': abs(change['traffic_change_absolute']),
                        'position_change': change['position_change'],
                        'likely_cause': self._determine_likely_cause(change)
                    })
                elif drop_percent < -25:
                    severity = 'moderate'
                    traffic_drops['moderate_drops'].append({
                        'brand': brand,
                        'keyword': change['keyword'],
                        'drop_percent': drop_percent,
                        'traffic_lost': abs(change['traffic_change_absolute']),
                        'position_change': change['position_change'],
                        'likely_cause': self._determine_likely_cause(change)
                    })
                elif drop_percent < -10:
                    severity = 'minor'
                    traffic_drops['minor_drops'].append({
                        'brand': brand,
                        'keyword': change['keyword'],
                        'drop_percent': drop_percent,
                        'traffic_lost': abs(change['traffic_change_absolute']),
                        'position_change': change['position_change'],
                        'likely_cause': self._determine_likely_cause(change)
                    })
            
            # Analyze drop patterns
            traffic_drops['drop_patterns'][brand] = self._analyze_drop_patterns(changes)
        
        return traffic_drops
    
    def _determine_likely_cause(self, change: Dict[str, Any]) -> str:
        """Determine the likely cause of a traffic change"""
        
        traffic_change = change['change_percent']
        position_change = change['position_change']
        
        if position_change > 10:
            return 'ranking_drop'
        elif position_change > 5:
            return 'minor_ranking_decline'
        elif traffic_change < -30 and abs(position_change) < 3:
            return 'ctr_decline_or_serp_changes'
        elif traffic_change < -50:
            return 'potential_penalty_or_algorithm_impact'
        else:
            return 'market_or_seasonal_factors'
    
    def _analyze_drop_patterns(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in traffic drops"""
        
        patterns = {
            'total_drops': 0,
            'avg_drop_severity': 0,
            'most_affected_keywords': [],
            'pattern_type': 'isolated'
        }
        
        drops = [change for change in changes if change['change_percent'] < -10]
        patterns['total_drops'] = len(drops)
        
        if drops:
            patterns['avg_drop_severity'] = sum(drop['change_percent'] for drop in drops) / len(drops)
            patterns['most_affected_keywords'] = sorted(
                drops, key=lambda x: x['change_percent']
            )[:10]
            
            # Determine pattern type
            if len(drops) > len(changes) * 0.5:
                patterns['pattern_type'] = 'widespread'
            elif len(drops) > len(changes) * 0.2:
                patterns['pattern_type'] = 'moderate'
            else:
                patterns['pattern_type'] = 'isolated'
        
        return patterns
    
    def _analyze_seasonal_patterns(self, current_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze seasonal traffic patterns"""
        
        seasonal_analysis = {
            'current_season': self._identify_current_season(),
            'seasonal_expectations': {},
            'anomalies': [],
            'seasonal_opportunities': []
        }
        
        current_month = datetime.now().strftime('%B').lower()
        current_season = seasonal_analysis['current_season']
        
        # Analyze expected vs actual performance for current season
        for brand, df in current_data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            seasonal_keywords = self._identify_seasonal_keywords(df, current_season)
            
            seasonal_analysis['seasonal_expectations'][brand] = {
                'seasonal_keyword_count': len(seasonal_keywords),
                'seasonal_traffic_share': seasonal_keywords['Traffic (%)'].sum() if not seasonal_keywords.empty else 0,
                'expected_performance': self._get_seasonal_expectations(current_season),
                'performance_vs_expectation': 'meeting'  # Would calculate with historical data
            }
        
        return seasonal_analysis
    
    def _identify_current_season(self) -> str:
        """Identify current seasonal period"""
        
        current_month = datetime.now().month
        
        if current_month in [8, 9]:
            return 'back_to_school'
        elif current_month in [11, 12]:
            return 'holiday_shopping'
        elif current_month in [1, 2]:
            return 'new_year_planning'
        elif current_month in [3, 4]:
            return 'spring_refresh'
        elif current_month in [6, 7]:
            return 'summer_slowdown'
        else:
            return 'regular_period'
    
    def _identify_seasonal_keywords(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Identify keywords relevant to current season"""
        
        if 'Keyword' not in df.columns:
            return pd.DataFrame()
        
        seasonal_terms = {
            'back_to_school': ['school', 'student', 'college', 'university', 'academic'],
            'holiday_shopping': ['gift', 'holiday', 'christmas', 'black friday', 'cyber monday'],
            'new_year_planning': ['budget', 'planning', 'new year', 'resolution'],
            'spring_refresh': ['spring', 'refresh', 'upgrade', 'new'],
            'summer_slowdown': ['summer', 'vacation', 'travel']
        }
        
        terms = seasonal_terms.get(season, [])
        if not terms:
            return pd.DataFrame()
        
        pattern = '|'.join(terms)
        seasonal_keywords = df[df['Keyword'].str.contains(pattern, case=False, na=False)]
        
        return seasonal_keywords
    
    def _get_seasonal_expectations(self, season: str) -> Dict[str, str]:
        """Get expected performance for seasonal periods"""
        
        expectations = {
            'back_to_school': {'traffic': 'increase', 'competition': 'high'},
            'holiday_shopping': {'traffic': 'significant_increase', 'competition': 'very_high'},
            'new_year_planning': {'traffic': 'moderate_increase', 'competition': 'moderate'},
            'spring_refresh': {'traffic': 'moderate_increase', 'competition': 'moderate'},
            'summer_slowdown': {'traffic': 'decrease', 'competition': 'low'},
            'regular_period': {'traffic': 'stable', 'competition': 'moderate'}
        }
        
        return expectations.get(season, {'traffic': 'stable', 'competition': 'moderate'})
    
    def _analyze_algorithm_correlation(self, current_data: Dict[str, pd.DataFrame],
                                     historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze correlation with Google algorithm updates"""
        
        algorithm_analysis = {
            'recent_updates': [],
            'potential_correlations': [],
            'update_impact_assessment': {}
        }
        
        # Check for recent algorithm updates
        recent_updates = [
            update for update in self.google_updates
            if update['date'] >= datetime.now() - timedelta(days=60)
        ]
        
        algorithm_analysis['recent_updates'] = recent_updates
        
        if not historical_data or not recent_updates:
            return algorithm_analysis
        
        # Analyze potential correlations with traffic changes
        for update in recent_updates:
            update_date = update['date']
            update_impact = self._assess_update_impact(current_data, historical_data, update)
            
            if update_impact['correlation_strength'] > 0.3:
                algorithm_analysis['potential_correlations'].append({
                    'update_name': update['name'],
                    'update_date': update_date,
                    'correlation_strength': update_impact['correlation_strength'],
                    'affected_keywords': update_impact['affected_keywords'],
                    'impact_summary': update_impact['summary']
                })
            
            algorithm_analysis['update_impact_assessment'][update['name']] = update_impact
        
        return algorithm_analysis
    
    def _assess_update_impact(self, current_data: Dict[str, pd.DataFrame],
                            historical_data: Dict[str, pd.DataFrame],
                            update: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of a specific algorithm update"""
        
        impact_assessment = {
            'correlation_strength': 0,
            'affected_keywords': [],
            'summary': '',
            'confidence_level': 'low'
        }
        
        # This would be implemented with proper temporal analysis
        # For now, providing structure for update impact assessment
        
        affected_keywords = 0
        total_impact = 0
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords':
                continue
            
            historical_df = historical_data.get(brand, pd.DataFrame())
            if historical_df.empty:
                continue
            
            changes = self._calculate_traffic_changes(current_df, historical_df)
            
            # Check keywords that align with update impact areas
            for change in changes:
                if abs(change['change_percent']) > 20:  # Significant change threshold
                    if self._keyword_aligns_with_update(change['keyword'], update):
                        affected_keywords += 1
                        total_impact += abs(change['change_percent'])
                        impact_assessment['affected_keywords'].append({
                            'keyword': change['keyword'],
                            'brand': brand,
                            'impact': change['change_percent']
                        })
        
        if affected_keywords > 0:
            impact_assessment['correlation_strength'] = min(affected_keywords / 20, 1.0)
            impact_assessment['summary'] = f"Potential correlation detected with {affected_keywords} keywords affected"
            
            if impact_assessment['correlation_strength'] > 0.7:
                impact_assessment['confidence_level'] = 'high'
            elif impact_assessment['correlation_strength'] > 0.4:
                impact_assessment['confidence_level'] = 'medium'
        
        return impact_assessment
    
    def _keyword_aligns_with_update(self, keyword: str, update: Dict[str, Any]) -> bool:
        """Check if a keyword aligns with algorithm update impact areas"""
        
        keyword_lower = str(keyword).lower()
        impact_areas = update.get('impact_areas', [])
        
        alignment_patterns = {
            'content_quality': ['review', 'guide', 'how to', 'best'],
            'user_experience': ['fast', 'mobile', 'responsive'],
            'eeat': ['expert', 'review', 'professional'],
            'product_reviews': ['review', 'rating', 'comparison', 'vs'],
            'affiliate_sites': ['best', 'top', 'deal', 'price']
        }
        
        for area in impact_areas:
            patterns = alignment_patterns.get(area, [])
            if any(pattern in keyword_lower for pattern in patterns):
                return True
        
        return False
    
    def _analyze_competitor_correlation(self, current_data: Dict[str, pd.DataFrame],
                                      historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze correlations with competitor movements"""
        
        competitor_analysis = {
            'correlated_movements': [],
            'inverse_correlations': [],
            'market_share_shifts': {},
            'competitive_pressure': {}
        }
        
        if not historical_data:
            return competitor_analysis
        
        brands = [brand for brand in current_data.keys() if brand != 'gap_keywords']
        
        # Analyze cross-brand correlations
        for i, brand1 in enumerate(brands):
            for brand2 in brands[i+1:]:
                correlation = self._calculate_brand_correlation(
                    current_data, historical_data, brand1, brand2
                )
                
                if correlation['correlation_coefficient'] > 0.7:
                    competitor_analysis['correlated_movements'].append({
                        'brand1': brand1,
                        'brand2': brand2,
                        'correlation': correlation['correlation_coefficient'],
                        'pattern': 'positive_correlation'
                    })
                elif correlation['correlation_coefficient'] < -0.7:
                    competitor_analysis['inverse_correlations'].append({
                        'brand1': brand1,
                        'brand2': brand2,
                        'correlation': correlation['correlation_coefficient'],
                        'pattern': 'inverse_correlation'
                    })
        
        return competitor_analysis
    
    def _calculate_brand_correlation(self, current_data: Dict[str, pd.DataFrame],
                                   historical_data: Dict[str, pd.DataFrame],
                                   brand1: str, brand2: str) -> Dict[str, Any]:
        """Calculate actual correlation between two brands' performance using real data"""
        
        correlation_data = {
            'correlation_coefficient': 0,
            'shared_keywords': 0,
            'movement_patterns': [],
            'traffic_correlation': 0,
            'position_correlation': 0,
            'significance_level': 'not_significant'
        }
        
        # Get current data for both brands
        brand1_current = current_data.get(brand1, pd.DataFrame())
        brand2_current = current_data.get(brand2, pd.DataFrame())
        
        if brand1_current.empty or brand2_current.empty:
            return correlation_data
        
        # Get historical data for comparison
        brand1_historical = historical_data.get(brand1, pd.DataFrame())
        brand2_historical = historical_data.get(brand2, pd.DataFrame())
        
        if brand1_historical.empty or brand2_historical.empty:
            return correlation_data
        
        # Find shared keywords between brands
        if 'Keyword' not in brand1_current.columns or 'Keyword' not in brand2_current.columns:
            return correlation_data
        
        brand1_keywords = set(brand1_current['Keyword'].str.lower())
        brand2_keywords = set(brand2_current['Keyword'].str.lower())
        shared_keywords = brand1_keywords.intersection(brand2_keywords)
        
        correlation_data['shared_keywords'] = len(shared_keywords)
        
        if len(shared_keywords) < 5:  # Need minimum overlap for meaningful correlation
            return correlation_data
        
        # Calculate traffic and position correlations for shared keywords
        brand1_changes = self._calculate_traffic_changes(brand1_current, brand1_historical)
        brand2_changes = self._calculate_traffic_changes(brand2_current, brand2_historical)
        
        # Create dictionaries for easy lookup
        brand1_change_dict = {str(change['keyword']).lower(): change for change in brand1_changes}
        brand2_change_dict = {str(change['keyword']).lower(): change for change in brand2_changes}
        
        # Collect correlation data for shared keywords
        traffic_changes_1 = []
        traffic_changes_2 = []
        position_changes_1 = []
        position_changes_2 = []
        movement_patterns = []
        
        for keyword in shared_keywords:
            if keyword in brand1_change_dict and keyword in brand2_change_dict:
                change1 = brand1_change_dict[keyword]
                change2 = brand2_change_dict[keyword]
                
                traffic_changes_1.append(change1['change_percent'])
                traffic_changes_2.append(change2['change_percent'])
                position_changes_1.append(change1['position_change'])
                position_changes_2.append(change2['position_change'])
                
                # Analyze movement patterns
                pattern = self._analyze_keyword_movement_pattern(change1, change2)
                movement_patterns.append({
                    'keyword': keyword,
                    'brand1_traffic_change': change1['change_percent'],
                    'brand2_traffic_change': change2['change_percent'],
                    'brand1_position_change': change1['position_change'],
                    'brand2_position_change': change2['position_change'],
                    'pattern_type': pattern
                })
        
        # Calculate correlations if we have enough data
        if len(traffic_changes_1) >= 5:
            # Traffic correlation
            traffic_corr, traffic_p_val = self._safe_pearson_correlation(traffic_changes_1, traffic_changes_2)
            correlation_data['traffic_correlation'] = traffic_corr
            
            # Position correlation  
            position_corr, position_p_val = self._safe_pearson_correlation(position_changes_1, position_changes_2)
            correlation_data['position_correlation'] = position_corr
            
            # Overall correlation (average of absolute traffic and position)
            # We use absolute values here because strong inverse correlation is still a strong correlation
            correlation_data['correlation_coefficient'] = (
                abs(traffic_corr) + 
                abs(position_corr)
            ) / 2
            
            # Determine significance
            min_p_value = min(traffic_p_val, position_p_val)
            
            if min_p_value < 0.01:
                correlation_data['significance_level'] = 'highly_significant'
            elif min_p_value < 0.05:
                correlation_data['significance_level'] = 'significant'
            else:
                correlation_data['significance_level'] = 'not_significant'
        
        correlation_data['movement_patterns'] = movement_patterns
        
        return correlation_data

    def _analyze_keyword_movement_pattern(self, change1: Dict[str, Any], change2: Dict[str, Any]) -> str:
        """Analyze movement pattern between two brands for a keyword"""
        
        traffic1 = change1['change_percent']
        traffic2 = change2['change_percent']
        position1 = change1['position_change']
        position2 = change2['position_change']
        
        # Determine pattern type
        if traffic1 > 10 and traffic2 > 10:
            return 'both_gaining_traffic'
        elif traffic1 < -10 and traffic2 < -10:
            return 'both_losing_traffic'
        elif (traffic1 > 10 and traffic2 < -10) or (traffic1 < -10 and traffic2 > 10):
            return 'inverse_traffic_movement'
        elif (position1 < -5 and position2 > 5) or (position1 > 5 and position2 < -5): # Lower position number is better
            return 'position_swap' # e.g. brand1 improves significantly, brand2 declines significantly
        else:
            return 'mixed_or_stable'
    
    def _generate_recovery_recommendations(self, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for traffic recovery"""
        
        recommendations = []
        
        # Critical drops recommendations
        critical_drops = diagnosis.get('traffic_drops', {}).get('critical_drops', [])
        if critical_drops:
            recommendations.append({
                'priority': 'immediate',
                'category': 'traffic_recovery',
                'action': 'Emergency traffic recovery plan for critical drops',
                'details': f'Address {len(critical_drops)} keywords with >50% traffic loss',
                'timeline': '0-7 days',
                'resources': ['seo_specialist', 'technical_team', 'content_team']
            })
        
        # Algorithm correlation recommendations
        correlations = diagnosis.get('algorithm_correlation', {}).get('potential_correlations', [])
        if correlations:
            for correlation in correlations[:2]:  # Top 2 correlations
                recommendations.append({
                    'priority': 'high',
                    'category': 'algorithm_response',
                    'action': f'Adapt to {correlation["update_name"]} impact',
                    'details': f'Optimize {len(correlation["affected_keywords"])} affected keywords',
                    'timeline': '1-4 weeks',
                    'resources': ['seo_specialist', 'content_team']
                })
        
                # Seasonal optimization recommendations
        seasonal_data = diagnosis.get('seasonal_analysis', {})
        current_season = seasonal_data.get('current_season', 'regular_period')
        
        if current_season != 'regular_period':
            seasonal_expectations = seasonal_data.get('seasonal_expectations', {})
            for brand, expectations in seasonal_expectations.items():
                performance_vs_expectation = expectations.get('performance_vs_expectation', 'meeting')
                if performance_vs_expectation == 'below':
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'seasonal_optimization',
                        'action': f'Optimize seasonal content for {current_season}',
                        'details': f'Performance below expectations for {brand}',
                        'timeline': '1-2 weeks',
                        'resources': ['content_team', 'seo_specialist']
                    })
        
        return recommendations
    
    def _setup_alert_triggers(self, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Setup automated alert triggers for traffic monitoring"""
        
        alerts = []
        
        # Recent changes alerts
        recent_changes = diagnosis.get('recent_changes', {})
        significant_drops = recent_changes.get('significant_drops', [])
        
        if len(significant_drops) > 5:
            alerts.append({
                'alert_type': 'traffic_drop_cluster',
                'severity': 'high',
                'trigger_condition': f'{len(significant_drops)} keywords with >20% traffic drop',
                'monitoring_frequency': 'daily',
                'notification_channels': ['email', 'slack', 'dashboard']
            })
        
        # Competitive movement alerts
        competitive_threats = diagnosis.get('competitor_correlation', {}).get('inverse_correlations', [])
        if competitive_threats:
            alerts.append({
                'alert_type': 'competitive_threat',
                'severity': 'medium',
                'trigger_condition': 'Inverse correlation detected with competitors',
                'monitoring_frequency': 'weekly',
                'notification_channels': ['email', 'dashboard']
            })
        
        # Algorithm correlation alerts
        algorithm_correlations = diagnosis.get('algorithm_correlation', {}).get('potential_correlations', [])
        if algorithm_correlations:
            alerts.append({
                'alert_type': 'algorithm_impact',
                'severity': 'high',
                'trigger_condition': 'High correlation with recent algorithm update',
                'monitoring_frequency': 'daily',
                'notification_channels': ['email', 'slack', 'sms']
            })
        
        # General monitoring alerts
        alerts.extend([
            {
                'alert_type': 'position_volatility',
                'severity': 'medium',
                'trigger_condition': '>10 keywords with >5 position changes in 24 hours',
                'monitoring_frequency': 'daily',
                'notification_channels': ['dashboard']
            },
            {
                'alert_type': 'traffic_anomaly',
                'severity': 'high',
                'trigger_condition': '>30% traffic change in any keyword with >1% traffic share',
                'monitoring_frequency': '4_hours',
                'notification_channels': ['email', 'slack']
            },
            {
                'alert_type': 'serp_feature_loss',
                'severity': 'medium',
                'trigger_condition': 'Loss of featured snippet or rich result',
                'monitoring_frequency': 'daily',
                'notification_channels': ['email', 'dashboard']
            }
        ])
        
        return alerts
    
    def get_traffic_drop_keywords(self, domain: str, date: str, threshold: float = 1.0) -> Dict[str, Any]:
        """Get keywords with significant traffic drops (Implementation for query 1)"""
        
        # This would be implemented with actual historical data comparison
        # For now, providing structure
        
        dropping_keywords = []
        
        return {
            'dropping_keywords': dropping_keywords[:20],
            'total_impact': 0,
            'recovery_timeline': {
                'immediate': [],
                'short_term': [],
                'long_term': []
            },
            'immediate_actions': [
                'Content audit and optimization',
                'Technical SEO review', 
                'Backlink analysis',
                'SERP feature optimization'
            ]
        }
    
    def compare_metrics(self, domain: str, competitor: str, metric: str, timeframe: str) -> Dict[str, Any]:
        """Compare specific metrics between domain and competitor (Implementation for query 2)"""
        
        comparison = {
            'metric_trends': {
                domain: {'current': 0, 'change': 0, 'trend': 'stable'},
                competitor: {'current': 0, 'change': 0, 'trend': 'stable'}
            },
            'performance_gaps': {
                'absolute_gap': 0,
                'percentage_gap': 0,
                'gap_trend': 'widening'
            },
            'competitive_advantages': {
                domain: [],
                competitor: []
            },
            'action_recommendations': []
        }
        
        return comparison
    
    def analyze_readability_ranking_mismatch(self, readability_threshold: int = 50) -> Dict[str, Any]:
        """Identify high-ranking pages with poor readability (Implementation for query 3)"""
        
        analysis = {
            'mismatched_pages': [],
            'optimization_priority': {},
            'content_improvement_plan': []
        }
        
        return analysis

    def _safe_pearson_correlation(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Safely calculate Pearson correlation with constant array handling"""
        
        # Check inputs
        if not x or not y or len(x) != len(y) or len(x) < 2:
            return 0.0, 1.0
        
        # Check for constant arrays
        if not self._has_variance(x) or not self._has_variance(y):
            return 0.0, 1.0 # Return 0 correlation, 1 p-value (not significant)
        
        # Check for NaN/infinite values and clean data
        try:
            # Create pairs and filter out any pair containing NaN or Inf
            pairs = [(xi, yi) for xi, yi in zip(x, y) if not (np.isnan(xi) or np.isinf(xi) or np.isnan(yi) or np.isinf(yi))]
            
            if len(pairs) < 2: # Not enough data points after cleaning
                return 0.0, 1.0
            
            x_clean, y_clean = zip(*pairs) # Unzip to cleaned lists
            
            correlation, p_value = pearsonr(list(x_clean), list(y_clean))
            
            return correlation if not np.isnan(correlation) else 0.0, p_value if not np.isnan(p_value) else 1.0
        except Exception: # pylint: disable=broad-except
            return 0.0, 1.0
    def _has_variance(self, array: List[float]) -> bool:
        """Check if array has variance (not constant)"""
        
        if not array or len(array) < 2:
            return False
        
        # Check if all values are the same (constant array)
        unique_values = len(set(array))
        
        if unique_values <= 1:
            return False
        
        # Additional check using numpy variance
        variance = np.var(array) # No try-except needed here as np.var handles empty/single element lists gracefully (returns nan or 0)
        return variance > 1e-10  # Small threshold to handle floating point precision

    def _assess_update_impact(self, current_data: Dict[str, pd.DataFrame],
                            historical_data: Dict[str, pd.DataFrame],
                            update: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the actual impact of a specific algorithm update using real data"""
        
        impact_assessment = {
            'correlation_strength': 0,
            'affected_keywords': [],
            'summary': '',
            'confidence_level': 'low',
            'total_keywords_analyzed': 0,
            'significant_changes': 0,
            'aligned_keywords': 0
        }
        
        # update_date = update['date'] # Not directly used in this logic, but good for context
        impact_areas = update.get('impact_areas', [])
        
        total_keywords_analyzed = 0
        significant_changes = 0
        aligned_keywords = 0
        total_impact_score = 0
        
        for brand, current_df in current_data.items():
            if brand == 'gap_keywords':
                continue
            
            historical_df = historical_data.get(brand, pd.DataFrame())
            if historical_df.empty:
                continue
            
            # Calculate changes for this brand
            changes = self._calculate_traffic_changes(current_df, historical_df)
            total_keywords_analyzed += len(changes)
            
            # Analyze each keyword change
            for change in changes:
                keyword = change['keyword']
                change_percent = change['change_percent']
                
                # Check if change is significant (>20% change)
                if abs(change_percent) > 20:
                    significant_changes += 1
                    
                    # Check if keyword aligns with update impact areas
                    if self._keyword_aligns_with_update(keyword, update):
                        aligned_keywords += 1
                        
                        # Calculate impact score for this keyword
                        impact_score = self._calculate_keyword_impact_score(
                            change, update, impact_areas
                        )
                        total_impact_score += impact_score
                        
                        impact_assessment['affected_keywords'].append({
                            'keyword': keyword,
                            'brand': brand,
                            'traffic_change_percent': change_percent,
                            'position_change': change['position_change'],
                            'impact_score': impact_score,
                            'alignment_reason': self._get_alignment_reason(keyword, update)
                        })
        
        # Calculate correlation strength
        impact_assessment['total_keywords_analyzed'] = total_keywords_analyzed
        impact_assessment['significant_changes'] = significant_changes
        impact_assessment['aligned_keywords'] = aligned_keywords
        
        if total_keywords_analyzed > 0 and significant_changes > 0 : # Avoid division by zero
            # Correlation strength based on proportion of aligned keywords with significant changes
            alignment_ratio = aligned_keywords / significant_changes # Of those that changed significantly, how many aligned?
            significance_ratio = significant_changes / total_keywords_analyzed # What proportion of all KWs changed significantly?
            
            # Combined correlation strength
            impact_assessment['correlation_strength'] = min(
                (alignment_ratio * 0.7 + significance_ratio * 0.3), 1.0
            )
            
            # Determine confidence level
            if impact_assessment['correlation_strength'] > 0.7 and aligned_keywords >= 5:
                impact_assessment['confidence_level'] = 'high'
            elif impact_assessment['correlation_strength'] > 0.4 and aligned_keywords >= 3:
                impact_assessment['confidence_level'] = 'medium'
            else:
                impact_assessment['confidence_level'] = 'low'
            
            # Generate summary
            if aligned_keywords > 0:
                avg_impact = total_impact_score / aligned_keywords
                impact_assessment['summary'] = (
                    f"Detected {aligned_keywords} keywords aligned with {update['name']} "
                    f"showing significant changes (avg impact score: {avg_impact:.1f}). "
                    f"Correlation strength: {impact_assessment['correlation_strength']:.3f}"
                )
            else:
                impact_assessment['summary'] = f"No significant correlation detected with {update['name']} based on keyword alignment and changes."
        else:
            impact_assessment['summary'] = f"Not enough data or significant changes to assess correlation with {update['name']}"
        
        return impact_assessment

    def _calculate_keyword_impact_score(self, change: Dict[str, Any], 
                                      update: Dict[str, Any], # pylint: disable=unused-argument
                                      impact_areas: List[str]) -> float:
        """Calculate impact score for a keyword based on change magnitude and alignment"""
        
        traffic_change = abs(change['change_percent'])
        position_change = abs(change['position_change'])
        
        # Base score from magnitude of change
        traffic_score = min(traffic_change / 10, 10)  # 0-10 scale, 100% change = 10 points
        position_score = min(position_change / 5, 10)  # 0-10 scale, 50 pos change = 10 points
        
        # Alignment bonus
        alignment_bonus = 0
        keyword = str(change['keyword']).lower()
        
        alignment_patterns = {
            'content_quality': ['review', 'guide', 'how to', 'best', 'tutorial'],
            'user_experience': ['fast', 'mobile', 'responsive', 'speed'],
            'eeat': ['expert', 'review', 'professional', 'authority'],
            'product_reviews': ['review', 'rating', 'comparison', 'vs', 'test'],
            'affiliate_sites': ['best', 'top', 'deal', 'price', 'buy']
        }
        
        for area in impact_areas:
            patterns = alignment_patterns.get(area, [])
            matches = sum(1 for pattern in patterns if pattern in keyword)
            alignment_bonus += matches * 2  # 2 points per matching pattern in an aligned area
        
        total_score = traffic_score + position_score + alignment_bonus
        return min(total_score, 25)  # Cap at 25 (e.g. 10 for traffic, 10 for pos, 5 for alignment)

    def _get_alignment_reason(self, keyword: str, update: Dict[str, Any]) -> str:
        """Get reason why keyword aligns with algorithm update"""
        
        keyword_lower = str(keyword).lower()
        impact_areas = update.get('impact_areas', [])
        reasons = []
        
        # Using the same patterns as _keyword_aligns_with_update for consistency
        alignment_patterns = {
            'content_quality': ['review', 'guide', 'how to', 'best'],
            'user_experience': ['fast', 'mobile', 'responsive'],
            'eeat': ['expert', 'review', 'professional'],
            'product_reviews': ['review', 'rating', 'comparison', 'vs'],
            'affiliate_sites': ['best', 'top', 'deal', 'price']
        }
        
        for area in impact_areas:
            patterns = alignment_patterns.get(area, [])
            matched_patterns = [pattern for pattern in patterns if pattern in keyword_lower]
            if matched_patterns:
                reasons.append(f"{area} (terms: {', '.join(matched_patterns)})")
        
        return '; '.join(reasons) if reasons else 'general_alignment_to_update_focus'

    def _analyze_seasonal_patterns(self, current_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze seasonal traffic patterns using actual performance data"""
        
        seasonal_analysis = {
            'current_season': self._identify_current_season(),
            'seasonal_expectations_vs_actual': {}, # Renamed for clarity
            'seasonal_anomalies': [],
            'seasonal_opportunities': []
        }
        
        current_season = seasonal_analysis['current_season']
        
        # Analyze expected vs actual performance for current season
        for brand, df in current_data.items():
            if brand == 'gap_keywords' or df.empty:
                continue
            
            seasonal_keywords_df = self._identify_seasonal_keywords(df, current_season)
            seasonal_performance = self._calculate_seasonal_performance(seasonal_keywords_df, df, current_season)
            
            seasonal_analysis['seasonal_expectations_vs_actual'][brand] = {
                'seasonal_keyword_count': len(seasonal_keywords_df),
                'seasonal_traffic_share_on_brand': seasonal_keywords_df['Traffic (%)'].sum() if not seasonal_keywords_df.empty else 0,
                'expected_performance_profile': self._get_seasonal_expectations(current_season),
                'actual_performance_vs_expectation': seasonal_performance['vs_expectation'],
                'seasonal_performance_metrics': seasonal_performance # Contains score, efficiency, etc.
            }
            
            # Identify seasonal anomalies for this brand's seasonal keywords
            anomalies = self._identify_seasonal_anomalies(seasonal_keywords_df, current_season, brand)
            seasonal_analysis['seasonal_anomalies'].extend(anomalies)
            
            # Find seasonal opportunities for this brand
            opportunities = self._find_seasonal_opportunities(df, current_season, brand)
            seasonal_analysis['seasonal_opportunities'].extend(opportunities)
        
        return seasonal_analysis

    def _calculate_seasonal_performance(self, seasonal_keywords_df: pd.DataFrame, 
                                      all_keywords_df: pd.DataFrame, season: str) -> Dict[str, Any]:
        """Calculate actual seasonal performance vs expectations"""
        
        performance = {
            'vs_expectation': 'meeting', # Default
            'performance_score': 0.0,    # Overall score for seasonal performance
            'seasonal_keyword_traffic_contribution': 0.0, # % of brand's total traffic from these seasonal KWs
            'avg_seasonal_keyword_position': 100.0, # Default to worst position
            'seasonal_keyword_top10_rate': 0.0, # % of seasonal KWs in top 10
            'seasonal_keyword_efficiency': 0.0 # Traffic per seasonal keyword
        }
        
        if seasonal_keywords_df.empty:
            performance['vs_expectation'] = 'no_seasonal_keyword_presence'
            # If the season expects an increase, 'no_seasonal_keyword_presence' is 'below' expectations.
            # If season expects decrease or stable, it might be 'meeting' or 'n/a'.
            # Let's refine this based on expected trend.
            expected_trend = self._get_seasonal_expectations(season).get('traffic', 'stable')
            if 'increase' in expected_trend:
                 performance['vs_expectation'] = 'below_due_to_no_presence'
            return performance
        
        seasonal_traffic_sum = seasonal_keywords_df['Traffic (%)'].sum()
        num_seasonal_keywords = len(seasonal_keywords_df)
        
        performance['avg_seasonal_keyword_position'] = seasonal_keywords_df['Position'].mean()
        performance['seasonal_keyword_top10_rate'] = \
            (len(seasonal_keywords_df[seasonal_keywords_df['Position'] <= 10]) / num_seasonal_keywords) * 100
        performance['seasonal_keyword_efficiency'] = seasonal_traffic_sum / num_seasonal_keywords

        total_brand_traffic = all_keywords_df['Traffic (%)'].sum()
        if total_brand_traffic > 0:
            performance['seasonal_keyword_traffic_contribution'] = (seasonal_traffic_sum / total_brand_traffic) * 100
        
        # Scoring logic (example, can be refined)
        # Max score: 100
        score = 0
        # Top 10 rate (max 40 points)
        score += min(performance['seasonal_keyword_top10_rate'] * 0.4, 40)
        # Avg position (max 30 points, lower is better, 1 is best)
        # Score = 30 * (1 - (avg_pos - 1) / 99)) for avg_pos in [1, 100]
        score += max(0, 30 * (1 - (performance['avg_seasonal_keyword_position'] - 1) / 99.0))
        # Traffic contribution (max 30 points) - this depends on how important seasonal is for the brand
        # Let's say >10% contribution is good for 30 points.
        score += min(performance['seasonal_keyword_traffic_contribution'] * 3, 30) 
        
        performance['performance_score'] = round(score, 2)
        
        # Determine vs_expectation based on score and seasonal context
        expected_performance_profile = self._get_seasonal_expectations(season)
        expected_traffic_trend = expected_performance_profile.get('traffic', 'stable')
        
        # Define thresholds for meeting/exceeding/below based on expected trend
        # These are illustrative and can be tuned
        thresholds = {'exceeding': 70, 'meeting': 50} # Default for stable/increase
        if expected_traffic_trend == 'significant_increase':
            thresholds = {'exceeding': 75, 'meeting': 55}
        elif expected_traffic_trend == 'decrease': # If decrease is expected, good performance is maintaining or slow decline
            thresholds = {'exceeding': 60, 'meeting': 40} # Lower bar for "exceeding" if slowdown expected

        if performance['performance_score'] >= thresholds['exceeding']:
            performance['vs_expectation'] = 'exceeding_expectations'
        elif performance['performance_score'] >= thresholds['meeting']:
            performance['vs_expectation'] = 'meeting_expectations'
        else:
            performance['vs_expectation'] = 'below_expectations'
            
        return performance

    def _identify_seasonal_anomalies(self, seasonal_keywords_df: pd.DataFrame, 
                                   season: str, brand: str) -> List[Dict[str, Any]]:
        """Identify seasonal performance anomalies for a specific brand's seasonal keywords"""
        
        anomalies = []
        if seasonal_keywords_df.empty:
            return anomalies
        
        # Anomaly 1: High-ranking seasonal keyword with very low traffic
        # (Could indicate SERP feature changes, poor snippet, or low search volume despite ranking)
        high_rank_low_traffic_df = seasonal_keywords_df[
            (seasonal_keywords_df['Position'] <= 5) & (seasonal_keywords_df['Traffic (%)'] < 0.1)
        ]
        for _, row in high_rank_low_traffic_df.iterrows():
            anomalies.append({
                'brand': brand,
                'type': 'high_ranking_low_seasonal_traffic',
                'keyword': row['Keyword'],
                'position': row['Position'],
                'traffic_percent': row['Traffic (%)'],
                'season': season,
                'severity': 'medium',
                'details': 'Keyword ranks well for seasonal term but gets minimal traffic. Investigate SERP, CTR.'
            })
            
        # Anomaly 2: Seasonal keyword with poor ranking (missed opportunity)
        # Only flag if it has some (even if low) traffic, indicating search volume exists
        poor_rank_seasonal_df = seasonal_keywords_df[
            (seasonal_keywords_df['Position'] > 20) & (seasonal_keywords_df['Traffic (%)'] > 0.01) # Position >20, but some traffic
        ]
        for _, row in poor_rank_seasonal_df.iterrows():
            anomalies.append({
                'brand': brand,
                'type': 'poor_ranking_for_relevant_seasonal_keyword',
                'keyword': row['Keyword'],
                'position': row['Position'],
                'traffic_percent': row['Traffic (%)'],
                'season': season,
                'severity': 'high',
                'details': 'Relevant seasonal keyword ranks poorly. Significant missed opportunity.'
            })
        return anomalies

    def _find_seasonal_opportunities(self, all_keywords_df: pd.DataFrame, 
                                   season: str, brand: str) -> List[Dict[str, Any]]:
        """Find seasonal optimization opportunities for a brand"""
        
        opportunities = []
        if all_keywords_df.empty or 'Keyword' not in all_keywords_df.columns:
            return opportunities
        
        # Terms defining the season (should match _identify_seasonal_keywords closely or use its output)
        seasonal_terms_map = {
            'back_to_school': ['school', 'student', 'college', 'university', 'academic', 'education'],
            'holiday_shopping': ['gift', 'holiday', 'christmas', 'black friday', 'cyber monday', 'deals', 'xmas'],
            'new_year_planning': ['budget', 'planning', 'new year', 'resolution', 'goals'],
            'spring_refresh': ['spring', 'refresh', 'upgrade', 'new', 'clean', 'garden'],
            'summer_slowdown': ['summer', 'vacation', 'travel', 'holiday', 'trip'] # 'holiday' can be ambiguous
        }
        
        current_season_terms = seasonal_terms_map.get(season, [])
        if not current_season_terms:
            return opportunities
            
        # Identify keywords that contain seasonal terms but are NOT currently high-performing seasonal keywords
        # This means they are either not in `seasonal_keywords_df` or are in it but performing poorly.
        
        # Create a regex pattern for current seasonal terms
        # Example: r'\b(school|student|college)\b' for case-insensitive whole word match
        pattern = r'\b(' + '|'.join(current_season_terms) + r')\b'
        
        potential_seasonal_df = all_keywords_df[
            all_keywords_df['Keyword'].str.contains(pattern, case=False, na=False)
        ]
        
        for _, row in potential_seasonal_df.iterrows():
            # Opportunity if:
            # 1. It's a seasonal term.
            # 2. Its current position is not great (e.g., > 10 or even > 5 for competitive terms).
            # 3. It has some traffic, indicating search volume (or use a separate volume metric if available).
            if row['Position'] > 10 and row['Traffic (%)'] > 0.05 : # Example thresholds
                opportunity_score = self._calculate_seasonal_opportunity_score(row, season)
                if opportunity_score > 40: # Only add if score is reasonably high
                    opportunities.append({
                        'brand': brand,
                        'keyword': row['Keyword'],
                        'current_position': row['Position'],
                        'current_traffic_percent': row['Traffic (%)'],
                        'season': season,
                        'opportunity_score': round(opportunity_score, 2),
                        'optimization_type': 'seasonal_content_or_ranking_improvement',
                        'details': f"Potential to improve ranking/traffic for this seasonal keyword. Score: {opportunity_score:.1f}"
                    })
        
        # Sort by opportunity score, highest first
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        return opportunities[:10] # Return top 10

    def _calculate_seasonal_opportunity_score(self, keyword_row: pd.Series, season: str) -> float:
        """Calculate opportunity score for seasonal optimization of a specific keyword"""
        
        position = keyword_row['Position']
        # current_traffic = keyword_row['Traffic (%)'] # Not directly used in this score, but could be
        
        # Score based on current position (higher score for worse positions, indicating more room for improvement)
        # Max 50 points for position. If pos=100, score=50. If pos=1, score=0.
        position_improvement_potential = max(0, (position - 1) / 99.0 * 50) if position > 1 else 0
        
        # Score based on seasonal relevance (how strongly does it match the season?)
        # Max 50 points for relevance.
        keyword_lower = str(keyword_row['Keyword']).lower()
        seasonal_relevance_score = 0
        
        seasonal_terms_map = { # Duplicated for direct use, consider centralizing
            'back_to_school': ['school', 'student', 'college', 'university', 'academic'],
            'holiday_shopping': ['gift', 'holiday', 'christmas', 'deals', 'xmas'],
            'new_year_planning': ['budget', 'planning', 'new year', 'resolution'],
            'spring_refresh': ['spring', 'refresh', 'upgrade', 'new'],
            'summer_slowdown': ['summer', 'vacation', 'travel', 'trip']
        }
        current_season_terms = seasonal_terms_map.get(season, [])
        
        matches = 0
        for term in current_season_terms:
            if term in keyword_lower:
                matches +=1
        # Give more points for more matches, up to a cap.
        # E.g., 1 match = 20 pts, 2 matches = 35 pts, 3+ matches = 50 pts
        if matches == 1: seasonal_relevance_score = 20
        elif matches == 2: seasonal_relevance_score = 35
        elif matches >=3: seasonal_relevance_score = 50
            
        total_score = position_improvement_potential + seasonal_relevance_score
        return min(total_score, 100) # Ensure score is capped at 100
